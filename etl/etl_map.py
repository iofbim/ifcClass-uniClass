import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
import psycopg
from psycopg.rows import dict_row
from rapidfuzz import fuzz
import yaml


@dataclass
class Settings:
    db_url: str
    ifc_json: Path
    ifc_schema_version: str
    # Whether to expand IFC text with parent-chain context
    ifc_use_parent_context: bool
    uniclass_revision: str
    autodetect_uniclass_revision: bool
    enforce_monotonic_revision: bool
    uniclass_csvs: Dict[str, Path]
    uniclass_xlsx_dir: Optional[Path]
    output_dir: Path
    viewer_json: Path
    top_k: int
    auto_accept: float
    review_threshold: float
    synonyms: List[List[str]]
    # Matching direction: 'ifc_to_uniclass' (default) or 'uniclass_to_ifc'
    match_direction: str
    embed_model: str
    embed_endpoint: str
    embed_batch_size: int
    embed_timeout_s: float
    embedding_weight: float
    embedding_top_k: int


def _load_dotenv(dotenv_path: Optional[Path] = None) -> None:
    """Best-effort .env loader to populate os.environ without extra deps.

    - Reads KEY=VALUE lines, ignoring comments and blanks.
    - Does not handle advanced quoting/escaping; adequate for simple secrets.
    - Only sets keys that are not already in the environment.
    """
    try:
        p = dotenv_path or Path.cwd() / ".env"
        if not p.exists():
            # Try repo root relative to this file
            p = Path(__file__).resolve().parent.parent / ".env"
            if not p.exists():
                return
        for raw in p.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            key = k.strip()
            val = v.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val
    except Exception:
        # Non-fatal: config can still come from YAML
        pass


def load_settings(path: Path) -> Settings:
    # Load environment file if present, then allow env to override YAML
    _load_dotenv()
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    syns = cfg.get("matching", {}).get("synonyms", []) or []
    # Prefer DATABASE_URL if provided; else compose from POSTGRES_*; else YAML
    env_db_url = os.getenv("DATABASE_URL")
    if not env_db_url:
        host = os.getenv("POSTGRES_HOST", "localhost")
        port = os.getenv("POSTGRES_PORT", "5432")
        user = os.getenv("POSTGRES_USER")
        pwd = os.getenv("POSTGRES_PASSWORD")
        db = os.getenv("POSTGRES_DB")
        if user and pwd and db:
            env_db_url = f"postgresql://{user}:{pwd}@{host}:{port}/{db}"
    return Settings(
        db_url=env_db_url or cfg["db_url"],
        ifc_json=Path(cfg["ifc"]["json_path"]).expanduser(),
        ifc_schema_version=cfg["ifc"]["schema_version"],
        ifc_use_parent_context=bool(cfg.get("ifc", {}).get("use_parent_context", True)),
        uniclass_revision=str(cfg["uniclass"]["revision"]),
        autodetect_uniclass_revision=bool(cfg["uniclass"].get("autodetect_revision", True)),
        enforce_monotonic_revision=bool(cfg["uniclass"].get("enforce_monotonic_revision", True)),
        uniclass_csvs={k: Path(v) for k, v in (cfg["uniclass"].get("csv_paths", {}) or {}).items()},
        uniclass_xlsx_dir=Path(cfg["uniclass"].get("xlsx_dir")).expanduser() if cfg["uniclass"].get("xlsx_dir") else None,
        output_dir=Path(cfg["output"]["dir"]).expanduser(),
        viewer_json=Path(cfg["output"]["viewer_json"]).expanduser(),
        top_k=int(cfg["matching"].get("top_k", 5)),
        auto_accept=float(cfg["matching"].get("auto_accept_threshold", 0.85)),
        review_threshold=float(cfg["matching"].get("review_threshold", 0.6)),
        synonyms=syns,
        match_direction=str((cfg.get("matching", {}) or {}).get("direction", "ifc_to_uniclass")),
        embed_model=str((cfg.get("embedding", {}) or {}).get("model", os.getenv("EMBED_MODEL", "nomic-embed-text"))),
        embed_endpoint=str((cfg.get("embedding", {}) or {}).get("endpoint", os.getenv("EMBED_ENDPOINT", "http://localhost:11434"))),
        embed_batch_size=int((cfg.get("embedding", {}) or {}).get("batch_size", os.getenv("EMBED_BATCH_SIZE", 16))),
        embed_timeout_s=float((cfg.get("embedding", {}) or {}).get("timeout_s", os.getenv("EMBED_TIMEOUT_S", 30))),
        embedding_weight=float((cfg.get("matching", {}) or {}).get("embedding_weight", os.getenv("EMBEDDING_WEIGHT", 0.4))),
        embedding_top_k=int((cfg.get("matching", {}) or {}).get("embedding_top_k", os.getenv("EMBEDDING_TOP_K", 50))),
    )


def _chunked(seq: List, n: int) -> List[List]:
    return [seq[i:i + n] for i in range(0, len(seq), max(1, n))]


def _ollama_embed(text: str, model: str, endpoint: str, timeout_s: float) -> Optional[List[float]]:
    try:
        import requests
    except Exception:
        raise RuntimeError("requests is required for --embed; install via `pip install requests`")
    url = endpoint.rstrip("/") + "/api/embeddings"
    try:
        resp = requests.post(url, json={"model": model, "prompt": text}, timeout=timeout_s)
        resp.raise_for_status()
        data = resp.json()
        vec = data.get("embedding") or data.get("data", {}).get("embedding")
        if not isinstance(vec, list):
            raise ValueError("Unexpected embeddings response format")
        return [float(x) for x in vec]
    except Exception as e:
        print(f"[embed] error from Ollama for text len={len(text)}: {e}")
        return None


def _ensure_vector_literal(vec: List[float], expected_dim: Optional[int] = None) -> str:
    if expected_dim is not None and len(vec) != expected_dim:
        raise ValueError(f"Embedding dim {len(vec)} != expected {expected_dim}")
    # pgvector accepts '[v1,v2,...]'
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"


def _existing_embedding_ids(conn: psycopg.Connection, entity_type: str) -> set:
    with conn.cursor() as cur:
        cur.execute("SELECT entity_id FROM text_embedding WHERE entity_type=%s", (entity_type,))
        rows = cur.fetchall()
    return set((r[0] if not isinstance(r, dict) else r["entity_id"]) for r in rows)


def generate_and_store_embeddings(
    conn: psycopg.Connection,
    entity_type: str,
    items: List[Tuple[str, str]],  # (entity_id, text)
    model: str,
    endpoint: str,
    batch_size: int,
    timeout_s: float,
    expected_dim: Optional[int] = 768,
) -> int:
    # Skip ones that already exist
    existing = _existing_embedding_ids(conn, entity_type)
    todo = [(eid, txt) for (eid, txt) in items if eid not in existing and (txt or "").strip()]
    inserted = 0
    if not todo:
        return 0
    for chunk in _chunked(todo, batch_size):
        rows = []
        for eid, text in chunk:
            vec = _ollama_embed(text, model=model, endpoint=endpoint, timeout_s=timeout_s)
            if vec is None:
                continue
            lit = _ensure_vector_literal(vec, expected_dim=expected_dim)
            rows.append((entity_type, eid, lit))
        if not rows:
            continue
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO text_embedding (entity_type, entity_id, embedding)
                VALUES (%s, %s, %s::vector)
                ON CONFLICT (entity_type, entity_id) DO UPDATE SET embedding = EXCLUDED.embedding
                """,
                rows,
            )
        inserted += len(rows)
    return inserted


def _parse_revision_token(s: str) -> Tuple[str, Tuple[int, int]]:
    """Parse a revision string into a comparable tuple.
    Returns (scheme, (a,b)). Supported:
      - 'vA_B' -> ("vab", (A,B))
      - 'YYYY-MM' -> ("ym", (YYYY, MM))
    Raises ValueError if unparseable.
    """
    s = (s or "").strip()
    import re
    m = re.search(r"_v(\d+)_(\d+)", s)
    if m:
        return ("vab", (int(m.group(1)), int(m.group(2))))
    m2 = re.fullmatch(r"v?(\d+)[_\.](\d+)", s)
    if m2:
        return ("vab", (int(m2.group(1)), int(m2.group(2))))
    m3 = re.fullmatch(r"(\d{4})[-/](\d{1,2})", s)
    if m3:
        return ("ym", (int(m3.group(1)), int(m3.group(2))))
    raise ValueError(f"Unrecognized revision format: {s}")


def _detect_uniclass_revision_from_paths(paths: List[Path]) -> Optional[str]:
    import re
    revs = []
    for p in paths:
        name = p.name
        m = re.search(r"_v(\d+)_(\d+)", name)
        if m:
            revs.append(f"v{m.group(1)}_{m.group(2)}")
    if not revs:
        return None
    # Pick the max by parsed tuple
    def keyfn(r: str):
        try:
            _, t = _parse_revision_token(r)
            return t
        except Exception:
            return (0, 0)
    return sorted(revs, key=keyfn)[-1]


def _get_current_max_uniclass_revision(conn: psycopg.Connection) -> Optional[str]:
    # Prefer history table; fallback to live table
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT revision FROM uniclass_item_revision")
            revs = [r[0] if isinstance(r, tuple) else r for r in cur.fetchall()]
            if not revs:
                cur.execute("SELECT DISTINCT revision FROM uniclass_item")
                revs = [r[0] if isinstance(r, tuple) else r for r in cur.fetchall()]
    except Exception:
        return None
    if not revs:
        return None
    # Group by scheme and select max per scheme; return the max of the dominant scheme
    parsed = []
    for r in revs:
        try:
            scheme, t = _parse_revision_token(str(r))
            parsed.append((scheme, t, str(r)))
        except Exception:
            continue
    if not parsed:
        return None
    # Choose scheme with most entries, then max in that scheme
    from collections import Counter
    scheme = Counter(x[0] for x in parsed).most_common(1)[0][0]
    cands = [x for x in parsed if x[0] == scheme]
    cands.sort(key=lambda x: x[1])
    return cands[-1][2]


def ensure_output_dirs(s: Settings):
    s.output_dir.mkdir(parents=True, exist_ok=True)
    s.viewer_json.parent.mkdir(parents=True, exist_ok=True)


def read_ifc_json(path: Path, schema_version: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rows = []
    parent_map: Dict[str, Optional[str]] = {}
    label_map: Dict[str, str] = {}
    defn_map: Dict[str, str] = {}
    # Support structure: { "classes": { "IfcWall": { ... } } }
    classes = data.get("classes") if isinstance(data, dict) else None
    if isinstance(classes, dict):
        for ifc_id, item in classes.items():
            label = item.get("description") or item.get("name") or ifc_id
            desc = item.get("definition") or item.get("description") or ""
            parent = item.get("parent")
            parent_map[str(ifc_id)] = str(parent) if parent else None
            label_map[str(ifc_id)] = str(label) if label is not None else str(ifc_id)
            defn_map[str(ifc_id)] = str(desc) if desc is not None else ""
            rows.append({
                "ifc_id": str(ifc_id),
                "schema_version": schema_version,
                "label": str(label) if label is not None else str(ifc_id),
                "description": str(desc) if desc is not None else "",
                "parent": str(parent) if parent else "",
            })
    elif isinstance(data, list):
        # Fallback: list of objects with id/name/description
        for item in data:
            ifc_id = item.get("id") or item.get("name")
            if not ifc_id:
                continue
            rows.append({
                "ifc_id": str(ifc_id),
                "schema_version": schema_version,
                "label": item.get("label") or item.get("name") or str(ifc_id),
                "description": item.get("description") or "",
                "parent": str(item.get("parent") or ""),
            })
    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["ifc_id","schema_version","label","description","parent"]) 
    if df.empty:
        return df

    # Build augmented text using parent chain context
    def build_chain_text(node: str) -> str:
        seen = set()
        parts: List[str] = []
        cur = node
        depth = 0
        while cur and cur not in seen and depth < 10:
            seen.add(cur)
            lbl = label_map.get(cur, cur)
            dsc = defn_map.get(cur, "")
            if dsc:
                parts.append(f"{lbl}: {dsc}")
            else:
                parts.append(f"{lbl}")
            cur = parent_map.get(cur)
            depth += 1
        # Most-specific first already; join
        return ". ".join([p for p in parts if p]).strip()

    df["aug_text"] = df["ifc_id"].map(lambda x: build_chain_text(str(x)))
    return df.drop_duplicates(subset=["ifc_id"]) 


def read_uniclass_csvs(csvs: Dict[str, Path], revision: str) -> pd.DataFrame:
    frames = []
    for facet, p in csvs.items():
        if not p.exists():
            continue
        df = pd.read_csv(p)
        # Try to be robust to column headings
        code_col = next((c for c in df.columns if c.lower() in ("code", "uniclass code", "item code")), None)
        title_col = next((c for c in df.columns if c.lower() in ("title", "name", "item title")), None)
        if not code_col or not title_col:
            raise ValueError(f"CSV {p} lacks Code/Title columns")
        frames.append(pd.DataFrame({
            "code": df[code_col].astype(str),
            "facet": str(facet).upper(),
            "title": df[title_col].astype(str),
            "description": "",
            "revision": revision,
        }))
    if not frames:
        return pd.DataFrame(columns=["code","facet","title","description","revision"]) 
    return pd.concat(frames, ignore_index=True).drop_duplicates(subset=["code"]) 


FACETS = ["EF","SS","PR","TE","PM","AC","EN","SL","RO","CO","MA","FI","PC","RK","ZZ"]


def facet_from_filename(name: str) -> Optional[str]:
    up = name.upper()
    # Prefer exact facet token matches
    for fac in FACETS:
        token = f"_{fac}_"
        if token in up:
            return fac
        if up.startswith(f"{fac}_") or up.endswith(f"_{fac}"):
            return fac
    # Last resort: search any facet substring
    for fac in FACETS:
        if fac in up:
            return fac
    return None


def read_uniclass_xlsx_dir(dir_path: Path, revision: str) -> pd.DataFrame:
    def _coerce_with_header_search(xlsx_path: Path) -> Optional[pd.DataFrame]:
        """Read first sheet and try to locate the header row if the file has a title banner.

        Strategy:
        - First, try the straightforward read with inferred header.
        - If expected columns not found, read with header=None, then scan the first 25 rows
          for a row containing tokens like 'Uniclass Code' and 'Title'. Use that row as header.
        """
        try:
            df = pd.read_excel(xlsx_path, sheet_name=0, dtype=str, engine="openpyxl")
        except Exception:
            df = pd.read_excel(xlsx_path, sheet_name=0, dtype=str)

        def _find_cols(frame: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
            cols = [c for c in frame.columns if isinstance(c, str)]
            lower = {c.lower(): c for c in cols}
            code_col = next((lower[k] for k in [
                "uniclass code", "code", "item code", "pr code", "ef code", "ss code"
            ] if k in lower), None)
            title_col = next((lower[k] for k in [
                "title", "item title", "name"
            ] if k in lower), None)
            return code_col, title_col

        code_col, title_col = _find_cols(df)
        if code_col and title_col:
            return df

        # Fallback: search header row
        try:
            raw = pd.read_excel(xlsx_path, sheet_name=0, header=None, dtype=str, engine="openpyxl")
        except Exception:
            raw = pd.read_excel(xlsx_path, sheet_name=0, header=None, dtype=str)
        header_row = None
        want_code = {"uniclass code", "code", "item code", "pr code", "ef code", "ss code"}
        want_title = {"title", "item title", "name"}
        max_scan = min(len(raw), 25)
        for i in range(max_scan):
            row_vals = [str(v).strip().lower() for v in list(raw.iloc[i].values)]
            if any(v in want_code for v in row_vals) and any(v in want_title for v in row_vals):
                header_row = i
                break
        if header_row is None:
            return None
        # Rebuild dataframe with detected header
        new_cols = [str(v).strip() for v in raw.iloc[header_row].values]
        data = raw.iloc[header_row + 1 :].copy()
        data.columns = new_cols
        code_col, title_col = _find_cols(data)
        if code_col and title_col:
            return data
        return None

    frames = []
    for p in sorted(dir_path.glob("*.xlsx")):
        facet = facet_from_filename(p.stem) or ""
        df = _coerce_with_header_search(p)
        if df is None:
            continue
        # Normalize columns
        cols = [c for c in df.columns if isinstance(c, str)]
        lower = {c.lower(): c for c in cols}
        code_col = next((lower[k] for k in ["uniclass code","code","item code","pr code","ef code","ss code"] if k in lower), None)
        title_col = next((lower[k] for k in ["title","item title","name"] if k in lower), None)
        if not code_col or not title_col:
            continue
        fval = facet or ""
        frames.append(pd.DataFrame({
            "code": df[code_col].astype(str).str.strip(),
            "facet": fval,
            "title": df[title_col].astype(str).str.strip(),
            "description": "",
            "revision": revision,
        }))
    if not frames:
        return pd.DataFrame(columns=["code","facet","title","description","revision"]) 
    out = pd.concat(frames, ignore_index=True)
    # If facet was empty, try infer from code prefix before first underscore
    mask = out["facet"] == ""
    out.loc[mask, "facet"] = out.loc[mask, "code"].str.extract(r"^([A-Za-z]{1,2})_", expand=False).fillna("")
    return out.drop_duplicates(subset=["code"]) 


def connect(db_url: str) -> psycopg.Connection:
    return psycopg.connect(db_url, row_factory=dict_row, autocommit=True)


def upsert_ifc(conn: psycopg.Connection, df: pd.DataFrame):
    if df.empty:
        return
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TEMP TABLE tmp_ifc AS SELECT * FROM ifc_class WITH NO DATA;
            """
        )
        cols = ["ifc_id","schema_version","label","description"]
        csv_bytes = df[cols].to_csv(index=False, header=False).encode("utf-8")
        with cur.copy(
            "COPY tmp_ifc (ifc_id,schema_version,label,description) FROM STDIN WITH (FORMAT CSV, HEADER FALSE)"
        ) as cp:
            cp.write(csv_bytes)
        cur.execute(
            """
            INSERT INTO ifc_class AS t (ifc_id,schema_version,label,description)
            SELECT ifc_id,schema_version,label,description FROM tmp_ifc
            ON CONFLICT (ifc_id) DO UPDATE
            SET schema_version = EXCLUDED.schema_version,
                label = EXCLUDED.label,
                description = EXCLUDED.description;
            DROP TABLE tmp_ifc;
            """
        )


def upsert_uniclass(conn: psycopg.Connection, df: pd.DataFrame):
    if df.empty:
        return
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TEMP TABLE tmp_uc AS SELECT * FROM uniclass_item WITH NO DATA;
            """
        )
        cols = ["code","facet","title","description","revision"]
        csv_bytes = df[cols].to_csv(index=False, header=False).encode("utf-8")
        with cur.copy(
            "COPY tmp_uc (code,facet,title,description,revision) FROM STDIN WITH (FORMAT CSV, HEADER FALSE)"
        ) as cp:
            cp.write(csv_bytes)
        cur.execute(
            """
            INSERT INTO uniclass_item AS u (code,facet,title,description,revision)
            SELECT code,facet,title,description,revision FROM tmp_uc
            ON CONFLICT (code) DO UPDATE
            SET facet = EXCLUDED.facet,
                title = EXCLUDED.title,
                description = EXCLUDED.description,
                revision = EXCLUDED.revision;
            -- Also record a historical snapshot per (code, revision)
            INSERT INTO uniclass_item_revision (code, revision, facet, title, description)
            SELECT code, revision, facet, title, description FROM tmp_uc
            ON CONFLICT (code, revision) DO NOTHING;
            DROP TABLE tmp_uc;
            """
        )


def normalize_text(s: str) -> str:
    return " ".join((s or "").lower().replace("_", " ").split())


def expand_with_synonyms(text: str, synonyms: List[List[str]]) -> str:
    t = text
    for group in synonyms:
        base = group[0]
        for alt in group[1:]:
            t = t.replace(alt, base)
    return t


def facet_heuristics(ifc_id: str) -> List[str]:
    f = []
    if ifc_id.startswith("Ifc") and (ifc_id.endswith("Element") or "Element" in ifc_id):
        f.extend(["EF", "SS"])
    if "System" in ifc_id:
        f.append("SS")
    if "Property" in ifc_id or "Pset" in ifc_id:
        f.append("PR")
    return list(dict.fromkeys(f))


def score_pair(a: str, b: str) -> float:
    return max(
        fuzz.token_set_ratio(a, b),
        fuzz.partial_token_set_ratio(a, b),
        fuzz.WRatio(a, b),
    ) / 100.0


def generate_candidates(
    ifc_df: pd.DataFrame,
    uc_df: pd.DataFrame,
    synonyms: List[List[str]],
    top_k: int,
) -> pd.DataFrame:
    uc_df = uc_df.copy()
    uc_df["norm_title"] = uc_df["title"].fillna("").map(normalize_text).map(lambda s: expand_with_synonyms(s, synonyms))

    rows = []
    for _, r in ifc_df.iterrows():
        ifc_id = r["ifc_id"]
        desc = normalize_text(str(r.get("description", "")))
        # Fallback to label if description/definition is empty
        if not desc:
            desc = normalize_text(str(r.get("label", "")))
        q = expand_with_synonyms(f"{desc}".strip(), synonyms)
        facets = facet_heuristics(ifc_id)
        pool = uc_df if not facets else uc_df[uc_df["facet"].isin(facets)]
        if pool.empty:
            pool = uc_df
        scores = []
        for _, u in pool.iterrows():
            s = score_pair(q, u["norm_title"]) * 1.0
            scores.append((u["code"], u["facet"], u["title"], u["description"], float(s)))
        scores.sort(key=lambda x: x[4], reverse=True)
        for code, facet, title, udesc, s in scores[:top_k]:
            rows.append({
                "ifc_id": ifc_id,
                "code": code,
                "facet": facet,
                "uniclass_title": title,
                "uniclass_description": udesc,
                "score": round(s, 4),
            })
    return pd.DataFrame(rows)


def _fetch_ifc_embedding(conn: psycopg.Connection, ifc_id: str) -> Optional[str]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT embedding::text FROM text_embedding WHERE entity_type='ifc' AND entity_id=%s",
            (ifc_id,),
        )
        row = cur.fetchone()
    if not row:
        return None
    val = row[0] if not isinstance(row, dict) else row.get("embedding")
    return str(val) if val is not None else None


def _fetch_embedding(conn: psycopg.Connection, entity_type: str, entity_id: str) -> Optional[str]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT embedding::text FROM text_embedding WHERE entity_type=%s AND entity_id=%s",
            (entity_type, entity_id),
        )
        row = cur.fetchone()
    if not row:
        return None
    val = row[0] if not isinstance(row, dict) else row.get("embedding")
    return str(val) if val is not None else None


def _embedding_neighbors(conn: psycopg.Connection, target_entity_type: str, qvec_text: str, limit: int) -> Dict[str, float]:
    # Returns mapping neighbor_id -> emb_score in [0,1] for the given target entity type
    sql = (
        "SELECT entity_id, 1 - (embedding <=> %s::vector) AS score "
        "FROM text_embedding WHERE entity_type=%s "
        "ORDER BY embedding <-> %s::vector LIMIT %s"
    )
    with conn.cursor() as cur:
        cur.execute(sql, (qvec_text, target_entity_type, qvec_text, limit))
        rows = cur.fetchall()
    out: Dict[str, float] = {}
    for r in rows:
        code = r[0] if not isinstance(r, dict) else r["entity_id"]
        score = float(r[1] if not isinstance(r, dict) else r["score"])
        out[str(code)] = max(0.0, min(1.0, score))
    return out


def generate_candidates_blended(
    conn: psycopg.Connection,
    ifc_df: pd.DataFrame,
    uc_df: pd.DataFrame,
    synonyms: List[List[str]],
    top_k: int,
    embedding_weight: float,
    embedding_top_k: int,
) -> pd.DataFrame:
    # Precompute lex features on uc_df
    uc_work = uc_df.copy()
    uc_work["norm_title"] = uc_work["title"].fillna("").map(normalize_text).map(lambda s: expand_with_synonyms(s, synonyms))
    # Dict for quick lookup by code
    uc_info = {
        str(r["code"]): (r["facet"], r["title"], r.get("description", ""))
        for _, r in uc_work.iterrows()
    }

    rows = []
    base_alpha = float(embedding_weight)
    for _, r in ifc_df.iterrows():
        ifc_id = r["ifc_id"]
        desc = normalize_text(str(r.get("description", "")))
        if not desc:
            desc = normalize_text(str(r.get("label", "")))
        q = expand_with_synonyms(f"{desc}".strip(), synonyms)
        facets = facet_heuristics(ifc_id)
        pool = uc_work if not facets else uc_work[uc_work["facet"].isin(facets)]
        if pool.empty:
            pool = uc_work

        # Embedding neighbor scores (if available)
        emb_scores: Dict[str, float] = {}
        qvec_text = _fetch_ifc_embedding(conn, ifc_id)
        if qvec_text:
            emb_scores = _embedding_neighbors(conn, 'uniclass', qvec_text, embedding_top_k)
            # Optionally filter by facets
            if facets:
                emb_scores = {c: s for c, s in emb_scores.items() if uc_info.get(c, ("", "", ""))[0] in facets}

        # If we have embedding neighbors, restrict lexical scoring to those neighbors only.
        # This avoids O(|pool|) lexical comparisons for every IFC class.
        restrict_codes = set(emb_scores.keys()) if emb_scores else None

        # Lexical scores
        lex_scores: Dict[str, float] = {}
        if restrict_codes is not None and len(restrict_codes) > 0:
            # Score only the neighbor codes
            for code in restrict_codes:
                facet, title, _udesc = uc_info.get(code, ("", "", ""))
                # Skip if facet-filtering removed it
                if facets and facet not in facets:
                    continue
                # Look up normalized title from uc_work
                norm_t = uc_work.loc[uc_work["code"] == code, "norm_title"].values
                if len(norm_t) == 0:
                    continue
                lex_scores[code] = float(score_pair(q, norm_t[0]) * 1.0)
        else:
            # No embeddings found for this IFC; fall back to full lexical over the pool
            for _, u in pool.iterrows():
                lex_scores[str(u["code"])] = float(score_pair(q, u["norm_title"]) * 1.0)

        # Blend scores; if no embeddings, act like alpha=0 (pure lexical)
        alpha = base_alpha if emb_scores else 0.0
        codes = set(lex_scores.keys()) | set(emb_scores.keys())
        scored = []
        for code in codes:
            lex = lex_scores.get(code, 0.0)
            emb = emb_scores.get(code, 0.0)
            final = (1 - alpha) * lex + alpha * emb
            facet, title, udesc = uc_info.get(code, ("", "", ""))
            scored.append((code, facet, title, udesc, float(final)))
        scored.sort(key=lambda x: x[4], reverse=True)
        for code, facet, title, udesc, s in scored[:top_k]:
            rows.append({
                "ifc_id": ifc_id,
                "code": code,
                "facet": facet,
                "uniclass_title": title,
                "uniclass_description": udesc,
                "score": round(s, 4),
            })
    return pd.DataFrame(rows)


def generate_candidates_uc2ifc_blended(
    conn: psycopg.Connection,
    ifc_df: pd.DataFrame,
    uc_df: pd.DataFrame,
    synonyms: List[List[str]],
    top_k: int,
    embedding_weight: float,
    embedding_top_k: int,
) -> pd.DataFrame:
    # Prepare IFC searchable text (prefer augmented parent-chain text if present)
    ifc_work = ifc_df.copy()
    if "aug_text" in ifc_work.columns:
        base_text = ifc_work["aug_text"].fillna("")
    else:
        base_text = ifc_work["description"].fillna("")
    ifc_work["search_text"] = base_text.where(base_text.str.len() > 0, ifc_work["label"].fillna(""))
    ifc_work["norm_text"] = ifc_work["search_text"].map(normalize_text).map(lambda s: expand_with_synonyms(s, synonyms))
    ifc_info = {
        str(r["ifc_id"]): (r.get("label", ""), r.get("description", ""))
        for _, r in ifc_work.iterrows()
    }

    rows = []
    base_alpha = float(embedding_weight)
    for _, u in uc_df.iterrows():
        code = str(u["code"])
        facet = str(u.get("facet", ""))
        title = str(u.get("title", ""))
        q = expand_with_synonyms(normalize_text(title), synonyms)

        # Embedding neighbors from Uniclass -> IFC
        emb_scores: Dict[str, float] = {}
        qvec_text = _fetch_embedding(conn, "uniclass", code)
        if qvec_text:
            emb_scores = _embedding_neighbors(conn, 'ifc', qvec_text, embedding_top_k)

        # Lexical: restrict to embedding neighbors if present, else full IFC set
        restrict_ids = set(emb_scores.keys()) if emb_scores else None
        lex_scores: Dict[str, float] = {}
        if restrict_ids:
            for ifc_id in restrict_ids:
                norm_t_vals = ifc_work.loc[ifc_work["ifc_id"] == ifc_id, "norm_text"].values
                if len(norm_t_vals) == 0:
                    continue
                lex_scores[ifc_id] = float(score_pair(q, norm_t_vals[0]) * 1.0)
        else:
            for _, r in ifc_work.iterrows():
                lex_scores[str(r["ifc_id"])] = float(score_pair(q, r["norm_text"]) * 1.0)

        alpha = base_alpha if emb_scores else 0.0
        ids = set(lex_scores.keys()) | set(emb_scores.keys())
        scored = []
        for ifc_id in ids:
            lex = lex_scores.get(ifc_id, 0.0)
            emb = emb_scores.get(ifc_id, 0.0)
            final = (1 - alpha) * lex + alpha * emb
            lbl, _desc = ifc_info.get(ifc_id, ("", ""))
            scored.append((ifc_id, lbl, float(final)))
        scored.sort(key=lambda x: x[2], reverse=True)
        for ifc_id, _lbl, s in scored[:top_k]:
            rows.append({
                "ifc_id": ifc_id,
                "code": code,
                "facet": facet,
                "uniclass_title": title,
                "uniclass_description": u.get("description", ""),
                "score": round(s, 4),
            })
    return pd.DataFrame(rows)


def infer_relation_type(ifc_id: str, facet: str, score: float) -> str:
    # Uniclass 'PR' are Products in Uniclass 2015 -> relate as typical_of to IFC element/type
    if facet == "PR":
        return "typical_of"
    if facet == "SS":
        if "System" in ifc_id:
            return "equivalent" if score >= 0.8 else "broader"
        return "part_of"
    if facet == "EF":
        return "equivalent" if score >= 0.85 else ("broader" if score >= 0.7 else "narrower")
    return "broader"


def write_candidate_edges(
    conn: psycopg.Connection,
    candidates: pd.DataFrame,
    source_ifc_version: str,
    source_uniclass_revision: str,
    auto_accept: float,
):
    if candidates.empty:
        return
    payload = []
    for _, r in candidates.iterrows():
        rel = infer_relation_type(r["ifc_id"], r["facet"], r["score"])
        if r["score"] < auto_accept:
            continue
        payload.append((
            r["ifc_id"], r["code"], rel, float(r["score"]),
            f"auto-accepted by lexical heuristic top-k; facet={r['facet']}",
            source_ifc_version, source_uniclass_revision
        ))
    if not payload:
        return
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO ifc_uniclass_map (ifc_id, code, relation_type, confidence, rationale, source_ifc_version, source_uniclass_revision)
            VALUES (%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (ifc_id, code, relation_type, source_uniclass_revision) DO UPDATE
            SET confidence = EXCLUDED.confidence,
                rationale = EXCLUDED.rationale,
                source_ifc_version = EXCLUDED.source_ifc_version;
            """,
            payload,
        )


def export_viewer_json(conn: psycopg.Connection, out_path: Path):
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT m.ifc_id, m.relation_type, m.code, m.confidence, u.title AS uniclass_title
            FROM ifc_uniclass_map m
            JOIN uniclass_item u ON u.code = m.code
            ORDER BY m.ifc_id, m.relation_type, m.confidence DESC
            """
        )
        rows = cur.fetchall()
    grouped: Dict[str, Dict[str, List[dict]]] = {}
    for r in rows:
        ifc_id = r["ifc_id"]
        rel = r["relation_type"]
        grouped.setdefault(ifc_id, {}).setdefault(rel, []).append({
            "code": r["code"],
            "title": r["uniclass_title"],
            "confidence": float(r["confidence"]) if r["confidence"] is not None else None,
        })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(grouped, f, ensure_ascii=False, indent=2)


def main():
    ap = argparse.ArgumentParser(description="IFC ↔ Uniclass ETL and candidate mapping")
    ap.add_argument("--config", required=True, help="Path to YAML settings")
    ap.add_argument("--load", action="store_true", help="Load IFC and Uniclass into DB")
    ap.add_argument("--candidates", action="store_true", help="Generate top-k candidate links and auto-accept above threshold")
    ap.add_argument("--export", action="store_true", help="Export viewer JSON from accepted mappings")
    ap.add_argument("--embed", action="store_true", help="Generate embeddings via Ollama and store in DB")
    args = ap.parse_args()

    s = load_settings(Path(args.config))
    ensure_output_dirs(s)

    ifc_df = read_ifc_json(s.ifc_json, s.ifc_schema_version)

    # Optionally detect Uniclass revision from filenames and override YAML
    detected_rev: Optional[str] = None
    if s.autodetect_uniclass_revision:
        paths: List[Path] = []
        if s.uniclass_xlsx_dir and s.uniclass_xlsx_dir.exists():
            paths.extend(sorted(s.uniclass_xlsx_dir.glob("*.xlsx")))
        if s.uniclass_csvs:
            paths.extend([p for p in s.uniclass_csvs.values() if p.exists()])
        detected_rev = _detect_uniclass_revision_from_paths(paths) if paths else None
        if detected_rev:
            print(f"[uniclass] Detected revision from filenames: {detected_rev} (overriding {s.uniclass_revision})")
            s.uniclass_revision = detected_rev

    if s.uniclass_xlsx_dir and s.uniclass_xlsx_dir.exists():
        uc_df = read_uniclass_xlsx_dir(s.uniclass_xlsx_dir, s.uniclass_revision)
    else:
        uc_df = read_uniclass_csvs(s.uniclass_csvs, s.uniclass_revision)

    with connect(s.db_url) as conn:
        # Enforce monotonic revision if enabled
        if s.enforce_monotonic_revision:
            current = _get_current_max_uniclass_revision(conn)
            if current:
                try:
                    sch_new, t_new = _parse_revision_token(s.uniclass_revision)
                    sch_cur, t_cur = _parse_revision_token(current)
                    if sch_new == sch_cur and not (t_new > t_cur):
                        raise SystemExit(
                            f"Refusing to load Uniclass revision '{s.uniclass_revision}' <= current '{current}' (monotonic enforcement).\n"
                            f"Update revision or set uniclass.enforce_monotonic_revision=false in config."
                        )
                except ValueError:
                    print(f"[warn] Could not compare revisions ('{s.uniclass_revision}' vs '{current}'); proceeding.")
        if args.load:
            upsert_ifc(conn, ifc_df)
            upsert_uniclass(conn, uc_df)

        if args.embed:
            # Prepare texts: IFC uses description/definition optionally augmented with parent chain; Uniclass uses title only
            ifc_items: List[Tuple[str, str]] = []
            for _, r in ifc_df.iterrows():
                eid = str(r["ifc_id"]) if r.get("ifc_id") is not None else None
                if not eid:
                    continue
                if s.ifc_use_parent_context and (r.get('aug_text') or ''):
                    text = str(r.get('aug_text') or '').strip()
                else:
                    desc = str(r.get('description','') or '').strip()
                    text = desc if desc else str(r.get('label','') or '').strip()
                ifc_items.append((eid, text))

            uc_items: List[Tuple[str, str]] = []
            for _, r in uc_df.iterrows():
                eid = str(r["code"]) if r.get("code") is not None else None
                if not eid:
                    continue
                text = f"{str(r.get('title','') or '')}".strip()
                uc_items.append((eid, text))

            print(f"[embed] Generating embeddings using model '{s.embed_model}' at {s.embed_endpoint}")
            n_ifc = generate_and_store_embeddings(conn, "ifc", ifc_items, s.embed_model, s.embed_endpoint, s.embed_batch_size, s.embed_timeout_s, expected_dim=768)
            n_uc = generate_and_store_embeddings(conn, "uniclass", uc_items, s.embed_model, s.embed_endpoint, s.embed_batch_size, s.embed_timeout_s, expected_dim=768)
            print(f"[embed] Upserted {n_ifc} IFC and {n_uc} Uniclass embeddings")

        if args.candidates:
            use_embed = s.embedding_weight and float(s.embedding_weight) > 0
            direction = (s.match_direction or 'ifc_to_uniclass').lower()
            if direction == 'uniclass_to_ifc':
                if use_embed:
                    cands = generate_candidates_uc2ifc_blended(conn, ifc_df, uc_df, s.synonyms, s.top_k, s.embedding_weight, s.embedding_top_k)
                else:
                    # Lexical-only reverse direction using the same blended helper with alpha=0 via parameters
                    cands = generate_candidates_uc2ifc_blended(conn, ifc_df, uc_df, s.synonyms, s.top_k, 0.0, s.embedding_top_k)
            else:
                if use_embed:
                    cands = generate_candidates_blended(conn, ifc_df, uc_df, s.synonyms, s.top_k, s.embedding_weight, s.embedding_top_k)
                else:
                    cands = generate_candidates(ifc_df, uc_df, s.synonyms, s.top_k)
            cands_path = s.output_dir / "candidates.csv"
            cands.to_csv(cands_path, index=False)
            write_candidate_edges(conn, cands, s.ifc_schema_version, s.uniclass_revision, s.auto_accept)

        if args.export:
            export_viewer_json(conn, s.viewer_json)


if __name__ == "__main__":
    main()
