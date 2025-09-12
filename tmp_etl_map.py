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
    # Optional facet-aware rules
    matching_rules: Dict[str, dict]
    # Optional convenience list of facets to skip entirely
    skip_tables: List[str]
    # Optional scoring boosts driven by IFC features
    feature_boosts: Dict[str, float]
    embed_model: str
    embed_endpoint: str
    embed_batch_size: int
    embed_timeout_s: float
    embed_expected_dim: Optional[int]
    embed_ivf_lists: int
    embed_ivf_probes: int
    embed_query_timeout_ms: int
    embedding_weight: float
    embedding_top_k: int
    # Anchor guidance from Uniclass IFC mapping columns
    anchor_bonus: float
    anchor_use_ancestors: bool
    # LLM rerank settings (Ollama)
    rerank_top_n: int
    rerank_model: str
    rerank_endpoint: str
    rerank_temperature: float
    rerank_max_tokens: int
    rerank_fewshot_per_facet: int
    rerank_timeout_s: float
    


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
    # Build matching rules with optional skip_tables merged in (uppercase facets)
    match_cfg = (cfg.get("matching", {}) or {})
    base_rules = (match_cfg.get("rules", {}) or {}).copy()
    skip_list = [str(x).upper() for x in (match_cfg.get("skip_tables", []) or [])]
    for fac in skip_list:
        r = (base_rules.get(fac) or {}).copy()
        r["skip"] = True
        base_rules[fac] = r

    fb_cfg = (match_cfg.get("feature_boosts", {}) or {})
    # Global discipline gating options tucked under matching (soft by default)
    glob_rules = {
        "discipline_filter": str(match_cfg.get("discipline_filter", os.getenv("DISCIPLINE_FILTER", "soft"))).lower(),
        "discipline_penalty": float(match_cfg.get("discipline_penalty", os.getenv("DISCIPLINE_PENALTY", 0.5))),
        "discipline_source": str(match_cfg.get("discipline_source", os.getenv("DISCIPLINE_SOURCE", "heuristic"))).lower(),
    }
    # Stash into rules dict under a special key for downstream functions
    base_rules["__global__"] = glob_rules
    # Defaults are conservative multipliers (added to 1.0 later)
    feature_boosts = {
        "pr_predefined_type": float(fb_cfg.get("pr_predefined_type", 0.10)),
        "pr_manufacturer_ps": float(fb_cfg.get("pr_manufacturer_ps", 0.10)),
        "ss_group_system": float(fb_cfg.get("ss_group_system", 0.10)),
        "type_bias": float(fb_cfg.get("type_bias", 0.05)),
        "pset_count_scale": float(fb_cfg.get("pset_count_scale", 0.002)),  # +0.2 at 100 psets
        "max_pset_boost": float(fb_cfg.get("max_pset_boost", 0.25)),
    }

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
        match_direction=str(match_cfg.get("direction", "ifc_to_uniclass")),
        matching_rules=base_rules,
        skip_tables=skip_list,
        feature_boosts=feature_boosts,
        embed_model=str((cfg.get("embedding", {}) or {}).get("model", os.getenv("EMBED_MODEL", "nomic-embed-text"))),
        embed_endpoint=str((cfg.get("embedding", {}) or {}).get("endpoint", os.getenv("EMBED_ENDPOINT", "http://localhost:11434"))),
        embed_batch_size=int((cfg.get("embedding", {}) or {}).get("batch_size", os.getenv("EMBED_BATCH_SIZE", 16))),
        embed_timeout_s=float((cfg.get("embedding", {}) or {}).get("timeout_s", os.getenv("EMBED_TIMEOUT_S", 30))),
        embed_expected_dim=(lambda v: int(v) if (v is not None and str(v).strip() != "") else None)((cfg.get("embedding", {}) or {}).get("expected_dim", os.getenv("EMBED_EXPECTED_DIM", ""))),
        embed_ivf_lists=int((cfg.get("embedding", {}) or {}).get("ivf_lists", os.getenv("EMBED_IVF_LISTS", 100))),
        embed_ivf_probes=int((cfg.get("embedding", {}) or {}).get("ivf_probes", os.getenv("EMBED_IVF_PROBES", 10))),
        embed_query_timeout_ms=int((cfg.get("embedding", {}) or {}).get("query_timeout_ms", os.getenv("EMBED_QUERY_TIMEOUT_MS", 5000))),
        embedding_weight=float((cfg.get("matching", {}) or {}).get("embedding_weight", os.getenv("EMBEDDING_WEIGHT", 0.4))),
        embedding_top_k=int((cfg.get("matching", {}) or {}).get("embedding_top_k", os.getenv("EMBEDDING_TOP_K", 50))),
        anchor_bonus=float((cfg.get("matching", {}) or {}).get("anchor_bonus", os.getenv("ANCHOR_BONUS", 0.25))),
        anchor_use_ancestors=bool((cfg.get("matching", {}) or {}).get("anchor_use_ancestors", os.getenv("ANCHOR_USE_ANCESTORS", "true")).__str__().lower() in ("1","true","yes")),
        rerank_top_n=int((cfg.get("rerank", {}) or {}).get("top_n", os.getenv("RERANK_TOP_N", 0))),
        rerank_model=str((cfg.get("rerank", {}) or {}).get("model", os.getenv("RERANK_MODEL", "llama3.2:latest"))),
        rerank_endpoint=str((cfg.get("rerank", {}) or {}).get("endpoint", os.getenv("RERANK_ENDPOINT", os.getenv("EMBED_ENDPOINT", "http://localhost:11434")))),
        rerank_temperature=float((cfg.get("rerank", {}) or {}).get("temperature", os.getenv("RERANK_TEMPERATURE", 0))),
        rerank_max_tokens=int((cfg.get("rerank", {}) or {}).get("max_tokens", os.getenv("RERANK_MAX_TOKENS", 512))),
        rerank_fewshot_per_facet=int((cfg.get("rerank", {}) or {}).get("fewshot_per_facet", os.getenv("RERANK_FEWSHOT_PER_FACET", 3))),
        rerank_timeout_s=float((cfg.get("rerank", {}) or {}).get("timeout_s", os.getenv("RERANK_TIMEOUT_S", 20))),
    )


def _ollama_generate(prompt: str, model: str, endpoint: str, temperature: float = 0.0, max_tokens: int = 512, timeout_s: float = 60.0) -> Optional[str]:
    try:
        import requests
    except Exception:
        print("[rerank] requests is required. pip install requests")
        return None
    url = endpoint.rstrip("/") + "/api/generate"
    try:
        resp = requests.post(url, json={
            "model": model,
            "prompt": prompt,
            "options": {"temperature": temperature, "num_predict": max_tokens},
            "format": "json",
            "stream": False,
        }, timeout=timeout_s)
        resp.raise_for_status()
        data = resp.json()
        return data.get("response") or data.get("text")
    except Exception as e:
        print(f"[rerank] ollama generate error: {e}")
        return None


def _collect_anchor_examples(uc_df: pd.DataFrame, fewshot_per_facet: int = 3) -> Dict[str, List[Tuple[str, str, List[str]]]]:
    by_facet: Dict[str, List[Tuple[str, str, List[str]]]] = {}
    if "ifc_mappings" not in uc_df.columns:
        return by_facet
    for _, r in uc_df.iterrows():
        facet = str(r.get("facet", "")).upper()
        code = str(r.get("code", ""))
        title = str(r.get("title", ""))
        maps_raw = str(r.get("ifc_mappings", "") or "")
        ids = [m for m in maps_raw.split(";") if m]
        if not ids:
            continue
        by_facet.setdefault(facet, []).append((code, title, ids))
    # Trim per facet
    for f in list(by_facet.keys()):
        by_facet[f] = by_facet[f][:max(0, int(fewshot_per_facet))]
    return by_facet


def _format_rerank_prompt(direction: str, source_label: str, source_desc: str, source_facet: str, candidates: List[Tuple[str, str]], examples: Dict[str, List[Tuple[str, str, List[str]]]]) -> str:
    # candidates: list of (id, label)
    lines = []
    lines.append("You are an expert mapping assistant for IFC and Uniclass.")
    lines.append("Re-rank candidates by semantic suitability. Return ONLY JSON array: [{\"id\": \"...\", \"score\": 0..1}].")
    lines.append("Higher score means better match.")
    lines.append("")
    # Few-shot using examples from same facet first, then any facet if none
    shots = examples.get(source_facet.upper(), []) or next(iter(examples.values()), [])
    if shots:
        lines.append("Examples:")
        for code, title, ifc_ids in shots:
            if direction == 'uniclass_to_ifc':
                lines.append(f"- Uniclass [{source_facet}] {code}: {title} -> IFC: {', '.join(ifc_ids)}")
            else:
                # For ifc_to_uniclass we invert description; still useful hints
                lines.append(f"- IFC to Uniclass hint: IFC {', '.join(ifc_ids)} often maps to [{source_facet}] like '{title}'")
        lines.append("")
    lines.append(f"Task direction: {direction}")
    lines.append(f"Source facet: {source_facet}")
    lines.append(f"Source: {source_label}")
    if source_desc:
        lines.append(f"Source details: {source_desc}")
    lines.append("Candidates:")
    for cid, lbl in candidates:
        lines.append(f"- id={cid} | {lbl}")
    lines.append("")
    lines.append("Return JSON only: [{\"id\": \"...\", \"score\": 0.xx}] with same ids.")
    return "\n".join(lines)


def _parse_rerank_json(text: str) -> List[Tuple[str, float]]:
    import json, re
    if not text:
        return []
    s = text.strip()
    # Try to extract the first JSON array
    m = re.search(r"\[(?:.|\n)*\]", s)
    if m:
        s = m.group(0)
    try:
        arr = json.loads(s)
        out = []
        for it in arr:
            cid = str(it.get("id")) if isinstance(it, dict) else None
            sc = float(it.get("score", 0.0)) if isinstance(it, dict) else 0.0
            if cid:
                out.append((cid, max(0.0, min(1.0, sc))))
        return out
    except Exception:
        # Fallback: parse lines like id=..., score=...
        out = []
        for line in text.splitlines():
            if "id=" in line:
                cid = line.split("id=",1)[1].split()[0].strip().strip(',')
                out.append((cid, 0.5))
        return out


ALLOWED_LABELS = [
    # AECO disciplines
    "ARCH","STRUCT","CIVIL","MECH","PLUMB","ELEC","ICT","MAINT",
    # Subjects / roles
    "ROLE","PM","ACTIVITIES","COMPLEX",
    # Uniclass table facets (minus ZZ)
    "AC","CO","EF","EN","FI","MA","PC","PM","PR","RK","RO","SL","SS",
]

def _format_label_prompt(item_type: str, identifier: str, title: str, description: str, extra: str = "") -> str:
    lines = []
    lines.append("You are an AECO taxonomy classifier.")
    lines.append("Return ONLY a JSON array of labels drawn from this closed set, no text: ")
    lines.append(json.dumps(ALLOWED_LABELS))
    lines.append("")
    lines.append(f"ItemType: {item_type}")
    lines.append(f"Id: {identifier}")
    if title:
        lines.append(f"Title: {title}")
    if description:
        lines.append(f"Description: {description}")
    if extra:
        lines.append(f"Context: {extra}")
    lines.append("")
    lines.append("Rules: choose all applicable labels; output JSON array only. Examples not required.")
    return "\n".join(lines)

def _parse_label_array(text: Optional[str]) -> List[str]:
    if not text:
        return []
    try:
        # Extract first JSON array
        import re
        m = re.search(r"\[(?:.|\n)*\]", text)
        s = m.group(0) if m else text
        arr = json.loads(s)
        if not isinstance(arr, list):
            return []
        out = []
        for v in arr:
            if isinstance(v, str):
                val = v.strip().upper()
                if val in ALLOWED_LABELS:
                    out.append(val)
        # Dedup preserve order
        seen = set()
        res = []
        for v in out:
            if v not in seen:
                res.append(v); seen.add(v)
        return res
    except Exception:
        return []

def classify_items_with_llm(
    items: List[Tuple[str, str, str, str]],  # (key, item_type, title, description)
    model: str,
    endpoint: str,
    temperature: float = 0.0,
    max_tokens: int = 128,
    timeout_s: float = 30.0,
) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for key, item_type, title, desc in items:
        prompt = _format_label_prompt(item_type, key, title or "", desc or "")
        resp = _ollama_generate(prompt, model=model, endpoint=endpoint, temperature=temperature, max_tokens=max_tokens, timeout_s=timeout_s)
        labels = _parse_label_array(resp)
        out[key] = labels
    return out


def rerank_candidates_with_llm(
    cands: pd.DataFrame,
    ifc_df: pd.DataFrame,
    uc_df: pd.DataFrame,
    direction: str,
    top_n: int,
    model: str,
    endpoint: str,
    temperature: float,
    max_tokens: int,
    fewshot_per_facet: int,
    timeout_s: float,
) -> pd.DataFrame:
    if top_n <= 0 or cands.empty:
        return cands
    # Build lookups
    ifc_labels = {str(r["ifc_id"]): str(r.get("label", "")) for _, r in ifc_df.iterrows()}
    uc_titles = {str(r["code"]): (str(r.get("title", "")), str(r.get("facet", ""))) for _, r in uc_df.iterrows()}
    examples = _collect_anchor_examples(uc_df, fewshot_per_facet=fewshot_per_facet)
    rows = []
    direction = (direction or 'ifc_to_uniclass').lower()
    if direction == 'uniclass_to_ifc':
        for code, grp in cands.groupby("code"):
            title, facet = uc_titles.get(str(code), ("", ""))
            grp_sorted = grp.sort_values("score", ascending=False).head(top_n)
            cand_list = [(str(r.ifc_id), ifc_labels.get(str(r.ifc_id), str(r.ifc_id))) for r in grp_sorted.itertuples(index=False)]
            prompt = _format_rerank_prompt(direction, f"Uniclass {code}: {title}", "", facet, cand_list, examples)
            resp = _ollama_generate(prompt, model=model, endpoint=endpoint, temperature=temperature, max_tokens=max_tokens, timeout_s=timeout_s)
            order = _parse_rerank_json(resp)
            order_map = {cid: sc for cid, sc in order}
            for r in grp.itertuples(index=False):
                sc = order_map.get(str(r.ifc_id))
                new_score = float(sc) if sc is not None else float(r.score)
                rows.append({**r._asdict(), "score": max(0.0, min(1.0, new_score))})
    else:
        for ifc_id, grp in cands.groupby("ifc_id"):
            lbl = ifc_labels.get(str(ifc_id), str(ifc_id))
            grp_sorted = grp.sort_values("score", ascending=False).head(top_n)
            cand_list = [(str(r.code), f"[{r.facet}] {r.uniclass_title}") for r in grp_sorted.itertuples(index=False)]
            try:
                facet = grp_sorted["facet"].mode().iat[0]
            except Exception:
                facet = ""
            prompt = _format_rerank_prompt(direction, f"IFC {ifc_id}: {lbl}", "", facet, cand_list, examples)
            resp = _ollama_generate(prompt, model=model, endpoint=endpoint, temperature=temperature, max_tokens=max_tokens, timeout_s=timeout_s)
            order = _parse_rerank_json(resp)
            order_map = {cid: sc for cid, sc in order}
            for r in grp.itertuples(index=False):
                sc = order_map.get(str(r.code))
                new_score = float(sc) if sc is not None else float(r.score)
                rows.append({**r._asdict(), "score": max(0.0, min(1.0, new_score))})
    # Rebuild DataFrame
    out = pd.DataFrame(rows)
    # Ensure same column order if possible
    return out[cands.columns] if set(out.columns) >= set(cands.columns) else out


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
            parent = item.get("parent") or item.get("baseClass")
            parent_map[str(ifc_id)] = str(parent) if parent else None
            label_map[str(ifc_id)] = str(label) if label is not None else str(ifc_id)
            defn_map[str(ifc_id)] = str(desc) if desc is not None else ""
            rows.append({
                "ifc_id": str(ifc_id),
                "schema_version": schema_version,
                "label": str(label) if label is not None else str(ifc_id),
                "description": str(desc) if desc is not None else "",
                "parent": str(parent) if parent else "",
                "is_abstract": bool(item.get("isAbstract", False)),
                "role": str(item.get("role") or ""),
                "is_type": bool(item.get("isType", False)),
                "is_instance": bool(item.get("isInstance", False)),
                "counterpart": str(item.get("counterpart") or ""),
                "has_predefined_type": bool(item.get("predefinedType") is not None),
                "pset_count": int(len(item.get("psets") or [])),
                "has_manufacturer_ps": bool("Pset_ManufacturerTypeInformation" in (item.get("psets") or [])),
                "base_class": str(item.get("baseClass") or ""),
                "enum": str(item.get("enum") or ""),
                "enum_value": str(item.get("enumValue") or ""),
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
                "parent": str(item.get("parent") or item.get("baseClass") or ""),
                "is_abstract": bool(item.get("isAbstract", False)),
                "role": str(item.get("role") or ""),
                "is_type": bool(item.get("isType", False)),
                "is_instance": bool(item.get("isInstance", False)),
                "counterpart": str(item.get("counterpart") or ""),
                "has_predefined_type": bool(item.get("predefinedType") is not None),
                "pset_count": int(len(item.get("psets") or [])),
                "has_manufacturer_ps": bool("Pset_ManufacturerTypeInformation" in (item.get("psets") or [])),
                "base_class": str(item.get("baseClass") or ""),
                "enum": str(item.get("enum") or ""),
                "enum_value": str(item.get("enumValue") or ""),
            })
    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=[
        "ifc_id","schema_version","label","description","parent","is_abstract",
        "role","is_type","is_instance","counterpart","has_predefined_type","pset_count",
        "has_manufacturer_ps","base_class","enum","enum_value"
    ]) 
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


def _build_ifc_ancestors_map(ifc_df: pd.DataFrame) -> Dict[str, List[str]]:
    """Return mapping ifc_id -> list of ancestors including self, ordered most-specific to root."""
    pmap = {str(r["ifc_id"]): (str(r.get("parent")) if r.get("parent") else None) for _, r in ifc_df.iterrows()}
    cache: Dict[str, List[str]] = {}
    def ancestors(n: Optional[str]) -> List[str]:
        if not n:
            return []
        n = str(n)
        if n in cache:
            return cache[n]
        visited = set()
        chain: List[str] = []
        cur = n
        depth = 0
        while cur and cur not in visited and depth < 50:
            chain.append(cur)
            visited.add(cur)
            cur = pmap.get(cur)
            depth += 1
        cache[n] = chain
        return chain
    out: Dict[str, List[str]] = {}
    for k in pmap.keys():
        out[k] = ancestors(k)
    return out


def _facet_allows_ifc(
    ifc_id: str,
    facet: str,
    rules: Dict[str, dict],
    ancestors: Dict[str, List[str]],
    is_abstract_map: Dict[str, bool],
    flags_map: Optional[Dict[str, Dict[str, object]]] = None,
) -> bool:
    f = (facet or "").upper()
    r = (rules or {}).get(f, {}) or {}
    if not r:
        return True
    # Disallow by IFC id prefixes per facet
    try:
        for pfx in (r.get("disallow_prefixes") or []):
            if str(ifc_id).startswith(str(pfx)):
                return False
    except Exception:
        pass
    if r.get("skip"):
        return False
    if r.get("abstract_only") and not bool(is_abstract_map.get(ifc_id, False)):
        return False
    req = r.get("require_ifc_roots") or []
    if req:
        anc = ancestors.get(ifc_id, [ifc_id])
        # Accept if any required root is in ancestor chain
        if not any(root in anc for root in req):
            return False
    # Disqualifiers based on flags
    if flags_map is not None:
        flags = flags_map.get(str(ifc_id), {})
        # Example: don't map spatial structures to PR (products)
        if f == "PR" and bool(flags.get("is_spatial", False)):
            return False
    return True


def _compute_ifc_flags(ifc_df: pd.DataFrame, ancestors: Dict[str, List[str]]) -> Dict[str, Dict[str, object]]:
    roots_spatial = {"IfcSpatialElement", "IfcSpatialStructureElement", "IfcSite", "IfcBuilding", "IfcBuildingStorey", "IfcSpace"}
    roots_product = {"IfcProduct", "IfcElement", "IfcDistributionElement"}
    roots_process = {"IfcProcess", "IfcTask", "IfcProcedure"}
    roots_group_system = {"IfcGroup", "IfcSystem"}
    out: Dict[str, Dict[str, object]] = {}

    def infer_ifc_disciplines(iid: str, anc: set, label: str) -> List[str]:
        s = (label or iid or "").lower()
        discs: List[str] = []
        # Structural
        if any(k in anc for k in {"IfcStructuralItem", "IfcBeam", "IfcColumn", "IfcSlab", "IfcFooting"}):
            discs.append("STRUCT")
        # Civil/Infrastructure
        if any(k in anc for k in {"IfcRoad", "IfcBridge", "IfcRailway", "IfcAlignment", "IfcEarthworksElement"}):
            discs.append("CIVIL")
        # Architectural
        if any(k in anc for k in {"IfcDoor", "IfcWindow", "IfcWall", "IfcCovering", "IfcFurnishingElement", "IfcCurtainWall"}):
            discs.append("ARCH")
        # Mechanical (HVAC/process)
        if any(k in anc for k in {"IfcDuctSegment", "IfcDuctFitting", "IfcDuctAccessory", "IfcFlowTerminal", "IfcFlowTreatmentDevice", "IfcAirToAirHeatRecovery", "IfcChiller", "IfcBoiler"}) or ("duct" in s or "hvac" in s or "air" in s):
            discs.append("MECH")
        # Plumbing
        if any(k in anc for k in {"IfcPipeSegment", "IfcPipeFitting", "IfcPipeAccessory", "IfcWasteTerminal", "IfcSanitaryTerminal", "IfcInterceptor"}) or ("pipe" in s or "sanitary" in s or "plumb" in s):
            discs.append("PLUMB")
        # Electrical
        if any(k in anc for k in {"IfcElectricDistributionBoard", "IfcTransformer", "IfcSwitchingDevice", "IfcLightFixture", "IfcCableCarrierSegment", "IfcCableSegment"}) or ("elect" in s or "cable" in s or "switch" in s or "luminaire" in s):
            discs.append("ELEC")
        # Communications / IT
        if any(k in anc for k in {"IfcCommunicationsAppliance", "IfcAudioVisualAppliance"}) or ("data" in s or "network" in s or "it " in s or "comm" in s):
            discs.append("ICT")
        # Maintenance/Consumables
        if ("clean" in s or "chemical" in s or "detergent" in s or "gel" in s):
            discs.append("MAINT")
        # De-duplicate preserving order
        return list(dict.fromkeys(discs))

    def infer_ifc_subjects(iid: str, anc: set, label: str) -> List[str]:
        s = (label or iid or "").lower()
        subs: List[str] = []
        # Roles
        if any(k in anc for k in {"IfcActor"}) or "actor" in s:
            subs.append("ROLE")
        # Project management / process control
        if any(k in anc for k in {"IfcProcess", "IfcTask", "IfcProcedure", "IfcControl"}) or ("process" in s or "task" in s or "control" in s):
            subs.append("PM")
        # Activities (tasks)
        if any(k in anc for k in {"IfcTask"}) or "activity" in s:
            subs.append("ACTIVITIES")
        # Complex/group/system
        if any(k in anc for k in {"IfcGroup", "IfcSystem"}):
            subs.append("COMPLEX")
        return list(dict.fromkeys(subs))
    for _, r in ifc_df.iterrows():
        iid = str(r["ifc_id"]) 
        anc = set(ancestors.get(iid, []))
        is_spatial = any(root in anc for root in roots_spatial)
        is_product = any(root in anc for root in roots_product)
        is_process = any(root in anc for root in roots_process)
        is_group_system = any(root in anc for root in roots_group_system)
        discs = infer_ifc_disciplines(iid, anc, str(r.get("label", "")))
        subs = infer_ifc_subjects(iid, anc, str(r.get("label", "")))
        d = {
            "is_spatial": is_spatial,
            "is_product": is_product,
            "is_process": is_process,
            "is_group_system": is_group_system,
            "is_type": bool(r.get("is_type", False)),
            "is_instance": bool(r.get("is_instance", False)),
            "has_predefined_type": bool(r.get("has_predefined_type", False)),
            "pset_count": int(r.get("pset_count", 0) or 0),
            "has_manufacturer_ps": bool(r.get("has_manufacturer_ps", False)),
            "role": str(r.get("role", "") or ""),
            # Back-compat field name
            "disciplines": discs,
            # Unified labels = disciplines + subjects
            "labels": list(dict.fromkeys(list(discs) + list(subs))),
        }
        out[iid] = d
    return out


def _feature_multiplier(facet: str, flags: Dict[str, object], boosts: Dict[str, float]) -> float:
    f = (facet or "").upper()
    mult = 1.0
    if f == "PR":
        if bool(flags.get("has_predefined_type", False)):
            mult += boosts.get("pr_predefined_type", 0.0)
        if bool(flags.get("has_manufacturer_ps", False)):
            mult += boosts.get("pr_manufacturer_ps", 0.0)
    if f == "SS":
        if bool(flags.get("is_group_system", False)):
            mult += boosts.get("ss_group_system", 0.0)
    # Type bias across PR/EF
    if bool(flags.get("is_type", False)) and f in {"PR", "EF"}:
        mult += boosts.get("type_bias", 0.0)
    # Pset count small boost
    pc = int(flags.get("pset_count", 0) or 0)
    if pc > 0:
        mult += min(pc * boosts.get("pset_count_scale", 0.0), boosts.get("max_pset_boost", 0.25))
    return max(0.0, mult)


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
        # Attempt to harvest any IFC mapping hint columns (e.g., IFC 2x3, IFC 4x3)
        ifc_cols = [c for c in cols if isinstance(c, str) and c and "ifc" in c.lower()]
        ifc_map_series = None
        if ifc_cols:
            import re
            def extract_ifc_ids(val: object) -> str:
                s = str(val or "")
                ids = re.findall(r"Ifc[A-Za-z0-9_]+", s)
                return ";".join(sorted(set(ids))) if ids else ""
            merged = []
            for _, row in df[ifc_cols].iterrows():
                parts = []
                for c in ifc_cols:
                    parts.append(extract_ifc_ids(row.get(c)))
                uniq = sorted(set([p for p in ";".join(parts).split(";") if p]))
                merged.append(";".join(uniq))
            ifc_map_series = pd.Series(merged, index=df.index)
        fval = facet or ""
        base = {
            "code": df[code_col].astype(str).str.strip(),
            "facet": fval,
            "title": df[title_col].astype(str).str.strip(),
            "description": "",
            "revision": revision,
        }
        if ifc_map_series is not None:
            base["ifc_mappings"] = ifc_map_series
        frames.append(pd.DataFrame(base))
    if not frames:
        return pd.DataFrame(columns=["code","facet","title","description","revision"]) 
    out = pd.concat(frames, ignore_index=True)
    # If facet was empty, try infer from code prefix before first underscore
    mask = out["facet"] == ""
    out.loc[mask, "facet"] = out.loc[mask, "code"].str.extract(r"^([A-Za-z]{1,2})_", expand=False).fillna("")
    if "ifc_mappings" not in out.columns:
        out["ifc_mappings"] = ""
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


def _tokenize_set(s: str) -> set:
    import re as _re
    return set(t for t in _re.split(r"[^a-z0-9]+", normalize_text(s)) if t)

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
    rules: Optional[Dict[str, dict]] = None,
) -> pd.DataFrame:
    uc_df = uc_df.copy()
    uc_df["norm_title"] = uc_df["title"].fillna("").map(normalize_text).map(lambda s: expand_with_synonyms(s, synonyms))
    # Infer Uniclass disciplines once
    def infer_uc_disciplines(title: str) -> List[str]:
        t = (title or "").lower()
        discs: List[str] = []
        if any(k in t for k in ["duct", "hvac", "air", "fan", "coil", "damper"]):
            discs.append("MECH")
        if any(k in t for k in ["pipe", "sanitary", "plumb", "drain", "valve", "tap", "waste"]):
            discs.append("PLUMB")
        if any(k in t for k in ["elect", "cable", "switch", "socket", "luminaire", "lighting", "breaker", "panel"]):
            discs.append("ELEC")
        if any(k in t for k in ["beam", "column", "slab", "rebar", "struct"]):
            discs.append("STRUCT")
        if any(k in t for k in ["door", "window", "glaz", "partition", "flooring", "covering", "furniture"]):
            discs.append("ARCH")
        if any(k in t for k in ["road", "pavement", "asphalt", "drainage", "culvert", "embankment"]):
            discs.append("CIVIL")
        if any(k in t for k in ["data", "network", "router", "server", "ict", "comm"]):
            discs.append("ICT")
        if any(k in t for k in ["clean", "chemical", "detergent", "gel"]):
            discs.append("MAINT")
        return list(dict.fromkeys(discs))
    uc_df["disciplines"] = uc_df["title"].map(infer_uc_disciplines)

    def _disc_opts(rules: Optional[Dict[str, dict]]):
        r = (rules or {}).get("__global__", {})
        mode = str(r.get("discipline_filter", "soft")).lower()
        pen = float(r.get("discipline_penalty", 0.5))
        return mode, pen
    disc_mode, disc_penalty = _disc_opts(rules)
    # Precompute IFC ancestors and abstract flags
    ancestors = _build_ifc_ancestors_map(ifc_df)
    is_abs = {str(r["ifc_id"]): bool(r.get("is_abstract", False)) for _, r in ifc_df.iterrows()}

    rows = []
    flags_map = _compute_ifc_flags(ifc_df, ancestors)
    for _, r in ifc_df.iterrows():
        ifc_id = r["ifc_id"]
        # Prefer augmented text if available
        base_text = str(r.get("aug_text") or r.get("description") or "")
        desc = normalize_text(base_text)
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
            fac = str(u["facet"])
            if not _facet_allows_ifc(str(ifc_id), fac, rules or {}, ancestors, is_abs, flags_map):
                continue
            mult = _feature_multiplier(fac, flags_map.get(str(ifc_id), {}), {
                "pr_predefined_type": 0.10, "pr_manufacturer_ps": 0.10, "ss_group_system": 0.10, "type_bias": 0.05, "pset_count_scale": 0.002, "max_pset_boost": 0.25
            })
            # Discipline gating: if both sides have inferred disciplines and do not intersect
            if disc_mode in ("soft", "hard"):
                ifc_discs = set(flags_map.get(str(ifc_id), {}).get("disciplines", []) or [])
                uc_discs = set(u.get("disciplines", []) or [])
                if ifc_discs and uc_discs and ifc_discs.isdisjoint(uc_discs):
                    if disc_mode == "hard":
                        continue
                    else:
                        mult *= float(disc_penalty)
            s = score_pair(q, u["norm_title"]) * mult
            scores.append((u["code"], fac, u["title"], u["description"], float(s)))
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


def _fetch_embeddings_map(conn: psycopg.Connection, entity_type: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    with conn.cursor() as cur:
        cur.execute(
            "SELECT entity_id, embedding::text FROM text_embedding WHERE entity_type=%s",
            (entity_type,),
        )
        rows = cur.fetchall()
    for r in rows:
        if isinstance(r, dict):
            eid = str(r.get("entity_id"))
            vec = r.get("embedding")
        else:
            eid = str(r[0])
            vec = r[1]
        if eid and vec is not None:
            out[eid] = str(vec)
    return out

def _embedding_neighbors(
    conn: psycopg.Connection,
    target_entity_type: str,
    qvec_text: str,
    limit: int,
    facets: Optional[List[str]] = None,
    timeout_ms: Optional[int] = None,
) -> Dict[str, float]:
    """Return mapping neighbor_id -> cosine similarity in [0,1] using pgvector.

    - Uses ivfflat cosine distance (<=>) for ranking.
    - When `target_entity_type='uniclass'` and `facets` provided, restricts
      candidates to those facets via a join to `uniclass_item`.
    - Applies a per-statement timeout if provided.
    """
    if target_entity_type == 'uniclass' and facets:
        sql = (
            "SELECT te.entity_id, 1 - (te.embedding <=> %s::vector) AS score "
            "FROM text_embedding te JOIN uniclass_item u ON u.code = te.entity_id "
            "WHERE te.entity_type=%s AND u.facet = ANY(%s) "
            "ORDER BY te.embedding <=> %s::vector ASC LIMIT %s"
        )
        params = (qvec_text, target_entity_type, facets, qvec_text, limit)
    else:
        sql = (
            "SELECT entity_id, 1 - (embedding <=> %s::vector) AS score "
            "FROM text_embedding WHERE entity_type=%s "
            "ORDER BY embedding <=> %s::vector ASC LIMIT %s"
        )
        params = (qvec_text, target_entity_type, qvec_text, limit)

    with conn.cursor() as cur:
        # Optional safety timeout
        try:
            if timeout_ms is None:
                timeout_ms = int(os.getenv('EMBED_QUERY_TIMEOUT_MS', '5000'))
            if timeout_ms and timeout_ms > 0:
                cur.execute("SET LOCAL statement_timeout = %s", (int(timeout_ms),))
        except Exception:
            pass
        cur.execute(sql, params)
        rows = cur.fetchall()
    out: Dict[str, float] = {}
    for r in rows:
        code = r[0] if not isinstance(r, dict) else r["entity_id"]
        score = float(r[1] if not isinstance(r, dict) else r["score"])
        out[str(code)] = max(0.0, min(1.0, score))
    return out


def ensure_vector_indexes(conn: psycopg.Connection, lists: int = 100) -> None:
    """Create partial IVF Flat indexes for cosine distance if missing."""
    with conn.cursor() as cur:
        cur.execute(
            f"""
            CREATE INDEX IF NOT EXISTS text_embedding_ifc_cos_ivf
            ON text_embedding USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = {int(lists)})
            WHERE entity_type='ifc';
            """
        )
        cur.execute(
            f"""
            CREATE INDEX IF NOT EXISTS text_embedding_uc_cos_ivf
            ON text_embedding USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = {int(lists)})
            WHERE entity_type='uniclass';
            """
        )


def generate_candidates_blended(
    conn: psycopg.Connection,
    ifc_df: pd.DataFrame,
    uc_df: pd.DataFrame,
    synonyms: List[List[str]],
    top_k: int,
    embedding_weight: float,
    embedding_top_k: int,
    anchor_bonus: float,
    anchor_use_ancestors: bool,
    rules: Optional[Dict[str, dict]] = None,
) -> pd.DataFrame:
    # Precompute lex features on uc_df
    uc_work = uc_df.copy()
    uc_work["norm_title"] = uc_work["title"].fillna("").map(normalize_text).map(lambda s: expand_with_synonyms(s, synonyms))
    # Infer lightweight disciplines for Uniclass items
    def _infer_uc_disciplines(title: str) -> List[str]:
        t = (title or "").lower()
        discs: List[str] = []
        if any(k in t for k in ["duct", "hvac", "air ", "fan", "coil", "damper"]): discs.append("MECH")
        if any(k in t for k in ["pipe", "sanitary", "plumb", "drain", "valve", "tap", "waste"]): discs.append("PLUMB")
        if any(k in t for k in ["elect", "cable", "switch", "socket", "luminaire", "lighting", "breaker", "panel"]): discs.append("ELEC")
        if any(k in t for k in ["beam", "column", "slab", "rebar", "struct"]): discs.append("STRUCT")
        if any(k in t for k in ["door", "window", "glaz", "partition", "flooring", "covering", "furniture"]): discs.append("ARCH")
        if any(k in t for k in ["road", "pavement", "asphalt", "drainage", "culvert", "embankment"]): discs.append("CIVIL")
        if any(k in t for k in ["data", "network", "router", "server", "ict", "comm"]): discs.append("ICT")
        if any(k in t for k in ["clean", "chemical", "detergent", "gel"]): discs.append("MAINT")
        return list(dict.fromkeys(discs))
    uc_work["disciplines"] = uc_work["title"].map(_infer_uc_disciplines)
    # Add Uniclass facet as a subject label
    uc_work["subjects"] = uc_work["facet"].astype(str).str.upper().map(lambda v: [v] if v else [])
    # Dict for quick lookup by code (string keys)
    uc_info = {}
    for _, r in uc_work.iterrows():
        code = str(r["code"])
        facet = r["facet"]
        title = r["title"]
        desc = r.get("description", "")
        maps_raw = str(r.get("ifc_mappings", "") or "")
        map_set = set([m for m in maps_raw.split(";") if m])
        uc_info[code] = (facet, title, desc, map_set)
    # Fast lookup for normalized titles by code, avoid .loc in tight loops
    code_to_norm = pd.Series(uc_work["norm_title"].values, index=uc_work["code"].astype(str)).to_dict()
    uc_disc_map = pd.Series(uc_work["disciplines"].values, index=uc_work["code"].astype(str)).to_dict()
    uc_subj_map = pd.Series(uc_work["subjects"].values, index=uc_work["code"].astype(str)).to_dict()
    # Unified labels (disciplines + subjects) per Uniclass code
    uc_label_map = {k: list(dict.fromkeys((uc_disc_map.get(k) or []) + (uc_subj_map.get(k) or []))) for k in uc_work["code"].astype(str)}

    rows = []
    ancestors = _build_ifc_ancestors_map(ifc_df)
    is_abs = {str(r["ifc_id"]): bool(r.get("is_abstract", False)) for _, r in ifc_df.iterrows()}
    flags_map = _compute_ifc_flags(ifc_df, ancestors)
    base_alpha = float(embedding_weight)
    # Speed-ups: precompute IFC query text and facets map
    ifc_work = ifc_df.copy()
    if "aug_text" in ifc_work.columns:
        _bt = ifc_work["aug_text"].fillna("")
    else:
        _bt = ifc_work["description"].fillna("")
    _search = _bt.where(_bt.str.len() > 0, ifc_work["label"].fillna(""))
    ifc_work["norm_q"] = _search.map(normalize_text).map(lambda s: expand_with_synonyms(s, synonyms))
    ifc_q_map = pd.Series(ifc_work["norm_q"].values, index=ifc_work["ifc_id"].astype(str)).to_dict()
    facets_map = {str(r["ifc_id"]): facet_heuristics(str(r["ifc_id"])) for _, r in ifc_df.iterrows()}
    # Prefetch all IFC embeddings once to avoid per-IFC DB calls
    ifc_vecs = _fetch_embeddings_map(conn, 'ifc')
    # Discipline filter settings from rules.__global__ if present
    def _disc_opts(rules: Optional[Dict[str, dict]]):
        r = (rules or {}).get("__global__", {})
        mode = str(r.get("discipline_filter", "soft")).lower()
        pen = float(r.get("discipline_penalty", 0.5))
        return mode, pen
    disc_mode, disc_penalty = _disc_opts(rules)
    for r in ifc_df.itertuples(index=False):
        ifc_id = str(r.ifc_id)
        q = ifc_q_map.get(ifc_id, "")
        facets = facets_map.get(ifc_id, [])
        pool = uc_work if not facets else uc_work[uc_work["facet"].isin(facets)]
        if pool.empty:
            pool = uc_work

        # Embedding neighbor scores (if available)
        emb_scores: Dict[str, float] = {}
        qvec_text = ifc_vecs.get(ifc_id)
        if qvec_text:
            # Set ivfflat probes for better recall and ensure indexes
            with conn.cursor() as _c:
                try:
                    _c.execute("SET LOCAL ivfflat.probes = %s", (int(os.getenv('EMBED_IVF_PROBES', '10')),))
                except Exception:
                    pass
            emb_scores = _embedding_neighbors(conn, 'uniclass', qvec_text, embedding_top_k, facets=facets, timeout_ms=int(os.getenv('EMBED_QUERY_TIMEOUT_MS', '5000')))
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
                facet, title, _udesc, _maps = uc_info.get(code, ("", "", "", set()))
                # Skip if facet-filtering removed it
                if facets and facet not in facets:
                    continue
                # Rules: ensure this IFC is allowed for the facet
                if not _facet_allows_ifc(str(ifc_id), str(facet), rules or {}, ancestors, is_abs, flags_map):
                    continue
                # Use precomputed normalized title
                norm_t = code_to_norm.get(str(code))
                if not norm_t:
                    continue
                # Discipline/subject gating on lexical contribution
                mult_disc = 1.0
                if disc_mode in ("soft", "hard"):
                    ifc_labels = set(flags_map.get(str(ifc_id), {}).get("labels", []) or flags_map.get(str(ifc_id), {}).get("disciplines", []) or [])
                    uc_labels = set(uc_label_map.get(str(code), []) or [])
                    if ifc_labels and uc_labels and ifc_labels.isdisjoint(uc_labels):
                        if disc_mode == "hard":
                            continue
                        else:
                            mult_disc *= float(disc_penalty)
                lex_scores[code] = float(score_pair(q, norm_t) * mult_disc)
        else:
            # No embeddings found for this IFC; fall back to full lexical over the pool
            for _, u in pool.iterrows():
                fac = str(u["facet"])
                if not _facet_allows_ifc(str(ifc_id), fac, rules or {}, ancestors, is_abs, flags_map):
                    continue
                # Discipline/subject gating
                mult_disc = 1.0
                if disc_mode in ("soft", "hard"):
                    ifc_labels = set(flags_map.get(str(ifc_id), {}).get("labels", []) or flags_map.get(str(ifc_id), {}).get("disciplines", []) or [])
                    uc_labels = set(((u.get("disciplines") or [])) + ((u.get("subjects") or [])))
                    if ifc_labels and uc_labels and ifc_labels.isdisjoint(uc_labels):
                        if disc_mode == "hard":
                            continue
                        else:
                            mult_disc *= float(disc_penalty)
                lex_scores[str(u["code"])] = float(score_pair(q, u["norm_title"]) * mult_disc)

        # Blend scores; if no embeddings, act like alpha=0 (pure lexical)
        alpha = base_alpha if emb_scores else 0.0
        codes = set(lex_scores.keys()) | set(emb_scores.keys())
        scored = []
        for code in codes:
            lex = lex_scores.get(code, 0.0)
            emb = emb_scores.get(code, 0.0)
            # Retrieve Uniclass facet/title for this code first
            ufacet, title, udesc, umaps = uc_info.get(code, ("", "", "", set()))
            # Apply feature multiplier as post-combination adjuster
            mult = _feature_multiplier(ufacet, flags_map.get(str(ifc_id), {}), {
                "pr_predefined_type": 0.10, "pr_manufacturer_ps": 0.10, "ss_group_system": 0.10, "type_bias": 0.05, "pset_count_scale": 0.002, "max_pset_boost": 0.25
            })
            # Discipline/subject gating also applied at final score to catch embedding-only candidates
            if disc_mode in ("soft", "hard"):
                ifc_labels = set(flags_map.get(str(ifc_id), {}).get("labels", []) or flags_map.get(str(ifc_id), {}).get("disciplines", []) or [])
                uc_labels = set(uc_label_map.get(str(code), []) or [])
                if ifc_labels and uc_labels and ifc_labels.isdisjoint(uc_labels):
                    if disc_mode == "hard":
                        continue
                    else:
                        mult *= float(disc_penalty)
            final = ((1 - alpha) * lex + alpha * emb) * mult
            # Anchor guidance: boost if Uniclass row maps to this IFC (or its ancestors if enabled)
            if umaps:
                if anchor_use_ancestors:
                    anc = set(ancestors.get(str(ifc_id), [])) | {str(ifc_id)}
                    if any(m in anc for m in umaps):
                        final = min(1.0, final + anchor_bonus)
                else:
                    if str(ifc_id) in umaps:
                        final = min(1.0, final + anchor_bonus)
            scored.append((code, ufacet, title, udesc, float(final)))
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
    anchor_bonus: float,
    anchor_use_ancestors: bool,
    rules: Optional[Dict[str, dict]] = None,
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
    # Fast lookup for normalized IFC text by id to avoid .loc
    ifc_id_to_norm = pd.Series(ifc_work["norm_text"].values, index=ifc_work["ifc_id"].astype(str)).to_dict()
    ancestors = _build_ifc_ancestors_map(ifc_work)
    is_abs = {str(r["ifc_id"]): bool(r.get("is_abstract", False)) for _, r in ifc_work.iterrows()}
    flags_map = _compute_ifc_flags(ifc_work, ancestors)

    rows = []
    # Discipline options
    def _disc_opts(rules: Optional[Dict[str, dict]]):
        r = (rules or {}).get("__global__", {})
        mode = str(r.get("discipline_filter", "soft")).lower()
        pen = float(r.get("discipline_penalty", 0.5))
        return mode, pen
    disc_mode, disc_penalty = _disc_opts(rules)
    base_alpha = float(embedding_weight)
    for _, u in uc_df.iterrows():
        code = str(u["code"])
        facet = str(u.get("facet", ""))
        title = str(u.get("title", ""))
        q = expand_with_synonyms(normalize_text(title), synonyms)
        u_tokens = _tokenize_set(title)
        umaps = set([m for m in str(u.get("ifc_mappings", "") or "").split(";") if m])

        # If facet is globally skipped, then there will be no allowed IFCs
        if (rules or {}).get(facet.upper(), {}).get("skip"):
            continue

        # Embedding neighbors from Uniclass -> IFC
        emb_scores: Dict[str, float] = {}
        qvec_text = _fetch_embedding(conn, "uniclass", code)
        if qvec_text:
            with conn.cursor() as _c:
                try:
                    _c.execute("SET LOCAL ivfflat.probes = %s", (int(os.getenv('EMBED_IVF_PROBES', '10')),))
                except Exception:
                    pass
            emb_scores = _embedding_neighbors(conn, 'ifc', qvec_text, embedding_top_k)

        # Lexical: restrict to embedding neighbors if present, else full IFC set
        restrict_ids = set(emb_scores.keys()) if emb_scores else None
        lex_scores: Dict[str, float] = {}
        if restrict_ids:
            for ifc_id in restrict_ids:
                if not _facet_allows_ifc(ifc_id, facet, rules or {}, ancestors, is_abs, flags_map):
                    continue
                norm_t = ifc_id_to_norm.get(str(ifc_id))
                if not norm_t:
                    continue
                # Discipline gating: infer UC disciplines from title text on the fly
                uc_discs = set()
                lt = title.lower()
                if any(k in lt for k in ["duct", "hvac", "air ", "fan", "coil", "damper"]): uc_discs.add("MECH")
                if any(k in lt for k in ["pipe", "sanitary", "plumb", "drain", "valve", "tap", "waste"]): uc_discs.add("PLUMB")
                if any(k in lt for k in ["elect", "cable", "switch", "socket", "luminaire", "lighting", "breaker", "panel"]): uc_discs.add("ELEC")
                if any(k in lt for k in ["beam", "column", "slab", "rebar", "struct"]): uc_discs.add("STRUCT")
                if any(k in lt for k in ["door", "window", "glaz", "partition", "flooring", "covering", "furniture"]): uc_discs.add("ARCH")
                if any(k in lt for k in ["road", "pavement", "asphalt", "drainage", "culvert", "embankment"]): uc_discs.add("CIVIL")
                if any(k in lt for k in ["data", "network", "router", "server", "ict", "comm"]): uc_discs.add("ICT")
                if any(k in lt for k in ["clean", "chemical", "detergent", "gel"]): uc_discs.add("MAINT")
                mult_disc = 1.0
                if disc_mode in ("soft", "hard"):
                    # include Uniclass facet as subject label
                    try:
                        uc_discs.add(facet.upper())
                    except Exception:
                        pass
                    ifc_labels = set(flags_map.get(str(ifc_id), {}).get("labels", []) or flags_map.get(str(ifc_id), {}).get("disciplines", []) or [])
                    if uc_discs and ifc_labels and ifc_labels.isdisjoint(uc_discs):
                        if disc_mode == "hard":
                            continue
                        else:
                            mult_disc *= float(disc_penalty)
                lex_scores[ifc_id] = float(score_pair(q, norm_t) * mult_disc)
        else:
            for _, r in ifc_work.iterrows():
                ifc_id = str(r["ifc_id"])
                if not _facet_allows_ifc(ifc_id, facet, rules or {}, ancestors, is_abs, flags_map):
                    continue
                lt = title.lower()
                uc_discs = set()
                if any(k in lt for k in ["duct", "hvac", "air ", "fan", "coil", "damper"]): uc_discs.add("MECH")
                if any(k in lt for k in ["pipe", "sanitary", "plumb", "drain", "valve", "tap", "waste"]): uc_discs.add("PLUMB")
                if any(k in lt for k in ["elect", "cable", "switch", "socket", "luminaire", "lighting", "breaker", "panel"]): uc_discs.add("ELEC")
                if any(k in lt for k in ["beam", "column", "slab", "rebar", "struct"]): uc_discs.add("STRUCT")
                if any(k in lt for k in ["door", "window", "glaz", "partition", "flooring", "covering", "furniture"]): uc_discs.add("ARCH")
                if any(k in lt for k in ["road", "pavement", "asphalt", "drainage", "culvert", "embankment"]): uc_discs.add("CIVIL")
                if any(k in lt for k in ["data", "network", "router", "server", "ict", "comm"]): uc_discs.add("ICT")
                if any(k in lt for k in ["clean", "chemical", "detergent", "gel"]): uc_discs.add("MAINT")
                mult_disc = 1.0
                if disc_mode in ("soft", "hard"):
                    try:
                        uc_discs.add(facet.upper())
                    except Exception:
                        pass
                    ifc_labels = set(flags_map.get(str(ifc_id), {}).get("labels", []) or flags_map.get(str(ifc_id), {}).get("disciplines", []) or [])
                    if uc_discs and ifc_labels and ifc_labels.isdisjoint(uc_discs):
                        if disc_mode == "hard":
                            continue
                        else:
                            mult_disc *= float(disc_penalty)
                lex_scores[ifc_id] = float(score_pair(q, r["norm_text"]) * mult_disc)

        alpha = base_alpha if emb_scores else 0.0
        ids = set(lex_scores.keys()) | set(emb_scores.keys())
        scored = []
        for ifc_id in ids:
            lex = lex_scores.get(ifc_id, 0.0)
            emb = emb_scores.get(ifc_id, 0.0)
            mult = _feature_multiplier(facet, flags_map.get(str(ifc_id), {}), {
                "pr_predefined_type": 0.10, "pr_manufacturer_ps": 0.10, "ss_group_system": 0.10, "type_bias": 0.05, "pset_count_scale": 0.002, "max_pset_boost": 0.25
            })
            # Discipline/subject gating at final combination stage
            if disc_mode in ("soft", "hard"):
                lt = title.lower()
                uc_discs = set()
                if any(k in lt for k in ["duct", "hvac", "air ", "fan", "coil", "damper"]): uc_discs.add("MECH")
                if any(k in lt for k in ["pipe", "sanitary", "plumb", "drain", "valve", "tap", "waste"]): uc_discs.add("PLUMB")
                if any(k in lt for k in ["elect", "cable", "switch", "socket", "luminaire", "lighting", "breaker", "panel"]): uc_discs.add("ELEC")
                if any(k in lt for k in ["beam", "column", "slab", "rebar", "struct"]): uc_discs.add("STRUCT")
                if any(k in lt for k in ["door", "window", "glaz", "partition", "flooring", "covering", "furniture"]): uc_discs.add("ARCH")
                if any(k in lt for k in ["road", "pavement", "asphalt", "drainage", "culvert", "embankment"]): uc_discs.add("CIVIL")
                if any(k in lt for k in ["data", "network", "router", "server", "ict", "comm"]): uc_discs.add("ICT")
                if any(k in lt for k in ["clean", "chemical", "detergent", "gel"]): uc_discs.add("MAINT")
                try:
                    uc_discs.add(facet.upper())
                except Exception:
                    pass
                ifc_labels = set(flags_map.get(str(ifc_id), {}).get("labels", []) or flags_map.get(str(ifc_id), {}).get("disciplines", []) or [])
                if uc_discs and ifc_labels and ifc_labels.isdisjoint(uc_discs):
                    if disc_mode == "hard":
                        continue
                    else:
                        mult *= float(disc_penalty)
            final = ((1 - alpha) * lex + alpha * emb) * mult
            # Token-overlap penalty: if Uniclass title shares no tokens with IFC id/label/desc, downweight
            try:
                if not (u_tokens & set(_tokenize_set(f"{ifc_id} {ifc_info.get(ifc_id, ('',''))[0]} {ifc_info.get(ifc_id, ('',''))[1]}"))):
                    final *= 0.6
            except Exception:
                pass
            if umaps:
                if anchor_use_ancestors:
                    anc = set(ancestors.get(str(ifc_id), [])) | {str(ifc_id)}
                    if any(m in anc for m in umaps):
                        final = min(1.0, final + anchor_bonus)
                else:
                    if str(ifc_id) in umaps:
                        final = min(1.0, final + anchor_bonus)
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
        # Clamp confidence into [0.0, 1.0] to satisfy DB constraint
        conf = float(r["score"])
        if conf < 0.0:
            conf = 0.0
        elif conf > 1.0:
            conf = 1.0
        payload.append((
            r["ifc_id"], r["code"], rel, conf,
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


def reset_mappings(
    conn: psycopg.Connection,
    uniclass_revision: Optional[str] = None,
    facets: Optional[List[str]] = None,
) -> int:
    """Delete existing mappings, scoped by Uniclass revision and optional facets.

    - If uniclass_revision is provided, deletes only rows with matching
      source_uniclass_revision.
    - If facets are provided, deletes only rows whose Uniclass code belongs to
      those facets (via join to uniclass_item).
    Returns number of deleted rows (best-effort estimate).
    """
    conds = []
    params: List[object] = []
    if uniclass_revision:
        conds.append("m.source_uniclass_revision = %s")
        params.append(uniclass_revision)
    facet_list: List[str] = []
    if facets:
        facet_list = [str(f).upper().strip() for f in facets if str(f).strip()]
    deleted = 0
    with conn.cursor() as cur:
        if facet_list:
            where = (" AND ".join(conds)) if conds else "TRUE"
            placeholders = ",".join(["%s"] * len(facet_list))
            sql = f"""
                DELETE FROM ifc_uniclass_map m
                USING uniclass_item u
                WHERE u.code = m.code
                  AND u.facet = ANY (ARRAY[{placeholders}])
                  AND {where}
            """
            cur.execute(sql, [*facet_list, *params])
        else:
            where = (" AND ".join(conds)) if conds else "TRUE"
            sql = f"DELETE FROM ifc_uniclass_map m WHERE {where}"
            cur.execute(sql, params)
        # Best-effort rowcount
        try:
            deleted = cur.rowcount or 0
        except Exception:
            deleted = 0
    return int(deleted)

def main():
    ap = argparse.ArgumentParser(description="IFC ↔ Uniclass ETL and candidate mapping")
    ap.add_argument("--config", required=True, help="Path to YAML settings")
    ap.add_argument("--load", action="store_true", help="Load IFC and Uniclass into DB")
    ap.add_argument("--candidates", action="store_true", help="Generate top-k candidate links and auto-accept above threshold")
    ap.add_argument("--export", action="store_true", help="Export viewer JSON from accepted mappings")
    # Optional maintenance flags
    ap.add_argument("--reset-mappings", action="store_true", help="Delete existing mappings for current Uniclass revision (optionally filter by --reset-facets)")
    ap.add_argument("--reset-facets", type=str, default="", help="Comma-separated Uniclass facets to reset (e.g., PR,SS)")
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

        # Optional cleanup before generating candidates
        if args.reset_mappings:
            facets = [x.strip() for x in (args.reset_facets or "").split(",") if x.strip()]
            n = reset_mappings(conn, uniclass_revision=s.uniclass_revision, facets=facets)
            print(f"[cleanup] Deleted {n} mapping rows for revision '{s.uniclass_revision}'" + (f" and facets {facets}" if facets else ""))

        if args.embed:
            # Prepare texts: IFC uses description/definition optionally augmented with parent chain; Uniclass uses enriched title/context
            ifc_items: List[Tuple[str, str]] = []
            # Prepare flags to append concise feature hints into embedding text
            anc_map = _build_ifc_ancestors_map(ifc_df)
            flags_map = _compute_ifc_flags(ifc_df, anc_map)
            for _, r in ifc_df.iterrows():
                eid = str(r["ifc_id"]) if r.get("ifc_id") is not None else None
                if not eid:
                    continue
                if s.ifc_use_parent_context and (r.get('aug_text') or ''):
                    base = str(r.get('aug_text') or '').strip()
                else:
                    desc = str(r.get('description','') or '').strip()
                    base = desc if desc else str(r.get('label','') or '').strip()
                # Append compact feature hints
                fl = flags_map.get(eid, {})
                feat_bits = []
                if fl.get('is_type'): feat_bits.append('role:Type')
                if fl.get('is_instance'): feat_bits.append('role:Instance')
                if fl.get('is_product'): feat_bits.append('kind:Product')
                if fl.get('is_spatial'): feat_bits.append('kind:Spatial')
                if fl.get('is_process'): feat_bits.append('kind:Process')
                if fl.get('is_group_system'): feat_bits.append('kind:System')
                if fl.get('has_predefined_type'): feat_bits.append('predefinedType')
                pc = int(fl.get('pset_count', 0) or 0)
                if pc:
                    feat_bits.append(f'psets:{pc}')
                if bool(fl.get('has_manufacturer_ps', False)):
                    feat_bits.append('pset:ManufacturerTypeInformation')
                extra = (" | " + ", ".join(feat_bits)) if feat_bits else ""
                # Include the IFC identifier token to help matching (e.g., 'IfcAirTerminalBox')
                text = (f"{eid}. " + base + extra).strip()
                ifc_items.append((eid, text))

            uc_items: List[Tuple[str, str]] = []
            for _, r in uc_df.iterrows():
                eid = str(r["code"]) if r.get("code") is not None else None
                if not eid:
                    continue
                title = str(r.get('title','') or '').strip()
                facet = str(r.get('facet','') or '').upper()
                # Tokenize code to expose hierarchy (e.g., 'Pr 20 93' instead of 'Pr_20_93')
                code_tokens = eid.replace('_', ' ')
                # Build enriched text with facet and code context to improve semantic alignment
                text = f"[{facet}] {code_tokens}: {title}"
                uc_items.append((eid, text))

            # Ensure vector indexes exist (safe to call repeatedly)
            try:
                ensure_vector_indexes(conn, lists=int(s.embed_ivf_lists))
            except Exception as e:
                print(f"[embed] index ensure failed (non-fatal): {e}")

            print(f"[embed] Generating embeddings using model '{s.embed_model}' at {s.embed_endpoint}")
            n_ifc = generate_and_store_embeddings(conn, "ifc", ifc_items, s.embed_model, s.embed_endpoint, s.embed_batch_size, s.embed_timeout_s, expected_dim=s.embed_expected_dim)
            n_uc = generate_and_store_embeddings(conn, "uniclass", uc_items, s.embed_model, s.embed_endpoint, s.embed_batch_size, s.embed_timeout_s, expected_dim=s.embed_expected_dim)
            print(f"[embed] Upserted {n_ifc} IFC and {n_uc} Uniclass embeddings")

        if args.candidates:
            # Ensure vector indexes exist before neighbor queries
            try:
                ensure_vector_indexes(conn, lists=int(s.embed_ivf_lists))
            except Exception as e:
                print(f"[candidates] index ensure failed (non-fatal): {e}")
            # Set probes for this session for better recall/latency tradeoff
            try:
                with conn.cursor() as cur:
                    cur.execute("SET ivfflat.probes = %s", (int(s.embed_ivf_probes),))
                    cur.execute("SET statement_timeout = %s", (int(s.embed_query_timeout_ms),))
            except Exception:
                pass
            use_embed = s.embedding_weight and float(s.embedding_weight) > 0
            direction = (s.match_direction or 'uniclass_to_ifc').lower()
            if direction != 'uniclass_to_ifc':
                print("[info] ifc_to_uniclass is disabled; forcing direction=uniclass_to_ifc")
                direction = 'uniclass_to_ifc'
            if use_embed:
                cands = generate_candidates_uc2ifc_blended(conn, ifc_df, uc_df, s.synonyms, s.top_k, s.embedding_weight, s.embedding_top_k, s.anchor_bonus, s.anchor_use_ancestors, s.matching_rules)
            else:
                # Lexical-only reverse direction using the same blended helper with alpha=0 via parameters
                cands = generate_candidates_uc2ifc_blended(conn, ifc_df, uc_df, s.synonyms, s.top_k, 0.0, s.embedding_top_k, s.anchor_bonus, s.anchor_use_ancestors, s.matching_rules)

            # Optional LLM rerank
            if int(s.rerank_top_n or 0) > 0:
                try:
                    # Quick health check to avoid long hangs
                    import requests  # noqa: F401
                    ok = True
                except Exception:
                    ok = False
                if not ok:
                    print("[rerank] requests not available; skipping rerank.")
                else:
                    # Proceed with rerank, per-call timeout is enforced
                    try:
                        cands = rerank_candidates_with_llm(
                            cands,
                            ifc_df,
                            uc_df,
                            direction=direction,
                            top_n=int(s.rerank_top_n),
                            model=s.rerank_model,
                            endpoint=s.rerank_endpoint,
                            temperature=float(s.rerank_temperature),
                            max_tokens=int(s.rerank_max_tokens),
                            fewshot_per_facet=int(s.rerank_fewshot_per_facet),
                            timeout_s=float(s.rerank_timeout_s),
                        )
                    except Exception as e:
                        print(f"[rerank] skipped due to error: {e}")
            cands_path = s.output_dir / "candidates.csv"
            cands.to_csv(cands_path, index=False)
            write_candidate_edges(conn, cands, s.ifc_schema_version, s.uniclass_revision, s.auto_accept)

        if args.export:
            export_viewer_json(conn, s.viewer_json)


if __name__ == "__main__":
    main()
