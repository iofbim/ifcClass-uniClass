import argparse
import json
import os
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from uuid import UUID, uuid4
import time

import pandas as pd
import psycopg
from psycopg.rows import dict_row
from rapidfuzz import fuzz
import yaml
import re


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
    features_enabled: bool
    feature_attribute_tokens: List[str]
    evaluation_baseline_run_id: Optional[str]
    evaluation_max_precision_drop: float
    evaluation_min_recall: float
    evaluation_fail_on_regression: bool
    evaluation_accept_threshold: float
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
    # Optional OpenAI-specific overrides
    openai_embed_model: Optional[str]
    openai_embed_expected_dim: Optional[int]
    openai_rerank_model: Optional[str]
    # Optional: future room for provider-specific defaults (kept minimal here)
    

ATTRIBUTE_COLUMN_TOKENS: Tuple[str, ...] = (
    "attribute",
    "attributes",
    "group",
    "sub group",
    "sub-group",
    "subgroup",
    "section",
    "object",
    "sub object",
    "sub-object",
    "cobie",
    "nrm",
    "nrm1",
    "nbs",
    "ifc",
    "cesmm",
    "classification",
    "mapping",
    "alias",
    "aka",
    "synonym",
    "reference",
    "category",
    "type",
    "includes",
    "excludes",
    "notes",
)


FACETS: Tuple[str, ...] = (
    "EF",
    "SS",
    "PR",
    "TE",
    "PM",
    "AC",
    "EN",
    "SL",
    "RO",
    "CO",
    "MA",
    "FI",
    "PC",
    "RK",
    "ZZ",
)


def facet_from_filename(name: str) -> Optional[str]:
    """Best-effort facet extraction from a Uniclass filename stem."""
    if not name:
        return None
    up = name.upper()
    # Prefer exact token matches such as `_PR_` or prefix/suffix forms like `PR_`
    for fac in FACETS:
        token = f"_{fac}_"
        if token in up:
            return fac
        if up.startswith(f"{fac}_") or up.endswith(f"_{fac}"):
            return fac
    # Fall back to any substring hit
    for fac in FACETS:
        if fac in up:
            return fac
    return None

def _looks_like_attribute_column(name: str) -> bool:
    if not isinstance(name, str):
        return False
    lowered = name.strip().lower()
    if not lowered:
        return False
    core_exclusions = {
        "code",
        "uniclass code",
        "item code",
        "pr code",
        "ef code",
        "ss code",
        "title",
        "item title",
        "name",
        "description",
        "item description",
        "facet",
        "revision",
    }
    if lowered in core_exclusions:
        return False
    if lowered.startswith("unnamed:"):
        return False
    normalized = lowered.replace("_", " ").replace("-", " ")
    return any(token in normalized for token in ATTRIBUTE_COLUMN_TOKENS)

def _split_feature_values(value) -> List[str]:
    if value is None:
        return []
    try:
        if pd.isna(value):
            return []
    except Exception:
        pass
    if isinstance(value, float):
        try:
            if math.isnan(value):
                return []
        except TypeError:
            pass
    if isinstance(value, (list, tuple, set)):
        raw_parts = list(value)
    else:
        text_value = str(value).strip()
        if not text_value:
            return []
        lowered_val = text_value.lower()
        if lowered_val in {"nan", "<na>"}:
            return []
        import re
        raw_parts = re.split(r"[\n\r;,|]+", text_value)
    out: List[str] = []
    seen = set()
    for part in raw_parts:
        part_str = str(part).strip()
        if not part_str:
            continue
        key = part_str.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(part_str)
    return out
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

    feature_cfg = cfg.get("features", {}) or {}
    features_enabled = bool(feature_cfg.get("enabled", True))
    raw_attr_tokens = feature_cfg.get("attribute_tokens") or []
    attr_tokens: List[str] = []
    for tok in raw_attr_tokens:
        st = str(tok).strip().lower()
        if st:
            attr_tokens.append(st)
    if attr_tokens:
        global ATTRIBUTE_COLUMN_TOKENS
        ATTRIBUTE_COLUMN_TOKENS = tuple(dict.fromkeys(list(ATTRIBUTE_COLUMN_TOKENS) + attr_tokens))
    effective_attr_tokens = list(ATTRIBUTE_COLUMN_TOKENS)

    eval_cfg = (cfg.get("evaluation", {}) or {})
    eval_baseline = str(eval_cfg.get("baseline_run_id") or "").strip() or None
    eval_max_drop = float(eval_cfg.get("max_precision_drop", 0.05))
    eval_min_recall = float(eval_cfg.get("min_recall", 0.7))
    eval_fail = bool(eval_cfg.get("fail_on_regression", False))
    eval_accept_threshold = float(eval_cfg.get("accept_threshold", match_cfg.get("auto_accept_threshold", 0.7)))
    # OpenAI-specific optional overrides
    emb_openai_model = str((cfg.get("embedding", {}) or {}).get("openai_model", os.getenv("OPENAI_EMBED_MODEL", ""))) or None
    emb_openai_expected_dim = (lambda v: int(v) if (v is not None and str(v).strip() != "") else None)((cfg.get("embedding", {}) or {}).get("openai_expected_dim", os.getenv("OPENAI_EMBED_EXPECTED_DIM", "")))
    rr_openai_model = str((cfg.get("rerank", {}) or {}).get("openai_model", os.getenv("OPENAI_RERANK_MODEL", ""))) or None

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
        features_enabled=features_enabled,
        feature_attribute_tokens=effective_attr_tokens,
        evaluation_baseline_run_id=eval_baseline,
        evaluation_max_precision_drop=eval_max_drop,
        evaluation_min_recall=eval_min_recall,
        evaluation_fail_on_regression=eval_fail,
        evaluation_accept_threshold=eval_accept_threshold,
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
        openai_embed_model=emb_openai_model,
        openai_embed_expected_dim=emb_openai_expected_dim,
        openai_rerank_model=rr_openai_model,
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


def _openai_generate(prompt: str, model: str, temperature: float = 0.0, max_tokens: int = 512, timeout_s: float = 60.0) -> Optional[str]:
    """Call OpenAI Chat Completions (v1 SDK) to generate JSON text.

    Requires OPENAI_API_KEY in environment. Optional OPENAI_BASE_URL supported for Azure/proxy.
    """
    try:
        from openai import OpenAI  # type: ignore
    except (ImportError, ModuleNotFoundError):
        print("[llm] OpenAI SDK not installed. Run: pip install openai")
        return None
    try:
        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL") or None,
            timeout=timeout_s,  # type: ignore
        )
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful AI that returns JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=float(temperature or 0.0),
            max_tokens=int(max_tokens or 256),
        )
        return (resp.choices[0].message.content or "") if resp and resp.choices else None
    except Exception as e:
        print(f"[llm] OpenAI generate error: {e}")
        return None

def _ollama_warmup(model: str, endpoint: str, timeout_s: float = 180.0) -> None:
    """Warm up Ollama model load with a trivial prompt and generous timeout.

    This helps avoid timeouts on the first real request when the model is still loading.
    Non-fatal on failure.
    """
    try:
        _ = _ollama_generate("OK", model=model, endpoint=endpoint, temperature=0.0, max_tokens=8, timeout_s=timeout_s)
    except Exception:
        pass


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
    "ARCH", "STRUCT", "CIVIL", "MECH", "PLUMB", "ELEC", "ICT", "MAINT",
    # Subjects / roles (used for IFC subject tagging)
    "ROLE","PM","ACTIVITIES","COMPLEX", "ENTITIES", "FORMSOFINFORMATION", "MATERIALS", "PROPERTIESANDCHARACTERISTICS", "ELEMENTSANDFUNCTIONS", "PRODUCTS", "RISK", "SPACE/LOCATIONS", "SYSTEMS", "TOOLSANDEQUIPMENT"
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
        # Extract first JSON-ish array with normalization
        import re
        s = text
        # Normalize curly quotes and stray unicode to plain ASCII quotes to help JSON parser
        s = s.replace("\u201C", '"').replace("\u201D", '"').replace("\u2018", "'").replace("\u2019", "'")
        s = s.replace("“", '"').replace("”", '"').replace("’", "'")
        # Strip JS-style comments that some models append
        s = re.sub(r"//.*", "", s)
        # Extract first bracketed array
        m = re.search(r"\[(?:.|\n)*?\]", s)
        if m:
            s = m.group(0)
        arr = json.loads(s)
        if isinstance(arr, list):
            out = []
            for v in arr:
                if isinstance(v, str):
                    val = v.strip().upper()
                    if val in ALLOWED_LABELS:
                        out.append(val)
            # Dedup preserve order
            seen = set(); res = []
            for v in out:
                if v not in seen:
                    res.append(v); seen.add(v)
            return res
    except Exception:
        pass
    # Fallback: heuristic extraction from free text
    try:
        t = (text or "").upper()
        # Map common variants to allowed labels
        synonyms = {
            "PROJECT MANAGEMENT": "PM",
            "PROJECT-MANAGEMENT": "PM",
            "PM": "PM",
            "TASK": "ACTIVITIES",
            "TASKS": "ACTIVITIES",
            "OPERATION": "ACTIVITIES",
            "OPERATIONS": "ACTIVITIES",
            "MAINTENANCE": "MAINT",
            "MECHANICAL": "MECH",
            "ELECTRICAL": "ELEC",
            "PLUMBING": "PLUMB",
            "STRUCTURAL": "STRUCT",
            "ARCHITECTURE": "ARCH",
            "ARCHITECTURAL": "ARCH",
            "INFORMATION TECHNOLOGY": "ICT",
            "INFORMATION AND COMMUNICATIONS TECHNOLOGY": "ICT",
        }
        found: List[str] = []
        # Exact label hits first
        for lab in ALLOWED_LABELS:
            if lab in t:
                found.append(lab)
        # Synonym hits
        for k, v in synonyms.items():
            if k in t and v not in found:
                found.append(v)
        # Dedup preserve order
        res = []
        seen = set()
        for v in found:
            if v in ALLOWED_LABELS and v not in seen:
                res.append(v); seen.add(v)
        return res
    except Exception:
        return []

def classify_items_with_llm(
    items: List[Tuple[str, str, str, str]],  # (key, item_type, title, description)
    model: str,
    endpoint: str,
    temperature: float = 0.0,
    max_tokens: int = 96,
    timeout_s: float = 45.0,
    warmup_timeout_s: float = 180.0,
    progress_every: int = 200,
    sleep_between: float = 0.0,
    max_retries: int = 2,
    provider: str = "ollama",
    cache_path: Optional[Path] = None,
    flush_every: int = 50,
    skip_cached: bool = True,
    alias_map: Optional[Dict[str, List[str]]] = None,  # master_code -> list of alias codes to mirror
    extras_map: Optional[Dict[str, str]] = None,        # key -> extra context for prompt
    debug: bool = False,
) -> Dict[str, List[str]]:
    """Classify items via Ollama or OpenAI, with retries and basic rate limiting.

    When cache_path is provided, this function will:
      - Load existing taxonomy cache
      - Skip items already present (if skip_cached=True)
      - Write incremental updates to cache after every flush_every items
    """
    # Load existing cache when live-updating
    existing: Dict[str, List[str]] = {}
    if cache_path is not None:
        try:
            existing = _load_taxonomy_cache(cache_path)
        except Exception:
            existing = {}
    out: Dict[str, List[str]] = dict(existing) if existing else {}
    # If alias_map provided and we already have master entries, propagate to aliases
    if alias_map and out and cache_path is not None:
        changed = False
        for master, aliases in alias_map.items():
            mkey = f"uc:{master}"
            labs = out.get(mkey)
            if not labs:
                continue
            for a in (aliases or []):
                akey = f"uc:{a}"
                if akey not in out:
                    out[akey] = list(labs)
                    changed = True
        if changed:
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(out, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"[classify] cache propagate error: {e}")
    # Warm up Ollama once to load the model
    if (provider or "ollama").lower() != "openai":
        _ollama_warmup(model=model, endpoint=endpoint, timeout_s=warmup_timeout_s)
    from time import sleep
    # Optionally filter items that are already in cache
    work_items = items
    if skip_cached and out:
        work_items = [(k, t, ti, d) for (k, t, ti, d) in items if k not in out]
    total = len(work_items)
    done = 0
    for idx, (key, item_type, title, desc) in enumerate(work_items, start=1):
        extra = ""
        if extras_map is not None:
            extra = str(extras_map.get(key, "") or "")
        prompt = _format_label_prompt(item_type, key, title or "", desc or "", extra=extra)
        if debug:
            try:
                title_preview = (title or "").replace("\n"," ")
                if len(title_preview) > 100:
                    title_preview = title_preview[:100] + "..."
                desc_used = bool((desc or "").strip())
                desc_len = len((desc or "").strip())
                extra_preview = (extra or "").replace("\n"," ")
                if len(extra_preview) > 120:
                    extra_preview = extra_preview[:120] + "..."
                print(f"[debug.classify] key={key} type={item_type} desc_used={desc_used} desc_len={desc_len} title=\"{title_preview}\" extra=\"{extra_preview}\"")
            except Exception:
                pass
        resp: Optional[str] = None
        for attempt in range(max(1, int(max_retries) + 1)):
            tmo = float(timeout_s) * (1.5 ** attempt)
            if (provider or "ollama").lower() == "openai":
                resp = _openai_generate(prompt, model=model, temperature=temperature, max_tokens=max_tokens, timeout_s=tmo)
            else:
                resp = _ollama_generate(prompt, model=model, endpoint=endpoint, temperature=temperature, max_tokens=max_tokens, timeout_s=tmo)
            if resp:
                break
            if sleep_between > 0:
                sleep(min(2.0, float(sleep_between)))
        labels = _parse_label_array(resp)
        out[key] = labels
        # If key is a Uniclass master and alias_map provided, propagate to aliases
        if alias_map and key.startswith("uc:"):
            base = key.split(":",1)[1]
            for a in (alias_map.get(base) or []):
                out[f"uc:{a}"] = list(labels)
        done += 1
        if cache_path is not None and (flush_every <= 1 or (done % int(max(1, flush_every)) == 0)):
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(out, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"[classify] cache write error: {e}")
        if progress_every and idx % int(progress_every) == 0:
            print(f"[classify] progress: {idx}/{total}")
        if sleep_between > 0:
            sleep(float(sleep_between))
    # Final flush
    if cache_path is not None:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[classify] cache final write error: {e}")
    return out

def _load_taxonomy_cache(path: Optional[Path] = None) -> Dict[str, List[str]]:
    try:
        p = path or (Path("output") / "taxonomy_cache.json")
        if not p.exists():
            return {}
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        clean: Dict[str, List[str]] = {}
        for k, v in (data.items() if isinstance(data, dict) else []):
            if isinstance(v, list):
                vals = [str(x).upper() for x in v if isinstance(x, (str,)) and str(x).strip()]
                # keep only allowed labels
                vals = [x for x in vals if x in ALLOWED_LABELS]
                clean[str(k)] = list(dict.fromkeys(vals))
        return clean
    except Exception:
        return {}


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
    provider: str = "ollama",
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
            if (provider or "ollama").lower() == "openai":
                resp = _openai_generate(prompt, model=model, temperature=temperature, max_tokens=max_tokens, timeout_s=timeout_s)
            else:
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
            if (provider or "ollama").lower() == "openai":
                resp = _openai_generate(prompt, model=model, temperature=temperature, max_tokens=max_tokens, timeout_s=timeout_s)
            else:
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


def _openai_embed(text: str, model: str, timeout_s: float) -> Optional[List[float]]:
    """Create an embedding via OpenAI Embeddings API (v1 SDK).

    Uses OPENAI_API_KEY and optional OPENAI_BASE_URL from environment.
    """
    try:
        from openai import OpenAI  # type: ignore
    except (ImportError, ModuleNotFoundError):
        print("[embed] OpenAI SDK not installed. Run: pip install openai")
        return None
    try:
        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL") or None,
            timeout=timeout_s,  # type: ignore
        )
        resp = client.embeddings.create(model=model, input=text)
        vec = resp.data[0].embedding if resp and resp.data else None
        if not isinstance(vec, list):
            raise ValueError("Unexpected OpenAI embedding response format")
        return [float(x) for x in vec]
    except Exception as e:
        print(f"[embed] OpenAI embed error (len={len(text)}): {e}")
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
    provider: str = "ollama",
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
            if (provider or "ollama").lower() == "openai":
                vec = _openai_embed(text, model=model, timeout_s=timeout_s)
            else:
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


def read_ifc_json(path: Path, schema_version: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rows = []
    feature_rows: List[Dict[str, object]] = []
    seen_features: set = set()

    def add_feature(fid: str, key: str, value: object, source: str, confidence: float = 1.0) -> None:
        if value is None:
            return
        val = str(value).strip()
        if not val:
            return
        token = (fid, key, val)
        if token in seen_features:
            return
        seen_features.add(token)
        feature_rows.append({
            "ifc_id": fid,
            "feature_key": key,
            "feature_value": val,
            "source": source,
            "confidence": float(confidence),
        })

    parent_map: Dict[str, Optional[str]] = {}
    label_map: Dict[str, str] = {}
    defn_map: Dict[str, str] = {}
    classes = data.get("classes") if isinstance(data, dict) else None
    if isinstance(classes, dict):
        for ifc_id, item in classes.items():
            label = item.get("description") or item.get("name") or ifc_id
            desc = item.get("definition") or item.get("description") or ""
            parent = item.get("parent") or item.get("baseClass")
            fid = str(ifc_id)
            parent_map[fid] = str(parent) if parent else None
            label_map[fid] = str(label) if label is not None else fid
            defn_map[fid] = str(desc) if desc is not None else ""
            rows.append({
                "ifc_id": fid,
                "schema_version": schema_version,
                "label": str(label) if label is not None else fid,
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
            if item.get("role"):
                add_feature(fid, "role", item.get("role"), "role")
            if item.get("baseClass"):
                add_feature(fid, "base_class", item.get("baseClass"), "base_class")
            if item.get("counterpart"):
                add_feature(fid, "counterpart", item.get("counterpart"), "counterpart")
            predef = item.get("predefinedType")
            if isinstance(predef, dict):
                enum_name = predef.get("enum")
                if enum_name:
                    add_feature(fid, "predefined_enum", enum_name, "predefined_type")
                values = predef.get("values")
                if isinstance(values, (list, tuple, set)):
                    for v in values:
                        add_feature(fid, "predefined_value", v, "predefined_type")
            if item.get("enum"):
                add_feature(fid, "enum", item.get("enum"), "enum")
            if item.get("enumValue"):
                add_feature(fid, "enum_value", item.get("enumValue"), "enum")
            psets = item.get("psets") or []
            if isinstance(psets, (list, tuple, set)):
                for pset in psets:
                    add_feature(fid, "pset", pset, "pset_list")
    elif isinstance(data, list):
        for item in data:
            ifc_id = item.get("id") or item.get("name")
            if not ifc_id:
                continue
            fid = str(ifc_id)
            label = item.get("label") or item.get("name") or fid
            desc = item.get("description") or ""
            parent = item.get("parent") or item.get("baseClass")
            parent_map[fid] = str(parent) if parent else None
            label_map[fid] = str(label) if label is not None else fid
            defn_map[fid] = str(desc) if desc is not None else ""
            rows.append({
                "ifc_id": fid,
                "schema_version": schema_version,
                "label": label,
                "description": desc,
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
            if item.get("role"):
                add_feature(fid, "role", item.get("role"), "role")
            if item.get("baseClass"):
                add_feature(fid, "base_class", item.get("baseClass"), "base_class")
            if item.get("counterpart"):
                add_feature(fid, "counterpart", item.get("counterpart"), "counterpart")
            predef = item.get("predefinedType")
            if isinstance(predef, dict):
                enum_name = predef.get("enum")
                if enum_name:
                    add_feature(fid, "predefined_enum", enum_name, "predefined_type")
                values = predef.get("values")
                if isinstance(values, (list, tuple, set)):
                    for v in values:
                        add_feature(fid, "predefined_value", v, "predefined_type")
            if item.get("enum"):
                add_feature(fid, "enum", item.get("enum"), "enum")
            if item.get("enumValue"):
                add_feature(fid, "enum_value", item.get("enumValue"), "enum")
            psets = item.get("psets") or []
            if isinstance(psets, (list, tuple, set)):
                for pset in psets:
                    add_feature(fid, "pset", pset, "pset_list")
    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=[
        "ifc_id","schema_version","label","description","parent","is_abstract",
        "role","is_type","is_instance","counterpart","has_predefined_type","pset_count",
        "has_manufacturer_ps","base_class","enum","enum_value"
    ])
    feature_df = pd.DataFrame(feature_rows) if feature_rows else pd.DataFrame(columns=["ifc_id","feature_key","feature_value","source","confidence"])
    if df.empty:
        return df, feature_df

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
        return ". ".join([p for p in parts if p]).strip()

    df["aug_text"] = df["ifc_id"].map(lambda x: build_chain_text(str(x)))
    return df.drop_duplicates(subset=["ifc_id"]), feature_df.drop_duplicates(subset=["ifc_id","feature_key","feature_value"])

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


def read_uniclass_csvs(csvs: Dict[str, Path], revision: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    feature_rows: List[Dict[str, object]] = []
    seen_features: set = set()

    def add_feature(code: str, key: str, value: object, source: str, confidence: float = 1.0) -> None:
        if value is None:
            return
        val = str(value).strip()
        if not val:
            return
        token = (code, key, val)
        if token in seen_features:
            return
        seen_features.add(token)
        feature_rows.append({
            "code": code,
            "feature_key": key,
            "feature_value": val,
            "source": source,
            "confidence": float(confidence),
        })

    for facet, p in csvs.items():
        if not p.exists():
            continue
        df = pd.read_csv(p)
        code_col = next((c for c in df.columns if isinstance(c, str) and c.lower() in ("code", "uniclass code", "item code")), None)
        title_col = next((c for c in df.columns if isinstance(c, str) and c.lower() in ("title", "name", "item title")), None)
        if not code_col or not title_col:
            raise ValueError(f"CSV {p} lacks Code/Title columns")
        desc_candidates = [c for c in df.columns if isinstance(c, str) and c.lower() in ("description", "item description", "notes")]
        if desc_candidates:
            desc_col = desc_candidates[0]
            desc_series = df[desc_col].astype(str).fillna("").str.strip()
        else:
            desc_series = pd.Series([""] * len(df), index=df.index)
        frames.append(pd.DataFrame({
            "code": df[code_col].astype(str),
            "facet": str(facet).upper(),
            "title": df[title_col].astype(str),
            "description": desc_series,
            "revision": revision,
        }))
        attribute_cols = [c for c in df.columns if isinstance(c, str) and _looks_like_attribute_column(c)]
        if attribute_cols:
            for _, row in df.iterrows():
                code_val = str(row.get(code_col, "") or "").strip()
                if not code_val:
                    continue
                for col in attribute_cols:
                    values = _split_feature_values(row.get(col))
                    for val in values:
                        add_feature(code_val, "attribute", val, str(col))
    if not frames:
        df_out = pd.DataFrame(columns=["code","facet","title","description","revision"])
    else:
        df_out = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["code"])
    feature_df = pd.DataFrame(feature_rows) if feature_rows else pd.DataFrame(columns=["code","feature_key","feature_value","source","confidence"])
    return df_out, feature_df.drop_duplicates(subset=["code","feature_key","feature_value"])

def read_uniclass_xlsx_dir(dir_path: Path, revision: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    def _coerce_with_header_search(xlsx_path: Path) -> Optional[pd.DataFrame]:
        """Read first sheet and try to locate the header row if the file has a title banner."""
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
        new_cols = [str(v).strip() for v in raw.iloc[header_row].values]
        data = raw.iloc[header_row + 1 :].copy()
        data.columns = new_cols
        code_col, title_col = _find_cols(data)
        if code_col and title_col:
            return data
        return None

    frames: List[pd.DataFrame] = []
    feature_rows: List[Dict[str, object]] = []
    seen_features: set = set()

    def add_feature(code: str, key: str, value: object, source: str, confidence: float = 1.0) -> None:
        if value is None:
            return
        val = str(value).strip()
        if not val:
            return
        token = (code, key, val)
        if token in seen_features:
            return
        seen_features.add(token)
        feature_rows.append({
            "code": code,
            "feature_key": key,
            "feature_value": val,
            "source": source,
            "confidence": float(confidence),
        })

    for p in sorted(dir_path.glob("*.xlsx")):
        facet = facet_from_filename(p.stem) or ""
        df = _coerce_with_header_search(p)
        if df is None:
            continue
        cols = [c for c in df.columns if isinstance(c, str)]
        lower = {c.lower(): c for c in cols}
        code_col = next((lower[k] for k in ["uniclass code","code","item code","pr code","ef code","ss code"] if k in lower), None)
        title_col = next((lower[k] for k in ["title","item title","name"] if k in lower), None)
        if not code_col or not title_col:
            continue
        ifc_cols = [c for c in cols if isinstance(c, str) and c and "ifc" in c.lower()]
        ifc_map_series = None
        if ifc_cols:
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
        desc_candidates = [c for c in cols if isinstance(c, str) and c.lower() in ("description", "item description", "notes")]
        if desc_candidates:
            desc_col = desc_candidates[0]
            desc_series = df[desc_col].astype(str).str.strip()
        else:
            desc_series = pd.Series([""] * len(df), index=df.index)
        base = {
            "code": df[code_col].astype(str).str.strip(),
            "facet": fval,
            "title": df[title_col].astype(str).str.strip(),
            "description": desc_series,
            "revision": revision,
        }
        if ifc_map_series is not None:
            base["ifc_mappings"] = ifc_map_series
        frames.append(pd.DataFrame(base))
        attribute_cols = [c for c in cols if c not in {code_col, title_col} and _looks_like_attribute_column(c)]
        if attribute_cols:
            for idx in df.index:
                code_val = str(df.at[idx, code_col] or "").strip()
                if not code_val:
                    continue
                for col in attribute_cols:
                    raw = df.at[idx, col] if col in df.columns else None
                    for val in _split_feature_values(raw):
                        add_feature(code_val, "attribute", val, str(col))
    if not frames:
        out = pd.DataFrame(columns=["code","facet","title","description","revision"])
    else:
        out = pd.concat(frames, ignore_index=True)
        mask = out["facet"] == ""
        out.loc[mask, "facet"] = out.loc[mask, "code"].str.extract(r"^([A-Za-z]{1,2})_", expand=False).fillna("")
        if "ifc_mappings" not in out.columns:
            out["ifc_mappings"] = ""
        out = out.drop_duplicates(subset=["code"])
    feature_df = pd.DataFrame(feature_rows) if feature_rows else pd.DataFrame(columns=["code","feature_key","feature_value","source","confidence"])
    return out, feature_df.drop_duplicates(subset=["code","feature_key","feature_value"])


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


def upsert_ifc_features(conn: psycopg.Connection, df: pd.DataFrame):
    if df.empty:
        return
    rows = []
    for ifc_id, key, value, source, confidence in df[['ifc_id','feature_key','feature_value','source','confidence']].itertuples(index=False, name=None):
        fid = str(ifc_id or '').strip()
        fkey = str(key or '').strip()
        fval = str(value or '').strip()
        src = str(source or '').strip()
        if not fid or not fkey or not fval:
            continue
        try:
            conf = float(confidence)
        except (TypeError, ValueError):
            conf = 1.0
        if not (0 <= conf <= 1):
            conf = max(0.0, min(1.0, conf))
        rows.append((fid, fkey, fval, src or None, conf))
    if not rows:
        return
    ids = sorted({r[0] for r in rows})
    with conn.cursor() as cur:
        cur.execute("DELETE FROM ifc_feature WHERE ifc_id = ANY(%s)", (ids,))
        cur.executemany(
            """
            INSERT INTO ifc_feature (ifc_id, feature_key, feature_value, source, confidence)
            VALUES (%s,%s,%s,%s,%s)
            """,
            rows,
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
            INSERT INTO uniclass_item_revision (code, revision, facet, title, description)
            SELECT code, revision, facet, title, description FROM tmp_uc
            ON CONFLICT (code, revision) DO NOTHING;
            DROP TABLE tmp_uc;
            """
        )

def upsert_uniclass_features(conn: psycopg.Connection, df: pd.DataFrame):
    if df.empty:
        return
    rows = []
    for code, key, value, source, confidence in df[['code','feature_key','feature_value','source','confidence']].itertuples(index=False, name=None):
        code_val = str(code or '').strip()
        fkey = str(key or '').strip()
        fval = str(value or '').strip()
        src = str(source or '').strip()
        if not code_val or not fkey or not fval:
            continue
        try:
            conf = float(confidence)
        except (TypeError, ValueError):
            conf = 1.0
        if not (0 <= conf <= 1):
            conf = max(0.0, min(1.0, conf))
        rows.append((code_val, fkey, fval, src or None, conf))
    if not rows:
        return
    codes = sorted({r[0] for r in rows})
    with conn.cursor() as cur:
        cur.execute("DELETE FROM uniclass_feature WHERE code = ANY(%s)", (codes,))
        cur.executemany(
            """
            INSERT INTO uniclass_feature (code, feature_key, feature_value, source, confidence)
            VALUES (%s,%s,%s,%s,%s)
            """,
            rows,
        )




    if df.empty:
        return
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TEMP TABLE tmp_uc AS SELECT * FROM uniclass_item WITH NO DATA;
            """
        )
        cols = ["code","facet","title","revision"]
        csv_bytes = df[cols].to_csv(index=False, header=False).encode("utf-8")
        with cur.copy(
            "COPY tmp_uc (code,facet,title,revision) FROM STDIN WITH (FORMAT CSV, HEADER FALSE)"
        ) as cp:
            cp.write(csv_bytes)
        cur.execute(
            """
            INSERT INTO uniclass_item AS u (code,facet,title,revision)
            SELECT code,facet,title,revision FROM tmp_uc
            ON CONFLICT (code) DO UPDATE
            SET facet = EXCLUDED.facet,
                title = EXCLUDED.title,
                revision = EXCLUDED.revision;
            -- Also record a historical snapshot per (code, revision)
            INSERT INTO uniclass_item_revision (code, revision, facet, title)
            SELECT code, revision, facet, title FROM tmp_uc
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
            scores.append((u["code"], fac, u["title"], float(s)))
        scores.sort(key=lambda x: x[4], reverse=True)
        for code, facet, title, s in scores[:top_k]:
            rows.append({
                "ifc_id": ifc_id,
                "code": code,
                "facet": facet,
                "uniclass_title": title,
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
    debug: bool = False,
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

    t0 = time.perf_counter()
    rows = []
    with conn.cursor() as cur:
        # Optional safety timeout
        try:
            if timeout_ms is None:
                timeout_ms = int(os.getenv('EMBED_QUERY_TIMEOUT_MS', '5000'))
            if timeout_ms and timeout_ms > 0:
                cur.execute(f"SET LOCAL statement_timeout = {int(timeout_ms)}")
        except Exception as e:
            if debug:
                print(f"[_embedding_neighbors] failed to set statement_timeout: {e}")
        try:
            cur.execute(sql, params)
            rows = cur.fetchall()
        except Exception as e:
            if debug:
                print(f"[_embedding_neighbors] query failed for entity_type={target_entity_type}, limit={limit}, facets={facets}: {e}")
            # On failure, return empty result so outer code falls back to lexical-only
            rows = []
    t1 = time.perf_counter()
    if debug:
        try:
            print(f"[_embedding_neighbors] fetched {len(rows)} neighbors for {target_entity_type} in {1000*(t1-t0):.1f} ms")
        except Exception:
            pass
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
    debug: bool = False,
    stream_to_csv: Optional[Path] = None,
    stream_every: int = 0,
) -> pd.DataFrame:
    # Precompute lex features on uc_df
    uc_work = uc_df.copy()
    uc_work["norm_title"] = uc_work["title"].fillna("").map(normalize_text).map(lambda s: expand_with_synonyms(s, synonyms))
    # Streaming setup
    header_cols = [
        "ifc_id", "code", "facet", "uniclass_title", "score",
        "lexical_score", "embedding_score", "feature_multiplier",
        "discipline_multiplier", "token_overlap_multiplier", "anchor_applied",
    ]
    header_written = False
    buffered_rows: List[dict] = []
    processed_uc = 0
    total_uc = int(len(uc_work))
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
        maps_raw = str(r.get("ifc_mappings", "") or "")
        map_set = set([m for m in maps_raw.split(";") if m])
        uc_info[code] = (facet, title, map_set)
    # Fast lookup for normalized titles by code, avoid .loc in tight loops
    code_to_norm = pd.Series(uc_work["norm_title"].values, index=uc_work["code"].astype(str)).to_dict()
    uc_disc_map = pd.Series(uc_work["disciplines"].values, index=uc_work["code"].astype(str)).to_dict()
    uc_subj_map = pd.Series(uc_work["subjects"].values, index=uc_work["code"].astype(str)).to_dict()
    # Unified labels (disciplines + subjects) per Uniclass code
    uc_label_map = {k: list(dict.fromkeys((uc_disc_map.get(k) or []) + (uc_subj_map.get(k) or []))) for k in uc_work["code"].astype(str)}
    # Integrate optional LLM-provided labels for Uniclass (overrides or augments)
    disc_source = str((rules or {}).get("__global__", {}).get("discipline_source", "heuristic")).lower()
    # Load cached LLM labels if available (optional)
    llm_labels = _load_taxonomy_cache()
    if llm_labels and disc_source in ("llm", "llm_then_heuristic"):
        for _code in list(uc_label_map.keys()):
            _key = f"uc:{_code}"
            _labs = llm_labels.get(_key)
            if _labs:
                if disc_source == "llm":
                    uc_label_map[_code] = list(dict.fromkeys(_labs))
                else:
                    uc_label_map[_code] = list(dict.fromkeys(list(uc_label_map[_code]) + list(_labs)))

    rows = []
    ancestors = _build_ifc_ancestors_map(ifc_df)
    is_abs = {str(r["ifc_id"]): bool(r.get("is_abstract", False)) for _, r in ifc_df.iterrows()}
    flags_map = _compute_ifc_flags(ifc_df, ancestors)
    # Ensure labels key present and merge LLM labels for IFC
    try:
        _ = next(iter(flags_map.values()))
        for _iid in list(flags_map.keys()):
            flags_map[_iid]["labels"] = list(dict.fromkeys(flags_map[_iid].get("labels", []) or flags_map[_iid].get("disciplines", []) or []))
    except Exception:
        pass
    if llm_labels and disc_source in ("llm", "llm_then_heuristic"):
        for _iid in list(flags_map.keys()):
            _key = f"ifc:{_iid}"
            _labs = llm_labels.get(_key)
            if _labs:
                if disc_source == "llm":
                    flags_map[_iid]["labels"] = list(dict.fromkeys(_labs))
                else:
                    _prev = flags_map[_iid].get("labels", [])
                    flags_map[_iid]["labels"] = list(dict.fromkeys(list(_prev) + list(_labs)))
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
                    _c.execute(f"SET LOCAL ivfflat.probes = {int(os.getenv('EMBED_IVF_PROBES', '10'))}")
                except Exception:
                    pass
            emb_scores = _embedding_neighbors(conn, 'uniclass', qvec_text, embedding_top_k, facets=facets, timeout_ms=int(os.getenv('EMBED_QUERY_TIMEOUT_MS', '5000')), debug=debug)
            # Optionally filter by facets
            if facets:
                emb_scores = {c: s for c, s in emb_scores.items() if uc_info.get(c, ("", "", ""))[0] in facets}
        elif debug:
            print(f"[candidates] missing IFC embedding for {ifc_id}; using lexical only")

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
            ufacet, title, umaps = uc_info.get(code, ("", "", set()))
            base_mult = _feature_multiplier(ufacet, flags_map.get(str(ifc_id), {}), {
                "pr_predefined_type": 0.10, "pr_manufacturer_ps": 0.10, "ss_group_system": 0.10, "type_bias": 0.05, "pset_count_scale": 0.002, "max_pset_boost": 0.25
            })
            mult = base_mult
            discipline_multiplier = 1.0
            # Discipline/subject gating also applied at final score to catch embedding-only candidates
            if disc_mode in ("soft", "hard"):
                ifc_labels = set(flags_map.get(str(ifc_id), {}).get("labels", []) or flags_map.get(str(ifc_id), {}).get("disciplines", []) or [])
                uc_labels = set(uc_label_map.get(str(code), []) or [])
                if ifc_labels and uc_labels and ifc_labels.isdisjoint(uc_labels):
                    if disc_mode == "hard":
                        continue
                    else:
                        factor = float(disc_penalty)
                        mult *= factor
                        discipline_multiplier *= factor
            anchor_hit = False
            final = ((1 - alpha) * lex + alpha * emb) * mult
            # Anchor guidance: boost if Uniclass row maps to this IFC (or its ancestors if enabled)
            if umaps:
                if anchor_use_ancestors:
                    anc = set(ancestors.get(str(ifc_id), [])) | {str(ifc_id)}
                    if any(m in anc for m in umaps):
                        final = min(1.0, final + anchor_bonus)
                        anchor_hit = True
                else:
                    if str(ifc_id) in umaps:
                        final = min(1.0, final + anchor_bonus)
                        anchor_hit = True
            scored.append((code, ufacet, title, float(final), float(lex), float(emb), float(base_mult), float(discipline_multiplier), anchor_hit))

        scored.sort(key=lambda x: x[3], reverse=True)
        for code, facet, title, s, lex_raw, emb_raw, base_mult, disc_mult, anchor_hit in scored[:top_k]:
            rec = {
                "ifc_id": ifc_id,
                "code": code,
                "facet": facet,
                "uniclass_title": title,
                "score": round(s, 4),
                "lexical_score": round(lex_raw, 4),
                "embedding_score": round(emb_raw, 4),
                "feature_multiplier": round(base_mult, 4),
                "discipline_multiplier": round(disc_mult, 4),
                "token_overlap_multiplier": 1.0,
                "anchor_applied": bool(anchor_hit),
            }
            rows.append(rec)
            if stream_to_csv is not None:
                buffered_rows.append(rec)
                if stream_every and len(buffered_rows) >= int(stream_every):
                    # Flush batch to CSV
                    try:
                        stream_to_csv.parent.mkdir(parents=True, exist_ok=True)
                        dfb = pd.DataFrame(buffered_rows)
                        for col in header_cols:
                            if col not in dfb.columns:
                                dfb[col] = None
                        dfb = dfb[header_cols]
                        dfb.to_csv(stream_to_csv, mode=('a' if header_written else 'w'), index=False, header=(not header_written))
                        header_written = True
                        buffered_rows.clear()
                    except Exception as e:
                        if debug:
                            print(f"[candidates-stream] failed to write batch: {e}")
        processed_uc += 1
        if debug and (processed_uc % 50 == 0 or processed_uc == total_uc):
            print(f"[candidates] processed {processed_uc}/{total_uc} Uniclass items")

    # Final flush for any remaining buffered rows
    if stream_to_csv is not None and buffered_rows:
        try:
            stream_to_csv.parent.mkdir(parents=True, exist_ok=True)
            dfb = pd.DataFrame(buffered_rows)
            for col in header_cols:
                if col not in dfb.columns:
                    dfb[col] = None
            dfb = dfb[header_cols]
            dfb.to_csv(stream_to_csv, mode=('a' if header_written else 'w'), index=False, header=(not header_written))
            header_written = True
            buffered_rows.clear()
        except Exception as e:
            if debug:
                print(f"[candidates-stream] failed to write final batch: {e}")
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
    debug: bool = False,
    stream_to_csv: Optional[Path] = None,
    stream_every: int = 0,
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
    # Load LLM taxonomy cache (optional) and prepare label maps
    llm_labels = _load_taxonomy_cache()
    disc_source = str((rules or {}).get("__global__", {}).get("discipline_source", "heuristic")).lower()
    try:
        for _iid in list(flags_map.keys()):
            flags_map[_iid]["labels"] = list(dict.fromkeys(flags_map[_iid].get("labels", []) or flags_map[_iid].get("disciplines", []) or []))
    except Exception:
        pass
    if llm_labels and disc_source in ("llm", "llm_then_heuristic"):
        for _iid in list(flags_map.keys()):
            _labs = llm_labels.get(f"ifc:{_iid}")
            if _labs:
                if disc_source == "llm":
                    flags_map[_iid]["labels"] = list(dict.fromkeys(_labs))
                else:
                    prev = flags_map[_iid]["labels"]
                    flags_map[_iid]["labels"] = list(dict.fromkeys(list(prev) + list(_labs)))
    uc_llm_map: Dict[str, List[str]] = {}
    if llm_labels and disc_source in ("llm", "llm_then_heuristic"):
        for _, _u in uc_df.iterrows():
            _code = str(_u.get("code", ""))
            if _code:
                labs = llm_labels.get(f"uc:{_code}")
                if labs:
                    uc_llm_map[_code] = list(dict.fromkeys(labs))

    # Streaming setup for incremental CSV writes
    header_cols = [
        "ifc_id", "code", "facet", "uniclass_title", "score",
        "lexical_score", "embedding_score", "feature_multiplier",
        "discipline_multiplier", "token_overlap_multiplier", "anchor_applied",
    ]
    header_written = False
    buffered_rows: List[dict] = []
    processed_uc = 0
    total_uc = int(len(uc_df))
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
        if not qvec_text and debug:
            print(f"[candidates] missing embedding for Uniclass code {code}; using lexical only")
        if qvec_text:
            with conn.cursor() as _c:
                try:
                    _c.execute(f"SET LOCAL ivfflat.probes = {int(os.getenv('EMBED_IVF_PROBES', '10'))}")
                except Exception:
                    pass
            emb_scores = _embedding_neighbors(conn, 'ifc', qvec_text, embedding_top_k, facets=None, timeout_ms=int(os.getenv('EMBED_QUERY_TIMEOUT_MS', '5000')), debug=debug)

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
                # Discipline/subject labels for UC (LLM first if available)
                uc_discs = set(uc_llm_map.get(code, []) or [])
                if not uc_discs or disc_source == "heuristic":
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
                uc_discs = set(uc_llm_map.get(code, []) or [])
                if not uc_discs or disc_source == "heuristic":
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
            base_mult = _feature_multiplier(facet, flags_map.get(str(ifc_id), {}), {
                "pr_predefined_type": 0.10, "pr_manufacturer_ps": 0.10, "ss_group_system": 0.10, "type_bias": 0.05, "pset_count_scale": 0.002, "max_pset_boost": 0.25
            })
            mult = base_mult
            discipline_multiplier = 1.0
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
                        factor = float(disc_penalty)
                        mult *= factor
                        discipline_multiplier *= factor
            anchor_hit = False
            token_multiplier = 1.0
            final = ((1 - alpha) * lex + alpha * emb) * mult
            # Token-overlap penalty: if Uniclass title shares no tokens with IFC id/label/desc, downweight
            try:
                if not (u_tokens & set(_tokenize_set(f"{ifc_id} {ifc_info.get(ifc_id, ('',''))[0]} {ifc_info.get(ifc_id, ('',''))[1]}"))):
                    final *= 0.6
                    token_multiplier *= 0.6
            except Exception:
                pass
            if umaps:
                if anchor_use_ancestors:
                    anc = set(ancestors.get(str(ifc_id), [])) | {str(ifc_id)}
                    if any(m in anc for m in umaps):
                        final = min(1.0, final + anchor_bonus)
                        anchor_hit = True
                else:
                    if str(ifc_id) in umaps:
                        final = min(1.0, final + anchor_bonus)
                        anchor_hit = True
            lbl, _desc = ifc_info.get(ifc_id, ("", ""))
            scored.append((code, facet, title, ifc_id, lbl, float(final), float(lex), float(emb), float(base_mult), float(discipline_multiplier), float(token_multiplier), anchor_hit))

        scored.sort(key=lambda x: x[5], reverse=True)
        for code, facet, title, ifc_id, _lbl, s, lex_raw, emb_raw, base_mult, disc_mult, token_mult, anchor_hit in scored[:top_k]:
            rec = {
                "ifc_id": ifc_id,
                "code": code,
                "facet": facet,
                "uniclass_title": title,
                "score": round(s, 4),
                "lexical_score": round(lex_raw, 4),
                "embedding_score": round(emb_raw, 4),
                "feature_multiplier": round(base_mult, 4),
                "discipline_multiplier": round(disc_mult, 4),
                "token_overlap_multiplier": round(token_mult, 4),
                "anchor_applied": bool(anchor_hit),
            }
            rows.append(rec)
            if stream_to_csv is not None:
                buffered_rows.append(rec)
                if stream_every and len(buffered_rows) >= int(stream_every):
                    try:
                        stream_to_csv.parent.mkdir(parents=True, exist_ok=True)
                        dfb = pd.DataFrame(buffered_rows)
                        for col in header_cols:
                            if col not in dfb.columns:
                                dfb[col] = None
                        dfb = dfb[header_cols]
                        dfb.to_csv(stream_to_csv, mode=('a' if header_written else 'w'), index=False, header=(not header_written))
                        header_written = True
                        buffered_rows.clear()
                    except Exception as e:
                        if debug:
                            print(f"[candidates-stream] failed to write batch: {e}")
        processed_uc += 1
        if debug and (processed_uc % 50 == 0 or processed_uc == total_uc):
            print(f"[candidates] processed {processed_uc}/{total_uc} Uniclass items")

    if stream_to_csv is not None and buffered_rows:
        try:
            stream_to_csv.parent.mkdir(parents=True, exist_ok=True)
            dfb = pd.DataFrame(buffered_rows)
            for col in header_cols:
                if col not in dfb.columns:
                    dfb[col] = None
            dfb = dfb[header_cols]
            dfb.to_csv(stream_to_csv, mode=('a' if header_written else 'w'), index=False, header=(not header_written))
            header_written = True
            buffered_rows.clear()
        except Exception as e:
            if debug:
                print(f"[candidates-stream] failed to write final batch: {e}")
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


def _coerce_float(val: object) -> Optional[float]:
    if val is None:
        return None
    try:
        f = float(val)
        if math.isnan(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


def _coerce_bool(val: object) -> Optional[bool]:
    if val is None:
        return None
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    if isinstance(val, str):
        low = val.strip().lower()
        if not low:
            return None
        if low in {"true", "t", "1", "yes", "y"}:
            return True
        if low in {"false", "f", "0", "no", "n"}:
            return False
    return None


def log_candidate_history(
    conn: psycopg.Connection,
    candidates: pd.DataFrame,
    run_id: UUID,
    source_ifc_version: str,
    source_uniclass_revision: str,
    direction: str,
) -> None:
    if candidates.empty:
        return
    rows = []
    for _, r in candidates.iterrows():
        ifc_id = str(r.get("ifc_id") or "").strip()
        code = str(r.get("code") or "").strip()
        if not ifc_id or not code:
            continue
        facet = str(r.get("facet") or "").strip()
        rows.append((
            str(run_id),
            ifc_id,
            code,
            facet,
            _coerce_float(r.get("score")),
            _coerce_float(r.get("lexical_score")),
            _coerce_float(r.get("embedding_score")),
            _coerce_float(r.get("feature_multiplier")),
            _coerce_float(r.get("discipline_multiplier")),
            _coerce_float(r.get("token_overlap_multiplier")),
            _coerce_bool(r.get("anchor_applied")),
            direction,
            source_ifc_version,
            source_uniclass_revision,
        ))
    if not rows:
        return
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO ifc_uniclass_candidate_history (
                run_id, ifc_id, code, facet, score, lexical_score, embedding_score,
                feature_multiplier, discipline_multiplier, token_overlap_multiplier,
                anchor_applied, direction, source_ifc_version, source_uniclass_revision
            )
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """,
            rows,
        )



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
    ap.add_argument("--candidates-debug", action="store_true", help="Verbose debug logging during candidate generation")
    ap.add_argument("--candidates-stream-every", type=int, default=0, help="Append to candidates.csv every N rows (0 = write at end)")
    ap.add_argument("--export", action="store_true", help="Export viewer JSON from accepted mappings")
    # Optional maintenance flags
    ap.add_argument("--reset-mappings", action="store_true", help="Delete existing mappings for current Uniclass revision (optionally filter by --reset-facets)")
    ap.add_argument("--reset-facets", type=str, default="", help="Comma-separated Uniclass facets to reset (e.g., PR,SS)")
    ap.add_argument("--embed", action="store_true", help="Generate embeddings via AI backend and store in DB")
    ap.add_argument("--openai", action="store_true", help="Use OpenAI API for embedding, classification, and rerank (default is Ollama)")
    ap.add_argument("--classify-disciplines", action="store_true", help="Classify IFC and Uniclass into labels via LLM and cache to output/taxonomy_cache.json")
    ap.add_argument("--classify-scope", choices=["both","ifc","uniclass"], default="both", help="Limit classification to IFC, Uniclass, or both")
    ap.add_argument("--classify-facets", type=str, default="", help="Comma-separated Uniclass facets to classify (e.g., PR,SS). Empty = all")
    ap.add_argument("--classify-limit", type=int, default=0, help="Stop after N items (0 = no limit)")
    ap.add_argument("--classify-model", type=str, default="", help="Override model for classification (defaults to rerank.model)")
    ap.add_argument("--classify-timeout", type=float, default=0.0, help="Per-call timeout seconds (0 = use rerank.timeout_s)")
    ap.add_argument("--classify-warmup-timeout", type=float, default=180.0, help="Warmup timeout seconds for first model load")
    ap.add_argument("--classify-sleep", type=float, default=0.0, help="Sleep seconds between calls for rate limiting")
    ap.add_argument("--debug", action="store_true", help="Log prompts and request metadata sent to the AI backend")
    # Resume control: skip already cached items (default true); allow override via --no-classify-skip-cached
    ap.add_argument("--classify-skip-cached", dest="classify_skip_cached", action="store_true", help="Skip items already in output/taxonomy_cache.json (default)")
    ap.add_argument("--no-classify-skip-cached", dest="classify_skip_cached", action="store_false", help="Do not skip cached items; reclassify all")
    ap.set_defaults(classify_skip_cached=True)
    args = ap.parse_args()

    s = load_settings(Path(args.config))
    ensure_output_dirs(s)

    ifc_df, ifc_features = read_ifc_json(s.ifc_json, s.ifc_schema_version)

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
        uc_df, uc_features = read_uniclass_xlsx_dir(s.uniclass_xlsx_dir, s.uniclass_revision)
    else:
        uc_df, uc_features = read_uniclass_csvs(s.uniclass_csvs, s.uniclass_revision)

    # Optional: LLM discipline/subject classification
    if args.classify_disciplines:
        print("[classify] Preparing items for LLM classification...")
        items: List[Tuple[str, str, str, str]] = []
        scope = (args.classify_scope or "both").lower()
        facet_filter = [f.strip().upper() for f in (args.classify_facets or "").split(',') if f.strip()]
        # IFC items
        if scope in ("both", "ifc"):
            for _, r in ifc_df.iterrows():
                iid = str(r.get("ifc_id", ""))
                if not iid:
                    continue
                title = str(r.get("label", ""))
                desc = str(r.get("aug_text") or r.get("description") or "")
                items.append((f"ifc:{iid}", "IFC", title, desc))
        # Uniclass items: classify at Section level (Facet_Group_Subgroup_Section) and propagate to child codes
        alias_map: Dict[str, List[str]] = {}
        extras_map: Dict[str, str] = {}
        if scope in ("both", "uniclass"):
            def _section_key(code: str) -> str:
                parts = str(code or "").split("_")
                return "_".join(parts[:4]) if len(parts) >= 4 else str(code or "")
            def _facet_name(facet: str) -> str:
                m = {
                    'AC': 'Activities', 'PM': 'Project Management', 'PR': 'Products', 'SS': 'Systems',
                    'EF': 'Elements/Functions', 'EN': 'Entities', 'FI': 'Forms of Information', 'MA': 'Materials',
                    'PC': 'Properties and Characteristics', 'RK': 'Risk', 'RO': 'Roles', 'SL': 'Spaces/Locations',
                    'CO': 'Complexes', 'TE': 'Tools and Equipment', 'ZZ': 'Unassigned'
                }
                return m.get(facet.upper(), facet)
            uc_src = uc_df
            if facet_filter:
                uc_src = uc_src[uc_src["facet"].astype(str).str.upper().isin(facet_filter)]
            reps: Dict[str, Tuple[str, str]] = {}
            for _, r in uc_src.iterrows():
                code = str(r.get("code", ""))
                if not code:
                    continue
                sec = _section_key(code)
                alias_map.setdefault(sec, []).append(code)
                title = str(r.get("title", ""))
                # Many Uniclass rows have empty or unhelpful descriptions; omit to improve model reliability
                desc = ""
                facet = str(r.get("facet", ""))
                if sec and sec not in extras_map:
                    extras_map[f"uc:{sec}"] = f"Facet: {facet} ({_facet_name(facet)}) | Section: {sec}"
                # Prefer exact section row if present; otherwise keep first seen child
                if code == sec or not reps.get(sec):
                    reps[sec] = (title, desc)
            for sec, (title, desc) in reps.items():
                items.append((f"uc:{sec}", "Uniclass", title, desc))
        if int(args.classify_limit or 0) > 0:
            items = items[: int(args.classify_limit)]
        # Choose model based on backend; args.classify_model overrides
        if args.openai:
            default_cls_model = s.openai_rerank_model or s.rerank_model
        else:
            default_cls_model = s.rerank_model
        c_model = (args.classify_model or default_cls_model)
        per_call_timeout = float(args.classify_timeout or 0.0) or float(s.rerank_timeout_s) or 45.0
        backend = "openai" if args.openai else "ollama"
        target = ("OpenAI" if backend == "openai" else s.rerank_endpoint)
        out_path = s.output_dir / "taxonomy_cache.json"
        print(f"[classify] Classifying {len(items)} section items using {c_model} via {backend} ({target}); live-writing to {out_path}")
        labels_map = classify_items_with_llm(
            items,
            model=c_model,
            endpoint=s.rerank_endpoint,
            temperature=s.rerank_temperature,
            max_tokens=min(128, s.rerank_max_tokens),
            timeout_s=max(20.0, per_call_timeout),
            warmup_timeout_s=max(60.0, float(args.classify_warmup_timeout or 180.0)),
            progress_every=200,
            sleep_between=float(args.classify_sleep or 0.0),
            max_retries=2,
            provider=backend,
            cache_path=out_path,
            flush_every=25,
            skip_cached=bool(getattr(args, 'classify_skip_cached', True)),
            alias_map=alias_map,
            extras_map=extras_map,
            debug=bool(getattr(args, 'debug', False)),
        )
        # A final confirmation message; file was live-written during the run
        print(f"[classify] Live-updated {out_path} with {len(labels_map)} entries")
        # Exit early if only classification requested
        if not (args.load or args.candidates or args.export or args.embed or args.reset_mappings):
            return

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
            if s.features_enabled:
                upsert_ifc_features(conn, ifc_features)
                upsert_uniclass_features(conn, uc_features)

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

            backend = "openai" if args.openai else "ollama"
            target = ("OpenAI" if backend == "openai" else s.embed_endpoint)
            # Choose model/expected_dim by backend
            if backend == "openai":
                e_model = s.openai_embed_model or s.embed_model
                e_dim = s.openai_embed_expected_dim if s.openai_embed_expected_dim is not None else s.embed_expected_dim
            else:
                e_model = s.embed_model
                e_dim = s.embed_expected_dim
            print(f"[embed] Generating embeddings using model '{e_model}' via {backend} ({target})")
            n_ifc = generate_and_store_embeddings(conn, "ifc", ifc_items, e_model, s.embed_endpoint, s.embed_batch_size, s.embed_timeout_s, expected_dim=e_dim, provider=backend)
            n_uc = generate_and_store_embeddings(conn, "uniclass", uc_items, e_model, s.embed_endpoint, s.embed_batch_size, s.embed_timeout_s, expected_dim=e_dim, provider=backend)
            print(f"[embed] Upserted {n_ifc} IFC and {n_uc} Uniclass embeddings")

        if args.candidates:
            run_id = uuid4()
            # Ensure vector indexes exist before neighbor queries
            try:
                ensure_vector_indexes(conn, lists=int(s.embed_ivf_lists))
            except Exception as e:
                print(f"[candidates] index ensure failed (non-fatal): {e}")
            # Set probes for this session for better recall/latency tradeoff
            try:
                with conn.cursor() as cur:
                    cur.execute(f"SET ivfflat.probes = {int(s.embed_ivf_probes)}")
                    cur.execute(f"SET statement_timeout = {int(s.embed_query_timeout_ms)}")
            except Exception:
                pass
            use_embed = s.embedding_weight and float(s.embedding_weight) > 0
            direction = (s.match_direction or 'uniclass_to_ifc').lower()
            if direction != 'uniclass_to_ifc':
                print("[info] ifc_to_uniclass is disabled; forcing direction=uniclass_to_ifc")
                direction = 'uniclass_to_ifc'
            print(f"[candidates] run id: {run_id} (direction={direction})")
            # Configure streaming and debug flags
            cands_path = s.output_dir / "candidates.csv"
            stream_every = int(getattr(args, "candidates_stream_every", 0) or 0)
            stream_path = cands_path if stream_every > 0 else None
            cand_debug = bool(getattr(args, "candidates_debug", False) or getattr(args, "debug", False))
            if use_embed:
                cands = generate_candidates_uc2ifc_blended(
                    conn, ifc_df, uc_df, s.synonyms, s.top_k, s.embedding_weight, s.embedding_top_k,
                    s.anchor_bonus, s.anchor_use_ancestors, s.matching_rules,
                    debug=cand_debug, stream_to_csv=stream_path, stream_every=stream_every,
                )
            else:
                # Lexical-only reverse direction using the same blended helper with alpha=0 via parameters
                cands = generate_candidates_uc2ifc_blended(
                    conn, ifc_df, uc_df, s.synonyms, s.top_k, 0.0, s.embedding_top_k,
                    s.anchor_bonus, s.anchor_use_ancestors, s.matching_rules,
                    debug=cand_debug, stream_to_csv=stream_path, stream_every=stream_every,
                )

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
                        backend = "openai" if args.openai else "ollama"
                        # Choose rerank model per backend
                        r_model = (s.openai_rerank_model or s.rerank_model) if backend == 'openai' else s.rerank_model
                        cands = rerank_candidates_with_llm(
                            cands,
                            ifc_df,
                            uc_df,
                            direction=direction,
                            top_n=int(s.rerank_top_n),
                            model=r_model,
                            endpoint=s.rerank_endpoint,
                            temperature=float(s.rerank_temperature),
                            max_tokens=int(s.rerank_max_tokens),
                            fewshot_per_facet=int(s.rerank_fewshot_per_facet),
                            timeout_s=float(s.rerank_timeout_s),
                            provider=backend,
                        )
                    except Exception as e:
                        print(f"[rerank] skipped due to error: {e}")
            needed_cols = ["lexical_score", "embedding_score", "feature_multiplier", "discipline_multiplier", "token_overlap_multiplier", "anchor_applied"]
            for col in needed_cols:
                if col not in cands.columns:
                    cands[col] = None
            log_candidate_history(conn, cands, run_id, s.ifc_schema_version, s.uniclass_revision, direction)
            # If streaming is disabled, write once at the end
            if not stream_path:
                cands.to_csv(cands_path, index=False)
            write_candidate_edges(conn, cands, s.ifc_schema_version, s.uniclass_revision, s.auto_accept)
        if args.export:
            export_viewer_json(conn, s.viewer_json)


if __name__ == "__main__":
    main()









