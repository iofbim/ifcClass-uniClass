# ifcClass-uniClass

This repository establishes structured mappings between Uniclass 2015 classification tables and IFC 4x3 schema classes.

## Overview

- Goal: Load IFC classes and Uniclass tables into Postgres, generate candidate mappings with lexical + vector similarity, optionally re-rank with a local LLM, and export a JSON for a viewer.
- Core: `etl/etl_map.py` orchestrates loading, embeddings, candidate generation, optional LLM re-rank, and export via CLI flags.
- Schema: `sql/schema.sql` defines tables (`ifc_class`, `uniclass_item`, `uniclass_item_revision`, `ifc_uniclass_map`, `text_embedding`) and enables `pgcrypto` and `vector` (pgvector).

## Data Model

- IFC: Classes with `ifc_id`, `schema_version`, `label`, `description`.
- Uniclass: Items with `code`, `facet` (EF/SS/PR/... inferred), `title`, `description`, `revision`.
- Mappings: `ifc_uniclass_map` stores edges with `relation_type` (typical_of, part_of, equivalent), `confidence`, and provenance (IFC version, Uniclass revision).

## ETL Flow

- Load (`--load`): Upserts IFC classes and Uniclass items.
- Embed (`--embed`): Generates embeddings with Ollama and stores in `text_embedding`.
  - IFC text: identifier + description/definition with parent-chain context (optional), plus compact feature hints (e.g., role, kind, psets).
  - Uniclass text: enriched with `[FACET] Code tokens: Title` (e.g., `[PR] Pr 20 93: Unit structure and general products`).
  - Recommended embed model: `mxbai-embed-large` (1024‑dim); set `embedding.expected_dim: 1024`.
- Candidates (`--candidates`):
  - Lexical: fuzzy match between normalized IFC text and Uniclass titles, filtered by facet rules.
  - Vector blend: nearest neighbors via pgvector using cosine distance (consistent rank + score), blended by `matching.embedding_weight`.
  - Anchor guidance: if Uniclass XLSX includes IFC mapping columns (e.g., “IFC 4x3”), extracted `ifc_mappings` boost those IFCs (or ancestors) by `matching.anchor_bonus`.
  - Direction: `matching.direction` `ifc_to_uniclass` (default) or `uniclass_to_ifc` (ensures every Uniclass gets candidates from IFC).
  - Auto-accept: rows above `matching.auto_accept_threshold` are written to `ifc_uniclass_map` (confidence clamped to [0,1]).
- Re-rank (optional): If `rerank.top_n > 0`, per-item top‑N are re-scored by a local LLM (Ollama) using few-shot examples from the extracted anchors per facet.
- Export (`--export`): Writes `output/viewer_mapping.json` summarizing accepted mappings.

### LLM Classification + Label-Guided Matching

- Classification (`--classify-disciplines`): Uses a local LLM (configured under `rerank.*`) to assign multi-label tags to each IFC class and Uniclass item.
  - Labels include AECO disciplines and subjects/facets: `ARCH, STRUCT, CIVIL, MECH, PLUMB, ELEC, ICT, MAINT, ROLE, PM, ACTIVITIES, COMPLEX, AC, CO, EF, EN, FI, MA, PC, PM, PR, RK, RO, SL, SS` (excludes `ZZ`).
  - Results are cached at `output/taxonomy_cache.json` and auto-loaded by candidate generation.
- Label gating: Candidate generation compares IFC labels to Uniclass labels and penalizes or excludes mismatches.
  - Configure in `config/settings.yaml` under `matching`:
    - `discipline_filter`: `none | soft | hard`
    - `discipline_penalty`: penalty multiplier when `soft`
    - `discipline_source`: `heuristic | llm | llm_then_heuristic`
  - With `llm_then_heuristic`, LLM labels are used when present; otherwise keyword/ancestor heuristics fill gaps.

#### Classification CLI Options

- `--classify-disciplines`: Run the classifier and write `output/taxonomy_cache.json`.
- `--classify-scope`: `both | ifc | uniclass` (default `both`).
- `--classify-facets`: Comma-separated Uniclass facets to include (e.g., `PR,SS`). Empty = all.
- `--classify-limit`: Stop after N items (0 = no limit).
- `--classify-model`: Override model just for classification (defaults to `rerank.model`).
- `--classify-timeout`: Per-call timeout seconds (0 = use `rerank.timeout_s`).
- `--classify-warmup-timeout`: Timeout seconds for the initial model load.
- `--classify-sleep`: Sleep seconds between calls (simple rate limiting).

Notes:
- The classifier uses the same Ollama endpoint configured for reranking (`rerank.endpoint`).
- The cache is id-keyed: `ifc:<IfcId>` and `uc:<Code>`, with values as arrays of uppercased labels.
- Matching honors `matching.discipline_source` to pick `heuristic`, `llm`, or `llm_then_heuristic` when labels are present.

#### Quick Examples

- Classify all IFC + Uniclass items and stop:
  - `py etl/etl_map.py --config config/settings.yaml --classify-disciplines`
- Classify only Uniclass PR/SS and then generate candidates using those labels:
  - `py etl/etl_map.py --config config/settings.yaml --classify-disciplines --classify-scope uniclass --classify-facets PR,SS`
  - `py etl/etl_map.py --config config/settings.yaml --candidates`
-
  Rebuild the cache with a different model and a 30s timeout:
  - `py etl/etl_map.py --config config/settings.yaml --classify-disciplines --classify-model llama3.2:latest --classify-timeout 30`

## Configuration

- YAML: Copy `config/settings.example.yaml` → `config/settings.yaml` and adjust file paths, thresholds, synonyms.
- Env precedence: The ETL auto-loads `.env` at repo root and prefers env vars for DB over YAML.
  - Preferred: `DATABASE_URL=postgresql://USER:PASSWORD@HOST:PORT/DBNAME`
  - Alternative: `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`
  - Fallback: `db_url` from YAML if env not set.

### Uniclass Revision Handling

- `uniclass.revision`: Fallback revision tag (string) in YAML.
- `uniclass.autodetect_revision` (default true): Parses revision from filenames (e.g., `..._v1_24.xlsx`) and overrides the YAML value.
- `uniclass.enforce_monotonic_revision` (default true): Prevents loading if the incoming revision is not greater than the latest stored.
- History: Each load writes snapshots to `uniclass_item_revision (code, revision, ...)` while `uniclass_item` holds the latest per code.

### Matching, Anchors, Rerank

- Facet rules: control eligibility with `matching.rules`, and disable facets with `matching.skip_tables`.
- Feature boosts: `_feature_multiplier` favors relevant IFCs (type bias, predefined type, pset count, group/system, etc.).
- Anchors from Uniclass: XLSX loader extracts IFC IDs from any “IFC …” columns into `ifc_mappings`; candidate scoring adds `matching.anchor_bonus` (optionally including ancestors) to those pairs.
- Rerank with Ollama: `rerank.top_n`, `rerank.model`, `rerank.temperature`, `rerank.max_tokens`, `rerank.fewshot_per_facet`.
  - Use a concise instruct model (e.g., `llama3.2:latest`, `llama3.1:8b-instruct`, `dolphin3:latest`).
  - Few-shot examples are drawn from `ifc_mappings` for the same facet (if present).

## Dependencies

- Python: `pandas`, `psycopg[binary]`, `rapidfuzz`, `pyyaml`, `openpyxl`, `requests` (for `--embed`)
- Postgres: 14+ with `pgcrypto` and `pgvector` (extension `vector` must be installed on the server). The schema creates both extensions if available.
- Optional (embeddings): [Ollama](https://ollama.com) with a local embedding model. Recommended: `mxbai-embed-large` (1024 dims).
- Optional (rerank): Ollama with a local instruct model (e.g., `llama3.2:latest`).

## Outputs

- `output/candidates.csv`: Candidate list with blended or reranked scores.
- `output/viewer_mapping.json`: Edges grouped by IFC class for a viewer.

## CLI Reference

- `--load`                Load IFC + Uniclass into DB (upsert)
- `--embed`               Generate and store embeddings via Ollama
- `--candidates`          Generate candidate mappings and auto-accept above threshold
- `--export`              Export accepted mappings to `viewer_mapping.json`
- `--classify-disciplines` Classify IFC + Uniclass into labels (disciplines/subjects) via LLM; writes `output/taxonomy_cache.json`
- `--reset-mappings`      Delete existing mappings for current Uniclass revision (see below)
- `--reset-facets PR,SS`  Limit reset to listed Uniclass facets

Classification flags:

- `--classify-scope both|ifc|uniclass`  Limit which side to classify
- `--classify-facets PR,SS`             Limit Uniclass facets to classify
- `--classify-limit N`                  Stop after N items (0 = no limit)
- `--classify-model name`               Override Ollama model for classification
- `--classify-timeout SECONDS`          Per-call timeout (falls back to `rerank.timeout_s`)
- `--classify-warmup-timeout SECONDS`   Warmup timeout for initial model load
- `--classify-sleep SECONDS`            Sleep between requests for rate limiting

Reset examples:

- Reset all facets for current revision: `--reset-mappings`
- Reset only PR/SS: `--reset-mappings --reset-facets PR,SS`

## Getting Started

PowerShell quickstart (Windows):

```powershell
# 1) Create and activate venv with latest python
py -3 -m venv .venv
. .\.venv\Scripts\Activate.ps1
pip install pandas psycopg[binary] rapidfuzz pyyaml openpyxl requests

# 2) Copy config and adjust paths if needed
cp config/settings.example.yaml config/settings.yaml

# 3) Database connection (or put the same value in .env)
$env:DATABASE_URL = "postgresql://postgres:***@host.docker.internal:5432/iob_ifc-uniclass"

# 4) Apply schema (requires pgvector installed on the server)
psql "$env:DATABASE_URL" -f sql/schema.sql

# 5) Optional: start Ollama for embeddings and rerank (once)
# docker run -d --name ollama -p 11434:11434 ollama/ollama
# ollama pull mxbai-embed-large
# ollama pull llama3.2:latest   # for rerank (or another small instruct model)

# 6) Load source data into Postgres
py etl/etl_map.py --config config/settings.yaml --load

# 7) Generate embeddings (re-run if model/text changes)
py etl/etl_map.py --config config/settings.yaml --embed

# 8) (Optional) Reset previous mappings for current revision
py etl/etl_map.py --config config/settings.yaml --reset-mappings
py etl/etl_map.py --config config/settings.yaml --reset-facets PR,SS
py etl/etl_map.py --config config/settings.yaml --reset-mappings --reset-facets PR,SS

# 9) Generate candidates (blended or with rerank if rerank.top_n > 0)
py etl/etl_map.py --config config/settings.yaml --candidates

# 10) Export viewer JSON
py etl/etl_map.py --config config/settings.yaml --export
```

Tips:

- Revision: Filenames like `..._v1_24.xlsx` auto-detect Uniclass revision; monotonic enforcement stops downgrades. Override in `config/settings.yaml`.
- Embeddings: If you change model or embedded text format, clear old rows and re-embed:
  - `DELETE FROM text_embedding WHERE entity_type IN ('ifc','uniclass');`
  - Ensure `embedding.expected_dim` matches your model (e.g., 1024 for `mxbai-embed-large`).

## Current State Summary

- Performance: Tight loops optimized (dictionary lookups instead of per-iteration DataFrame `.loc`); cosine-based pgvector ranking aligned between ORDER BY and score.
- Scoring: Blended lexical + embedding with feature multipliers; Uniclass anchors from XLSX boost known pairs; confidence clamped to [0,1] to satisfy DB constraints.
- Embeddings: Enriched texts on both sides; supports `mxbai-embed-large` via `embedding.expected_dim`. Note: LLM classification affects matching (gating/penalties), not the embedding text. Re-embed only if you change the embedding model or decide to include labels into the embedded text (not enabled by default).
- Rerank: Optional Ollama instruct-model re-rank with few-shot examples from anchors; controlled by `rerank.*` in config.
- Maintenance: `--reset-mappings` and `--reset-facets` flags to safely clear mapping rows by revision/facet before regeneration.
