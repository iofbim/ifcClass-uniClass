# ifcClass-uniClass

This repository establishes structured mappings between Uniclass 2015 classification tables and IFC 4x3 schema classes.

## Overview

- Goal: Load IFC classes and Uniclass tables into Postgres, generate candidate mappings, and export a JSON for a viewer.
- Core Script: `etl/etl_map.py` orchestrates loading, matching, and exporting via CLI flags.
- Schema: `sql/schema.sql` defines tables (`ifc_class`, `uniclass_item`, `uniclass_item_revision`, `ifc_uniclass_map`, `mapping_flag`, `text_embedding`) and enables `pgcrypto` and `vector` (pgvector).

## Data Model

- IFC: Classes with `ifc_id`, `schema_version`, `label`, `description`.
- Uniclass: Items with `code`, `facet` (EF/SS/PR/... inferred), `title`, `description`, `revision`.
- Mappings: `ifc_uniclass_map` stores edges with `relation_type` (typical_of, part_of, equivalent), `confidence`, and provenance (IFC version, Uniclass revision).

## ETL Flow

- Load (`--load`): Inserts/updates IFC classes and Uniclass items in bulk using psycopg copy/UPSERT.
- Candidates (`--candidates`):
  - Lexical: compares IFC description/definition (fallback to label) against Uniclass title only; blocks by facet heuristics.
  - Vector: optionally blends with pgvector similarity (IFC description/definition embeddings vs Uniclass title embeddings) controlled by `matching.embedding_weight`.
  - Auto-accepts above threshold with rationale.
- Embed (`--embed`): Generates embeddings for IFC/Uniclass text via Ollama and stores them in `text_embedding`.
- Export (`--export`): Produces `output/viewer_mapping.json` aggregating accepted mappings for the viewer.

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

### Matching Rules

- IFC side: uses class description/definition text; falls back to label if description is empty.
- Uniclass side: uses title only (no description/notes used).
- Embeddings: built from the same signals (IFC description/definition vs Uniclass title).

## Dependencies

- Python: `pandas`, `psycopg[binary]`, `rapidfuzz`, `pyyaml`, `openpyxl`, `requests` (for `--embed`)
- Postgres: 14+ with `pgcrypto` and `pgvector` (extension `vector` must be installed on the server). The schema creates both extensions if available.
- Optional (embeddings): [Ollama](https://ollama.com) with a local embedding model (default: `nomic-embed-text`, 768 dims).

## Outputs

- `output/candidates.csv`: Top-k Uniclass candidates per IFC class with scores.
- `output/viewer_mapping.json`: Edges grouped by IFC class for a viewer.

## Getting Started

PowerShell quickstart (Windows):

```powershell
# 1) Create and activate venv
py -3.11 -m venv .venv
. .\.venv\Scripts\Activate.ps1
pip install pandas psycopg[binary] rapidfuzz pyyaml openpyxl requests

# 2) Copy config and adjust paths if needed
cp config/settings.example.yaml config/settings.yaml

# 3) Database connection (or put the same value in .env)
$env:DATABASE_URL = "postgresql://postgres:***@host.docker.internal:5432/iob_ifc-uniclass"

# 4) Apply schema (requires pgvector installed on the server)
psql "$env:DATABASE_URL" -f sql/schema.sql

# 5) Optional: start Ollama for embeddings (once)
# docker run -d --name ollama -p 11434:11434 ollama/ollama
# ollama pull nomic-embed-text

# 6) Load source data into Postgres
py etl/etl_map.py --config config/settings.yaml --load

# 7) Generate embeddings (optional, improves candidates)
py etl/etl_map.py --config config/settings.yaml --embed

# 8) Generate candidate mappings (uses blended scoring if matching.embedding_weight > 0)
py etl/etl_map.py --config config/settings.yaml --candidates

# 9) Export viewer JSON
py etl/etl_map.py --config config/settings.yaml --export
```

Tips:
- Revision: Filenames like `..._v1_24.xlsx` auto-detect Uniclass revision; monotonic enforcement stops downgrades. Override in `config/settings.yaml`.
- Embeddings: Default model `nomic-embed-text` (768 dims). If you change models, update `sql/schema.sql` (`vector(768)`) to match.
