# ifcClass-uniClass

A reproducible pipeline for aligning IFC 4x3 entities with Uniclass 2015 classifications using Postgres, hybrid similarity scoring, and optional LLM assistance.

## What this project delivers

- Normalized Postgres schema (`sql/schema.sql`) capturing IFC classes, Uniclass items, embeddings, anchors, and accepted relationships.
- Single CLI entrypoint (`etl/etl_map.py`) that orchestrates loading, embedding, classification, candidate generation, re-ranking, and export.
- Hybrid scoring that blends lexical similarity, pgvector nearest neighbours, Uniclass-provided IFC anchors, and discipline labels.
- Incremental discipline/subject tagging with a cached taxonomy (`output/taxonomy_cache.json`) to gate candidates and reduce repeated LLM calls.
- Exportable viewer payloads in `output/viewer_mapping.json` for downstream review tooling or visualization layers.
- Structured feature caches in `ifc_feature` and `uniclass_feature` tables for property-level alignment.
- Logged candidate runs and reviewer feedback (`ifc_uniclass_candidate_history`, `review_decision`) for downstream evaluation and regression checks.

## Repository layout

- `etl/etl_map.py` – main orchestration script with helpers for embedding, classification, candidate generation, and export.
- `config/settings.yaml` – active configuration; `settings.example.yaml` documents defaults and optional tweaks.
- `sql/schema.sql` – Postgres schema, extensions (`pgcrypto`, `vector`), indexes, and helper views.
- `Samples/` – example IFC JSON and Uniclass spreadsheets for development and testing.
- `output/viewer_mapping.json` – generated mapping for viewers; `output/taxonomy_cache.json` – cached LLM discipline labels.
- `scripts/` – utility notebooks/scripts for diagnostics (if provided, not required for core flow).

## End-to-end flow

1. **Configure** – ensure `config/settings.yaml` points to the IFC JSON, Uniclass source, and database. Environment variables in `.env` or your shell override equivalent values.
2. **Load source data**  
   `py etl/etl_map.py --config config/settings.yaml --load`  
   Upserts IFC classes (`ifc_class`) and Uniclass items (`uniclass_item` + revisions) while capturing anchors from spreadsheet metadata.
3. **Generate embeddings**  
   `py etl/etl_map.py --config config/settings.yaml --embed`  
   Creates contextualized text for each entity, calls Ollama or OpenAI to embed, and stores vectors in `text_embedding`. Re-run when text formatting or model changes.
4. **Classify disciplines (optional but recommended)**  
   `py etl/etl_map.py --config config/settings.yaml --classify-disciplines`  
   Uses the configured AI backend for multi-label discipline tagging. Section-level Uniclass calls are cached in `output/taxonomy_cache.json` and replayed on subsequent runs.
5. **Generate candidates**  
   `py etl/etl_map.py --config config/settings.yaml --candidates`  
   Produces ranked IFC to Uniclass suggestions, blending lexical scores, embedding distances, feature boosts, anchors, and label gating. Rows above `matching.auto_accept_threshold` are written to `ifc_uniclass_map`.
   - Set `matching.direction` to `uniclass_to_ifc` to ensure coverage from the Uniclass side.
   - Use `--reset-mappings` (optionally `--reset-facets PR,SS`) before re-generating to avoid stale rows from older revisions.
6. **Optional LLM re-rank**  
   Enable by setting `rerank.top_n > 0` in the config; the CLI will automatically call the reranker during candidate generation to refine ordering inside the top-N shortlist per item.
7. **Export viewer payload**  
   `py etl/etl_map.py --config config/settings.yaml --export`  
   Writes a compact JSON summarising accepted links plus metadata for UI consumption.

## Classification cache and review workflow

- `output/taxonomy_cache.json` stores discipline labels keyed by entity identifier. Delete the file to force a fresh labelling pass.
- Cached labels are merged into candidate generation, optionally penalising or removing mismatched disciplines depending on `matching.discipline_filter`.
- Use the viewer JSON alongside manual QA tools to inspect `confidence`, `relation_type`, and the provenance stored in `ifc_uniclass_map`.
- The candidate CLI prints a UUID per run and persists the full score breakdown to `ifc_uniclass_candidate_history` for later audit.
- Record reviewer outcomes in `review_decision` so evaluation runs can compare predictions against curated decisions.
- Generate precision/recall scorecards with `python scripts/evaluate.py --config config/settings.yaml --run-id <RUN_UUID>` (optionally provide `--baseline-run-id` to compare against a previous run).

## Configuration highlights

- **Database**: `db_url` (or `DATABASE_URL`) plus optional `POSTGRES_*` fallbacks for convenience.
- **Features**: toggle structured extraction with `features.enabled` and extend detection keywords via `features.attribute_tokens`.
- **Evaluation**: configure `evaluation.*` (baseline run, precision drop tolerance, recall floor, acceptance threshold, fail-on-regression) to automate scorecard checks.
- **IFC input**: `ifc.json_path`, `ifc.schema_version`, and `ifc.use_parent_context` to decide whether ancestor descriptions enrich embedding text.
- **Uniclass input**: configure an `xlsx_dir` for bulk ingestion or explicit `csv_paths`. `autodetect_revision` and `enforce_monotonic_revision` defend against downgrading revisions.
- **Matching**:
  - `top_k`, `embedding_weight`, `embedding_top_k` tune the lexical/vector blend.
  - `rules` and `skip_tables` gate which IFC roots or facets participate.
  - `anchor_bonus` and `anchor_use_ancestors` leverage spreadsheet-provided IFC references.
  - `synonyms` supplies domain-specific term equivalences that augment lexical token sets.
  - Discipline gating: `discipline_filter`, `discipline_penalty`, and `discipline_source`.
- **Embedding backend**: Select Ollama or OpenAI models, vector dimensions, batch sizes, and ANN index parameters (`embedding.ivf_lists`, `embedding.ivf_probes`, `embedding.query_timeout_ms`).
- **Rerank backend**: Configure `rerank.model`, `rerank.top_n`, `fewshot_per_facet`, and timeouts; override with OpenAI equivalents when needed.
- **Output**: `output.dir` and `output.viewer_json`.

## Running the CLI

Common command snippets (PowerShell on Windows shown; swap to `python`/`pip` on other platforms):

```powershell
# Environment prep
py -3 -m venv .venv
. .\.venv\Scripts\Activate.ps1
pip install pandas psycopg[binary] rapidfuzz pyyaml openpyxl requests

# Database schema
psql "$env:DATABASE_URL" -f sql/schema.sql

# Core workflow
py etl/etl_map.py --config config/settings.yaml --load
py etl/etl_map.py --config config/settings.yaml --embed
py etl/etl_map.py --config config/settings.yaml --classify-disciplines --classify-scope both
py etl/etl_map.py --config config/settings.yaml --candidates
py etl/etl_map.py --config config/settings.yaml --export
```

Reset helpers:

```powershell
py etl/etl_map.py --config config/settings.yaml --reset-mappings
py etl/etl_map.py --config config/settings.yaml --reset-mappings --reset-facets PR,SS
```

## Automated evaluation

Use the recorded run id from candidate generation to score a run and compare against a baseline.

```powershell
python scripts/evaluate.py --config config/settings.yaml --run-id <RUN_UUID> --baseline-run-id <BASELINE_UUID> --fail-on-regression
```

Adjust thresholds via the `evaluation` block when you want the CLI to fail builds on regressions.

Set `--openai`, `--classify-facets`, `--classify-limit`, or `--classify-model` to override defaults for targeted runs.

## Improving IFC to Uniclass matching accuracy

1. **Structured feature alignment**  
   Parse IFC property sets, type enumerations (`PredefinedType`), and element assemblies into a feature table (e.g., `ifc_feature` with key/value pairs). Map Uniclass attribute tokens (from "Attributes" worksheet columns) into the same canonical vocabulary and introduce feature overlap scores. Matching models benefit from explicit signals like duct shape, flow medium, or system role beyond free text.
2. **Facet-specific scoring models**  
   Persist training snapshots of accepted mappings and fit lightweight classifiers (logistic regression or gradient-boosted trees) per facet that consume lexical similarity, embedding cosine, discipline overlap, feature overlap, anchor presence, and hierarchy distance. Replace the current linear blend with learned weights tuned to each facet noise profile.
3. **Graph-propagated anchors**  
   Use IFC inheritance and relationship graphs (`IfcRelAssignsToGroup`, `IfcRelAggregates`, `IfcRelPorts`) to propagate high-confidence anchors along structural edges. Penalise candidate pairs that violate graph constraints (e.g., subsystem relationships) or contradict known parent-child matches. This reduces drift when two Uniclass codes share similar names but belong to different systems.
4. **Token canonicalisation pipeline**  
   Maintain a `normalized_term` table containing lemmatized, domain-expanded tokens (unit conversions, regional spellings). Feed it into both lexical matching and embedding text generation to ensure consistent vocabulary before scoring. This also enables deterministic matching passes for highly structured tokens such as `Pr_23_75` or `IfcChillerType`.
5. **Incremental evaluation harness**  
   Record precision/recall metrics by facet after each ETL run, using reviewer decisions to generate gold labels. Automate regression detection so configuration tweaks can be measured before full-scale reruns.

Combining the structured features with facet-specific models provides a path toward more precise, explainable matches while keeping the pipeline efficient inside Postgres.

## Tips and troubleshooting

- Run `ensure_vector_indexes()` (automatic inside the CLI) after creating new embeddings so pgvector uses ANN search with the configured IVF parameters.
- Always bump the Uniclass revision in the database when ingesting fresh spreadsheets; monotonic enforcement prevents accidental downgrades.
- If embeddings fail due to dimension mismatch, update `embedding.expected_dim` (Ollama) or `embedding.openai_expected_dim` (OpenAI) and re-run `--embed` after clearing `text_embedding` for the affected entity types.
- Set `PYTHONWARNINGS=ignore` or adjust logging within the script if repeated retries clutter the console during long runs.
- See `agents.md` for a breakdown of the AI-driven components and hand-off points inside the pipeline.
