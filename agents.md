# Pipeline Agents

`etl/etl_map.py` orchestrates a set of focused agents that move data from raw IFC and Uniclass sources to reviewed mapping records. This note outlines each agent, the primary helper functions, and the artefacts they read or write.

## Settings and context loader

- Responsibilities: read `.env`, merge configuration defaults, and assemble a `Settings` object consumed by later stages.
- Key functions: `_load_dotenv`, `load_settings`.
- Outputs: populated `Settings` dataclass, resolved paths, connection strings.

## Source ingestion agent

- Responsibilities: read IFC JSON, gather ancestor context, parse Uniclass CSV or XLSX files, normalise structured attributes, and load them into Postgres.
- Key functions: `read_ifc_json`, `_build_ifc_ancestors_map`, `_compute_ifc_flags`, `read_uniclass_xlsx_dir`, `facet_from_filename`, `upsert_ifc`, `upsert_ifc_features`, `upsert_uniclass`, `upsert_uniclass_features`.
- Outputs: `ifc_class`, `ifc_feature`, `uniclass_item`, `uniclass_feature`, `uniclass_item_revision`, and anchor hints stored in staging columns.

## Embedding agent

- Responsibilities: build contextual text snippets, request embeddings from Ollama or OpenAI, and persist vectors for efficient nearest neighbour search.
- Key functions: `generate_and_store_embeddings`, `_ollama_embed`, `_openai_embed`, `_ensure_vector_literal`, `_existing_embedding_ids`, `ensure_vector_indexes`.
- Outputs: rows in `text_embedding`, pgvector indexes, enriched debug logging when vectors are skipped or retried.

## Discipline classifier agent

- Responsibilities: prompt the configured LLM for multi-label disciplines, reuse cached section labels, and stream updates into the taxonomy cache.
- Key functions: `classify_items_with_llm`, `_format_label_prompt`, `_parse_label_array`, `_load_taxonomy_cache`.
- Outputs: `output/taxonomy_cache.json`, optional console summaries, label dictionaries reused by matching.

## Candidate generator agent

- Responsibilities: compute lexical scores, expand synonyms, fetch embedding neighbours, blend scoring dimensions, infer relation types, and persist candidate edges.
- Key functions: `normalize_text`, `expand_with_synonyms`, `score_pair`, `_embedding_neighbors`, `generate_candidates`, `generate_candidates_blended`, `generate_candidates_uc2ifc_blended`, `infer_relation_type`, `log_candidate_history`, `write_candidate_edges`.
- Outputs: candidate lists for review, per-run snapshots in `ifc_uniclass_candidate_history`, accepted rows in `ifc_uniclass_map`, logging for filtered facets and skipped matches.

## LLM reranker agent

- Responsibilities: prepare facet-specific few-shot prompts, call the reranking model for the configured top-N items, and merge revised scores with the candidate list.
- Key functions: `_collect_anchor_examples`, `_format_rerank_prompt`, `_ollama_generate`, `_openai_generate`, `_parse_rerank_json`, `rerank_candidates_with_llm`.
- Outputs: reranked confidence scores, trace logging for prompt tokens, fallback behaviour when models are unavailable.

## Export and maintenance agent

- Responsibilities: reset stale mappings, export viewer-friendly JSON, and ensure downstream consumers receive consistent revision metadata.
- Key functions: `reset_mappings`, `export_viewer_json`, `ensure_output_dirs`.
- Outputs: pruned `ifc_uniclass_map` rows per revision, `output/viewer_mapping.json`, and directories ready for UI assets.

## Evaluation agent

- Responsibilities: log each candidate generation run, join reviewer outcomes, compute precision/recall scorecards, and flag regressions.
- Key functions: `log_candidate_history`, `scripts/evaluate.py` (`evaluate_run`, `build_report`).
- Outputs: `ifc_uniclass_candidate_history`, `review_decision`, JSON/Markdown reports under `output/evaluations/`.

## Proposed future agents

- **Property feature extraction agent**: Parse IFC property sets, product type enumerations, and quantitative attributes into a normalized feature table (`ifc_feature`). Map Uniclass attribute columns into the same vocabulary so the candidate generator can reference explicit feature overlap metrics rather than relying solely on text.
- **Facet calibration agent**: Train and apply lightweight classification or regression models per facet to combine lexical, vector, label, feature, and anchor signals. Persist learned weights and updates alongside revision metadata to justify acceptance thresholds.

Together these agents provide a modular framing for extending the pipeline, enabling focused improvements without rewriting the entire ETL script.
