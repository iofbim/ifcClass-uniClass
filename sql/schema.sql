-- Schema for IFC <-> Uniclass mapping (versioned, typed edges)
-- Requires: Postgres 14+; optional pgvector

CREATE EXTENSION IF NOT EXISTS pgcrypto; -- for gen_random_uuid
CREATE EXTENSION IF NOT EXISTS vector; -- enable if using embeddings

CREATE TABLE IF NOT EXISTS ifc_class (
  ifc_id TEXT PRIMARY KEY,
  schema_version TEXT NOT NULL,
  label TEXT,
  description TEXT
);

CREATE TABLE IF NOT EXISTS uniclass_item (
  code TEXT PRIMARY KEY,
  facet TEXT NOT NULL,
  title TEXT NOT NULL,
  revision TEXT NOT NULL
);

-- Historical snapshots of Uniclass rows per revision (immutable once inserted)
CREATE TABLE IF NOT EXISTS uniclass_item_revision (
  code TEXT NOT NULL,
  revision TEXT NOT NULL,
  facet TEXT NOT NULL,
  title TEXT NOT NULL,
  inserted_at TIMESTAMPTZ DEFAULT now(),
  PRIMARY KEY (code, revision)
);
CREATE INDEX IF NOT EXISTS idx_uc_rev_revision ON uniclass_item_revision(revision);

-- Optional embeddings
-- Adjust dimension to your model if enabling pgvector
CREATE TABLE IF NOT EXISTS text_embedding (
  id SERIAL PRIMARY KEY,
  entity_type TEXT NOT NULL CHECK (entity_type IN ('ifc','uniclass')),
  entity_id TEXT NOT NULL,
  embedding vector(1024),
  UNIQUE(entity_type, entity_id)
);

-- Nearest neighbor index to accelerate embedding searches (recommended)
-- Note: requires pgvector 0.5.0+ and Postgres 14+. Adjust lists per data size.
CREATE INDEX IF NOT EXISTS idx_te_uniclass_ivfflat
  ON text_embedding USING ivfflat (embedding vector_cosine_ops)
  WITH (lists = 100)
  WHERE entity_type = 'uniclass';


-- Normalized feature key/value pairs for structured matching
CREATE TABLE IF NOT EXISTS ifc_feature (
  ifc_id TEXT NOT NULL REFERENCES ifc_class(ifc_id) ON DELETE CASCADE,
  feature_key TEXT NOT NULL,
  feature_value TEXT NOT NULL,
  source TEXT,
  confidence NUMERIC DEFAULT 1 CHECK (confidence BETWEEN 0 AND 1),
  created_at TIMESTAMPTZ DEFAULT now(),
  PRIMARY KEY (ifc_id, feature_key, feature_value)
);
CREATE INDEX IF NOT EXISTS idx_ifc_feature_key ON ifc_feature(feature_key);
CREATE INDEX IF NOT EXISTS idx_ifc_feature_value ON ifc_feature(feature_value);

CREATE TABLE IF NOT EXISTS uniclass_feature (
  code TEXT NOT NULL REFERENCES uniclass_item(code) ON DELETE CASCADE,
  feature_key TEXT NOT NULL,
  feature_value TEXT NOT NULL,
  source TEXT,
  confidence NUMERIC DEFAULT 1 CHECK (confidence BETWEEN 0 AND 1),
  created_at TIMESTAMPTZ DEFAULT now(),
  PRIMARY KEY (code, feature_key, feature_value)
);
CREATE INDEX IF NOT EXISTS idx_uc_feature_key ON uniclass_feature(feature_key);
CREATE INDEX IF NOT EXISTS idx_uc_feature_value ON uniclass_feature(feature_value);
-- Historical record of candidate scores per generation run
CREATE TABLE IF NOT EXISTS ifc_uniclass_candidate_history (
  history_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  run_id UUID NOT NULL,
  generated_at TIMESTAMPTZ DEFAULT now(),
  ifc_id TEXT NOT NULL REFERENCES ifc_class(ifc_id) ON DELETE CASCADE,
  code TEXT NOT NULL REFERENCES uniclass_item(code) ON DELETE CASCADE,
  facet TEXT,
  score NUMERIC,
  lexical_score NUMERIC,
  embedding_score NUMERIC,
  feature_multiplier NUMERIC,
  discipline_multiplier NUMERIC,
  token_overlap_multiplier NUMERIC,
  anchor_applied BOOLEAN,
  direction TEXT,
  source_ifc_version TEXT,
  source_uniclass_revision TEXT
);
CREATE INDEX IF NOT EXISTS idx_candidate_history_run ON ifc_uniclass_candidate_history(run_id);
CREATE INDEX IF NOT EXISTS idx_candidate_history_facet ON ifc_uniclass_candidate_history(facet);

-- Reviewer feedback captured per IFC <-> Uniclass pair
CREATE TABLE IF NOT EXISTS review_decision (
  decision_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  ifc_id TEXT NOT NULL REFERENCES ifc_class(ifc_id) ON DELETE CASCADE,
  code TEXT NOT NULL REFERENCES uniclass_item(code) ON DELETE CASCADE,
  relation_type TEXT,
  decision TEXT NOT NULL CHECK (decision IN ('accept','reject','needs_review')),
  reviewer TEXT,
  notes TEXT,
  decided_at TIMESTAMPTZ DEFAULT now(),
  run_id UUID,
  score NUMERIC,
  lexical_score NUMERIC,
  embedding_score NUMERIC,
  feature_multiplier NUMERIC,
  discipline_multiplier NUMERIC,
  token_overlap_multiplier NUMERIC,
  anchor_applied BOOLEAN,
  facet TEXT,
  source_uniclass_revision TEXT
);
CREATE INDEX IF NOT EXISTS idx_review_decision_pair ON review_decision(ifc_id, code);
CREATE INDEX IF NOT EXISTS idx_review_decision_facet ON review_decision(facet);
CREATE INDEX IF NOT EXISTS idx_review_decision_revision ON review_decision(source_uniclass_revision);

CREATE TABLE IF NOT EXISTS ifc_uniclass_map (
  map_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  ifc_id TEXT REFERENCES ifc_class(ifc_id) ON DELETE CASCADE,
  code  TEXT REFERENCES uniclass_item(code) ON DELETE CASCADE,
  relation_type TEXT NOT NULL CHECK (
    relation_type IN ('equivalent','broader','narrower','typical_of','part_of','has_property')
  ),
  confidence NUMERIC CHECK (confidence BETWEEN 0 AND 1),
  rationale TEXT,
  created_by TEXT,
  created_at TIMESTAMPTZ DEFAULT now(),
  source_ifc_version TEXT,
  source_uniclass_revision TEXT,
  UNIQUE (ifc_id, code, relation_type, source_uniclass_revision)
);

-- Helpful indexes
CREATE INDEX IF NOT EXISTS idx_uniclass_facet ON uniclass_item(facet);
CREATE INDEX IF NOT EXISTS idx_uniclass_revision ON uniclass_item(revision);
CREATE INDEX IF NOT EXISTS idx_map_ifc ON ifc_uniclass_map(ifc_id);
CREATE INDEX IF NOT EXISTS idx_map_code ON ifc_uniclass_map(code);
CREATE INDEX IF NOT EXISTS idx_map_type ON ifc_uniclass_map(relation_type);


