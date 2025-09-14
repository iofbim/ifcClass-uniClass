-- Schema for IFC ↔ Uniclass mapping (versioned, typed edges)
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
