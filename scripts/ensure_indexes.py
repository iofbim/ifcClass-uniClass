from pathlib import Path

import psycopg

from etl.etl_map import load_settings


SQL = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE INDEX IF NOT EXISTS idx_te_uniclass_ivfflat
  ON text_embedding USING ivfflat (embedding vector_cosine_ops)
  WITH (lists = 100)
  WHERE entity_type = 'uniclass';
ANALYZE text_embedding;
"""


def main():
    s = load_settings(Path("config/settings.yaml"))
    with psycopg.connect(s.db_url, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(SQL)
    print("Indexes ensured and ANALYZE done.")


if __name__ == "__main__":
    main()

