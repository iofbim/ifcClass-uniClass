from pathlib import Path
import psycopg
from etl.etl_map import load_settings


def main():
    s = load_settings(Path("config/settings.yaml"))
    with psycopg.connect(s.db_url, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT embedding::text FROM text_embedding WHERE entity_type='ifc' LIMIT 1")
            row = cur.fetchone()
            if not row or not row[0]:
                print("No IFC embedding found")
                return
            qvec = row[0]
            cur.execute(
                "EXPLAIN (ANALYZE, BUFFERS, VERBOSE) "
                "SELECT entity_id, 1 - (embedding <=> %s::vector) AS score "
                "FROM text_embedding WHERE entity_type='uniclass' "
                "ORDER BY embedding <-> %s::vector LIMIT 50",
                (qvec, qvec),
            )
            plan = "\n".join(r[0] for r in cur.fetchall())
            print(plan)


if __name__ == "__main__":
    main()

