import os
from pathlib import Path

import sys
import psycopg

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from etl.etl_map import load_settings


def main():
    s = load_settings(Path("config/settings.yaml"))
    print(f"DB URL: {s.db_url}")
    with psycopg.connect(s.db_url, autocommit=True) as conn:
        with conn.cursor() as cur:
            def count(tbl: str) -> int:
                try:
                    cur.execute(f"SELECT COUNT(*) FROM {tbl}")
                    return int(cur.fetchone()[0])
                except Exception as e:
                    print(f"[warn] count {tbl}: {e}")
                    return -1

            for t in [
                "ifc_class",
                "uniclass_item",
                "uniclass_item_revision",
                "text_embedding",
                "ifc_uniclass_map",
            ]:
                print(f"{t}: {count(t)}")

            try:
                cur.execute("SELECT entity_type, COUNT(*) FROM text_embedding GROUP BY entity_type")
                for et, cnt in cur.fetchall():
                    print(f"text_embedding[{et}]: {cnt}")
            except Exception as e:
                print(f"[warn] text_embedding breakdown: {e}")

    cand_path = Path("output/candidates.csv")
    if cand_path.exists():
        print(f"candidates.csv exists, size={cand_path.stat().st_size} bytes")
    else:
        print("candidates.csv missing")


if __name__ == "__main__":
    main()
