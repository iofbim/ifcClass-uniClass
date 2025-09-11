import json, csv
from pathlib import Path
data = json.load(open("output/viewer_mapping.json", "r", encoding="utf-8"))
rows = []
for ifc_id, rels in data.items():
    for rel, items in rels.items():
        for it in items:
            rows.append({
            "ifc_id": ifc_id,
            "relation": rel,
            "code": it["code"],
            "title": it.get("title",""),
            "confidence": it.get("confidence",""),
            })

out = Path("output/viewer_mapping.csv")

with out.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["ifc_id","relation","code","title","confidence"])
    w.writeheader(); w.writerows(rows)

print("wrote", out)