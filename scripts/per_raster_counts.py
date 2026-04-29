from pathlib import Path
import csv

p = Path("output/tile_labels")
files = sorted([f for f in p.glob("*_labels.csv") if f.is_file() and not f.name.endswith(".bak")])
print(f"Found {len(files)} label CSVs")
summary = []
for f in files:
    dedup = {}
    with open(f, newline="") as fh:
        rdr = csv.DictReader(fh)
        for row in rdr:
            key = (row.get("row_off", ""), row.get("col_off", ""))
            dedup[key] = row.get("label", "")
    counts = {"cdw": 0, "no_cdw": 0, "unknown": 0}
    for v in dedup.values():
        if v in counts:
            counts[v] += 1
    total = sum(counts.values())
    summary.append((f.name, total, counts["cdw"], counts["no_cdw"], counts["unknown"]))
    print(
        f"{f.name}: total={total}  cdw={counts['cdw']}  no_cdw={counts['no_cdw']}  unknown={counts['unknown']}"
    )
# Grand totals
gt = {"total": 0, "cdw": 0, "no_cdw": 0, "unknown": 0}
for _, t, cdw, no, unk in summary:
    gt["total"] += t
    gt["cdw"] += cdw
    gt["no_cdw"] += no
    gt["unknown"] += unk
print("\nGrand total tiles:", gt["total"])
print("Grand CDW:", gt["cdw"], "No CDW:", gt["no_cdw"], "Unknown:", gt["unknown"])
