import csv
import json
from collections import Counter
from pathlib import Path

chunks = json.loads(Path("data/chunks.json").read_text(encoding="utf-8"))
known = {(c["file"], c["name"]) for c in chunks}

with Path("data/golden.csv").open(encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

print("Questions:", len(rows))
print("By kind:", dict(Counter(r["kind"] for r in rows)))

problems = 0
for r in rows:
    paths = r["expected_path"].split(";") if r["expected_path"] else []
    symbols = r["expected_symbol"].split(";") if r["expected_symbol"] else []

    if len(paths) != len(symbols):
        print(f"Row {r['id']}: {len(paths)} paths but {len(symbols)} symbols")
        problems += 1
        continue
    if r["kind"] == "unanswerable" and symbols:
        print(f"Row {r['id']}: unanswerable question has expected chunks")
        problems += 1
    if r["kind"] != "unanswerable" and not symbols:
        print(f"Row {r['id']}: answerable question has no expected chunk")
        problems += 1
    for path, symbol in zip(paths, symbols):
        if (path, symbol) not in known:
            print(f"Row {r['id']}: no chunk named {symbol} in {path}")
            problems += 1

print("Problems found:", problems)