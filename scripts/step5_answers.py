import csv
import json
import sys
from pathlib import Path

from rag import ask, refused

LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 20

with Path("data/golden.csv").open(encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))
step = max(1, len(rows) // LIMIT)
rows = rows[::step][:LIMIT]  # an even spread across the file, so every kind is included

results = []
for row in rows:
    result = ask(row["question"])
    expected = set(zip(row["expected_path"].split(";"), row["expected_symbol"].split(";"))) if row["expected_symbol"] else set()
    searched = {(c["file"], c["name"]) for c in result["retrieved"][: result["n_search"]]}
    result.update(
        id=row["id"],
        kind=row["kind"],
        reference_answer=row["reference_answer"],
        found_expected_chunk=bool(expected & searched),
        refused=refused(result["answer"]),
    )
    results.append(result)
    print(f"row {row['id']:>2} [{row['kind']}]  chunk found: {str(result['found_expected_chunk']):<5}  refused: {result['refused']}")

Path("data/answers_final.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
print()
print("Questions:", len(results), "  Total tokens:", sum(r["tokens"] for r in results))
print("Saved to data/answers_final.json")