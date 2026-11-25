import csv
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer, util

K = 5

chunks = json.loads(Path("data/chunks.json").read_text(encoding="utf-8"))
embeddings = np.load("data/embeddings.npy")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

with Path("data/golden.csv").open(encoding="utf-8-sig", newline="") as f:
    rows = [r for r in csv.DictReader(f) if r["kind"] != "unanswerable"]

question_vectors = embedder.encode([r["question"] for r in rows])
all_scores = util.cos_sim(question_vectors, embeddings)

hits = 0
recalls = []
reciprocal_ranks = []
by_kind = {}
misses = []

for row, scores in zip(rows, all_scores):
    expected = set(zip(row["expected_path"].split(";"), row["expected_symbol"].split(";")))
    order = scores.argsort(descending=True)
    ranked = [(chunks[int(i)]["file"], chunks[int(i)]["name"]) for i in order]

    found = expected & set(ranked[:K])
    if found:
        hits += 1
    recalls.append(len(found) / len(expected))
    by_kind.setdefault(row["kind"], []).append(1 if found else 0)

    best_rank = min(ranked.index(e) + 1 for e in expected)
    reciprocal_ranks.append(1 / best_rank if best_rank <= 10 else 0)

    if not found:
        misses.append((best_rank, row["id"], row["kind"], row["question"]))

n = len(rows)
print(f"Answerable questions: {n}")
print(f"Hit rate@{K}: {hits / n:.2f}")
print(f"Recall@{K}:   {sum(recalls) / n:.2f}")
print(f"MRR@10:       {sum(reciprocal_ranks) / n:.2f}")

print()
print("Hit rate by kind:")
for kind, values in by_kind.items():
    print(f"  {kind}: {sum(values)}/{len(values)}")

print()
print(f"Missed (no expected chunk in top {K}): {len(misses)}")
for best_rank, row_id, kind, question in sorted(misses):
    print(f"  row {row_id} [{kind}] best rank {best_rank}: {question}")