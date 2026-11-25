import csv
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer, util

K = 5
LIMIT = 256

chunks = json.loads(Path("data/chunks.json").read_text(encoding="utf-8"))
embeddings = np.load("data/embeddings.npy")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

token_counts = {
    (c["file"], c["name"]): len(embedder.tokenizer.encode(c["code"], add_special_tokens=False))
    for c in chunks
}

with Path("data/golden.csv").open(encoding="utf-8-sig", newline="") as f:
    rows = [r for r in csv.DictReader(f) if r["kind"] != "unanswerable"]

question_vectors = embedder.encode([r["question"] for r in rows])
all_scores = util.cos_sim(question_vectors, embeddings)

hit_group = []
miss_group = []

for row, scores in zip(rows, all_scores):
    expected = list(zip(row["expected_path"].split(";"), row["expected_symbol"].split(";")))
    order = scores.argsort(descending=True)
    ranked = [(chunks[int(i)]["file"], chunks[int(i)]["name"]) for i in order]
    ranks = {e: ranked.index(e) + 1 for e in expected}
    best = min(ranks, key=ranks.get)
    record = (row["id"], ranks[best], token_counts[best], best[1])
    if ranks[best] <= K:
        hit_group.append(record)
    else:
        miss_group.append(record)


def share_long(group):
    long_count = sum(1 for r in group if r[2] > LIMIT)
    return long_count, len(group)


print(f"Model reads at most {LIMIT} tokens per chunk.")
long_hits, n_hits = share_long(hit_group)
long_misses, n_misses = share_long(miss_group)
print(f"Hit questions whose best chunk is longer than that:    {long_hits} of {n_hits}")
print(f"Missed questions whose best chunk is longer than that: {long_misses} of {n_misses}")

print()
print("Missed questions (row, rank, tokens in the expected chunk, chunk):")
for row_id, rank, tokens, name in sorted(miss_group, key=lambda r: r[1]):
    flag = "  LONG" if tokens > LIMIT else ""
    print(f"  row {row_id:>2}  rank {rank:>2}  {tokens:>4} tokens  {name}{flag}")