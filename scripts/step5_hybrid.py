import csv
import json
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util

K = 5
RRF_K = 60

chunks = json.loads(Path("data/chunks.json").read_text(encoding="utf-8"))
embeddings = np.load("data/embeddings_context.npy")
embedder = SentenceTransformer("all-MiniLM-L6-v2")


def tokenize(text):
    """Split code text into lowercase words. Also splits get_adapter and SessionMixin into parts."""
    tokens = []
    for word in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text):
        tokens.append(word.lower())
        spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", word).replace("_", " ")
        parts = spaced.lower().split()
        if len(parts) > 1:
            tokens.extend(parts)
    return tokens


texts = [f"File: {c['file']}\nSymbol: {c['name']}\n{c['code']}" for c in chunks]
bm25 = BM25Okapi([tokenize(t) for t in texts])

with Path("data/golden.csv").open(encoding="utf-8-sig", newline="") as f:
    rows = [r for r in csv.DictReader(f) if r["kind"] != "unanswerable"]

question_vectors = embedder.encode([r["question"] for r in rows])
vector_scores = util.cos_sim(question_vectors, embeddings)

vector_rankings = []
keyword_rankings = []
hybrid_rankings = []

for row, v_scores in zip(rows, vector_scores):
    v_order = [int(i) for i in v_scores.argsort(descending=True)]
    k_scores = bm25.get_scores(tokenize(row["question"]))
    k_order = [int(i) for i in np.argsort(-k_scores)]

    fused = {i: 0.0 for i in range(len(chunks))}
    for order in (v_order, k_order):
        for rank, index in enumerate(order, start=1):
            fused[index] += 1 / (RRF_K + rank)
    h_order = sorted(fused, key=lambda i: -fused[i])

    vector_rankings.append(v_order)
    keyword_rankings.append(k_order)
    hybrid_rankings.append(h_order)


def evaluate(name, rankings):
    hits = 0
    recalls = []
    reciprocal_ranks = []
    missed = []
    for row, order in zip(rows, rankings):
        expected = set(zip(row["expected_path"].split(";"), row["expected_symbol"].split(";")))
        ranked = [(chunks[i]["file"], chunks[i]["name"]) for i in order]
        found = expected & set(ranked[:K])
        if found:
            hits += 1
        recalls.append(len(found) / len(expected))
        best_rank = min(ranked.index(e) + 1 for e in expected)
        reciprocal_ranks.append(1 / best_rank if best_rank <= 10 else 0)
        if not found:
            missed.append((best_rank, row["id"], row["kind"], row["question"]))
    n = len(rows)
    print(
        f"{name:<14} hit@{K} {hits / n:.2f}   recall@{K} {sum(recalls) / n:.2f}"
        f"   MRR@10 {sum(reciprocal_ranks) / n:.2f}"
    )
    return missed


evaluate("vector only", vector_rankings)
evaluate("keyword only", keyword_rankings)
missed = evaluate("hybrid (RRF)", hybrid_rankings)

print()
print(f"Hybrid missed {len(missed)} questions:")
for best_rank, row_id, kind, question in sorted(missed):
    print(f"  row {row_id} [{kind}] best rank {best_rank}: {question}")