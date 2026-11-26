import csv
import json
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer, util

K = 5
RRF_K = 60
RERANK_DEPTH = 10
CODE_BUDGET = 200  # tokens of code per piece (the model reads 256 in total)

chunks = json.loads(Path("data/chunks.json").read_text(encoding="utf-8"))
old_embeddings = np.load("data/embeddings_context.npy")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)


def count_tokens(text):
    return len(embedder.tokenizer.encode(text, add_special_tokens=False))


def split_into_pieces(code):
    """Cut code into pieces on line boundaries, each at most CODE_BUDGET tokens."""
    pieces = []
    current = []
    used = 0
    for line in code.splitlines():
        size = count_tokens(line) + 1
        if current and used + size > CODE_BUDGET:
            pieces.append("\n".join(current))
            current = []
            used = 0
        current.append(line)
        used += size
    if current:
        pieces.append("\n".join(current))
    return pieces


def tokenize(text):
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

piece_texts = []
piece_owner = []
for index, c in enumerate(chunks):
    for piece in split_into_pieces(c["code"]):
        piece_texts.append(f"File: {c['file']}\nSymbol: {c['name']}\n{piece}")
        piece_owner.append(index)
print("Chunks:", len(chunks), "  Pieces:", len(piece_texts))
piece_embeddings = embedder.encode(piece_texts, batch_size=32, show_progress_bar=False)

with Path("data/golden.csv").open(encoding="utf-8-sig", newline="") as f:
    rows = [r for r in csv.DictReader(f) if r["kind"] != "unanswerable"]

question_vectors = embedder.encode([r["question"] for r in rows])
old_scores = util.cos_sim(question_vectors, old_embeddings).numpy()
piece_scores = util.cos_sim(question_vectors, piece_embeddings).numpy()

new_scores = np.full(old_scores.shape, -1.0)
for piece_index, chunk_index in enumerate(piece_owner):
    new_scores[:, chunk_index] = np.maximum(new_scores[:, chunk_index], piece_scores[:, piece_index])


def fuse(v_order, k_order):
    fused = {i: 0.0 for i in range(len(chunks))}
    for order in (v_order, k_order):
        for rank, index in enumerate(order, start=1):
            fused[index] += 1 / (RRF_K + rank)
    return sorted(fused, key=lambda i: -fused[i])


names = [
    "vector, 5.1 embeddings",
    "hybrid, 5.1 embeddings",
    "vector, pieces (best piece)",
    "hybrid, pieces",
    f"hybrid, pieces + rerank {RERANK_DEPTH}",
]
rankings = {name: [] for name in names}

for row, old_row, new_row in zip(rows, old_scores, new_scores):
    k_scores = bm25.get_scores(tokenize(row["question"]))
    k_order = [int(i) for i in np.argsort(-k_scores)]
    old_order = [int(i) for i in np.argsort(-old_row)]
    new_order = [int(i) for i in np.argsort(-new_row)]

    hybrid_old = fuse(old_order, k_order)
    hybrid_new = fuse(new_order, k_order)

    pool = hybrid_new[:RERANK_DEPTH]
    pair_scores = reranker.predict([(row["question"], texts[i]) for i in pool])
    score_of = dict(zip(pool, pair_scores))
    reranked = sorted(pool, key=lambda i: -score_of[i]) + hybrid_new[RERANK_DEPTH:]

    rankings[names[0]].append(old_order)
    rankings[names[1]].append(hybrid_old)
    rankings[names[2]].append(new_order)
    rankings[names[3]].append(hybrid_new)
    rankings[names[4]].append(reranked)
    print(".", end="", flush=True)
print()


def evaluate(order_list):
    hits = 0
    recalls = []
    reciprocal_ranks = []
    missed = []
    for row, order in zip(rows, order_list):
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
    return hits / n, sum(recalls) / n, sum(reciprocal_ranks) / n, missed


final_missed = []
for name in names:
    hit, recall, mrr, missed = evaluate(rankings[name])
    print(f"{name:<32} hit@{K} {hit:.2f}   recall@{K} {recall:.2f}   MRR@10 {mrr:.2f}")
    final_missed = missed

print()
print(f"Final pipeline still misses {len(final_missed)}:")
for best_rank, row_id, kind, question in sorted(final_missed):
    print(f"  row {row_id} [{kind}] best rank {best_rank}: {question}")