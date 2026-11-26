import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer, util

K = 5
RRF_K = 60
RERANK_DEPTH = 10
CODE_BUDGET = 200

chunks_file = sys.argv[1] if len(sys.argv) > 1 else "data/chunks_v2.json"
all_chunks = json.loads(Path(chunks_file).read_text(encoding="utf-8"))

embedder = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)

embed_cache = {}
rerank_cache = {}


def embed_many(texts):
    missing = [t for t in dict.fromkeys(texts) if t not in embed_cache]
    if missing:
        vectors = embedder.encode(missing, batch_size=32, show_progress_bar=False)
        for text, vector in zip(missing, vectors):
            embed_cache[text] = vector
    return np.array([embed_cache[t] for t in texts])


def rerank_scores(question, texts):
    missing = [t for t in texts if (question, t) not in rerank_cache]
    if missing:
        scores = reranker.predict([(question, t) for t in missing])
        for text, score in zip(missing, scores):
            rerank_cache[(question, text)] = float(score)
    return [rerank_cache[(question, t)] for t in texts]


def count_tokens(text):
    return len(embedder.tokenizer.encode(text, add_special_tokens=False))


def split_into_pieces(code):
    pieces, current, used = [], [], 0
    for line in code.splitlines():
        size = count_tokens(line) + 1
        if current and used + size > CODE_BUDGET:
            pieces.append("\n".join(current))
            current, used = [], 0
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


def build_index(chunks):
    texts = [f"File: {c['file']}\nSymbol: {c['name']}\n{c['code']}" for c in chunks]
    bm25 = BM25Okapi([tokenize(t) for t in texts])
    piece_texts, owners = [], []
    for i, c in enumerate(chunks):
        for piece in split_into_pieces(c["code"]):
            piece_texts.append(f"File: {c['file']}\nSymbol: {c['name']}\n{piece}")
            owners.append(i)
    return {
        "chunks": chunks,
        "texts": texts,
        "bm25": bm25,
        "piece_vectors": embed_many(piece_texts),
        "owners": owners,
    }


def search(index, question, q_vector):
    chunks = index["chunks"]
    piece_scores = util.cos_sim(q_vector, index["piece_vectors"])[0].numpy()
    scores = np.full(len(chunks), -1.0)
    for piece_index, chunk_index in enumerate(index["owners"]):
        scores[chunk_index] = max(scores[chunk_index], piece_scores[piece_index])
    v_order = [int(i) for i in np.argsort(-scores)]

    k_scores = index["bm25"].get_scores(tokenize(question))
    k_order = [int(i) for i in np.argsort(-k_scores)]

    fused = {i: 0.0 for i in range(len(chunks))}
    for order in (v_order, k_order):
        for rank, i in enumerate(order, start=1):
            fused[i] += 1 / (RRF_K + rank)
    h_order = sorted(fused, key=lambda i: -fused[i])

    pool = h_order[:RERANK_DEPTH]
    pool_scores = rerank_scores(question, [index["texts"][i] for i in pool])
    score_of = dict(zip(pool, pool_scores))
    return sorted(pool, key=lambda i: -score_of[i]) + h_order[RERANK_DEPTH:]


def evaluate(index, rows, q_vectors):
    chunks = index["chunks"]
    hits = set()
    recalls, reciprocal_ranks = [], []
    for row, q_vector in zip(rows, q_vectors):
        order = search(index, row["question"], q_vector)
        expected = set(zip(row["expected_path"].split(";"), row["expected_symbol"].split(";")))
        ranked = [(chunks[i]["file"], chunks[i]["name"]) for i in order]
        found = expected & set(ranked[:K])
        if found:
            hits.add(row["id"])
        recalls.append(len(found) / len(expected))
        best_rank = min(ranked.index(e) + 1 for e in expected)
        reciprocal_ranks.append(1 / best_rank if best_rank <= 10 else 0)
    n = len(rows)
    return len(hits) / n, sum(recalls) / n, sum(reciprocal_ranks) / n, hits


def is_docstring(c):
    return c["name"] == "module docstring"


def is_constants(c):
    return c["name"].startswith("constants: ")


variants = {
    "core only (control)": lambda c: not is_docstring(c) and not is_constants(c),
    "+ docstring chunks": lambda c: not is_constants(c),
    "+ constants chunks": lambda c: not is_docstring(c),
    "+ both": lambda c: True,
}

probes = [
    ("Which hook events exist by default?", "src/requests/hooks.py", "constants: HOOKS"),
    ("What does the hooks module do?", "src/requests/hooks.py", "module docstring"),
    ("Which status codes count as redirects?", "src/requests/models.py", "constants: REDIRECT_STATI"),
    ("What is the default redirect limit?", "src/requests/models.py", "constants: REDIRECT_STATI"),
]

with Path("data/golden.csv").open(encoding="utf-8-sig", newline="") as f:
    rows = [r for r in csv.DictReader(f) if r["kind"] != "unanswerable"]
question_of = {r["id"]: r["question"] for r in rows}
q_vectors = embedder.encode([r["question"] for r in rows])
probe_vectors = embedder.encode([p[0] for p in probes])

results = {}
probe_ranks = {}
for name, keep in variants.items():
    index = build_index([c for c in all_chunks if keep(c)])
    hit, recall, mrr, hits = evaluate(index, rows, q_vectors)
    results[name] = (hit, recall, mrr, hits)
    print(f"{name:<22} chunks {len(index['chunks']):>3}   hit@{K} {hit:.2f}   recall@{K} {recall:.2f}   MRR@10 {mrr:.2f}")

    ranks = []
    for (question, file, name_start), p_vector in zip(probes, probe_vectors):
        target = [i for i, c in enumerate(index["chunks"]) if c["file"] == file and c["name"].startswith(name_start)]
        if not target:
            ranks.append("-")
            continue
        order = search(index, question, p_vector)
        ranks.append(str(order.index(target[0]) + 1))
    probe_ranks[name] = ranks

print()
print("Probe questions: rank of the chunk that answers them ('-' = chunk not in this index)")
print(f"{'':<22} " + "  ".join(f"P{i + 1}" for i in range(len(probes))))
for name, ranks in probe_ranks.items():
    print(f"{name:<22} " + "  ".join(f"{r:>2}" for r in ranks))
for i, probe in enumerate(probes):
    print(f"  P{i + 1}: {probe[0]}  -> {probe[2]}")

core_hits = results["core only (control)"][3]
for name, (hit, recall, mrr, hits) in results.items():
    if name == "core only (control)":
        continue
    print()
    print(f"{name} versus control:")
    for row_id in sorted(core_hits - hits, key=int):
        print(f"  NEWLY MISSED row {row_id}: {question_of[row_id]}")
    for row_id in sorted(hits - core_hits, key=int):
        print(f"  newly hit    row {row_id}: {question_of[row_id]}")
    if hits == core_hits:
        print("  no change in which questions are hit")