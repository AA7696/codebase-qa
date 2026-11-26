import ast
import json
import os
import re
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer, util

K = 5
RRF_K = 60
RERANK_DEPTH = 10
CODE_BUDGET = 200
MAX_CONST_LINES = 20
MAX_EXTRA = 3

load_dotenv()
client = OpenAI()
llm_model = os.environ["OPENAI_MODEL"]

all_chunks = json.loads(Path("data/chunks_v2.json").read_text(encoding="utf-8"))
const_chunks = [c for c in all_chunks if c["name"].startswith("constants: ")]
chunks = [c for c in all_chunks if not c["name"].startswith("constants: ")]

embedder = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)

SYSTEM_PROMPT = (
    "You answer questions about the Python 'requests' library. "
    "Use ONLY the numbered code chunks you are given. "
    "After each claim, cite the chunk number like [1]. "
    "If the chunks do not contain the answer, say exactly: "
    "\"I don't know based on the retrieved code.\" Do not guess."
)


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


texts = [f"File: {c['file']}\nSymbol: {c['name']}\n{c['code']}" for c in chunks]
bm25 = BM25Okapi([tokenize(t) for t in texts])

piece_texts, owners = [], []
for i, c in enumerate(chunks):
    for piece in split_into_pieces(c["code"]):
        piece_texts.append(f"File: {c['file']}\nSymbol: {c['name']}\n{piece}")
        owners.append(i)
piece_vectors = embedder.encode(piece_texts, batch_size=32, show_progress_bar=False)

# Map each constant name (like HOOKS) to the chunk that defines it
const_of = {}
for c in const_chunks:
    if c["end"] - c["start"] + 1 > MAX_CONST_LINES:
        continue
    for node in ast.parse(c["code"]).body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        target = node.targets[0] if isinstance(node, ast.Assign) else node.target
        if isinstance(target, ast.Name):
            const_of.setdefault(target.id, c)


def search(question):
    q_vector = embedder.encode(question)
    piece_scores = util.cos_sim(q_vector, piece_vectors)[0].numpy()
    scores = np.full(len(chunks), -1.0)
    for piece_index, chunk_index in enumerate(owners):
        scores[chunk_index] = max(scores[chunk_index], piece_scores[piece_index])
    v_order = [int(i) for i in np.argsort(-scores)]

    k_scores = bm25.get_scores(tokenize(question))
    k_order = [int(i) for i in np.argsort(-k_scores)]

    fused = {i: 0.0 for i in range(len(chunks))}
    for order in (v_order, k_order):
        for rank, i in enumerate(order, start=1):
            fused[i] += 1 / (RRF_K + rank)
    h_order = sorted(fused, key=lambda i: -fused[i])

    pool = h_order[:RERANK_DEPTH]
    pool_scores = reranker.predict([(question, texts[i]) for i in pool])
    score_of = dict(zip(pool, pool_scores))
    return sorted(pool, key=lambda i: -score_of[i])[:K]


def expand(top):
    """Constants that the retrieved chunks mention, as extra context."""
    extras = []
    for index in top:
        for word in dict.fromkeys(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", chunks[index]["code"])):
            c = const_of.get(word)
            if c is not None and c not in extras:
                extras.append(c)
    return extras[:MAX_EXTRA]


def build_prompt(question, items):
    parts = []
    for number, c in enumerate(items, start=1):
        header = f"[{number}] {c['file']} lines {c['start']}-{c['end']} ({c['name']})"
        parts.append(header + "\n" + c["code"])
    return "Code chunks:\n\n" + "\n\n".join(parts) + f"\n\nQuestion: {question}"


questions = [
    "Which hook events exist by default?",
    "What does the hooks module do?",
    "Which status codes count as redirects?",
    "What is the default redirect limit?",
]

for question in questions:
    top = search(question)
    extras = expand(top)
    items = [chunks[i] for i in top] + extras
    response = client.chat.completions.create(
        model=llm_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(question, items)},
        ],
    )
    print()
    print("QUESTION:", question)
    print("RETRIEVED:", ", ".join(chunks[i]["name"] for i in top))
    print("EXTRA CONSTANTS:", ", ".join(f"{c['file']}:{c['start']}-{c['end']}" for c in extras) or "none")
    print("ANSWER:", response.choices[0].message.content)