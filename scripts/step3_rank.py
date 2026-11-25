import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer, util

chunks = json.loads(Path("data/chunks.json").read_text(encoding="utf-8"))
embeddings = np.load("data/embeddings.npy")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

question = "What does Session.mount do?"
target = "Session.mount"

scores = util.cos_sim(embedder.encode(question), embeddings)[0]
order = scores.argsort(descending=True)

print("Top 10 for:", question)
for rank, index in enumerate(order[:10], start=1):
    chunk = chunks[int(index)]
    marker = "  <-- the right one" if chunk["name"] == target else ""
    print(f"{rank:>3}  {float(scores[index]):.2f}  {chunk['name']}{marker}")

for rank, index in enumerate(order, start=1):
    if chunks[int(index)]["name"] == target:
        print()
        print(f"{target} is ranked {rank} out of {len(chunks)}")

for chunk in chunks:
    if chunk["name"] == target:
        print()
        print("Text that was embedded for this chunk:")
        print(chunk["code"])