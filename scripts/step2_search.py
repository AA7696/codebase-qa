import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer, util

chunks = json.loads(Path("data/chunks.json").read_text(encoding="utf-8"))
embeddings = np.load("data/embeddings.npy")
model = SentenceTransformer("all-MiniLM-L6-v2")


def search(question, top_k=5):
    question_vector = model.encode(question)
    scores = util.cos_sim(question_vector, embeddings)[0]
    best = scores.argsort(descending=True)[:top_k]
    results = []
    for index in best:
        results.append((float(scores[index]), chunks[int(index)]))
    return results


questions = [
    "How does requests handle redirects?",
    "How do I add authentication to a request?",
    "Where are cookies stored in a session?",
]

for question in questions:
    print()
    print("QUESTION:", question)
    for score, chunk in search(question):
        print(f"  {score:.2f}  {chunk['file']}  {chunk['name']}  (lines {chunk['start']}-{chunk['end']})")