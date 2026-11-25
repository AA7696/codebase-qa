import csv
import json
import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer, util

MAX_PER_GROUP = 12
K = 5

load_dotenv()
client = OpenAI()
llm_model = os.environ["OPENAI_MODEL"]

chunks = json.loads(Path("data/chunks.json").read_text(encoding="utf-8"))
embeddings = np.load("data/embeddings.npy")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

SYSTEM_PROMPT = (
    "You answer questions about the Python 'requests' library. "
    "Use ONLY the numbered code chunks you are given. "
    "After each claim, cite the chunk number like [1]. "
    "If the chunks do not contain the answer, say exactly: "
    "\"I don't know based on the retrieved code.\" Do not guess."
)


def build_prompt(question, found):
    parts = []
    for number, chunk in enumerate(found, start=1):
        header = f"[{number}] {chunk['file']} lines {chunk['start']}-{chunk['end']} ({chunk['name']})"
        parts.append(header + "\n" + chunk["code"])
    context = "\n\n".join(parts)
    return f"Code chunks:\n\n{context}\n\nQuestion: {question}"


def ask(question, top):
    response = client.chat.completions.create(
        model=llm_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(question, top)},
        ],
    )
    return response.choices[0].message.content


def refused(text):
    return "i don't know based on the retrieved code" in text.lower().replace("\u2019", "'")



with Path("data/golden.csv").open(encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

question_vectors = embedder.encode([r["question"] for r in rows])
all_scores = util.cos_sim(question_vectors, embeddings)

unanswerable, retrieved, not_retrieved = [], [], []
for row, scores in zip(rows, all_scores):
    order = scores.argsort(descending=True)
    top = [chunks[int(i)] for i in order[:K]]
    if row["kind"] == "unanswerable":
        unanswerable.append((row, top))
    else:
        expected = set(zip(row["expected_path"].split(";"), row["expected_symbol"].split(";")))
        got = {(c["file"], c["name"]) for c in top}
        (retrieved if expected & got else not_retrieved).append((row, top))

groups = {
    "unanswerable": unanswerable[:MAX_PER_GROUP],
    "answerable, chunk retrieved": retrieved[:MAX_PER_GROUP],
    "answerable, chunk NOT retrieved": not_retrieved[:MAX_PER_GROUP],
}

results = []
for group_name, items in groups.items():
    for row, top in items:
        text = ask(row["question"], top)
        results.append(
            {
                "id": row["id"],
                "group": group_name,
                "question": row["question"],
                "refused": refused(text),
                "answer": text,
            }
        )
        print(".", end="", flush=True)
print()

print()
for group_name in groups:
    group = [r for r in results if r["group"] == group_name]
    count = sum(r["refused"] for r in group)
    print(f"{group_name}: refused {count} of {len(group)}")

print()
print("Look at these:")
for r in results:
    if r["group"] == "unanswerable" and not r["refused"]:
        print(f"  ANSWERED an unanswerable question (row {r['id']}): {r['question']}")
        print("      " + r["answer"][:300].replace("\n", " "))
    if r["group"] == "answerable, chunk retrieved" and r["refused"]:
        print(f"  REFUSED although the chunk was retrieved (row {r['id']}): {r['question']}")

Path("data/refusal_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
print()
print("Saved all answers to data/refusal_results.json")