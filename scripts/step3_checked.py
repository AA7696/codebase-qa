import json
import os
import re
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer, util

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


def search(question, top_k=5):
    question_vector = embedder.encode(question)
    scores = util.cos_sim(question_vector, embeddings)[0]
    best = scores.argsort(descending=True)[:top_k]
    return [chunks[int(i)] for i in best]


def build_prompt(question, found):
    parts = []
    for number, chunk in enumerate(found, start=1):
        header = f"[{number}] {chunk['file']} lines {chunk['start']}-{chunk['end']} ({chunk['name']})"
        parts.append(header + "\n" + chunk["code"])
    context = "\n\n".join(parts)
    return f"Code chunks:\n\n{context}\n\nQuestion: {question}"


def answer(question):
    found = search(question)
    response = client.chat.completions.create(
        model=llm_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(question, found)},
        ],
    )
    return response.choices[0].message.content, found, response.usage.total_tokens


def check_citations(text, found):
    """Find [n] citations in the answer. Split them into real and non-existent."""
    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text)})
    valid = [n for n in cited if 1 <= n <= len(found)]
    invalid = [n for n in cited if n not in valid]
    return valid, invalid


questions = [
    "How does requests handle redirects?",
    "Where are cookies stored in a session?",
    "How does requests use GraphQL?",
    "What does Session.mount do?",
]

for question in questions:
    text, found, tokens = answer(question)
    valid, invalid = check_citations(text, found)

    print()
    print("QUESTION:", question)
    print("ANSWER:", text)
    print("RETRIEVED:", ", ".join(chunk["name"] for chunk in found))
    print("CITED:")
    for n in valid:
        chunk = found[n - 1]
        print(f"  [{n}] {chunk['file']}:{chunk['start']}-{chunk['end']}  {chunk['name']}")
    if invalid:
        print("WARNING: answer cites chunks that do not exist:", invalid)
    if not valid and "I don't know" not in text:
        print("WARNING: answer has no citations")
    print("Tokens used:", tokens)