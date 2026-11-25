import json
import os
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


questions = [
    "How does requests handle redirects?",
    "Where are cookies stored in a session?",
    "How does requests use GraphQL?",
]

for question in questions:
    text, found, tokens = answer(question)
    print()
    print("QUESTION:", question)
    print("ANSWER:", text)
    print("SOURCES:")
    for number, chunk in enumerate(found, start=1):
        print(f"  [{number}] {chunk['file']}:{chunk['start']}-{chunk['end']}  {chunk['name']}")
    print("Tokens used:", tokens)