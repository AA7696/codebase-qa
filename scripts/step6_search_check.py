import numpy as np
from sentence_transformers import SentenceTransformer
from sqlalchemy import select

from app.db import SessionLocal
from app.models import Chunk, Piece, Repo, User
from app.search import vector_search

QUESTIONS = [
    "What does Session.mount do?",
    "How does requests handle redirects?",
    "Where are cookies stored in a session?",
]

embedder = SentenceTransformer("all-MiniLM-L6-v2")

with SessionLocal() as db:
    repo = db.scalar(
        select(Repo).join(User).where(User.email == "test@example.com", Repo.name == "requests")
    )
    names = {c.id: c.name for c in db.scalars(select(Chunk).where(Chunk.repo_id == repo.id))}

    rows = db.execute(
        select(Piece.chunk_id, Piece.embedding)
        .join(Chunk, Chunk.id == Piece.chunk_id)
        .where(Chunk.repo_id == repo.id)
    ).all()
    chunk_ids = [int(r[0]) for r in rows]
    matrix = np.vstack([np.asarray(r[1], dtype=np.float32) for r in rows])
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)

    for question in QUESTIONS:
        q = embedder.encode(question)
        sql_ids = vector_search(db, repo.id, q, limit=5)

        scores = matrix @ (q / np.linalg.norm(q))
        best = {}
        for cid, score in zip(chunk_ids, scores):
            best[cid] = max(best.get(cid, -1.0), float(score))
        numpy_ids = sorted(best, key=lambda c: -best[c])[:5]

        print()
        print("QUESTION:", question)
        for rank, cid in enumerate(sql_ids, start=1):
            print(f"  {rank}. {names[cid]}")
        print("Same top 5 as the numpy check:", sql_ids == numpy_ids)

    print()
    print("Results for a repo id that does not exist:", vector_search(db, -1, q, limit=5))