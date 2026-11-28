import numpy as np
from rank_bm25 import BM25Okapi
from sqlalchemy import select

from app.db import SessionLocal
from app.models import Chunk, Repo, User
from app.search import keyword_search, query_terms

QUESTIONS = [
    "What does Session.mount do?",
    "How does requests limit redirects?",
    "Where are cookies stored in a session?",
    "How does requests support SOCKS proxies?",
]

with SessionLocal() as db:
    repo = db.scalar(
        select(Repo).join(User).where(User.email == "test@example.com", Repo.name == "requests")
    )
    chunks = db.scalars(select(Chunk).where(Chunk.repo_id == repo.id).order_by(Chunk.id)).all()
    names = {c.id: c.name for c in chunks}
    bm25 = BM25Okapi([c.search_tokens.split() for c in chunks])

    for question in QUESTIONS:
        terms = query_terms(question)
        pg_ids = keyword_search(db, repo.id, question, limit=5)
        scores = bm25.get_scores(terms)
        bm_ids = [chunks[i].id for i in np.argsort(-scores)[:5]]
        print()
        print("QUESTION:", question)
        print("  terms:", terms)
        print("  Postgres ts_rank:", [names[i] for i in pg_ids])
        print("  Python BM25:     ", [names[i] for i in bm_ids])
        print("  overlap in top 5:", len(set(pg_ids) & set(bm_ids)))