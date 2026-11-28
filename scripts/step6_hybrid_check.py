from sentence_transformers import SentenceTransformer
from sqlalchemy import select

from app.db import SessionLocal
from app.models import Chunk, Repo, User
from app.search import hybrid_search, keyword_search, query_terms, term_doc_count, vector_search

QUESTIONS = [
    "What does Session.mount do?",
    "How does requests limit redirects?",
    "Where are cookies stored in a session?",
    "How does requests support SOCKS proxies?",
]

embedder = SentenceTransformer("all-MiniLM-L6-v2")

with SessionLocal() as db:
    repo = db.scalar(
        select(Repo).join(User).where(User.email == "test@example.com", Repo.name == "requests")
    )
    names = {c.id: c.name for c in db.scalars(select(Chunk).where(Chunk.repo_id == repo.id))}

    for question in QUESTIONS:
        q = embedder.encode(question)
        terms = query_terms(question)
        print()
        print("QUESTION:", question)
        counts = {t: term_doc_count(db, repo.id, t) for t in terms}
        print(f"  chunks containing each word (of {len(names)}):", counts)

        def show(label, ids):
            print(f"  {label:<22}", [names[i] for i in ids[:5]])

        show("vector", vector_search(db, repo.id, q, limit=5))
        show("keyword", keyword_search(db, repo.id, question, limit=5))
        show("keyword, max_df 0.5", keyword_search(db, repo.id, question, limit=5, max_df=0.5))
        show("hybrid", hybrid_search(db, repo.id, question, q))
        show("hybrid, max_df 0.5", hybrid_search(db, repo.id, question, q, max_df=0.5))