from sqlalchemy import func, literal_column, select
from sqlalchemy.orm import Session

from app.models import Chunk, Piece
from app.text import STOPWORDS, tokenize

RRF_K = 60


def _matches(term: str):
    query = func.to_tsquery(literal_column("'simple'::regconfig"), term)
    return Chunk.search_vector.op("@@")(query)


def term_doc_count(db: Session, repo_id: int, term: str) -> int:
    """How many chunks of the repo contain this word."""
    return db.scalar(
        select(func.count()).select_from(Chunk).where(Chunk.repo_id == repo_id, _matches(term))
    )


def vector_search(db: Session, repo_id: int, query_vector, limit: int = 50) -> list[int]:
    """Chunk ids of one repo, best first. A chunk scores as its best-matching piece."""
    distance = Piece.embedding.cosine_distance([float(x) for x in query_vector])
    best = func.min(distance).label("dist")
    stmt = (
        select(Piece.chunk_id, best)
        .join(Chunk, Chunk.id == Piece.chunk_id)
        .where(Chunk.repo_id == repo_id)
        .group_by(Piece.chunk_id)
        .order_by(best)
        .limit(limit)
    )
    return [chunk_id for chunk_id, _ in db.execute(stmt)]

def query_terms(question: str) -> list[str]:
    """Search words from a question: lowercase, no stopwords, no duplicates."""
    terms = []
    for token in tokenize(question):
        if token not in STOPWORDS and token not in terms:
            terms.append(token)
    return terms


def keyword_search(
    db: Session, repo_id: int, question: str, limit: int = 50, max_df: float | None = None
) -> list[int]:
    """Chunk ids of one repo that contain any search word, best ts_rank first.
    With max_df, words found in more than that share of the chunks are ignored."""
    terms = query_terms(question)
    if max_df is not None and terms:
        total = db.scalar(select(func.count()).select_from(Chunk).where(Chunk.repo_id == repo_id))
        terms = [t for t in terms if term_doc_count(db, repo_id, t) <= max_df * total]
    if not terms:
        return []
    query = func.to_tsquery(literal_column("'simple'::regconfig"), " | ".join(terms))
    rank = func.ts_rank(Chunk.search_vector, query).label("rank")
    stmt = (
        select(Chunk.id, rank)
        .where(Chunk.repo_id == repo_id, Chunk.search_vector.op("@@")(query))
        .order_by(rank.desc(), Chunk.id)
        .limit(limit)
    )
    return [chunk_id for chunk_id, _ in db.execute(stmt)]


def hybrid_search(
    db: Session, repo_id: int, question: str, query_vector, limit: int = 50, max_df: float | None = None
) -> list[int]:
    """Vector and keyword results merged with reciprocal rank fusion."""
    lists = [
        vector_search(db, repo_id, query_vector, limit=100),
        keyword_search(db, repo_id, question, limit=100, max_df=max_df),
    ]
    fused: dict[int, float] = {}
    for ids in lists:
        for rank, chunk_id in enumerate(ids, start=1):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1 / (RRF_K + rank)
    return sorted(fused, key=lambda c: -fused[c])[:limit]