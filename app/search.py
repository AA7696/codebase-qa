from sqlalchemy import func, literal_column, select
from sqlalchemy.orm import Session

from app.models import Chunk, Piece
from app.text import STOPWORDS, tokenize


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


def keyword_search(db: Session, repo_id: int, question: str, limit: int = 50) -> list[int]:
    """Chunk ids of one repo that contain any search word, best ts_rank first."""
    terms = query_terms(question)
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