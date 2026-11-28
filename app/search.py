from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import Chunk, Piece


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