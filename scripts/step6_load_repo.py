import json
import sys
from pathlib import Path

from sentence_transformers import SentenceTransformer
from sqlalchemy import func, select

from app.db import SessionLocal
from app.indexing import add_repo
from app.models import Chunk, Piece, Repo, User

EMAIL = sys.argv[1] if len(sys.argv) > 1 else "test@example.com"
REPO_NAME = "requests"

chunks = json.loads(Path("data/chunks_v2.json").read_text(encoding="utf-8"))
embedder = SentenceTransformer("all-MiniLM-L6-v2")

with SessionLocal() as db:
    user = db.scalar(select(User).where(User.email == EMAIL))
    if user is None:
        raise SystemExit(f"No user with email {EMAIL}. Sign up first.")
    existing = db.scalar(select(Repo).where(Repo.user_id == user.id, Repo.name == REPO_NAME))
    if existing:
        raise SystemExit(f"{EMAIL} already has a repo named {REPO_NAME} (id {existing.id}).")

    repo = add_repo(db, user.id, REPO_NAME, chunks, embedder)
    print("Created repo", repo.id, "for", EMAIL)

    n_chunks = db.scalar(select(func.count()).select_from(Chunk).where(Chunk.repo_id == repo.id))
    n_pieces = db.scalar(
        select(func.count())
        .select_from(Piece)
        .join(Chunk, Piece.chunk_id == Chunk.id)
        .where(Chunk.repo_id == repo.id)
    )
    print("Chunks:", n_chunks, " Pieces:", n_pieces)
    for kind, count in db.execute(
        select(Chunk.kind, func.count()).where(Chunk.repo_id == repo.id).group_by(Chunk.kind)
    ):
        print(" ", kind, count)