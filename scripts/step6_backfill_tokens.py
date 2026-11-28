from sqlalchemy import select

from app.db import SessionLocal
from app.models import Chunk
from app.text import tokenize

with SessionLocal() as db:
    chunks = db.scalars(select(Chunk)).all()
    for chunk in chunks:
        chunk.search_tokens = " ".join(tokenize(chunk.search_text))
    db.commit()
    print("Updated", len(chunks), "chunks")