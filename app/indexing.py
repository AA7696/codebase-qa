from sentence_transformers import SentenceTransformer
from sqlalchemy.orm import Session

from app.models import Chunk, Piece, Repo

CODE_BUDGET = 200  # tokens of code per piece (the embedding model reads 256 in total)


def chunk_kind(name: str) -> str:
    if name == "module docstring":
        return "docstring"
    if name.startswith("constants: "):
        return "constants"
    return "code"


def search_text_of(file: str, name: str, code: str) -> str:
    return f"File: {file}\nSymbol: {name}\n{code}"


def split_into_pieces(code: str, tokenizer) -> list[str]:
    """Cut code into pieces on line boundaries, each at most CODE_BUDGET tokens."""
    pieces, current, used = [], [], 0
    for line in code.splitlines():
        size = len(tokenizer.encode(line, add_special_tokens=False)) + 1
        if current and used + size > CODE_BUDGET:
            pieces.append("\n".join(current))
            current, used = [], 0
        current.append(line)
        used += size
    if current:
        pieces.append("\n".join(current))
    return pieces


def add_repo(
    db: Session, user_id: int, repo_name: str, chunks: list[dict], embedder: SentenceTransformer
) -> Repo:
    repo = Repo(user_id=user_id, name=repo_name)
    db.add(repo)
    db.flush()

    rows = [
        Chunk(
            repo_id=repo.id,
            file=c["file"],
            name=c["name"],
            kind=chunk_kind(c["name"]),
            start_line=c["start"],
            end_line=c["end"],
            code=c["code"],
            search_text=search_text_of(c["file"], c["name"], c["code"]),
        )
        for c in chunks
    ]
    db.add_all(rows)
    db.flush()  # gives every chunk its id

    piece_texts, owners = [], []
    for row in rows:
        if row.kind == "constants":
            continue
        for piece in split_into_pieces(row.code, embedder.tokenizer):
            piece_texts.append(f"File: {row.file}\nSymbol: {row.name}\n{piece}")
            owners.append(row.id)

    vectors = embedder.encode(piece_texts, batch_size=32, show_progress_bar=False)
    db.add_all([Piece(chunk_id=chunk_id, embedding=vector) for chunk_id, vector in zip(owners, vectors)])
    db.commit()
    return repo