from fastapi import Depends, FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import User
from app.rag import ask, refused
from app.security import hash_password

app = FastAPI(title="Codebase Q&A")


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


class Source(BaseModel):
    file: str
    name: str
    start: int
    end: int


class AskResponse(BaseModel):
    answer: str
    refused: bool
    sources: list[Source]


class SignupRequest(BaseModel):
    email: str = Field(min_length=3, max_length=255, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    password: str = Field(min_length=8, max_length=128)


class UserOut(BaseModel):
    id: int
    email: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/signup", response_model=UserOut, status_code=status.HTTP_201_CREATED)
def signup(body: SignupRequest, db: Session = Depends(get_db)):
    email = body.email.strip().lower()
    if db.scalar(select(User).where(User.email == email)):
        raise HTTPException(status_code=409, detail="Email already registered")
    user = User(email=email, password_hash=hash_password(body.password))
    db.add(user)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=409, detail="Email already registered")
    db.refresh(user)
    return UserOut(id=user.id, email=user.email)


@app.post("/ask", response_model=AskResponse)
def ask_question(body: AskRequest):
    result = ask(body.question)
    sources = [
        Source(file=c["file"], name=c["name"], start=c["start"], end=c["end"])
        for c in result["retrieved"]
    ]
    return AskResponse(
        answer=result["answer"],
        refused=refused(result["answer"]),
        sources=sources,
    )