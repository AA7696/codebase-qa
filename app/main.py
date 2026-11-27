from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import User
from app.rag import ask, refused
from app.security import (
    DUMMY_HASH,
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)

app = FastAPI(title="Codebase Q&A")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")


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


class TokenOut(BaseModel):
    access_token: str
    token_type: str


def get_current_user(
    token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)
) -> User:
    user_id = decode_access_token(token)
    user = db.get(User, user_id) if user_id is not None else None
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


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


@app.post("/login", response_model=TokenOut)
def login(form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.scalar(select(User).where(User.email == form.username.strip().lower()))
    # Hash against a dummy when the user does not exist, so both cases take similar time
    ok = verify_password(form.password, user.password_hash if user else DUMMY_HASH)
    if user is None or not ok:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return TokenOut(access_token=create_access_token(user.id), token_type="bearer")


@app.get("/me", response_model=UserOut)
def me(user: User = Depends(get_current_user)):
    return UserOut(id=user.id, email=user.email)


@app.post("/ask", response_model=AskResponse)
def ask_question(body: AskRequest, user: User = Depends(get_current_user)):
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