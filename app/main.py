from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.rag import ask, refused

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


@app.get("/health")
def health():
    return {"status": "ok"}


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