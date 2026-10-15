"""Web API:  uvicorn api:app --reload   then open http://127.0.0.1:8000/docs"""
from fastapi import FastAPI
from pydantic import BaseModel

from rag.pipeline import RAG

app = FastAPI(title="RAG Lab")
rag = RAG()


class Question(BaseModel):
    question: str


@app.post("/ask")
def ask(body: Question):
    ans = rag.ask(body.question)
    return {"answer": ans.text, "sources": ans.sources}
