"""Minimal HTTP API.  Run:  uvicorn api:app --port 8000
POST /ask {"question": "What is operant conditioning?"}
"""
from fastapi import FastAPI
from pydantic import BaseModel

from rag import pipeline

app = FastAPI(title="Psychology 2e RAG")


class AskRequest(BaseModel):
    question: str
    use_hyde: bool = True


@app.post("/ask")
def ask(req: AskRequest):
    res = pipeline.ask(req.question, use_hyde=req.use_hyde)
    return {
        "answer": res.answer,
        "references": res.references,
        "passages": [{"section": h.section, "page": h.page, "score": round(h.score, 3),
                      "text": h.text} for h in res.hits],
    }


@app.get("/health")
def health():
    return {"status": "ok"}
