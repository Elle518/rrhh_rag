from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.rag_backend import answer_with_grounding

app = FastAPI(title="RAG Chat API", version="1.0.0")


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


class GroundingItem(BaseModel):
    citation_id: str
    score: float
    doc_id: str | None = None
    source_file: str | None = None
    chunk_id: str | None = None
    chunk_index: int | None = None
    page_numbers: list[int] = []
    doc_item_refs: list[str] = []
    text: str


class ChatResponse(BaseModel):
    answer: str
    grounding: list[GroundingItem]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        result = answer_with_grounding(req.query, top_k=req.top_k)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
