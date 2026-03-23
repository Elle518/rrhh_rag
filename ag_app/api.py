"""API for the RAG agreement agent. Provides endpoints for interacting with the agent, including:
- `/health`: Check the service status.
- `/chat`: Send a message to the agent and receive a response with the grounding and UI status.
- `/reset/{session_id}`: Restart a specific session.
"""

from __future__ import annotations

import traceback
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ag_app.agent import run_agent_turn
from ag_app.rag_backend import ensure_qdrant_indexes


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_qdrant_indexes()
    yield


app = FastAPI(title="RAG Convenios Agent API", version="2.0.0", lifespan=lifespan)

# Simple memory in process; in production, switch to Redis/Postgres or another persistent store
SESSION_STORE: dict[str, dict[str, Any]] = {}


class Message(BaseModel):
    """Represents a single message in the conversation."""

    role: str
    content: str


class ChatRequest(BaseModel):
    """Request model for the /chat endpoint."""

    session_id: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


class GroundingItem(BaseModel):
    """Represents a single grounding item in the response."""

    citation_id: str
    score: float
    doc_id: str | None = None
    source_file: str | None = None
    chunk_id: str | None = None
    chunk_index: int | None = None
    page_numbers: list[int] = []
    doc_item_refs: list[str] = []
    text: str


class AgentUIState(BaseModel):
    """Represents the UI state of the agent."""

    convenio_id: str | None = None
    convenio_label: str | None = None
    pending_sector: str | None = None
    awaiting_field: str | None = None
    options: list[str] = []
    candidate_ids: list[str] = []
    finished: bool = False


class ChatResponse(BaseModel):
    """Response model for the /chat endpoint."""

    answer: str
    grounding: list[GroundingItem]
    state: AgentUIState


@app.get("/health")
def health():
    """Health check endpoint to verify the service is running."""
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """Endpoint to send a message to the agent and receive a response with the grounding and UI status."""
    try:
        session = SESSION_STORE.setdefault(
            req.session_id,
            {"messages": [], "state": {}},
        )

        session["messages"].append({"role": "user", "content": req.message})

        result = run_agent_turn(
            messages=session["messages"],
            top_k=req.top_k,
            current_state=session.get("state", {}),
        )

        session["messages"].append({"role": "assistant", "content": result["answer"]})
        session["state"] = result["state"]

        return result

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reset/{session_id}")
def reset_session(session_id: str):
    """Endpoint to restart a specific session, clearing its message history and state."""
    SESSION_STORE.pop(session_id, None)
    return {"status": "reset", "session_id": session_id}
