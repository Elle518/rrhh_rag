#!/usr/bin/env python3

"""
Query a Qdrant vector database with OpenAI embeddings and return grounded answers.

Input:
    Natural-language user query and retrieval settings from the project configuration.

Output:
    Retrieved vector search hits from Qdrant, an LLM-generated answer grounded on the
    recovered chunks, and structured citation metadata for traceability.

Usage example:
    # Run the example query defined in the script
    python -m scripts.query_vdb
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient

from rrhh_rag import conf
from rrhh_rag.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)

# =========================
# Config
# =========================
try:
    conf_settings = conf.load(file="settings.yaml")
except Exception as e:
    logger.error("Failed to load conf files: %s", e)

QDRANT_COLLECTION = conf_settings.vdb_index
EMBEDDING_MODEL = conf_settings.embeddings
EMBEDDING_DIMENSIONS = conf_settings.embeddings_dim
CHAT_MODEL = conf_settings.llm_workhorse
TOP_K = conf_settings.retrieve_k

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")


# =========================
# Clients
# =========================
openai_client = OpenAI(api_key=OPENAI_API_KEY)

qdrant = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)


# =========================
# Core functions
# =========================
def embed_query(query: str) -> list[float]:
    """Generates an embedding vector for a query string using the
    configured OpenAI embedding model."""
    kwargs = {
        "model": EMBEDDING_MODEL,
        "input": query,
    }
    if EMBEDDING_DIMENSIONS is not None:
        kwargs["dimensions"] = EMBEDDING_DIMENSIONS

    resp = openai_client.embeddings.create(**kwargs)
    return resp.data[0].embedding


def search_qdrant(query: str, top_k: int = TOP_K) -> list[dict[str, Any]]:
    """Embeds the query, searches Qdrant for the top matching
    chunks, and returns their scores and payload metadata."""
    query_vector = embed_query(query)

    resp = qdrant.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_vector,
        limit=top_k,
        with_payload=True,
        with_vectors=False,
    )

    results = resp.points

    hits: list[dict[str, Any]] = []
    for r in results:
        payload = r.payload or {}
        hits.append(
            {
                "score": float(r.score),
                "chunk_id": payload.get("chunk_id"),
                "doc_id": payload.get("doc_id"),
                "source_file": payload.get("source_file"),
                "chunk_index": payload.get("chunk_index"),
                "page_numbers": payload.get("page_numbers", []),
                "doc_item_refs": payload.get("doc_item_refs", []),
                # Exact for grounding
                "text": payload.get("text", ""),
                # Contextualized (optional, debug)
                "text_contextualized": payload.get("text_contextualized", ""),
            }
        )
    return hits


def build_context_for_llm(hits: list[dict[str, Any]]) -> str:
    """
    Build context using exact snippets and metadata. Include an
    internal citation ID [C1], [C2], ... for subsequent grounding
    mapping.
    """
    parts = []
    for i, h in enumerate(hits, start=1):
        pages = h.get("page_numbers") or []
        pages_str = ",".join(str(p) for p in pages) if pages else "N/A"

        parts.append(
            f"[C{i}] "
            f"doc_id={h.get('doc_id')} | source_file={h.get('source_file')} | "
            f"chunk_id={h.get('chunk_id')} | pages={pages_str}\n"
            f"{(h.get('text') or '').strip()}"
        )

    return "\n\n---\n\n".join(parts)


def answer_with_grounding(query: str, top_k: int = TOP_K) -> dict[str, Any]:
    """Retrieves relevant chunks from Qdrant and uses them to generate a grounded answer with structured citations."""
    hits = search_qdrant(query=query, top_k=top_k)

    if not hits:
        return {
            "answer": "No he encontrado fragmentos relevantes en la base vectorial.",
            "grounding": [],
        }

    context = build_context_for_llm(hits)

    system_prompt = (
        "Eres un asistente de QA con RAG.\n"
        "Responde SOLO con información contenida en el contexto proporcionado.\n"
        "Si el contexto no es suficiente, dilo explícitamente.\n"
        "Cuando afirmes algo, cita los fragmentos usando [C1], [C2], etc.\n"
        "No inventes información."
    )

    user_prompt = (
        f"Pregunta del usuario:\n{query}\n\n"
        f"Contexto recuperado:\n{context}\n\n"
        "Devuelve una respuesta clara en español y cita las fuentes [C#]."
    )

    chat_resp = openai_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )

    answer = chat_resp.choices[0].message.content or ""

    # Structured grounding (exact fragments recovered)
    grounding = []
    for i, h in enumerate(hits, start=1):
        grounding.append(
            {
                "citation_id": f"C{i}",
                "score": h["score"],
                "doc_id": h.get("doc_id"),
                "source_file": h.get("source_file"),
                "chunk_id": h.get("chunk_id"),
                "chunk_index": h.get("chunk_index"),
                "page_numbers": h.get("page_numbers", []),
                "doc_item_refs": h.get("doc_item_refs", []),
                # Exact fragment for user display
                "text": h.get("text", ""),
            }
        )

    return {
        "answer": answer,
        "grounding": grounding,
    }


# =========================
# Example usage
# =========================
if __name__ == "__main__":

    log_dir = Path("logs")
    setup_logging(log_dir=log_dir, log_name="query_vdb.log")

    query = "¿Cuántos días me corresponden por fallecimiento de familiar?"
    result = answer_with_grounding(query, top_k=5)

    print("\n=== RESPUESTA ===")
    print(result["answer"])

    print("\n=== GROUNDING ===")
    for g in result["grounding"]:
        print(
            f"\n[{g['citation_id']}] score={g['score']:.4f} | doc={g['doc_id']} | pages={g['page_numbers']}"
        )
        print(g["text"][:500], "..." if len(g["text"]) > 500 else "")
