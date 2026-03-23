"""This module implements the core RAG (Retrieval-Augmented Generation) logic for the application. It provides functions to:
- Embed user queries using OpenAI embeddings.
- Search for relevant document chunks in Qdrant based on the query embedding and optional filters.
- Build a context string from the retrieved chunks to feed into the LLM.
- Generate an answer from the LLM based on the retrieved context, ensuring that the answer is grounded in the retrieved information and cites the sources properly.
"""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    PayloadSchemaType,
)

QDRANT_COLLECTION = "rrhh_rag_chuncks"
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4.1-mini"
EMBEDDING_DIMENSIONS = None
TOP_K = 5

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

openai_client = OpenAI(api_key=OPENAI_API_KEY)
qdrant = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)


def debug_sample_hits(query: str, top_k: int = 20):
    """Utility function to debug the raw hits returned by Qdrant for a given query."""
    query_vector = embed_query(query)

    resp = qdrant.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_vector,
        limit=top_k,
        with_payload=True,
        with_vectors=False,
    )

    results = resp.points
    print("\n=== DEBUG SAMPLE HITS ===")
    for i, r in enumerate(results, start=1):
        payload = r.payload or {}
        print(
            i,
            {
                "score": float(r.score),
                "source_file": payload.get("source_file"),
                "doc_id": payload.get("doc_id"),
                "chunk_id": payload.get("chunk_id"),
            },
        )
    print("=== END DEBUG SAMPLE HITS ===\n")


def ensure_qdrant_indexes():
    """Ensure that the necessary indexes exist in Qdrant for efficient querying."""
    qdrant.create_payload_index(
        collection_name=QDRANT_COLLECTION,
        field_name="source_file",
        field_schema=PayloadSchemaType.KEYWORD,
    )


def embed_query(query: str) -> list[float]:
    """Embed the user query using OpenAI embeddings."""
    kwargs = {"model": EMBEDDING_MODEL, "input": query}
    if EMBEDDING_DIMENSIONS is not None:
        kwargs["dimensions"] = EMBEDDING_DIMENSIONS
    resp = openai_client.embeddings.create(**kwargs)
    return resp.data[0].embedding


def build_qdrant_filter(
    source_files: list[str] | None = None,
    doc_id: str | None = None,
) -> Filter | None:
    """Build a Qdrant filter based on the provided criteria."""
    must_conditions = []

    if source_files:
        if len(source_files) == 1:
            must_conditions.append(
                FieldCondition(
                    key="source_file",
                    match=MatchValue(value=source_files[0]),
                )
            )
        else:
            must_conditions.append(
                FieldCondition(
                    key="source_file",
                    match=MatchAny(any=source_files),
                )
            )

    if doc_id:
        must_conditions.append(
            FieldCondition(
                key="doc_id",
                match=MatchValue(value=doc_id),
            )
        )

    if not must_conditions:
        return None

    return Filter(must=must_conditions)


def search_qdrant(
    query: str,
    top_k: int = TOP_K,
    source_files: list[str] | None = None,
    doc_id: str | None = None,
) -> list[dict[str, Any]]:
    """Search for relevant document chunks in Qdrant based on the query embedding and optional filters."""
    query_vector = embed_query(query)
    query_filter = build_qdrant_filter(
        source_files=source_files,
        doc_id=doc_id,
    )

    resp = qdrant.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_vector,
        query_filter=query_filter,
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
                "text": payload.get("text", ""),
                "text_contextualized": payload.get("text_contextualized", ""),
            }
        )

    return hits


def build_context_for_llm(hits: list[dict[str, Any]]) -> str:
    """Build a context string from the retrieved chunks to feed into the LLM."""
    parts = []
    for i, h in enumerate(hits, start=1):
        pages = h.get("page_numbers") or []
        pages_str = ",".join(str(p) for p in pages) if pages else "N/A"
        text = (h.get("text_contextualized") or h.get("text") or "").strip()

        parts.append(
            f"[C{i}] doc_id={h.get('doc_id')} | source_file={h.get('source_file')} | "
            f"chunk_id={h.get('chunk_id')} | pages={pages_str}\n{text}"
        )
    return "\n\n---\n\n".join(parts)


def answer_with_grounding(
    query: str,
    top_k: int = TOP_K,
    source_files: list[str] | None = None,
) -> dict[str, Any]:
    """Generate an answer from the LLM based on the retrieved context, ensuring that the answer is grounded in the retrieved information and cites the sources properly."""
    debug_sample_hits(query)

    hits = search_qdrant(
        query=query,
        top_k=top_k,
        source_files=source_files,
    )

    if not hits and source_files:
        print(f"[DEBUG] No hits con filtro source_files={source_files}")

    if not hits:
        return {
            "answer": (
                "No he encontrado fragmentos relevantes del convenio seleccionado. "
                "Prueba a reformular la consulta."
            ),
            "grounding": [],
        }

    context = build_context_for_llm(hits)

    system_prompt = (
        "Eres un asistente especializado en convenios laborales en España.\n"
        "Responde SOLO con la información del contexto recuperado.\n"
        "El contexto pertenece al convenio ya seleccionado por el usuario.\n"
        "Si la información no es suficiente, dilo claramente.\n"
        "Si aparece una prórroga o actualización más específica, priorízala.\n"
        "Cita con [C1], [C2], etc.\n"
        "No inventes ni extrapoles."
    )

    user_prompt = (
        f"Pregunta del usuario: {query}\n\n"
        f"Contexto recuperado:\n{context}\n\n"
        "Responde en español y cita [C#]."
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
                "text": h.get("text", ""),
            }
        )

    return {"answer": answer, "grounding": grounding}
