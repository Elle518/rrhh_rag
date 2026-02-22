import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient

# =========================
# Config
# =========================
QDRANT_COLLECTION = "rrhh_rag_chuncks"

EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4.1-mini"  # puedes cambiarlo por el que uses
EMBEDDING_DIMENSIONS = None  # None = dimensión por defecto

TOP_K = 5

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
    kwargs = {
        "model": EMBEDDING_MODEL,
        "input": query,
    }
    if EMBEDDING_DIMENSIONS is not None:
        kwargs["dimensions"] = EMBEDDING_DIMENSIONS

    resp = openai_client.embeddings.create(**kwargs)
    return resp.data[0].embedding


def search_qdrant(query: str, top_k: int = TOP_K) -> list[dict[str, Any]]:
    query_vector = embed_query(query)

    # Compat con versiones recientes del cliente Qdrant
    resp = qdrant.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_vector,  # <-- antes query_vector=
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
                # exacto para grounding
                "text": payload.get("text", ""),
                # contextualizado (opcional, debug)
                "text_contextualized": payload.get("text_contextualized", ""),
            }
        )
    return hits


def build_context_for_llm(hits: list[dict[str, Any]]) -> str:
    """
    Construye contexto con fragmentos exactos + metadatos.
    Se incluye un ID de cita interno [C1], [C2], ... para luego mapear grounding.
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

    # Grounding estructurado (fragmentos exactos recuperados)
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
                # Fragmento exacto para mostrar al usuario
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
