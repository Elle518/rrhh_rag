import json
import os
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

# =========================
# Config
# =========================
JSONL_PATH = "data/processed/chunks/all_chunks.jsonl"
COLLECTION_NAME = "rrhh_rag_chuncks"

OPENAI_MODEL = "text-embedding-3-small"
# Si quieres reducir dimensión (opcional), pon por ejemplo 1024 o 512.
# Si lo dejas en None, usa la dimensión por defecto del modelo (1536 para 3-small).
OPENAI_DIMENSIONS = None

# Tamaño de lote para llamadas a OpenAI / Qdrant
BATCH_SIZE = 10

# Variables de entorno requeridas:
#   OPENAI_API_KEY
#   QDRANT_URL
#   QDRANT_API_KEY


# =========================
# Helpers
# =========================
def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] Línea {line_num} inválida, se omite: {e}")


def batched(items: list[dict[str, Any]], size: int) -> Iterator[list[dict[str, Any]]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def deterministic_point_id(chunk_id: str) -> str:
    # ID estable para re-upserts idempotentes
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def get_vector_size(model: str, dimensions: int | None) -> int:
    if dimensions is not None:
        return dimensions
    # Default OpenAI dims:
    # text-embedding-3-small -> 1536
    # text-embedding-3-large -> 3072
    if model == "text-embedding-3-small":
        return 1536
    if model == "text-embedding-3-large":
        return 3072
    raise ValueError(
        f"No sé la dimensión por defecto de {model}; especifica OPENAI_DIMENSIONS."
    )


# =========================
# Main
# =========================
def main():
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    QDRANT_URL = os.getenv("QDRANT_URL")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

    openai_client = OpenAI(api_key=openai_api_key)

    # Qdrant Cloud / online
    qdrant_client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
    )

    print(qdrant_client.get_collections())

    vector_size = get_vector_size(OPENAI_MODEL, OPENAI_DIMENSIONS)

    # Crear colección si no existe
    if not qdrant_client.collection_exists(COLLECTION_NAME):
        print("Entra aquí")
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )
        print(f"[OK] Colección creada: {COLLECTION_NAME} (size={vector_size})")
    else:
        print(f"[OK] Colección ya existe: {COLLECTION_NAME}")

    # Cargar todos los registros (si prefieres streaming puro, se puede adaptar)
    rows = list(iter_jsonl(JSONL_PATH))
    if not rows:
        print("[WARN] No hay registros en el JSONL")
        return

    total = 0

    for batch in batched(rows, BATCH_SIZE):
        # 1) Textos para embeddings (usa contextualizado)
        texts = []
        valid_rows = []
        for row in batch:
            txt = (row.get("text_contextualized") or "").strip()
            if not txt:
                print(
                    f"[WARN] chunk_id {row.get('chunk_id')} tiene texto contextualizado vacío, se omite."
                )
                # Evita inputs vacíos (OpenAI embeddings no acepta string vacío)
                continue
            texts.append(txt)
            valid_rows.append(row)

        if not texts:
            continue

        # 2) Embeddings OpenAI (batch)
        emb_kwargs = {
            "model": OPENAI_MODEL,
            "input": texts,
        }
        if OPENAI_DIMENSIONS is not None:
            emb_kwargs["dimensions"] = OPENAI_DIMENSIONS

        emb_resp = openai_client.embeddings.create(**emb_kwargs)

        # 3) Construir points para Qdrant
        points: list[PointStruct] = []
        for row, emb_item in zip(valid_rows, emb_resp.data):
            chunk_id = row["chunk_id"]
            point_id = deterministic_point_id(chunk_id)

            payload = {
                # clave para grounding / mostrar cita exacta
                "text": row.get("text", ""),
                # útil para debug / trazabilidad
                "text_contextualized": row.get("text_contextualized", ""),
                "doc_id": row.get("doc_id"),
                "source_file": row.get("source_file"),
                "chunk_id": chunk_id,
                "chunk_index": row.get("chunk_index"),
                "page_numbers": row.get("page_numbers", []),
                "doc_item_refs": row.get("doc_item_refs", []),
                "token_count_text": row.get("token_count_text"),
                "token_count_contextualized": row.get("token_count_contextualized"),
            }

            # Si en tu JSONL existe chunk_meta, lo conservas; si no, no pasa nada
            if "chunk_meta" in row:
                payload["chunk_meta"] = row["chunk_meta"]

            points.append(
                PointStruct(
                    id=point_id,
                    vector=emb_item.embedding,
                    payload=payload,
                )
            )

        # 4) Upsert en Qdrant
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
        )

        total += len(points)
        print(f"[OK] Upsert batch: {len(points)} | Total: {total}")

    print(f"\n✅ Ingesta completada. Total puntos subidos: {total}")


if __name__ == "__main__":
    main()
