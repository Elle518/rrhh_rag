#!/usr/bin/env python3

"""
Batch ingestion of contextualized chunks into Qdrant.

Reads a JSONL file with chunk records, generates embeddings with OpenAI,
creates the target Qdrant collection if it does not exist, and upserts
the resulting vectors with their payloads.

Usage examples
--------------
# Ingest chunks using default paths/config
python -m scripts.ingest_chunks_to_qdrant
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

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

JSONL_PATH = conf_settings.jsonl_path
COLLECTION_NAME = conf_settings.vdb_index

OPENAI_MODEL = conf_settings.embeddings
# To reduce the size (optional) enter, for example, 1024 or 512.
# If left as None, it uses model's default size (1536 for 3-small).
OPENAI_DIMENSIONS = conf_settings.embeddings_dim

# Batch size for OpenAI/Qdrant calls
BATCH_SIZE = conf_settings.batch_size


# =========================
# Helpers
# =========================
def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Iterates through a JSONL file, yielding each valid
    non-empty line as a parsed JSON object and skipping invalid
    lines with a warning."""
    with Path(path).open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning("Invalid JSONL line %d skipped: %s", line_num, e)


def batched(items: list[dict[str, Any]], size: int) -> Iterator[list[dict[str, Any]]]:
    """Splits a list of dictionaries into successive batches of
    the given size and yields each batch."""
    for i in range(0, len(items), size):
        yield items[i : i + size]


def deterministic_point_id(chunk_id: str) -> str:
    """Generates a stable, deterministic UUID from a chunk ID so
    the same input always produces the same point ID."""

    # Stable ID for idempotent re-upserts
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def get_vector_size(model: str, dimensions: int | None) -> int:
    """Returns the embedding vector size, using the provided
    dimensions when available or falling back to the model’s
    default size."""
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
        f"Default dimension of {model} it's not defined; specify OPENAI_DIMENSIONS."
    )


# =========================
# Main
# =========================
def main():
    log_dir = Path("logs")
    setup_logging(log_dir=log_dir, log_name="ingest_chunks_to_qdrant.log")

    logger.info("Starting Qdrant chunk ingestion")
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    QDRANT_URL = os.getenv("QDRANT_URL")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

    openai_client = OpenAI(api_key=OPENAI_API_KEY)

    # Qdrant Cloud / online
    qdrant_client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
    )

    logger.info("Available Qdrant collections: %s", qdrant_client.get_collections())

    vector_size = get_vector_size(OPENAI_MODEL, OPENAI_DIMENSIONS)

    # Create collection if it doesn't exist
    if not qdrant_client.collection_exists(COLLECTION_NAME):
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )
        logger.info("Collection created: %s (size=%d)", COLLECTION_NAME, vector_size)
    else:
        logger.info("Collection already exists: %s", COLLECTION_NAME)

    rows = list(iter_jsonl(JSONL_PATH))
    if not rows:
        logger.warning("No records found in JSONL: %s", JSONL_PATH)
        return

    total = 0

    for batch in batched(rows, BATCH_SIZE):
        # 1) Texts for embeddings (use contextualized)
        texts = []
        valid_rows = []
        for row in batch:
            txt = (row.get("text_contextualized") or "").strip()
            if not txt:
                # Avoid empty inputs (OpenAI embeddings
                # does not accept empty strings)
                logger.warning(
                    "chunk_id %s has empty contextualized text and will be skipped.",
                    row.get("chunk_id"),
                )
                continue
            texts.append(txt)
            valid_rows.append(row)

        if not texts:
            continue

        # 2) OpenAI Embeddings (batch)
        emb_kwargs = {
            "model": OPENAI_MODEL,
            "input": texts,
        }
        if OPENAI_DIMENSIONS is not None:
            emb_kwargs["dimensions"] = OPENAI_DIMENSIONS

        emb_resp = openai_client.embeddings.create(**emb_kwargs)

        # 3) Build points for Qdrant
        points: list[PointStruct] = []
        for row, emb_item in zip(valid_rows, emb_resp.data):
            chunk_id = row["chunk_id"]
            point_id = deterministic_point_id(chunk_id)

            payload = {
                # Key for grounding / display exact quote
                "text": row.get("text", ""),
                # Useful for debugging/traceability
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

            # If JSONL contains `chunk_meta`, keep it in the payload for Qdrant (optional metadata)
            if "chunk_meta" in row:
                payload["chunk_meta"] = row["chunk_meta"]

            points.append(
                PointStruct(
                    id=point_id,
                    vector=emb_item.embedding,
                    payload=payload,
                )
            )

        # 4) Upsert on Qdrant
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
        )

        total += len(points)
        logger.info("Upsert batch OK | batch_size=%d | total=%d", len(points), total)

    logger.info("Ingestion completed | total_points=%d", total)


if __name__ == "__main__":
    main()
