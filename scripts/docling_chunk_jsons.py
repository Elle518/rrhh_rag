#!/usr/bin/env python3
"""
Chunk DoclingDocument JSON files with HybridChunker + contextualization.

Input:
    Directory containing DoclingDocument JSON files.

Output:
    One JSONL file per input document (and optionally a merged JSONL), with:
        - exact chunk text (for grounding / exact quotes)
        - contextualized chunk text (for embeddings)
        - source metadata (doc item refs, pages, etc. when available)

Usage example:
    # Basic usage with defaults and merged output
    python -m scripts.docling_chunk_jsons \
        --input-dir data/interim/docling \
        --output-dir data/processed/chunks \
        --write-merged

    # Modified arguments
    python -m scripts.docling_chunk_jsons \
        --input-dir data/interim/docling \
        --output-dir data/processed/chunks \
        --pattern "*_final.json" \
        --max-tokens 512 \

    # Disable raw chunk metadata
    python -m scripts.docling_chunk_jsons \
        --input-dir data/interim/docling \
        --output-dir data/processed/chunks \
        --no-chunk-meta
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import tiktoken
from docling.datamodel.document import DoclingDocument
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer

from src.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chunk DoclingDocument JSONs with HybridChunker + contextualization."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing DoclingDocument JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where chunk JSONL files will be written.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.json",
        help="Glob pattern for input JSON files (default: *.json).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4.1",
        help="OpenAI model name for tokenizer selection (default: gpt-4.1).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Max tokens per chunk for the OpenAI tokenizer wrapper (default: 2048).",
    )
    parser.add_argument(
        "--write-merged",
        action="store_true",
        help="Also write a merged all_chunks.jsonl file with chunks from all documents.",
    )
    parser.add_argument(
        "--no-chunk-meta",
        action="store_true",
        help=(
            "Do not include raw chunk_meta in the output JSONL. "
            "By default chunk_meta is included."
        ),
    )
    return parser.parse_args()


def _safe_model_dump(obj: Any) -> Any:
    """Return a JSON-serializable representation of an object when possible."""
    if obj is None:
        return None

    # Pydantic v2
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump()
        except Exception:
            pass

    # Pydantic v1
    if hasattr(obj, "dict"):
        try:
            return obj.dict()
        except Exception:
            pass

    # Common simple types
    if isinstance(obj, (str, int, float, bool)):
        return obj

    if isinstance(obj, (list, tuple)):
        return [_safe_model_dump(x) for x in obj]

    if isinstance(obj, dict):
        return {str(k): _safe_model_dump(v) for k, v in obj.items()}

    # Fallback
    return str(obj)


def _extract_doc_item_refs(chunk: Any) -> list[str]:
    """Extract source doc item refs from chunk metadata when available."""
    refs: list[str] = []
    meta = getattr(chunk, "meta", None)
    doc_items = getattr(meta, "doc_items", None)

    if not doc_items:
        return refs

    for item in doc_items:
        ref = getattr(item, "self_ref", None)
        if ref is not None:
            refs.append(str(ref))

    # de-duplicate, preserve order
    seen = set()
    out: list[str] = []
    for r in refs:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def _extract_pages_from_chunk(chunk: Any) -> list[int]:
    """Extract page numbers from chunk metadata (if provenance exists)."""
    pages: list[int] = []
    meta = getattr(chunk, "meta", None)
    doc_items = getattr(meta, "doc_items", None)

    if not doc_items:
        return pages

    for item in doc_items:
        prov_list = getattr(item, "prov", None) or []
        for prov in prov_list:
            page_no = getattr(prov, "page_no", None)
            if isinstance(page_no, int):
                pages.append(page_no)

    # de-duplicate + sort
    return sorted(set(pages))


def _serialize_chunk(
    *,
    chunk: Any,
    chunk_idx: int,
    doc_id: str,
    file_name: str,
    chunker: HybridChunker,
    tokenizer: OpenAITokenizer,
    include_chunk_meta: bool = True,
) -> dict[str, Any]:
    """Build one JSONL record for a Docling chunk."""
    exact_text = getattr(chunk, "text", "") or ""
    contextualized_text = chunker.contextualize(chunk)

    # Token counts are useful later for filtering / batching embeddings
    exact_tokens = tokenizer.count_tokens(exact_text)
    contextualized_tokens = tokenizer.count_tokens(contextualized_text)

    record: dict[str, Any] = {
        "chunk_id": f"{doc_id}::chunk_{chunk_idx:05d}",
        "doc_id": doc_id,
        "source_file": file_name,
        "chunk_index": chunk_idx,
        "text": exact_text,  # exact fragment for grounding / citation
        "text_contextualized": contextualized_text,  # ideal for embeddings
        "token_count_text": exact_tokens,
        "token_count_contextualized": contextualized_tokens,
        "page_numbers": _extract_pages_from_chunk(chunk),
        "doc_item_refs": _extract_doc_item_refs(chunk),
    }

    if include_chunk_meta:
        # Keep raw chunk metadata (best effort) for traceability / debugging
        record["chunk_meta"] = _safe_model_dump(getattr(chunk, "meta", None))

    return record


def chunk_docling_json(
    json_path: Path,
    output_dir: Path,
    chunker: HybridChunker,
    tokenizer: OpenAITokenizer,
    include_chunk_meta: bool = True,
) -> tuple[Path, int]:
    """Load one DoclingDocument JSON, chunk it, contextualize chunks, and write JSONL."""
    doc = DoclingDocument.load_from_json(json_path)
    chunks = list(chunker.chunk(doc))

    doc_id = json_path.stem
    out_path = output_dir / f"{doc_id}.chunks.jsonl"

    with out_path.open("w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks):
            rec = _serialize_chunk(
                chunk=chunk,
                chunk_idx=i,
                doc_id=doc_id,
                file_name=json_path.name,
                chunker=chunker,
                tokenizer=tokenizer,
                include_chunk_meta=include_chunk_meta,
            )
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return out_path, len(chunks)


def iter_json_paths(input_dir: Path, pattern: str) -> Iterable[Path]:
    """Yield JSON files sorted by name."""
    return sorted(p for p in input_dir.glob(pattern) if p.is_file())


def main() -> int:
    args = parse_args()

    log_dir = Path("logs")
    setup_logging(log_dir=log_dir, log_name="docling_chunking.log")

    logger.info("Starting Docling chunking")
    logger.info(
        "Arguments: input_dir=%s | output_dir=%s | pattern=%s | model=%s | "
        "max_tokens=%s | include_chunk_meta=%s | write_merged=%s",
        args.input_dir,
        args.output_dir,
        args.pattern,
        args.model,
        args.max_tokens,
        not args.no_chunk_meta,
        args.write_merged,
    )

    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir
    pattern: str = args.pattern
    model_name: str = args.model
    max_tokens: int = args.max_tokens
    include_chunk_meta: bool = not args.no_chunk_meta

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = list(iter_json_paths(input_dir, pattern))
    if not json_files:
        logger.error("No files found in %s with pattern '%s'", input_dir, pattern)
        return 1

    # OpenAI tokenizer via tiktoken
    encoding = tiktoken.encoding_for_model(model_name)
    tokenizer = OpenAITokenizer(
        tokenizer=encoding,
        max_tokens=max_tokens,
    )

    chunker = HybridChunker(
        tokenizer=tokenizer,
    )

    merged_path = output_dir / "all_chunks.jsonl"
    merged_fh = None
    if args.write_merged:
        merged_fh = merged_path.open("w", encoding="utf-8")

    total_chunks = 0
    total_docs = 0

    try:
        for json_path in json_files:
            try:
                out_path, n_chunks = chunk_docling_json(
                    json_path=json_path,
                    output_dir=output_dir,
                    chunker=chunker,
                    tokenizer=tokenizer,
                    include_chunk_meta=include_chunk_meta,
                )
                total_docs += 1
                total_chunks += n_chunks
                logger.info(
                    "[OK] %s -> %s (%s chunks)", json_path.name, out_path.name, n_chunks
                )

                if merged_fh is not None:
                    with out_path.open("r", encoding="utf-8") as src_f:
                        for line in src_f:
                            merged_fh.write(line)

            except Exception as e:
                logger.exception("Error processing %s: %s", json_path.name, e)

    finally:
        if merged_fh is not None:
            merged_fh.close()

    logger.info("Processed documents: %s", total_docs)
    logger.info("Total chunks: %s", total_chunks)
    if args.write_merged:
        logger.info("Merged JSONL: %s", merged_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
