#!/usr/bin/env python3

"""
Batch PDF extractor/cleaner with Docling.

Given an input directory with PDFs, converts each PDF to a DoclingDocument,
removes selected non-relevant items (pictures, furniture, page headers/footers),
and saves the cleaned document as JSON and optionally Markdown.

Usage examples
--------------
# Remove pictures + furniture + headers + footers (default) and save JSON
python -m scripts.docling_extract_clean \
    --input-dir data/raw \
    --output-dir data/interim/docling/raw

# Also export Markdown
python -m scripts.docling_extract_clean \
    --input-dir data/raw \
    --output-dir data/interim/docling/raw \
    --export-markdown

# Keep furniture, but still remove pictures/headers/footers
python -m scripts.docling_extract_clean \
    --input-dir data/raw \
    --output-dir data/interim/docling/raw \
    --keep-furniture

# Keep page headers and footers
python -m scripts.docling_extract_clean \
    --input-dir data/raw \
    --output-dir data/interim/docling/raw \
    --keep-page-headers \
    --keep-page-footers
"""

from __future__ import annotations

import argparse
import logging
import time
from collections.abc import Iterable
from pathlib import Path

from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TableFormerMode,
    TableStructureOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import ContentLayer, DocItemLabel, DoclingDocument
from tqdm.auto import tqdm

from src.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)

PAGE_BREAK_PLACEHOLDER = "<!-- page_break -->"


def create_pdf_pipeline_options() -> PdfPipelineOptions:
    """Create Docling PDF pipeline options."""
    return PdfPipelineOptions(
        do_ocr=False,
        do_table_structure=True,
        table_structure_options=TableStructureOptions(
            mode=TableFormerMode.ACCURATE,
        ),
    )


def build_converter() -> DocumentConverter:
    """Build a Docling converter configured for PDFs."""
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=create_pdf_pipeline_options(),
                backend=PyPdfiumDocumentBackend,
            )
        }
    )


def _unique_items_by_ref(items: Iterable) -> list:
    """Return unique Docling items preserving order, deduplicated by self_ref."""
    seen: set[str] = set()
    out: list = []
    for item in items:
        ref = getattr(item, "self_ref", None)
        ref_key = str(ref) if ref is not None else id(item)
        if ref_key not in seen:
            seen.add(ref_key)
            out.append(item)
    return out


def collect_items_to_delete(
    doc: DoclingDocument,
    remove_furniture: bool = True,
    remove_page_headers: bool = True,
    remove_page_footers: bool = True,
) -> list:
    """
    Collect items to delete from a DoclingDocument.

    Pictures are always removed. The rest can be toggled with parameters.
    """
    items_to_delete = []

    # 1) Pictures (always remove)
    items_to_delete.extend(list(doc.pictures))

    # 2) Text-based removals
    for item in doc.texts:
        # Furniture layer
        if (
            remove_furniture
            and getattr(item, "content_layer", None) == ContentLayer.FURNITURE
        ):
            items_to_delete.append(item)
            continue

        # Page header / footer labels
        label = getattr(item, "label", None)
        if remove_page_headers and label == DocItemLabel.PAGE_HEADER:
            items_to_delete.append(item)
            continue
        if remove_page_footers and label == DocItemLabel.PAGE_FOOTER:
            items_to_delete.append(item)
            continue

    return _unique_items_by_ref(items_to_delete)


def clean_doc(
    doc: DoclingDocument,
    remove_furniture: bool = True,
    remove_page_headers: bool = True,
    remove_page_footers: bool = True,
) -> DoclingDocument:
    """Delete selected non-relevant items from a DoclingDocument in-place."""
    items_to_delete = collect_items_to_delete(
        doc=doc,
        remove_furniture=remove_furniture,
        remove_page_headers=remove_page_headers,
        remove_page_footers=remove_page_footers,
    )

    if items_to_delete:
        doc.delete_items(node_items=items_to_delete)

    return doc


def export_outputs(
    doc: DoclingDocument,
    out_dir: Path,
    stem: str,
    export_markdown: bool = False,
) -> None:
    """Save cleaned DoclingDocument as JSON and optionally Markdown."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    json_path = out_dir / f"{stem}.json"
    doc.save_as_json(json_path)

    # Optional Markdown
    if export_markdown:
        md_content = doc.export_to_markdown(
            page_break_placeholder=PAGE_BREAK_PLACEHOLDER,
        )
        md_path = out_dir / f"{stem}.md"
        md_path.write_text(md_content, encoding="utf-8")


def process_pdf(
    pdf_path: Path,
    converter: DocumentConverter,
    out_dir: Path,
    remove_furniture: bool = True,
    remove_page_headers: bool = True,
    remove_page_footers: bool = True,
    export_markdown: bool = False,
) -> None:
    """Convert, clean and export a single PDF."""
    result = converter.convert(pdf_path)
    doc = result.document

    clean_doc(
        doc=doc,
        remove_furniture=remove_furniture,
        remove_page_headers=remove_page_headers,
        remove_page_footers=remove_page_footers,
    )

    export_outputs(
        doc=doc,
        out_dir=out_dir,
        stem=pdf_path.stem,
        export_markdown=export_markdown,
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Extract PDFs with Docling and remove pictures/furniture/page headers/footers."
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing input PDFs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where cleaned JSON/Markdown will be saved.",
    )
    parser.add_argument(
        "--export-markdown",
        action="store_true",
        help="Also export cleaned Markdown (.md).",
    )

    # Toggles (default: remove them)
    parser.add_argument(
        "--keep-furniture",
        action="store_true",
        help="Do NOT remove items in ContentLayer.FURNITURE.",
    )
    parser.add_argument(
        "--keep-page-headers",
        action="store_true",
        help="Do NOT remove PAGE_HEADER items.",
    )
    parser.add_argument(
        "--keep-page-footers",
        action="store_true",
        help="Do NOT remove PAGE_FOOTER items.",
    )

    parser.add_argument(
        "--glob",
        default="*.pdf",
        help='Glob pattern to find PDFs (default: "*.pdf").',
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    log_dir = Path("logs")
    setup_logging(log_dir=log_dir, log_name="docling_extract_clean.log")

    logger.info("Starting Docling batch extractor")
    logger.info(
        "Arguments: input_dir=%s | output_dir=%s | glob=%s | export_markdown=%s",
        args.input_dir,
        args.output_dir,
        args.glob,
        args.export_markdown,
    )

    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir

    if not input_dir.exists() or not input_dir.is_dir():
        logger.error(
            "Input directory does not exist or is not a directory: %s", input_dir
        )
        return 1

    pdf_files = sorted(input_dir.glob(args.glob))
    if not pdf_files:
        logger.warning("No PDFs found in %s with pattern %s", input_dir, args.glob)
        return 0

    converter = build_converter()

    total_start = time.perf_counter()
    ok = 0
    failed = 0

    with tqdm(
        pdf_files, total=len(pdf_files), unit="pdf", desc="Converting PDFs"
    ) as pbar:
        for pdf_path in pbar:
            t0 = time.perf_counter()
            try:
                logger.info("Processing PDF: %s", pdf_path.name)

                process_pdf(
                    pdf_path=pdf_path,
                    converter=converter,
                    out_dir=output_dir,
                    remove_furniture=not args.keep_furniture,
                    remove_page_headers=not args.keep_page_headers,
                    remove_page_footers=not args.keep_page_footers,
                    export_markdown=args.export_markdown,
                )

                ok += 1
                elapsed = time.perf_counter() - t0
                logger.info("OK | %s | %.2fs", pdf_path.name, elapsed)

            except Exception:
                failed += 1
                # logger.exception incluye traceback completo automáticamente
                logger.exception("Failed processing %s", pdf_path.name)

    total_elapsed = time.perf_counter() - total_start
    logger.info("Done")
    logger.info("PDFs found: %d", len(pdf_files))
    logger.info("Processed OK: %d", ok)
    logger.info("Failed: %d", failed)
    logger.info("Output dir: %s", output_dir)
    logger.info("Total time: %.2fs", total_elapsed)

    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
