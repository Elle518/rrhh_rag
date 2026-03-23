#!/usr/bin/env python3
"""
Evaluate the RAG layer of the HR assistant with RAGAS.

Input:
    CSV file with evaluation questions and references. Expected columns:
        - code: numeric identifier of the question
        - question: user question text
        - answer: reference / gold answer
        - difficulty: difficulty label
        - source: source file identifier
        - cite: exact supporting quote from the convenio

Output:
    - One CSV file with per-question evaluation results, including:
        - generated response
        - retrieved contexts
        - resolved source files used for filtering
        - retrieval diagnostics
        - RAGAS metric scores
    - One JSON file with aggregated summary statistics
    - One Excel file generated from the JSON summary, split into sheets:
        - summary
        - retrieval
        - metrics
        - by_difficulty

Evaluation flow:
    - Resolve each CSV `source` value into one or more backend-compatible
      `source_files`
    - Run the real project RAG pipeline through `answer_with_grounding()`
    - Extract retrieved contexts from grounding
    - Compute the following RAGAS metrics with OpenAI models:
        - Faithfulness
        - Answer relevancy
        - Context precision
        - Context recall
        - Factual correctness
    - When no contexts are retrieved, compute only the metrics that do not
      require `retrieved_contexts` and mark the retrieval as failed

Usage example:
    # Basic usage with defaults
    python -m ag_app.eval_ragas \
        --csv resources/qa.csv \
        --out-csv output/eval/qa_ragas_results.csv \
        --out-json output/eval/qa_ragas_summary.json

    # Evaluate with custom retrieval depth and parallelism
    python -m ag_app.eval_ragas \
        --csv resources/qa.csv \
        --out-csv output/eval/qa_ragas_results.csv \
        --out-json output/eval/qa_ragas_summary.json \
        --top-k 3 \
        --concurrency 8

    # Verbose mode for debugging source resolution and retrieval
    python -m ag_app.eval_ragas \
        --csv resources/qa.csv \
        --out-csv output/eval/qa_ragas_results.csv \
        --out-json output/eval/qa_ragas_summary.json \
        --verbose

Notes:
    - This script evaluates the pure RAG path, not the conversational agent loop
    - It uses the same OpenAI models configured in the project backend
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

import pandas as pd
from openai import AsyncOpenAI
from ragas.embeddings import embedding_factory
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    FactualCorrectness,
    Faithfulness,
)

from ag_app.convenio_catalog import CATALOGO_CONVENIOS
from ag_app.rag_backend import CHAT_MODEL, EMBEDDING_MODEL, answer_with_grounding

CATALOG_BY_ID = {c["id"]: c for c in CATALOGO_CONVENIOS}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluación RAG con RAGAS")
    parser.add_argument(
        "--csv",
        type=str,
        default="resources/qa.csv",
        help="Ruta al CSV con preguntas y respuestas",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default="output/eval/qa_ragas_results.csv",
        help="Ruta del CSV de salida con resultados por fila",
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default="output/eval/qa_ragas_summary.json",
        help="Ruta del JSON de salida con resumen agregado",
    )
    parser.add_argument(
        "--out-xlsx",
        type=str,
        default="output/eval/qa_ragas_summary.xlsx",
        help="Ruta del archivo Excel de salida con resumen agregado",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top-K de recuperación",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Número máximo de filas evaluadas en paralelo",
    )
    parser.add_argument(
        "--strictness",
        type=int,
        default=1,
        help="Strictness para AnswerRelevancy",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Imprime trazas de depuración",
    )
    return parser.parse_args()


def build_metrics(strictness: int) -> dict[str, Any]:
    async_client = AsyncOpenAI()

    evaluator_llm = llm_factory(
        CHAT_MODEL,
        client=async_client,
    )
    evaluator_embeddings = embedding_factory(
        "openai",
        model=EMBEDDING_MODEL,
        client=async_client,
    )

    return {
        "faithfulness": Faithfulness(llm=evaluator_llm),
        "answer_relevancy": AnswerRelevancy(
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
            strictness=strictness,
        ),
        "context_precision": ContextPrecision(llm=evaluator_llm),
        "context_recall": ContextRecall(llm=evaluator_llm),
        "factual_correctness": FactualCorrectness(llm=evaluator_llm),
    }


def normalize_source(source_value: Any) -> list[str]:
    """
    Normaliza la columna `source` del CSV para convertirla en `source_files`
    compatibles con el filtro real del backend.

    Casos soportados:
    - "contactcenter" -> ["contactcenter.json"]
    - "contactcenter.json" -> ["contactcenter.json"]
    - "establecimientosanitarios_madrid" -> source_files del catálogo
    - '["a.json", "b.json"]' -> ["a.json", "b.json"]
    """
    if source_value is None or (
        isinstance(source_value, float) and pd.isna(source_value)
    ):
        return []

    if isinstance(source_value, list):
        raw_values = [str(x).strip() for x in source_value if str(x).strip()]
    else:
        text = str(source_value).strip()
        if not text:
            return []

        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    raw_values = [str(x).strip() for x in parsed if str(x).strip()]
                else:
                    raw_values = [text]
            except json.JSONDecodeError:
                raw_values = [text]
        else:
            raw_values = [text]

    resolved: list[str] = []

    for value in raw_values:
        convenio = CATALOG_BY_ID.get(value)
        if convenio:
            resolved.extend(convenio["source_files"])
            continue

        if Path(value).suffix:
            resolved.append(value)
            continue

        resolved.append(f"{value}.json")

    # deduplicar preservando orden
    return list(dict.fromkeys(resolved))


def build_retrieved_contexts(rag_result: dict[str, Any]) -> list[str]:
    contexts: list[str] = []
    for g in rag_result.get("grounding", []):
        text = (g.get("text") or "").strip()
        if text:
            contexts.append(text)
    return contexts


def cite_found_in_contexts(cite: str, retrieved_contexts: list[str]) -> bool:
    if not isinstance(cite, str):
        return False

    gold = cite.strip()
    if not gold:
        return False

    gold_norm = " ".join(gold.split()).lower()
    for ctx in retrieved_contexts:
        ctx_norm = " ".join(ctx.split()).lower()
        if gold_norm in ctx_norm:
            return True
    return False


def metric_value(result: Any) -> float | None:
    if result is None:
        return None
    value = getattr(result, "value", result)
    if value is None:
        return None
    return float(value)


async def evaluate_row(
    row: pd.Series,
    metrics: dict[str, Any],
    top_k: int,
    semaphore: asyncio.Semaphore,
    verbose: bool = False,
) -> dict[str, Any]:
    async with semaphore:
        code = row.get("code")
        question = str(row.get("question", "")).strip()
        reference = str(row.get("answer", "")).strip()
        difficulty = row.get("difficulty")
        source_raw = row.get("source")
        # source_files = normalize_source(source_raw)
        source_files = [source_raw + ".json"]
        gold_cite = str(row.get("cite", "")).strip()
        print(f"SOURCE_RAW={source_raw} -> SOURCE_FILES={source_files}")

        # if not question:
        #     return {
        #         "code": code,
        #         "question": question,
        #         "answer": reference,
        #         "difficulty": difficulty,
        #         "source": source_raw,
        #         "source_files_resolved": [],
        #         "cite": gold_cite,
        #         "response": "",
        #         "retrieved_contexts": [],
        #         "n_contexts": 0,
        #         "retrieval_failed": True,
        #         "gold_cite_found": False,
        #         "faithfulness": None,
        #         "response_relevance": None,
        #         "context_precision": None,
        #         "context_recall": None,
        #         "correctness": None,
        #         "error": "La fila no tiene question",
        #     }

        try:
            if verbose:
                print(
                    f"[EVAL DEBUG] code={code!r} source_original={source_raw!r} "
                    f"-> source_files={source_files}"
                )

            rag_result = await asyncio.to_thread(
                answer_with_grounding,
                query=question,
                top_k=top_k,
                source_files=source_files or None,
            )

            response = (rag_result.get("answer") or "").strip()
            retrieved_contexts = build_retrieved_contexts(rag_result)
            has_contexts = bool(retrieved_contexts)
            retrieval_failed = not has_contexts
            gold_cite_found = cite_found_in_contexts(gold_cite, retrieved_contexts)

            # Metrics always computable (not requiring retrieved_contexts)
            relevancy_task = metrics["answer_relevancy"].ascore(
                user_input=question,
                response=response,
            )
            correctness_task = metrics["factual_correctness"].ascore(
                response=response,
                reference=reference,
            )

            relevancy, correctness = await asyncio.gather(
                relevancy_task,
                correctness_task,
            )

            # Metrics that require retrieved_contexts
            faithfulness = None
            precision = None
            recall = None

            if has_contexts:
                faithfulness_task = metrics["faithfulness"].ascore(
                    user_input=question,
                    response=response,
                    retrieved_contexts=retrieved_contexts,
                )
                precision_task = metrics["context_precision"].ascore(
                    user_input=question,
                    reference=reference,
                    retrieved_contexts=retrieved_contexts,
                )
                recall_task = metrics["context_recall"].ascore(
                    user_input=question,
                    reference=reference,
                    retrieved_contexts=retrieved_contexts,
                )

                faithfulness, precision, recall = await asyncio.gather(
                    faithfulness_task,
                    precision_task,
                    recall_task,
                )

            return {
                "code": code,
                "question": question,
                "answer": reference,
                "difficulty": difficulty,
                "source": source_raw,
                "source_files_resolved": source_files,
                "cite": gold_cite,
                "response": response,
                "retrieved_contexts": retrieved_contexts,
                "n_contexts": len(retrieved_contexts),
                "retrieval_failed": retrieval_failed,
                "gold_cite_found": gold_cite_found,
                "faithfulness": metric_value(faithfulness),
                "response_relevance": metric_value(relevancy),
                "context_precision": metric_value(precision),
                "context_recall": metric_value(recall),
                "correctness": metric_value(correctness),
                "error": None,
            }

        except Exception as e:
            return {
                "code": code,
                "question": question,
                "answer": reference,
                "difficulty": difficulty,
                "source": source_raw,
                "source_files_resolved": source_files,
                "cite": gold_cite,
                "response": "",
                "retrieved_contexts": [],
                "n_contexts": 0,
                "retrieval_failed": True,
                "gold_cite_found": False,
                "faithfulness": None,
                "response_relevance": None,
                "context_precision": None,
                "context_recall": None,
                "correctness": None,
                "error": repr(e),
            }


def safe_mean(df: pd.DataFrame, col: str) -> float | None:
    if df.empty or col not in df.columns:
        return None
    series = pd.to_numeric(df[col], errors="coerce").dropna()
    if series.empty:
        return None
    return float(series.mean())


def build_summary(
    results_df: pd.DataFrame, top_k: int, concurrency: int
) -> dict[str, Any]:
    ok_df = results_df[results_df["error"].isna()].copy()
    with_context_df = ok_df[ok_df["n_contexts"] > 0].copy()
    without_context_df = ok_df[ok_df["n_contexts"] == 0].copy()

    summary: dict[str, Any] = {
        "n_rows": int(len(results_df)),
        "n_ok": int(len(ok_df)),
        "n_error": int(results_df["error"].notna().sum()),
        "top_k": top_k,
        "concurrency": concurrency,
        "retrieval": {
            "with_context": int(len(with_context_df)),
            "without_context": int(len(without_context_df)),
            "retrieval_failed_rate": (
                float(without_context_df.shape[0] / ok_df.shape[0])
                if len(ok_df)
                else None
            ),
            "gold_cite_found_rate": (
                float(ok_df["gold_cite_found"].mean()) if len(ok_df) else None
            ),
        },
        "metrics_mean_all_ok": {
            "faithfulness": safe_mean(ok_df, "faithfulness"),
            "response_relevance": safe_mean(ok_df, "response_relevance"),
            "context_precision": safe_mean(ok_df, "context_precision"),
            "context_recall": safe_mean(ok_df, "context_recall"),
            "correctness": safe_mean(ok_df, "correctness"),
        },
        "metrics_mean_only_with_context": {
            "faithfulness": safe_mean(with_context_df, "faithfulness"),
            "response_relevance": safe_mean(with_context_df, "response_relevance"),
            "context_precision": safe_mean(with_context_df, "context_precision"),
            "context_recall": safe_mean(with_context_df, "context_recall"),
            "correctness": safe_mean(with_context_df, "correctness"),
        },
        "metrics_mean_only_without_context": {
            "faithfulness": safe_mean(without_context_df, "faithfulness"),
            "response_relevance": safe_mean(without_context_df, "response_relevance"),
            "context_precision": safe_mean(without_context_df, "context_precision"),
            "context_recall": safe_mean(without_context_df, "context_recall"),
            "correctness": safe_mean(without_context_df, "correctness"),
        },
        "by_difficulty": {},
    }

    if not ok_df.empty:
        for difficulty, group in ok_df.groupby("difficulty", dropna=False):
            key = str(difficulty)
            group_with_context = group[group["n_contexts"] > 0]
            group_without_context = group[group["n_contexts"] == 0]

            summary["by_difficulty"][key] = {
                "n": int(len(group)),
                "with_context": int(len(group_with_context)),
                "without_context": int(len(group_without_context)),
                "retrieval_failed_rate": (
                    float(group_without_context.shape[0] / group.shape[0])
                    if len(group)
                    else None
                ),
                "gold_cite_found_rate": (
                    float(group["gold_cite_found"].mean()) if len(group) else None
                ),
                "faithfulness": safe_mean(group, "faithfulness"),
                "response_relevance": safe_mean(group, "response_relevance"),
                "context_precision": safe_mean(group, "context_precision"),
                "context_recall": safe_mean(group, "context_recall"),
                "correctness": safe_mean(group, "correctness"),
            }

    return summary


async def main_async(args: argparse.Namespace) -> None:
    csv_path = Path(args.csv)
    out_csv = Path(args.out_csv)
    out_json = Path(args.out_json)
    out_xlsx = Path(args.out_xlsx)

    if not csv_path.exists():
        raise FileNotFoundError(f"Missing CSV file: {csv_path}")

    df = pd.read_csv(csv_path, sep=";")

    # COMMENT THIS OUT TO EVALUATE THE FULL
    # CSV THIS IS JUST FOR QUICK TESTS
    df = df.iloc[:5]

    required_columns = {"code", "question", "answer", "difficulty", "source", "cite"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {sorted(missing)}")

    metrics = build_metrics(strictness=args.strictness)
    semaphore = asyncio.Semaphore(args.concurrency)

    tasks = [
        evaluate_row(
            row=row,
            metrics=metrics,
            top_k=args.top_k,
            semaphore=semaphore,
            verbose=args.verbose,
        )
        for _, row in df.iterrows()
    ]

    results = await asyncio.gather(*tasks)
    results_df = pd.DataFrame(results)

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # Serializar listas para CSV
    results_df_for_csv = results_df.copy()
    for col in ["retrieved_contexts", "source_files_resolved"]:
        results_df_for_csv[col] = results_df_for_csv[col].apply(
            lambda x: json.dumps(x, ensure_ascii=False)
        )

    results_df_for_csv.to_csv(out_csv, index=False)

    summary = build_summary(
        results_df=results_df,
        top_k=args.top_k,
        concurrency=args.concurrency,
    )

    out_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n=== RESUMEN ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nResultados guardados en: {out_csv}")
    print(f"Resumen guardado en: {out_json}")

    summary_general_df = pd.DataFrame(
        [
            {
                "n_rows": summary.get("n_rows"),
                "n_ok": summary.get("n_ok"),
                "n_error": summary.get("n_error"),
                "top_k": summary.get("top_k"),
                "concurrency": summary.get("concurrency"),
            }
        ]
    )

    retrieval_df = pd.DataFrame([summary.get("retrieval", {})])

    metrics_all_ok_df = pd.DataFrame(
        [
            {
                "scope": "all_ok",
                **summary.get("metrics_mean_all_ok", {}),
            }
        ]
    )

    metrics_with_context_df = pd.DataFrame(
        [
            {
                "scope": "only_with_context",
                **summary.get("metrics_mean_only_with_context", {}),
            }
        ]
    )

    metrics_without_context_df = pd.DataFrame(
        [
            {
                "scope": "only_without_context",
                **summary.get("metrics_mean_only_without_context", {}),
            }
        ]
    )

    metrics_df = pd.concat(
        [
            metrics_all_ok_df,
            metrics_with_context_df,
            metrics_without_context_df,
        ],
        ignore_index=True,
    )

    by_difficulty = summary.get("by_difficulty", {})
    difficulty_rows = []
    for difficulty, values in by_difficulty.items():
        difficulty_rows.append(
            {
                "difficulty": difficulty,
                **values,
            }
        )

    by_difficulty_df = pd.DataFrame(difficulty_rows)

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        summary_general_df.to_excel(writer, sheet_name="summary", index=False)
        retrieval_df.to_excel(writer, sheet_name="retrieval", index=False)
        metrics_df.to_excel(writer, sheet_name="metrics", index=False)
        by_difficulty_df.to_excel(writer, sheet_name="by_difficulty", index=False)

    print(f"Resumen Excel guardado en: {out_xlsx}")


def main() -> None:
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
