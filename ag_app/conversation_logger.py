from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

LOG_DIR = Path(os.getenv("CHAT_LOG_DIR", "logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)

TURN_LOG_PATH = LOG_DIR / "chat_turns.jsonl"
RAG_LOG_PATH = LOG_DIR / "rag_events.jsonl"

_lock = Lock()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    record = {"timestamp": _utc_now(), **payload}
    with _lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def log_chat_turn(payload: dict[str, Any]) -> None:
    append_jsonl(TURN_LOG_PATH, payload)


def log_rag_event(payload: dict[str, Any]) -> None:
    append_jsonl(RAG_LOG_PATH, payload)
