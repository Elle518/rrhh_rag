from __future__ import annotations

import logging
import sys
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path


class ProjectOnlyFilter(logging.Filter):
    def __init__(self, allowed_prefixes: tuple[str, ...]):
        super().__init__()
        self.allowed_prefixes = allowed_prefixes

    def filter(self, record: logging.LogRecord) -> bool:
        return record.name.startswith(self.allowed_prefixes)


def setup_logging(
    log_dir: Path,
    log_name: str = "app.log",
    level: int = logging.INFO,
) -> None:
    """
    Configure root logging with console + rotating file handlers.

    - Console output
    - Daily log rotation at midnight
    - Keeps 15 backup files (approx. 15 days if script runs daily)

    Args:
        log_dir: Directory where log files will be stored.
        log_name: Base log filename.
        level: Logging level (e.g. logging.INFO).
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / log_name

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Avoid duplicate handlers if setup_logging() is called more than once
    root_logger.handlers.clear()

    # Common formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Filter to only include logs from our project modules (e.g. "src" and "__main__") and exclude logs from external libraries
    project_filter = ProjectOnlyFilter(allowed_prefixes=("__main__", "src"))

    # 1) Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(project_filter)

    # 2) File handler (rotates daily at midnight, keeps 15 backups)
    file_handler = TimedRotatingFileHandler(
        filename=str(log_path),
        when="midnight",
        interval=1,
        backupCount=15,
        encoding="utf-8",
        utc=False,  # Use local time
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    file_handler.suffix = "%Y-%m-%d"
    file_handler.addFilter(project_filter)

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
