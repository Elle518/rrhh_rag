import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


# PATHS
path_root = Path(os.getcwd())

# Folders for data/
path_data = path_root / "data"
path_data_raw = path_data / "raw"
path_data_interim = path_data / "interim"
path_data_processed = path_data / "processed"

# Other Folders
path_conf = path_root / "conf"
path_models = path_root / "models"
path_logs = path_root / "logs"
path_output = path_root / "output"


# INITIALIZATION
def create_folders():
    """
    Create the project folder structure if it does not already exist.

    This function creates (recursively) the required directories used by the
    project (configuration, logs, outputs, and data subfolders). Existing
    folders are kept as-is.
    """
    path_conf.mkdir(parents=True, exist_ok=True)
    path_logs.mkdir(parents=True, exist_ok=True)
    path_output.mkdir(parents=True, exist_ok=True)
    path_data.mkdir(parents=True, exist_ok=True)
    path_data_raw.mkdir(parents=True, exist_ok=True)
    path_data_interim.mkdir(parents=True, exist_ok=True)
    path_data_processed.mkdir(parents=True, exist_ok=True)


# LOGS
def setup_logging(level=logging.INFO):
    """
    Configure the root logger to write formatted logs to stdout.

    The configuration:
      - Attaches a `logging.StreamHandler` pointing to `sys.stdout`.
      - Uses a timestamped format: "%(asctime)s [%(levelname)s] %(name)s: %(message)s".
      - Clears any existing handlers on the root logger to avoid duplicate logs.

    Args:
        level: Logging level for the root logger (e.g., `logging.INFO`, `logging.DEBUG`).

    Returns:
        None
    """
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()
    root_logger.addHandler(handler)


if __name__ == "__main__":

    setup_logging()
    create_folders()
    logger.info("Utils module executed successfully!")
