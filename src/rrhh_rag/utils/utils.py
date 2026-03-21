import logging
import os
from pathlib import Path

from rrhh_rag.utils.logging_utils import setup_logging

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


if __name__ == "__main__":

    setup_logging(log_dir=path_logs, log_name="info.log")

    create_folders()
    logger.info("Utils module executed successfully!")
