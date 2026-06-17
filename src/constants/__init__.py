import os
import shutil
import string
import tempfile
import torch
from datetime import datetime


def _get_dir_free_space(path: str) -> int:
    try:
        return shutil.disk_usage(path).free
    except Exception:
        return 0


def _choose_artifacts_root(preferred_root: str, min_free_bytes: int = 500 * 1024 * 1024) -> str:
    if os.path.exists(preferred_root):
        if _get_dir_free_space(preferred_root) >= min_free_bytes:
            return preferred_root
        preferred_parent = os.path.dirname(preferred_root) or preferred_root
        if os.path.exists(preferred_parent) and _get_dir_free_space(preferred_parent) >= min_free_bytes:
            return preferred_root

    for drive_letter in string.ascii_uppercase:
        drive_root = f"{drive_letter}:\\"
        if os.path.exists(drive_root):
            free_space = _get_dir_free_space(drive_root)
            if free_space >= min_free_bytes:
                return os.path.join(drive_root, "artifacts")

    return os.path.join(tempfile.gettempdir(), "artifacts")


# Common constants
CONFIG_PATH: str = os.path.join(os.getcwd(), "config", "config.yaml")
TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
DEFAULT_ARTIFACTS_ROOT: str = os.environ.get("ARTIFACTS_ROOT_DIR", os.path.join(os.getcwd(), "artifacts"))
ARTIFACTS_DIR = os.path.join(_choose_artifacts_root(DEFAULT_ARTIFACTS_ROOT), TIMESTAMP)
use_cuda = torch.cuda.is_available()
DEVICE = torch.device("cuda:0" if use_cuda else "cpu")



APP_HOST = "127.0.0.1"
APP_PORT = 8001


# Data Ingestion constants
DATA_INGESTION_ARTIFACTS_DIR = "DataIngestionArtifacts"



# Data transformation constants
DATA_TRANSFORMATION_ARTIFACTS_DIR = 'DataTransformationArtifacts'
DATA_TRANSFORMATION_TRAIN_FILE_NAME = "train_transformed.pkl"
DATA_TRANSFORMATION_VALID_FILE_NAME = "valid_transformed.pkl"
DATA_TRANSFORMATION_TEST_FILE_NAME = "test_transformed.pkl"


# Model trainer constants
MODEL_TRAINER_ARTIFACTS_DIR = "ModelTrainerArtifacts"
TRAINED_MODEL_PATH = "model.pt"


#Model evaluation constants
MODEL_EVALUATION_ARTIFACTS_DIR = "ModelEvaluationArtifacts"
BEST_MODEL_DIR = "best_model"
MODEL_NAME = "model.pt"