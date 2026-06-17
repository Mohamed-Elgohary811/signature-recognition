# Contains configuration classes for different project components using structured dataclasses.
import os
from src.constants import *

from dataclasses import dataclass
from src.utils.main_utils import read_yaml_file


@dataclass
class DataIngestionConfig:
    def __init__(self):
        self.config = read_yaml_file(CONFIG_PATH)
        di_cfg = self.config.get('data_ingestion_config', {})
        self.BUCKET_NAME: str = di_cfg.get("bucket_name")
        self.ZIP_FILE_NAME: str = di_cfg.get("zip_file_name")

        # Source path for the zip file (can be relative to project root or absolute)
        source_path = di_cfg.get("source_data_path", self.ZIP_FILE_NAME)
        if os.path.isabs(source_path):
            self.ZIP_SOURCE_PATH: str = source_path
        else:
            self.ZIP_SOURCE_PATH: str = os.path.join(os.getcwd(), source_path)

        # Destination artifact paths
        self.DATA_INGESTION_ARTIFACTS_DIR: str = os.path.join(os.getcwd(), ARTIFACTS_DIR, DATA_INGESTION_ARTIFACTS_DIR)
        self.ZIP_FILE_PATH: str = os.path.join(self.DATA_INGESTION_ARTIFACTS_DIR, self.ZIP_FILE_NAME)



@dataclass
class DataTransformationConfig:
    def __init__(self):
        self.config = read_yaml_file(CONFIG_PATH)
        dt_cfg = self.config.get('data_transformation_config', {})
        self.STD: list = dt_cfg.get("std")
        self.MEAN: list = dt_cfg.get("mean")
        self.IMG_SIZE: int = dt_cfg.get("img_size")
        self.DEGREE_N: int = dt_cfg.get("degree_n")
        self.DEGREE_P: int = dt_cfg.get("degree_p")
        self.TRAIN_RATIO: float = dt_cfg.get("train_ratio")
        self.VALID_RATIO: float = dt_cfg.get("valid_ratio")
        self.DATA_TRANSFORMATION_ARTIFACTS_DIR: str = os.path.join(os.getcwd(), ARTIFACTS_DIR, DATA_TRANSFORMATION_ARTIFACTS_DIR)
        self.TRAIN_TRANSFORM_OBJECT_FILE_PATH: str = os.path.join(self.DATA_TRANSFORMATION_ARTIFACTS_DIR, DATA_TRANSFORMATION_TRAIN_FILE_NAME)
        self.VALID_TRANSFORM_OBJECT_FILE_PATH: str = os.path.join(self.DATA_TRANSFORMATION_ARTIFACTS_DIR, DATA_TRANSFORMATION_VALID_FILE_NAME)
        self.TEST_TRANSFORM_OBJECT_FILE_PATH: str = os.path.join(self.DATA_TRANSFORMATION_ARTIFACTS_DIR, DATA_TRANSFORMATION_TEST_FILE_NAME)





@dataclass
class ModelTrainerConfig:

    def __init__(self):
        self.config = read_yaml_file(CONFIG_PATH)
        self.LR: float = self.config['model_trainer_config']['lr']
        self.EPOCHS: int = self.config['model_trainer_config']['epochs']
        self.NUM_WORKERS: int = self.config['model_trainer_config']['num_workers']
        self.BATCH_SIZE: int = self.config['model_trainer_config']['batch_size']
        self.MODEL_TRAINER_ARTIFACTS_DIR: str = os.path.join(os.getcwd(), ARTIFACTS_DIR, MODEL_TRAINER_ARTIFACTS_DIR)
        self.TRAINED_MODEL_PATH: str = os.path.join(self.MODEL_TRAINER_ARTIFACTS_DIR, TRAINED_MODEL_PATH)




@dataclass
class ModelEvaluationConfig:

    def __init__(self):
        self.config = read_yaml_file(CONFIG_PATH)
        self.MODEL_NAME: str = MODEL_NAME
        self.BUCKET_NAME: str = self.config['model_evaluation_config']["bucket_name"]
        self.BATCH_SIZE: int = self.config['model_evaluation_config'] ["batch_size"]
        self.NUM_WORKERS: int = self.config['model_evaluation_config']["num_workers"]
        self.MODEL_EVALUATION_ARTIFACTS_DIR: str = os.path.join(os.getcwd(), ARTIFACTS_DIR,
        MODEL_EVALUATION_ARTIFACTS_DIR)
        self.BEST_MODEL_DIR: str = os.path.join(self.MODEL_EVALUATION_ARTIFACTS_DIR, BEST_MODEL_DIR)




@dataclass
class ModelPusherConfig:

    def __init__(self):
        self.config = read_yaml_file(CONFIG_PATH)
        self.MODEL_NAME: str = MODEL_NAME
        self.bucket_name: str = self.config['model_pusher_config']["bucket_name"]


@dataclass
class PredictionPipelineConfig:

    def __init__(self):
        self.config = read_yaml_file(CONFIG_PATH)
        prediction_cfg = self.config.get('prediction_pipeline_config', {})
        self.INPUT_IMAGE: str = prediction_cfg.get("input_image", "input.jpg")
        self.BUCKET_NAME: str = prediction_cfg.get("bucket_name", prediction_cfg.get("pucket_name"))
        self.MODEL_NAME: str = prediction_cfg.get("model_name", MODEL_NAME)
        self.THRESHOLD: float = prediction_cfg.get("threshold", 0.5)
