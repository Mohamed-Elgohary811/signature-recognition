# Contains classes that represent the outputs (artifacts) generated from each pipeline stage such as model paths and metrics.

from dataclasses import dataclass

# Data Ingestion artifact   
@dataclass
class DataIngestionArtifacts:
    dataset_path: str

    def to_dict(self):
        return self.__dict__


# Data transformation artifacts
@dataclass
class DataTransformationArtifacts:
    train_transformed_object: str
    valid_transformed_object: str
    test_transformed_object: str
    classes: int

    def to_dict(self):
        return self.__dict__



# Model trainer artifacts
@dataclass
class ModelTrainerArtifacts:
    trained_model_path: str

    def to_dict(self):
        return self.__dict__


# Model evaluation artifacts
@dataclass
class ModelEvaluationArtifacts:
    is_model_accepted: bool

    def to_dict(self):
        return self.__dict__



# Model pusher artifacts
@dataclass
class ModelPusherArtifacts:
    bucket_name: str

    def to_dict(self):
        return self.__dict__