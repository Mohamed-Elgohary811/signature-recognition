import sys
from src.logger import logging
from src.exception import CustomException
from src.configurations.local_data_sync import LocalDataSync
from src.entity.config_entity import ModelPusherConfig
from src.entity.artifact_entity import ModelTrainerArtifacts, ModelPusherArtifacts




class ModelPusher:
    def __init__(self, model_pusher_config: ModelPusherConfig,
        model_trainer_artifacts: ModelTrainerArtifacts):
        """
        :param model_pusher_config: Configuration for model pusher
        :param model_trainer_artifacts: Output reference of model trainer artifact stage
        """
        self.model_pusher_config = model_pusher_config
        self.model_trainer_artifacts = model_trainer_artifacts
        self.local_data_sync = LocalDataSync()


    def initiate_model_pusher(self) -> ModelPusherArtifacts:
        """
        Method Name : initiate_model_pusher
        Description : This method initiates the model pusher process

        Output : Model pusher artifact
        """
        logging.info("Entered initiate_model_pusher method of ModelPusher class")
        try:
            logging.info("Saving the trained model to local storage")
            self.local_data_sync.sync_file_from_local(
                source_path=self.model_trainer_artifacts.trained_model_path,
                destination=self.model_pusher_config.bucket_name
            )
            logging.info("Saved model to local storage")
            logging.info("Creating model pusher artifacts")
            model_pusher_artifact = ModelPusherArtifacts(
                bucket_name=self.model_pusher_config.bucket_name
            )
            logging.info("Exited the initiate_model_pusher method of ModelPusher class")
            return model_pusher_artifact
        except Exception as e:
            raise CustomException(e, sys) from e