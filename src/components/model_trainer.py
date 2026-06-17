import os
import sys
import torch
from tqdm import tqdm
import torch.nn as nn
from torchvision import models
from src.logger import logging
from src.constants import DEVICE
from torch.utils.data import DataLoader
from src.exception import CustomException
from src.utils.main_utils import load_object
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import DataTransformationArtifacts, ModelTrainerArtifacts



class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig,
                 data_transformation_artifacts: DataTransformationArtifacts):
        """
        :param model_trainer_config: Configuration for model trainer
        :param data_transformation_artifacts: Artifacts for data transformation
        """

        self.model_trainer_config = model_trainer_config
        self.data_transformation_artifacts = data_transformation_artifacts
        self.learning_rate = self.model_trainer_config.LR
        self.epochs = self.model_trainer_config.EPOCHS
        self.batch_size = self.model_trainer_config.BATCH_SIZE
        self.num_workers = self.model_trainer_config.NUM_WORKERS

        if os.name == "nt" and self.num_workers > 0:
            logging.info("Running on Windows: setting num_workers to 0 to avoid multiprocessing spawn issues")
            self.num_workers = 0


    def train(self, model, criterion, optimizer, train_dataloader, valid_dataloader):
        """
        Method Name : train
        Description : This method takes pretrained model, loss, optimiser, train and valid data loaderto start training.
        """
        
        try:

            total_train_loss = 0
            total_test_loss = 0

            model.train()
            with tqdm(train_dataloader, unit='batch', leave=False) as pbar:
                pbar.set_description(f'training')
                for images, idxs in pbar:
                    images = images.to(DEVICE, non_blocking=True)
                    idxs = idxs.to(DEVICE, non_blocking=True)
                    output = model(images)

                    loss = criterion(output, idxs)
                    total_train_loss += loss.item()

                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            
            model.eval()
            with tqdm(valid_dataloader, unit='batch', leave=False) as pbar:
                pbar.set_description(f'testing')
                for images, idxs in pbar:
                    images = images.to(DEVICE, non_blocking=True)
                    idxs = idxs.to(DEVICE, non_blocking=True)

                    output = model(images)
                    loss = criterion(output, idxs)
                    total_test_loss += loss.item()

            train_loss = total_train_loss / len(train_dataloader)
            valid_loss = total_test_loss / len(valid_dataloader)
            print(f'Train loss: {train_loss:.6f} Test loss: {valid_loss:.6f} ')
        
        except Exception as e:
            raise CustomException(e, sys) from e

    def initiate_model_trainer(self) -> ModelTrainerArtifacts:
        """
        Method Name : initiate_model_trainer
        Description : This function initiates a model trainer steps

        Output :Returns model troiner artifact
        On Failure : Write an exception Log and then raise an exception
        """


        try:

            logging.info("Entered the initiate_model_trainer method of Model trainer class")

            train_dataset = load_object(self.data_transformation_artifacts.train_transformed_object)
            valid_dataset = load_object(self.data_transformation_artifacts.valid_transformed_object)
            base_train_dataset = getattr(train_dataset, "dataset", train_dataset)
            class_to_idx = getattr(base_train_dataset, "class_to_idx", {"forged": 0, "real": 1})
            idx_to_class = {
                idx: ("Genuine" if class_name.lower() in {"real", "genuine"} else "Forged")
                for class_name, idx in class_to_idx.items()
            }

            logging.info("Loaded dataset from data transformation artifacts")
            logging.info(f"class_to_idx used during training: {class_to_idx}")
            logging.info(f"idx_to_class saved with model: {idx_to_class}")
            train_loader = DataLoader(train_dataset,
                shuffle=True,
                batch_size=self.batch_size,
                num_workers=self.num_workers)

            valid_loader = DataLoader(valid_dataset,
                shuffle=True,
                batch_size=self.batch_size,
                num_workers=self.num_workers)
            
            logging.info("Loaded train and valid data loader")

            model = models.resnet34(weights='ResNet34_Weights.DEFAULT')
            logging.info("Loaded pretrained resnet34 model")

            model.fc = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(model.fc.in_features, self.data_transformation_artifacts.classes)
            )

            logging.info("Updated the last layer of the pretrained model")

            model = model.to(DEVICE)

            criterion = torch.nn.CrossEntropyLoss()
            logging.info("Cross entropy loss function is used.")
            optimizer = torch.optim.SGD(model.parameters(),
                        lr=self.learning_rate,
                        momentum=0.9)
            logging.info("SGD optimiser is used.")

            logging.info("Model training started")

            for i in range(self.epochs):
                logging.info(f"Model training at epoch: {i+1}")
                print(f"Epoch {i +1}/{self.epochs}")
                self.train(model, criterion, optimizer, train_loader, valid_loader)
            
            logging.info("Model training done. !!! ")

            os.makedirs(self.model_trainer_config.MODEL_TRAINER_ARTIFACTS_DIR, exist_ok=True)
            model.class_to_idx = class_to_idx
            model.idx_to_class = idx_to_class
            torch.save(model, self.model_trainer_config.TRAINED_MODEL_PATH)
            logging.info(f"saved trained model at {self.model_trainer_config.TRAINED_MODEL_PATH}")

            model_trainer_artifacts = ModelTrainerArtifacts(trained_model_path=self.model_trainer_config.TRAINED_MODEL_PATH)
            logging.info(f"Model trainer artifact: {model_trainer_artifacts}")

            logging.info("Exited the initiate_model_trainer method of Model trainer class")
            return model_trainer_artifacts
            
        except Exception as e:
            raise CustomException(e, sys) from e
