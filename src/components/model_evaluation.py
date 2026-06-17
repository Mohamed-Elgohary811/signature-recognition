import os
import sys
import torch
from tqdm import tqdm
from src.logger import logging
from src.exception import CustomException
from src.constants import DEVICE
from torch.utils.data import DataLoader
from src.utils.main_utils import load_object
from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import ModelTrainerArtifacts, DataTransformationArtifacts, ModelEvaluationArtifacts


class ModelEvaluation:
    def __init__(self,
                 model_evaluation_config: ModelEvaluationConfig,
                 model_trainer_artifacts: ModelTrainerArtifacts,
                 data_transformation_artifacts: DataTransformationArtifacts):
        self.model_evaluation_config = model_evaluation_config
        self.model_trainer_artifacts = model_trainer_artifacts
        self.data_transformation_artifacts = data_transformation_artifacts


    def _evaluate_loss_and_accuracy(self, model, criterion, dataloader):
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            with tqdm(dataloader, unit='batch', leave=False) as pbar:
                pbar.set_description('evaluating')
                for images, labels in pbar:
                    images = images.to(DEVICE, non_blocking=True)
                    labels = labels.to(DEVICE, non_blocking=True)

                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    total_loss += loss.item()

                    preds = torch.argmax(outputs, dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

        avg_loss = total_loss / max(1, len(dataloader))
        accuracy = correct / max(1, total)
        return avg_loss, accuracy


    def initiate_model_evaluation(self) -> ModelEvaluationArtifacts:
        try:
            logging.info("Entered the initiate_model_evaluation method")

            # load test dataset
            test_dataset_path = self.data_transformation_artifacts.test_transformed_object
            test_dataset = load_object(test_dataset_path)

            # dataloader
            num_workers = self.model_evaluation_config.NUM_WORKERS
            if os.name == 'nt' and num_workers > 0:
                logging.info("Running on Windows: setting num_workers to 0 to avoid multiprocessing spawn issues")
                num_workers = 0

            test_loader = DataLoader(test_dataset,
                                     shuffle=False,
                                     batch_size=self.model_evaluation_config.BATCH_SIZE,
                                     num_workers=num_workers)

            # load trained model
            trained_model_path = self.model_trainer_artifacts.trained_model_path
            if not os.path.isfile(trained_model_path):
                raise FileNotFoundError(f"Trained model not found at {trained_model_path}")

            model = torch.load(trained_model_path, map_location=DEVICE, weights_only=False)
            model.to(DEVICE)

            criterion = torch.nn.CrossEntropyLoss()

            trained_loss, trained_acc = self._evaluate_loss_and_accuracy(model, criterion, test_loader)
            logging.info(f"Trained model - Loss: {trained_loss:.6f}, Acc: {trained_acc:.4f}")

            # determine best model path (local-only)
            best_model_dir = self.model_evaluation_config.BEST_MODEL_DIR
            os.makedirs(best_model_dir, exist_ok=True)
            best_model_path = os.path.join(best_model_dir, self.model_evaluation_config.MODEL_NAME)

            is_model_accepted = True
            if os.path.isfile(best_model_path):
                logging.info("Found existing best model locally, comparing performance")
                best_model = torch.load(best_model_path, map_location=DEVICE, weights_only=False)
                best_model.to(DEVICE)
                best_loss, best_acc = self._evaluate_loss_and_accuracy(best_model, criterion, test_loader)
                logging.info(f"Best model - Loss: {best_loss:.6f}, Acc: {best_acc:.4f}")

                # accept new model if it has strictly lower loss or higher accuracy
                if trained_loss < best_loss or (trained_loss == best_loss and trained_acc > best_acc):
                    is_model_accepted = True
                    torch.save(model, best_model_path)
                    logging.info(f"New model saved as best model at {best_model_path}")
                else:
                    is_model_accepted = False
                    logging.info("Trained model did not outperform the current best model")
            else:
                # no best model found -> accept and save
                torch.save(model, best_model_path)
                logging.info(f"No existing best model found. Saved current model as best at {best_model_path}")
                is_model_accepted = True

            return ModelEvaluationArtifacts(is_model_accepted=is_model_accepted)

        except Exception as e:
            raise CustomException(e, sys) from e


if __name__ == "__main__":
    # simple CLI for quick local evaluation
    try:
        from src.entity.config_entity import DataIngestionConfig, DataTransformationConfig, ModelEvaluationConfig
        from src.entity.artifact_entity import DataIngestionArtifacts, DataTransformationArtifacts, ModelTrainerArtifacts

        # create configs
        di_cfg = DataIngestionConfig()
        dt_cfg = DataTransformationConfig()
        me_cfg = ModelEvaluationConfig()

        # Load artifacts
        trained_model_path = None
        for root, dirs, files in os.walk(os.path.join(os.getcwd(), 'artifacts')):
            if 'ModelTrainerArtifacts' in root and 'model.pt' in files:
                trained_model_path = os.path.join(root, 'model.pt')
                break

        if trained_model_path is None:
            raise FileNotFoundError('Could not find trained model under artifacts/*/ModelTrainerArtifacts/model.pt')

        dt_dir = None
        for root, dirs, files in os.walk(os.path.join(os.getcwd(), 'artifacts')):
            if 'DataTransformationArtifacts' in root and 'test_transformed.pkl' in files:
                dt_dir = root
                break

        if dt_dir is None:
            raise FileNotFoundError('Could not find DataTransformationArtifacts with test_transformed.pkl')

        data_trans_artifact = DataTransformationArtifacts(
            train_transformed_object=os.path.join(dt_dir, 'train_transformed.pkl'),
            valid_transformed_object=os.path.join(dt_dir, 'valid_transformed.pkl'),
            test_transformed_object=os.path.join(dt_dir, 'test_transformed.pkl'),
            classes=2
        )

        model_trainer_artifacts = ModelTrainerArtifacts(trained_model_path=trained_model_path)
        evaluator = ModelEvaluation(model_evaluation_config=me_cfg,
                                    model_trainer_artifacts=model_trainer_artifacts,
                                    data_transformation_artifacts=data_trans_artifact)

        result = evaluator.initiate_model_evaluation()
        logging.info(f"Model evaluation result: {result}")

    except Exception as e:
        logging.exception(e)
        raise
