import os
import sys
import torch
from PIL import Image
from torchvision import transforms
from src.logger import logging
from src.exception import CustomException
from src.constants import DEVICE
from src.entity.config_entity import ModelEvaluationConfig, PredictionPipelineConfig
from src.utils.main_utils import load_object


class PredictionPipeline:
    def __init__(self):
        """
        Initialize the prediction pipeline
        """
        self.model_evaluation_config = ModelEvaluationConfig()
        self.prediction_pipeline_config = PredictionPipelineConfig()
        self.model = None
        self.device = DEVICE
        self.class_to_idx = {"forged": 0, "real": 1}
        self.idx_to_class = {0: "Forged", 1: "Genuine"}
        self._load_model()
        self._load_class_mapping()

    @staticmethod
    def _display_name(class_name: str) -> str:
        normalized = str(class_name).strip().lower()
        if normalized in {"real", "genuine"}:
            return "Genuine"
        if normalized in {"forged", "forge", "fake"}:
            return "Forged"
        return str(class_name).strip().title()

    def _find_artifacts_root(self):
        return os.path.dirname(os.path.dirname(self.model_evaluation_config.MODEL_EVALUATION_ARTIFACTS_DIR))

    def _load_class_mapping(self):
        """
        Load the class mapping used by torchvision ImageFolder during training.
        ImageFolder sorts folder names alphabetically, so this project's dataset
        maps forged -> 0 and real -> 1.
        """
        try:
            model_class_to_idx = getattr(self.model, "class_to_idx", None)
            model_idx_to_class = getattr(self.model, "idx_to_class", None)
            if model_class_to_idx:
                self.class_to_idx = dict(model_class_to_idx)
                if model_idx_to_class:
                    self.idx_to_class = dict(model_idx_to_class)
                else:
                    self.idx_to_class = {
                        idx: self._display_name(class_name)
                        for class_name, idx in self.class_to_idx.items()
                    }
                logging.info(f"class_to_idx used during training: {self.class_to_idx}")
                logging.info(f"idx_to_class used during inference: {self.idx_to_class}")
                return

            artifacts_root = self._find_artifacts_root()
            mapping = None

            if os.path.exists(artifacts_root):
                timestamp_dirs = sorted([
                    d for d in os.listdir(artifacts_root)
                    if os.path.isdir(os.path.join(artifacts_root, d))
                ], reverse=True)

                for timestamp_dir in timestamp_dirs:
                    transformation_dir = os.path.join(
                        artifacts_root,
                        timestamp_dir,
                        "DataTransformationArtifacts"
                    )

                    for file_name in ("train_transformed.pkl", "valid_transformed.pkl", "test_transformed.pkl"):
                        transformed_path = os.path.join(transformation_dir, file_name)
                        if not os.path.exists(transformed_path):
                            continue

                        transformed_dataset = load_object(transformed_path)
                        base_dataset = getattr(transformed_dataset, "dataset", transformed_dataset)
                        mapping = getattr(base_dataset, "class_to_idx", None)
                        if mapping:
                            break

                    if mapping:
                        break

            if mapping:
                self.class_to_idx = dict(mapping)
                self.idx_to_class = {
                    idx: self._display_name(class_name)
                    for class_name, idx in self.class_to_idx.items()
                }

            logging.info(f"class_to_idx used during training: {self.class_to_idx}")
            logging.info(f"idx_to_class used during inference: {self.idx_to_class}")
        except Exception as e:
            logging.warning(f"Could not load class mapping from artifacts; using fallback mapping. Error: {e}")
            logging.info(f"class_to_idx used during training: {self.class_to_idx}")
            logging.info(f"idx_to_class used during inference: {self.idx_to_class}")

    def _find_latest_model(self):
        """
        Find the latest model file in artifacts directory
        """
        artifacts_root = self._find_artifacts_root()
        
        if not os.path.exists(artifacts_root):
            return None
        
        # Get all timestamp directories and sort by name (descending to get latest)
        timestamp_dirs = sorted([d for d in os.listdir(artifacts_root) 
                               if os.path.isdir(os.path.join(artifacts_root, d))], reverse=True)
        
        for timestamp_dir in timestamp_dirs:
            # Try best model first
            best_model_path = os.path.join(artifacts_root, timestamp_dir, "ModelEvaluationArtifacts", "best_model", "model.pt")
            if os.path.exists(best_model_path):
                logging.info(f"Found best model: {best_model_path}")
                return best_model_path
            
            # Try trainer model as fallback
            trainer_model_path = os.path.join(artifacts_root, timestamp_dir, "ModelTrainerArtifacts", "model.pt")
            if os.path.exists(trainer_model_path):
                logging.info(f"Found trainer model: {trainer_model_path}")
                return trainer_model_path
        
        return None

    def _load_model(self):
        """
        Load the trained model from the best model path
        """
        try:
            logging.info("Loading trained model for prediction")
            
            # First try the current timestamp path
            best_model_dir = self.model_evaluation_config.BEST_MODEL_DIR
            best_model_path = os.path.join(best_model_dir, self.model_evaluation_config.MODEL_NAME)
            
            if not os.path.exists(best_model_path):
                logging.warning(f"Model not found at {best_model_path}, searching for latest model...")
                best_model_path = self._find_latest_model()
                
                if best_model_path is None:
                    raise FileNotFoundError("No trained model found in artifacts directory")
            
            self.model = torch.load(best_model_path, map_location=self.device, weights_only=False)
            self.model.to(self.device)
            self.model.eval()
            logging.info(f"Model loaded successfully from {best_model_path}")
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise CustomException(e, sys) from e

    def _get_image_transform(self):
        """
        Get the image transformation pipeline
        """
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        return transform

    def predict_image(self, image_path: str) -> dict:
        """
        Predict if a signature image is genuine or forged
        
        :param image_path: Path to the image file
        :return: Dictionary with prediction results
        """
        try:
            logging.info(f"Starting prediction for image: {image_path}")
            
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            # Load and transform image
            image = Image.open(image_path).convert('RGB')
            transform = self._get_image_transform()
            image_tensor = transform(image).unsqueeze(0).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                output = self.model(image_tensor)
                probabilities = torch.softmax(output, dim=1)
                prediction = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][prediction].item()
            
            displayed_label = self.idx_to_class.get(prediction, "Unknown")
            genuine_idx = self.class_to_idx.get("genuine", self.class_to_idx.get("real"))
            forged_idx = self.class_to_idx.get("forged")
            result = {
                "prediction": displayed_label,
                "confidence": float(confidence),
                "probabilities": {
                    "genuine": float(probabilities[0][genuine_idx].item()) if genuine_idx is not None else 0.0,
                    "forged": float(probabilities[0][forged_idx].item()) if forged_idx is not None else 0.0
                },
                "debug": {
                    "class_to_idx": self.class_to_idx,
                    "idx_to_class": self.idx_to_class,
                    "predicted_index": prediction,
                    "displayed_label": displayed_label
                },
            }
            
            logging.info(f"predicted index: {prediction}")
            logging.info(f"displayed label: {displayed_label}")
            logging.info(f"Prediction completed: {result}")
            return result
            
        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            raise CustomException(e, sys) from e

    def predict_batch(self, image_paths: list) -> list:
        """
        Predict multiple images
        
        :param image_paths: List of image file paths
        :return: List of prediction results
        """
        try:
            logging.info(f"Starting batch prediction for {len(image_paths)} images")
            results = []
            
            for image_path in image_paths:
                result = self.predict_image(image_path)
                results.append({
                    "image_path": image_path,
                    **result
                })
            
            logging.info(f"Batch prediction completed for {len(results)} images")
            return results
            
        except Exception as e:
            logging.error(f"Error during batch prediction: {e}")
            raise CustomException(e, sys) from e

    def run_pipeline(self, image_bytes: bytes) -> dict:
        """
        Run the prediction pipeline on image bytes from API
        
        :param image_bytes: Image bytes from file upload
        :return: Dictionary with prediction results
        """
        tmp_path = None
        try:
            import tempfile
            
            logging.info("Running prediction pipeline on uploaded image")
            
            # Save bytes to temporary file
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                tmp_file.write(image_bytes)
                tmp_path = tmp_file.name
            
            # Predict on the temporary file
            result = self.predict_image(tmp_path)
            
            logging.info(f"Prediction pipeline completed successfully")
            return result
            
        except Exception as e:
            logging.error(f"Error in prediction pipeline: {e}")
            raise CustomException(e, sys) from e
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
