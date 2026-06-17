import logging
import os
from datetime import datetime

LOG_FILE_NAME = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

logs_dir_path = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_dir_path, exist_ok=True)

LOG_FILE_PATH = os.path.join(logs_dir_path, LOG_FILE_NAME)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)