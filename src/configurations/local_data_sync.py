import os
import shutil
import sys

from src.logger import logging
from src.exception import CustomException


class LocalDataSync:

    def sync_file_from_local(self, source_path, destination):

        """
        Copy file from local folder to destination

        param source_path: path of the source file
        param destination: destination folder
        """

        try:
            # Validate source exists before attempting copy
            if not os.path.exists(source_path):
                raise FileNotFoundError(f"Source file not found: {source_path}")

            os.makedirs(destination, exist_ok=True)

            shutil.copy(source_path, destination)

            logging.info(f"File copied from {source_path} to {destination}")

        except Exception as e:
            raise CustomException(e, sys) from e

    def sync_folder_from_local(self, source_folder, destination):

        """
        Copy full folder from local storage

        param source_folder: source folder path
        param destination: destination folder path
        """

        try:
            if os.path.exists(destination):
                shutil.rmtree(destination)

            shutil.copytree(source_folder, destination)

            logging.info(f"Folder copied from {source_folder} to {destination}")

        except Exception as e:
            raise CustomException(e, sys) from e