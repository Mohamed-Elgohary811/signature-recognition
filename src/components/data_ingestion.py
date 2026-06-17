import os
import sys
import shutil
import string
import tempfile
from zipfile import ZipFile
from src.logger import logging
from src.exception import CustomException

from src.configurations.local_data_sync import LocalDataSync

from src.entity.config_entity import DataIngestionConfig

from src.entity.artifact_entity import DataIngestionArtifacts


class DataIngestion:

    def __init__(self, data_ingestion_config : DataIngestionConfig):
    # Param data_ingestion_config: configration for data ingestion
        self.data_ingestion_config = data_ingestion_config
        self.local_data_sync = LocalDataSync()
        self.dataset_dir = None


    def get_data_from_local(self) -> None:

        """ 
            Method Name : get_data_from_local
  
          Description : This function fetch data from local storage

            Output : Returns data into DataIngestionArtifacts
            On Failure : Write an exception log and then raise an exception
        """
        try:
            logging.info("Entered the get_data_from_local method of DataIngestion class")

            os.makedirs(self.data_ingestion_config.DATA_INGESTION_ARTIFACTS_DIR, exist_ok=True)

            logging.info(f"Copying dataset from source: {self.data_ingestion_config.ZIP_SOURCE_PATH}")
            source_abs = os.path.abspath(self.data_ingestion_config.ZIP_SOURCE_PATH)
            artifacts_abs = os.path.abspath(self.data_ingestion_config.DATA_INGESTION_ARTIFACTS_DIR)

            if source_abs.startswith(artifacts_abs + os.sep) or source_abs == artifacts_abs:
                logging.info("Source zip is already within the artifacts directory; skipping copy.")
                self.actual_zip_path = self.data_ingestion_config.ZIP_SOURCE_PATH
            else:
                try:
                    self.local_data_sync.sync_file_from_local(
                        self.data_ingestion_config.ZIP_SOURCE_PATH,
                        self.data_ingestion_config.DATA_INGESTION_ARTIFACTS_DIR)

                    src_basename = os.path.basename(self.data_ingestion_config.ZIP_SOURCE_PATH)
                    dest_copied_path = os.path.join(self.data_ingestion_config.DATA_INGESTION_ARTIFACTS_DIR, src_basename)
                    expected_dest_path = os.path.join(self.data_ingestion_config.DATA_INGESTION_ARTIFACTS_DIR,
                                                      self.data_ingestion_config.ZIP_FILE_NAME)
                    if src_basename != self.data_ingestion_config.ZIP_FILE_NAME:
                        if os.path.exists(dest_copied_path):
                            os.replace(dest_copied_path, expected_dest_path)
                            logging.info(f"Renamed {dest_copied_path} to {expected_dest_path}")
                        else:
                            logging.warning(f"Expected copied file not found at {dest_copied_path}")

                    if os.path.exists(expected_dest_path):
                        self.actual_zip_path = expected_dest_path
                    elif os.path.exists(dest_copied_path):
                        self.actual_zip_path = dest_copied_path
                    else:
                        found = [os.path.join(self.data_ingestion_config.DATA_INGESTION_ARTIFACTS_DIR, f)
                                 for f in os.listdir(self.data_ingestion_config.DATA_INGESTION_ARTIFACTS_DIR)
                                 if f.lower().endswith('.zip')]
                        self.actual_zip_path = found[0] if found else None

                except Exception as e:
                    logging.warning(f"Failed to copy dataset into artifacts dir: {e}. Using source zip path directly.")
                    self.actual_zip_path = self.data_ingestion_config.ZIP_SOURCE_PATH

            if not self.actual_zip_path or not os.path.exists(self.actual_zip_path):
                raise FileNotFoundError(f"Zip file not found: {self.actual_zip_path}")

            logging.info(f"Actual zip path set to: {self.actual_zip_path}")

            logging.info("Exited the get_data_from_local method of DataIngestion class")

        except Exception as e:
            raise CustomException(e, sys) from e




    def _get_dir_free_space(self, path: str) -> int:
        try:
            return shutil.disk_usage(path).free
        except Exception:
            return 0


    def _estimate_uncompressed_size(self, zip_path: str) -> int:
        with ZipFile(zip_path, 'r') as zip_ref:
            return sum(zinfo.file_size for zinfo in zip_ref.infolist())


    def _choose_extraction_dir(self, zip_path: str, preferred_dir: str) -> str:
        estimated_size = self._estimate_uncompressed_size(zip_path)
        required_space = int(estimated_size * 1.1) + 5 * 1024 * 1024

        preferred_parent = preferred_dir if os.path.isdir(preferred_dir) else os.path.dirname(preferred_dir) or os.getcwd()
        preferred_free = self._get_dir_free_space(preferred_parent)
        logging.info(f"Preferred extraction location: {preferred_dir} (free space {preferred_free} bytes)")

        if preferred_free >= required_space:
            return preferred_dir

        source_parent = os.path.dirname(os.path.abspath(zip_path))
        fallback_dir = os.path.join(source_parent, f"{os.path.splitext(os.path.basename(zip_path))[0]}_data")
        fallback_free = self._get_dir_free_space(source_parent)
        logging.info(f"Fallback extraction candidate: {fallback_dir} (free space {fallback_free} bytes)")

        if fallback_free >= required_space:
            return fallback_dir

        # On Windows, attempt to locate any other drive with enough free space
        for drive_letter in string.ascii_uppercase:
            drive_root = f"{drive_letter}:\\"
            if os.path.exists(drive_root):
                drive_free = self._get_dir_free_space(drive_root)
                if drive_free >= required_space:
                    temp_dir = tempfile.mkdtemp(prefix="signature_data_", dir=drive_root)
                    logging.info(f"Using alternate drive for extraction: {temp_dir} (free space {drive_free} bytes)")
                    return temp_dir

        # Last resort: use system temp on current drive
        temp_dir = tempfile.mkdtemp(prefix="signature_data_")
        temp_parent = os.path.dirname(temp_dir)
        temp_free = self._get_dir_free_space(temp_parent)
        logging.info(f"Temp extraction candidate: {temp_dir} (free space {temp_free} bytes)")

        if temp_free >= required_space:
            return temp_dir

        raise OSError(
            f"Insufficient disk space to extract dataset. Need at least {required_space} bytes free on one drive.")


    def _normalize_dataset_dir(self, dataset_dir: str) -> str:
        if not os.path.isdir(dataset_dir):
            return dataset_dir

        entries = [entry for entry in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, entry))]
        if len(entries) == 1:
            single_path = os.path.join(dataset_dir, entries[0])
            nested_entries = [entry for entry in os.listdir(single_path) if os.path.isdir(os.path.join(single_path, entry))]
            if len(nested_entries) >= 1:
                logging.info(f"Normalized dataset path from {dataset_dir} to nested subdirectory {single_path}")
                return single_path

        return dataset_dir


    def unzip_and_clean(self) -> None:
        """ 
            Method Name : unzip_and_clean
            Description : This function unzips the data and then removes the zip file

            Output : Unzipped data in DataIngestionArtifacts and zip file will be removed
            On Failure : Write an exception log and then raise an exception
        """
        logging.info("Entered the unzip_and_clean method of Data ingestion class")
        try:
            # Use the actual copied zip file if available; otherwise fall back to configured path
            zip_path = getattr(self, 'actual_zip_path', None) or self.data_ingestion_config.ZIP_FILE_PATH
            if not os.path.exists(zip_path):
                # try to find any zip in the artifacts dir
                zips = [os.path.join(self.data_ingestion_config.DATA_INGESTION_ARTIFACTS_DIR, f)
                        for f in os.listdir(self.data_ingestion_config.DATA_INGESTION_ARTIFACTS_DIR)
                        if f.lower().endswith('.zip')]
                if zips:
                    zip_path = zips[0]
                else:
                    raise FileNotFoundError(f"No zip file found to extract in {self.data_ingestion_config.DATA_INGESTION_ARTIFACTS_DIR}")

            extract_dir = self._choose_extraction_dir(zip_path, self.data_ingestion_config.DATA_INGESTION_ARTIFACTS_DIR)
            os.makedirs(extract_dir, exist_ok=True)

            with ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)

            self.dataset_dir = extract_dir
            if os.path.abspath(extract_dir) != os.path.abspath(self.data_ingestion_config.DATA_INGESTION_ARTIFACTS_DIR):
                logging.warning(f"Extracted dataset to fallback directory: {extract_dir}")
            else:
                logging.info(f"Extracted dataset to artifacts directory: {extract_dir}")

            logging.info("Exited the unzip_and_clean method of Data ingestion class")
        except Exception as e:
            raise CustomException(e, sys) from e



    
    def initiate_data_ingestion(self) -> None:
        
        """ 
            Method Name : initiate_data_ingestion
            Description : This function initiotes a data ingestion steps

            Output : Returns data ingestion artifact
            On Failure : Write an exception log and then raise an exception
        """
                
        logging.info("Entered the initiate_data_ingestion method of Data ingestion class")
        try:
            self.get_data_from_local()
            logging.info("Fetched the zipped dataset from local storage")

            self.unzip_and_clean()
            logging.info("Unzipped the file fetched from local storage")

            # Delete the actual zip file used for extraction only if it was copied into artifacts
            zip_to_delete = getattr(self, 'actual_zip_path', None)
            artifacts_abs = os.path.abspath(self.data_ingestion_config.DATA_INGESTION_ARTIFACTS_DIR)
            if zip_to_delete and os.path.exists(zip_to_delete):
                zip_abs = os.path.abspath(zip_to_delete)
                if zip_abs.startswith(artifacts_abs + os.sep) or zip_abs == artifacts_abs:
                    logging.info(f"Deleting zip file {zip_to_delete}")
                    os.remove(zip_to_delete)
                else:
                    logging.info(f"Skipping deletion of external source zip file: {zip_to_delete}")
            else:
                logging.warning(f"Zip file to delete not found: {zip_to_delete}")

            normalized_dataset_dir = self._normalize_dataset_dir(self.dataset_dir or self.data_ingestion_config.DATA_INGESTION_ARTIFACTS_DIR)
            data_ingestion_artifacts = DataIngestionArtifacts(
                dataset_path=normalized_dataset_dir)

            logging.info("Exited the initiate_data_ingestion method of Data ingestion class")
            return data_ingestion_artifacts

        except Exception as e:
            raise CustomException(e, sys) from e