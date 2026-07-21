import logging
from pathlib import Path

import kagglehub
from google.cloud import storage

from common.helper.gcs_io import upload_file
from common.helper.zip_helpers import zip_images

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def download_dataset(source_path: str) -> Path:
    logger.info("Downloading dataset from Kaggle: %s", source_path)

    local_path = kagglehub.dataset_download(source_path)

    logger.info("Dataset available at: %s", local_path)

    return Path(local_path)

def upload_from_kaggle(
        bucket: storage.Bucket,
        kaggle_path: str, 
        bucket_prefix: str, 
        local_zip_name: str,
        overwrite: bool = False
    ):

    blob = bucket.blob(bucket_prefix)

    if blob.exists() and not overwrite:
        logger.info("Skipping %s (already exists). Use --overwrite argument to overwrite.", bucket_prefix)
        return
    
    dataset_dir = download_dataset(source_path=kaggle_path)
   
    zip_path = zip_images(dataset_dir, local_zip_name)

    if zip_path is None:
        return

    try:
        upload_file(zip_path, bucket_prefix)
    finally:
        # clean up local zip to save disk space if the script runs multiple times
        zip_path.unlink(missing_ok=True)