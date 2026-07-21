import os
import logging
from pathlib import Path
from typing import List
from dotenv import load_dotenv
from google.cloud import storage
import yaml

from common.helper.gcs_io import download_file, download_files
from common.helper.zip_helpers import unzip_file

load_dotenv()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def extract_raw(bucket_name: str, prefixes: List[str], work_dir: Path):
    client = storage.Client()
    bucket = client.bucket(bucket_name=bucket_name)

    for prefix in prefixes:
        download_files(bucket, prefix, work_dir / prefix)

def extract_zip(bucket_name: str, blob_paths: List[str], work_dir: Path):
    client = storage.Client()
    bucket = client.bucket(bucket_name=bucket_name)

    for blob_path in blob_paths:
        local_path = work_dir / blob_path
        download_file(bucket, blob_path, local_path)
        unzip_file(zip_path=local_path)

def extract_yolo(
    bucket: storage.Bucket,
    blob_path: str,
    work_dir: Path,
) -> str:
    local_path = work_dir / blob_path
    download_file(bucket, blob_path, local_path)
    dataset_dir = unzip_file(zip_path=local_path)

    data_yaml = dataset_dir / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found in extracted dataset: {data_yaml}")

    # replace path in data.yaml
    with open(data_yaml, "r") as f:
        data = yaml.safe_load(f)

    data["path"] = str(dataset_dir)

    with open(data_yaml, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)

    logger.info("Dataset ready at %s", dataset_dir)
    return str(data_yaml)
