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

def extract_raw(bucket_name: str, prefixes: List[str], dest_dir: Path):
    client = storage.Client()
    bucket = client.bucket(bucket_name=bucket_name)

    for prefix in prefixes:
        download_files(bucket, prefix, dest_dir / prefix)

def extract_zip(bucket_name: str, blob_paths: List[str], dest_dir: Path):
    client = storage.Client()
    bucket = client.bucket(bucket_name=bucket_name)

    for blob_path in blob_paths:
        local_path = dest_dir / blob_path
        download_file(bucket, blob_path, local_path)
        unzip_file(zip_path=local_path, extract_to=dest_dir)

#TODO refactor when used again
def extract_yolo(
    dataset_version: str,
    bucket: storage.Bucket,
    blob_prefix: str,
    dest_dir: Path,
) -> str:
    zip_path = download_file(dataset_version, bucket, blob_prefix, dest_dir)
    dataset_dir = unzip_file(zip_path, dest_dir)

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
