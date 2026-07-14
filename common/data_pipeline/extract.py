import os
import logging
from pathlib import Path
from typing import List
from dotenv import load_dotenv
from google.cloud import storage

load_dotenv()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

LOCAL_DATA_DIR  = Path(os.getenv("LOCAL_DATA_DIR"))

def download_from_gcs(bucket, prefix: str, local_dir: Path):
    if local_dir.exists() and any(local_dir.iterdir()):
        logger.info("Local data already exists at %s, skipping download.", local_dir)
        return

    logger.info("Downloading from GCS: %s → %s", prefix, local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    blobs = list(bucket.list_blobs(prefix=prefix))

    if not blobs:
        logger.warning("No files found in GCS under %s", prefix)
        return

    for i, blob in enumerate(blobs, 1):
        filename   = Path(blob.name).name
        local_path = local_dir / filename
        blob.download_to_filename(str(local_path))
        logger.info("[%d/%d] Downloaded %s", i, len(blobs), filename)

    logger.info("Download complete: %d files → %s", len(blobs), local_dir)

def main(bucket_name: str, prefixes: List[str]):
    client = storage.Client()
    bucket = client.bucket(bucket_name=bucket_name)

    for prefix in prefixes:
        download_from_gcs(bucket, prefix, LOCAL_DATA_DIR / prefix)
