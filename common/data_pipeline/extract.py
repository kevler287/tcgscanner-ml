import os
import logging
from pathlib import Path
from typing import List
from dotenv import load_dotenv
from google.cloud import storage

load_dotenv()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

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

    # Normalize prefix so relative paths are computed correctly
    prefix_norm = prefix.rstrip("/") + "/" if prefix else ""

    for i, blob in enumerate(blobs, 1):
        # Skip empty "directory marker" objects, if any
        if blob.name.endswith("/"):
            continue

        # Mirror the object name structure below the prefix, e.g.
        # prefix="yugioh" and blob.name="yugioh/first_ed_0/normal_en.jpeg"
        # -> relative_path="first_ed_0/normal_en.jpeg"
        relative_path = blob.name[len(prefix_norm):] if prefix_norm else blob.name

        local_path = local_dir / relative_path
        local_path.parent.mkdir(parents=True, exist_ok=True)

        blob.download_to_filename(str(local_path))
        logger.info("[%d/%d] Downloaded %s", i, len(blobs), relative_path)

    logger.info("Download complete: %d files → %s", len(blobs), local_dir)

def main(bucket_name: str, prefixes: List[str], dest_dir: Path):
    client = storage.Client()
    bucket = client.bucket(bucket_name=bucket_name)

    for prefix in prefixes:
        download_from_gcs(bucket, prefix, dest_dir / prefix)
