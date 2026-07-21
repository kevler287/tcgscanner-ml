import logging
from pathlib import Path
from google.cloud import storage

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def upload_file(bucket: storage.Bucket, file_path: Path, bucket_prefix: str, overwrite: bool = False):
    """Uploads a file to GCS."""

    blob = bucket.blob(bucket_prefix)

    if blob.exists() and not overwrite:
        logger.info("Skipping %s (already exists)", bucket_prefix)
        return

    logger.info("Uploading %s to gs://%s/%s ...", file_path.name, bucket.name, bucket_prefix)

    blob.upload_from_filename(str(file_path))

    logger.info("Upload complete: gs://%s/%s", bucket.name, bucket_prefix)

def download_file(
    file_name: str,
    bucket: storage.Bucket,
    blob_prefix: str,
    dest_dir: Path,
) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    file_path = dest_dir / file_name

    if file_path.exists():
        logger.info("Zip already present at %s, skipping download.", file_path)
        return file_path

    blob_path = f"{blob_prefix}{file_name}"
    blob = bucket.blob(blob_path)

    if not blob.exists():
        raise FileNotFoundError(f"Dataset not found: gs://{bucket.name}/{blob_path}")

    logger.info("Downloading gs://%s/%s -> %s", bucket.name, blob_path, file_path)
    blob.download_to_filename(str(file_path))
    logger.info("Download complete (%.1f MB)", file_path.stat().st_size / 1024 / 1024)

    return file_path

def download_files(bucket, prefix: str, local_dir: Path):
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