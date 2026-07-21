import logging
from pathlib import Path
import re
from google.cloud import storage
from google.oauth2 import service_account

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def upload_file(bucket: storage.Bucket, file_path: Path, blob_path: str, overwrite: bool = False):
    """Uploads a file to GCS."""

    blob = bucket.blob(blob_path)

    if blob.exists() and not overwrite:
        logger.info("Skipping %s (already exists)", blob_path)
        return

    logger.info("Uploading %s to gs://%s/%s ...", file_path.name, bucket.name, blob_path)

    blob.upload_from_filename(str(file_path))

    logger.info("Upload complete: gs://%s/%s", bucket.name, blob_path)

def download_file(
    bucket: storage.Bucket,
    blob_path: str,
    dest_dir: Path,
) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    file_path = dest_dir / blob_path

    if file_path.exists():
        logger.info("Zip already present at %s, skipping download.", file_path)
        return file_path

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

def get_model_version(
    dataset_version: str,
    testset_version: str,
    bucket_name: str,
    model_prefix: str,
    creds: service_account.Credentials,
) -> str:
    """
    Build the next model_version for a given dataset_version/testset_version
    combination, in the form "{testset_num}.{dataset_num}.{config_version}"
    (e.g. dataset_version="v1", testset_version="t2" -> "2.1.0").

    Looks for existing files "{prefix}*.pt" directly under
    gs://bucket_name/model_prefix/, takes the highest existing
    config_version, and increments it by 1. Starts at 0 if none exist.

    Expects files stored as gs://bucket_name/model_prefix/{model_version}.pt
    e.g. "models/ed_check/1.1.0.pt".
    """
    dataset_num = re.sub(r"\D", "", dataset_version)
    testset_num = re.sub(r"\D", "", testset_version)
    prefix = f"{testset_num}.{dataset_num}."
    full_prefix = f"{model_prefix.rstrip('/')}/{prefix}"

    client = storage.Client(credentials=creds)
    blobs = client.list_blobs(bucket_name, prefix=full_prefix)

    config_versions = []
    for blob in blobs:
        filename = blob.name.rsplit("/", 1)[-1]
        if not filename.endswith(".pt"):
            continue
        version_str = filename[:-len(".pt")]
        suffix = version_str[len(prefix):]
        if suffix.isdigit():
            config_versions.append(int(suffix))

    next_config_version = max(config_versions, default=-1) + 1
    return f"{prefix}{next_config_version}"