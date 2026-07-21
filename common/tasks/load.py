import logging
from pathlib import Path
from google.cloud import storage
from common.helper.gcs_io import upload_file
from common.helper.zip_helpers import zip_dir

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def load_zip_to_gcs(bucket_name: str, src_dir: str, dest_prefix: str, overwrite: bool = False):
    client = storage.Client()
    bucket = client.bucket(bucket_name=bucket_name)

    src_dir = Path(src_dir)

    if not src_dir.exists():
        raise FileNotFoundError(f"Output dir not found: {src_dir}")

    zip_path = zip_dir(src_dir)
    gcs_path = upload_file(bucket, zip_path, dest_prefix, overwrite=overwrite)

    logger.info("Done. Dataset available at %s", gcs_path)

    if zip_path.exists():
        zip_path.unlink()
        logger.info("Deleted local zip %s", zip_path)

def safe_float(value):
    """Convert value to float, returning None if conversion fails."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_int(value):
    """Convert value to int, returning None if conversion fails."""
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None