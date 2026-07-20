import logging
import zipfile
from pathlib import Path
from dotenv import load_dotenv
import kagglehub
from google.cloud import storage
from card_seg.src.config import CONFIG

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

load_dotenv()

client = storage.Client()
bucket = client.bucket(CONFIG.bucket.name)


def download_dataset(source_path: str) -> Path:
    logger.info("Downloading dataset from Kaggle: %s", source_path)

    local_path = kagglehub.dataset_download(source_path)

    logger.info("Dataset available at: %s", local_path)

    return Path(local_path)


def zip_images(dataset_dir: Path, zip_name: str) -> Path | None:
    """Finds all images in dataset_dir and packs them into a local ZIP."""

    images = (
        list(dataset_dir.rglob("*.jpg"))
        + list(dataset_dir.rglob("*.jpeg"))
    )

    if not images:
        logger.warning("No images found in %s", dataset_dir)
        return None

    logger.info("Found %d images, zipping...", len(images))

    zip_path = Path("/tmp") / zip_name

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
        for i, path in enumerate(images, 1):
            # arcname = filename only, so the zip does not include
            # kagglehub's deep cache folder structure
            zf.write(path, arcname=path.name)

            if i % 1000 == 0 or i == len(images):
                logger.info("[%d/%d] Added %s to zip", i, len(images), path.name)

    size_mb = zip_path.stat().st_size / (1024 * 1024)
    logger.info("Zip created: %s (%.1f MB)", zip_path, size_mb)

    return zip_path


def upload_zip(zip_path: Path, bucket_prefix: str, overwrite: bool = False):
    """Uploads the ZIP as a single file to GCS."""

    blob = bucket.blob(bucket_prefix)

    if blob.exists() and not overwrite:
        logger.info("Skipping %s (already exists)", bucket_prefix)
        return

    logger.info("Uploading %s to gs://%s/%s ...", zip_path.name, CONFIG.bucket.name, bucket_prefix)

    blob.upload_from_filename(str(zip_path))

    logger.info("Upload complete: gs://%s/%s", CONFIG.bucket.name, bucket_prefix)


def zip_and_upload(dataset_dir: Path, bucket_prefix: str, zip_name: str):
    zip_path = zip_images(dataset_dir, zip_name)

    if zip_path is None:
        return

    try:
        upload_zip(zip_path, bucket_prefix)
    finally:
        # clean up local zip to save disk space if the script runs multiple times
        zip_path.unlink(missing_ok=True)


if __name__ == "__main__":

    # ygo card templates
    dataset_dir = download_dataset(source_path="yelbuzz/yugioh-card-images-and-data")
    zip_and_upload(
        dataset_dir,
        bucket_prefix=CONFIG.pf_ygo_cards,
        zip_name="ygo_cards.zip",
    )

    # background images
    dataset_dir = download_dataset(source_path="haaroonafroz/material-dataset-new")
    zip_and_upload(
        dataset_dir,
        bucket_prefix=CONFIG.pf_bg,
        zip_name="backgrounds.zip",
    )