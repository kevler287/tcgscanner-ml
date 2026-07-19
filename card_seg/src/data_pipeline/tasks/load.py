import argparse
import logging
import zipfile
from pathlib import Path
from google.cloud import storage
from card_seg.src.config import CONFIG

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

OUTPUT_DIR  = CONFIG.transform.output_dir
BUCKET_NAME = CONFIG.bucket.name
PF_DATASETS = CONFIG.bucket.pf_datasets


def zip_dataset(output_dir: Path, version: str) -> Path:
    zip_path = output_dir.parent / f"{version}.zip"
    logger.info("Zipping %s → %s", output_dir, zip_path)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in output_dir.rglob("*"):
            if file.is_file():
                zf.write(file, file.relative_to(output_dir))

    logger.info("Zip created (%.1f MB)", zip_path.stat().st_size / 1024 / 1024)
    return zip_path


def upload_to_gcs(zip_path: Path, version: str) -> str:
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    blob_path = PF_DATASETS + f"{version}.zip"
    blob = bucket.blob(blob_path)

    if blob.exists():
        raise FileExistsError(
            f"Dataset {version} already exists at gs://{BUCKET_NAME}/{blob_path}. "
            "Choose a different version or delete the existing one first."
        )

    logger.info("Uploading to gs://%s/%s", BUCKET_NAME, blob_path)
    blob.upload_from_filename(str(zip_path))
    logger.info("Upload complete.")

    return f"gs://{BUCKET_NAME}/{blob_path}"

def load(dataset_version: str):
    output_dir = Path(OUTPUT_DIR)

    if not output_dir.exists():
        raise FileNotFoundError(f"Output dir not found: {output_dir}")

    zip_path = zip_dataset(output_dir, dataset_version)
    gcs_path = upload_to_gcs(zip_path, dataset_version)

    logger.info("Done. Dataset available at %s", gcs_path)

    if zip_path.exists():
        zip_path.unlink()
        logger.info("Deleted local zip %s", zip_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load card_seg dataset to GCS.")
    parser.add_argument("--version", required=True, help="Dataset version e.g. v1")
    args = parser.parse_args()

    load(dataset_version=args.version)