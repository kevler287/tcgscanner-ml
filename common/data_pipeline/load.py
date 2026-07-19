import logging
import zipfile
from pathlib import Path
from google.cloud import storage

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def zip_dataset(source_dir: Path) -> Path:
    zip_path = source_dir.parent / f"{source_dir.name}.zip"
    logger.info("Zipping %s → %s", source_dir, zip_path)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in source_dir.rglob("*"):
            if file.is_file():
                zf.write(file, file.relative_to(source_dir))

    logger.info("Zip created (%.1f MB)", zip_path.stat().st_size / 1024 / 1024)
    return zip_path


def upload_to_gcs(bucket_name: str, zip_path: Path, dest_prefix: str) -> str:
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    blob_path = str(Path(dest_prefix) / zip_path.name)
    blob = bucket.blob(blob_path)

    if blob.exists():
        raise FileExistsError(
            f"{zip_path.name} already exists at gs://{bucket_name}/{blob_path}. "
            "Choose a different version or delete the existing one first."
        )

    logger.info("Uploading to gs://%s/%s", bucket_name, blob_path)
    blob.upload_from_filename(str(zip_path))
    logger.info("Upload complete.")

    return f"gs://{bucket_name}/{blob_path}"


def load(bucket_name: str, src_dir: str, dest_dir: str):
    src_dir = Path(src_dir)

    if not src_dir.exists():
        raise FileNotFoundError(f"Output dir not found: {src_dir}")

    zip_path = zip_dataset(src_dir)
    gcs_path = upload_to_gcs(bucket_name, zip_path, dest_dir)

    logger.info("Done. Dataset available at %s", gcs_path)

    if zip_path.exists():
        zip_path.unlink()
        logger.info("Deleted local zip %s", zip_path)
