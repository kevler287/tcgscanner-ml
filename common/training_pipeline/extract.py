"""
Generic functions to download a dataset zip from Google Cloud Storage,
extract it, and patch the `path` field in its data.yaml to point to
the local extraction directory.

Usage (import and call directly):
    from extract_dataset import extract, load_credentials

    creds = load_credentials(Path("./service-account.json"))  # or None for ADC
    data_yaml_path = extract(
        dataset_version="v1.2.0",
        bucket_name="my-bucket",
        blob_prefix="datasets/pf/",
        work_dir=Path("./data"),
        creds=creds,
    )
"""

import logging
import zipfile
from pathlib import Path

import yaml
from google.cloud import storage
from google.oauth2 import service_account

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def load_credentials(credentials_path: Path | None) -> service_account.Credentials | None:
    """Loads credentials from a service account key file, or None to fall back to ADC."""
    if credentials_path is None:
        return None
    return service_account.Credentials.from_service_account_file(str(credentials_path))


def download_dataset(
    dataset_version: str,
    bucket_name: str,
    blob_prefix: str,
    dest_dir: Path,
    creds: service_account.Credentials | None,
) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / f"{dataset_version}.zip"

    if zip_path.exists():
        logger.info("Zip already present at %s, skipping download.", zip_path)
        return zip_path

    client = storage.Client(credentials=creds)
    bucket = client.bucket(bucket_name)

    blob_path = f"{blob_prefix}{dataset_version}.zip"
    blob = bucket.blob(blob_path)

    if not blob.exists():
        raise FileNotFoundError(f"Dataset not found: gs://{bucket_name}/{blob_path}")

    logger.info("Downloading gs://%s/%s -> %s", bucket_name, blob_path, zip_path)
    blob.download_to_filename(str(zip_path))
    logger.info("Download complete (%.1f MB)", zip_path.stat().st_size / 1024 / 1024)

    return zip_path


def unzip_dataset(zip_path: Path, extract_to: Path) -> Path:
    logger.info("Extracting %s -> %s", zip_path, extract_to)
    extract_to.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)

    logger.info("Extraction complete")

    return extract_to


def extract(
    dataset_version: str,
    bucket_name: str,
    blob_prefix: str,
    dest_dir: Path,
    creds_path: str,
) -> str:
    creds = load_credentials(credentials_path=creds_path)
    zip_path = download_dataset(dataset_version, bucket_name, blob_prefix, dest_dir, creds)
    dataset_dir = unzip_dataset(zip_path, dest_dir)
    return dataset_dir

def extract_yolo(
    dataset_version: str,
    bucket_name: str,
    blob_prefix: str,
    dest_dir: Path,
    creds_path: str,
) -> str:
    creds = load_credentials(credentials_path=creds_path)
    zip_path = download_dataset(dataset_version, bucket_name, blob_prefix, dest_dir, creds)
    dataset_dir = unzip_dataset(zip_path, dest_dir)

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