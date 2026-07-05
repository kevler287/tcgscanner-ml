import logging
import zipfile
from pathlib import Path

from google.cloud import storage
from google.oauth2 import service_account
import yaml

from card_seg.config import CONFIG

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

BUCKET_NAME = CONFIG.bucket.name
PF_DATASETS = CONFIG.bucket.pf_datasets


def download_dataset(dataset_version: str, local_dir: Path, creds: service_account.Credentials) -> Path:
    local_dir.mkdir(parents=True, exist_ok=True)
    zip_path = local_dir / f"{dataset_version}.zip"

    if zip_path.exists():
        logger.info("Zip already present at %s, skipping download.", zip_path)
        return zip_path

    client = storage.Client(credentials=creds)
    bucket = client.bucket(BUCKET_NAME)

    blob_path = PF_DATASETS + f"{dataset_version}.zip"
    blob = bucket.blob(blob_path)

    if not blob.exists():
        raise FileNotFoundError(
            f"Dataset not found: gs://{BUCKET_NAME}/{blob_path}"
        )

    logger.info("Downloading gs://%s/%s → %s", BUCKET_NAME, blob_path, zip_path)
    blob.download_to_filename(str(zip_path))
    logger.info("Download complete (%.1f MB)", zip_path.stat().st_size / 1024 / 1024)

    return zip_path


def unzip_dataset(zip_path: Path, extract_to: Path) -> Path:
    logger.info("Extracting %s → %s", zip_path, extract_to)
    extract_to.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)

    logger.info("Extraction complete")

    return extract_to


def extract(dataset_version: str, creds: service_account.Credentials, work_dir: Path) -> str:
    zip_path = download_dataset(dataset_version, work_dir, creds)
    dataset_dir = unzip_dataset(zip_path, work_dir)

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