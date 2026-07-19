import os
import logging
from pathlib import Path
from dotenv import load_dotenv
from google.cloud import storage
from card_seg.src.config import CONFIG

load_dotenv()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

LOCAL_CARDS_DIR  = Path(os.getenv("LOCAL_CARDS_DIR"))
LOCAL_BG_DIR     = Path(os.getenv("LOCAL_BG_DIR"))


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

def main():
    client = storage.Client()
    bucket = client.bucket(CONFIG.bucket.name)

    download_from_gcs(bucket, CONFIG.pf_ygo_cards, LOCAL_CARDS_DIR)
    download_from_gcs(bucket, CONFIG.pf_bg,    LOCAL_BG_DIR)

if __name__ == "__main__":
    main()