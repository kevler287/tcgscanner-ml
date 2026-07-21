import logging
from pathlib import Path
from dotenv import load_dotenv
from google.cloud import storage
from card_seg.src.config import CONFIG
from common.setup.kaggle_to_gcs import upload_from_kaggle

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

load_dotenv()

client = storage.Client()
bucket = client.bucket(CONFIG.bucket.name)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", required=False)
    args = parser.parse_args()

    upload_from_kaggle(
        bucket=bucket,
        kaggle_path="yelbuzz/yugioh-card-images-and-data",
        bucket_prefix=CONFIG.pf_ygo_cards,
        local_zip_name="ygo_cards.zip",
        overwrite=args.overwrite
    )

    upload_from_kaggle(
        bucket=bucket,
        kaggle_path="yelbuzz/yugioh-card-images-and-data",
        bucket_prefix=CONFIG.pf_ygo_cards,
        local_zip_name="ygo_cards.zip",
        overwrite=args.overwrite
    )