import os
from pathlib import Path
from dotenv import load_dotenv
from prefect import flow, task
from card_seg.src.config import CONFIG

load_dotenv()

WORK_DIR = Path(os.environ["LOCAL_DATA_DIR"])
TRANSFORM_TO = WORK_DIR / "datasets" / CONFIG.model_prefix

@task(name="Extract")
def extract():
    """Download cards and backgrounds from GCS to local cache."""
    from common.tasks.extract import extract_zip
    extract_zip(
        bucket_name=CONFIG.bucket.name, 
        blob_paths=[CONFIG.pf_ygo_cards, CONFIG.pf_bg],
        work_dir=WORK_DIR
    )

@task(name="Transform")
def transform():
    """Generate synthetic YOLO training dataset from raw cards and backgrounds."""
    from card_seg.src.data_pipeline.transform import transform as run_transform
    run_transform(
        cards_dir=WORK_DIR / CONFIG.pf_ygo_cards,
        bg_dir=WORK_DIR / CONFIG.pf_bg,
        dest_dir=TRANSFORM_TO
    )


@task(name="Load")
def load(overwrite: bool = False):
    """Zip dataset and upload back to Google Cloud Storage."""
    from common.tasks.load import load_zip_to_gcs
    load_zip_to_gcs(
        bucket_name=CONFIG.bucket.name, 
        src_dir=TRANSFORM_TO, 
        dest_prefix=CONFIG.bucket.pf_datasets,
        overwrite=overwrite
    )


@flow(name="Card Segmentation Data Pipeline")
def data_pipeline(overwrite: bool = False):
    extract()
    transform()
    load(overwrite)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--v", required=True)
    parser.add_argument("--overwrite", required=False)
    args = parser.parse_args()

    TRANSFORM_TO = TRANSFORM_TO / args.v

    data_pipeline()