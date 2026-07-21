import os
from pathlib import Path

from prefect import flow, task
from ed_check.src.config import CONFIG

WORK_DIR = Path(os.environ["LOCAL_DATA_DIR"])
TRANSFORM_TO = WORK_DIR / "datasets" / CONFIG.model_prefix

@task(name="Extract")
def extract():
    """Download cards and backgrounds from GCS to local cache."""
    from common.tasks.extract import extract_raw as run_extract
    run_extract(
        bucket_name=CONFIG.bucket.name, 
        prefixes=[CONFIG.pf_ed_types],
        dest_dir=WORK_DIR
    )


@task(name="Transform")
def transform():
    """Generate synthetic YOLO training dataset from raw cards and backgrounds."""
    from ed_check.src.data_pipeline.transform import run_crop_transform
    run_crop_transform(
        src_dir=WORK_DIR / CONFIG.pf_ed_types,
        dest_dir=TRANSFORM_TO
    )


@task(name="Load")
def load():
    """Zip dataset and upload back to Google Cloud Storage."""
    from common.tasks.load import load_zip_to_gcs as run_load
    run_load(
        bucket_name=CONFIG.bucket.name, 
        src_dir=TRANSFORM_TO, 
        dest_prefix=CONFIG.bucket.pf_datasets
    )


@flow(name="Edition Checker Data Pipeline")
def data_pipeline():
    extract()
    transform()
    load()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--v", required=True)
    args = parser.parse_args()

    TRANSFORM_TO = TRANSFORM_TO / args.v

    data_pipeline()