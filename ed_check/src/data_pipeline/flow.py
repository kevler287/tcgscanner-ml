import os
from pathlib import Path

from prefect import flow, task
from ed_check.src.config import CONFIG

WORK_DIR = Path(os.environ["LOCAL_DATA_DIR"]) / CONFIG.model_prefix / "data_pipeline"
EXTRACT_TO = WORK_DIR / "raw"
TRANSFORM_TO = WORK_DIR / "transformed"

@task(name="Extract")
def extract():
    """Download cards and backgrounds from GCS to local cache."""
    from common.data_pipeline.extract import main as run_extract
    run_extract(
        bucket_name=CONFIG.bucket.name, 
        prefixes=[CONFIG.pf_ed_types],
        dest_dir=EXTRACT_TO
    )


@task(name="Transform")
def transform():
    """Generate synthetic YOLO training dataset from raw cards and backgrounds."""
    from ed_check.src.data_pipeline.transform import run_crop_transform
    run_crop_transform(
        src_dir=EXTRACT_TO,
        dest_dir=TRANSFORM_TO
    )


@task(name="Load")
def load():
    """Zip dataset and upload back to Google Cloud Storage."""
    from common.data_pipeline.load import load as run_load
    run_load(
        bucket_name=CONFIG.bucket.name, 
        src_dir=TRANSFORM_TO, 
        dest_dir=CONFIG.bucket.pf_datasets
    )


@flow(name="Edition Checker Data Pipeline")
def data_pipeline():
    extract()
    transform()
    load()


if __name__ == "__main__":
    data_pipeline()