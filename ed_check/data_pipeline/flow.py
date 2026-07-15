from prefect import flow, task
from ed_check.config import CONFIG

@task(name="Extract")
def extract():
    """Download cards and backgrounds from GCS to local cache."""
    from common.data_pipeline.extract import main as run_extract
    run_extract(bucket_name=CONFIG.bucket.name, prefixes=[CONFIG.pf_ed_types])


@task(name="Transform")
def transform():
    """Generate synthetic YOLO training dataset from raw cards and backgrounds."""
    from ed_check.data_pipeline.transform import run_crop_transform
    run_crop_transform()


@task(name="Load")
def load(dataset_version: str):
    """Zip dataset and upload back to Google Cloud Storage."""
    from common.data_pipeline.load import load as run_load
    run_load(dataset_version)


@flow(name="Card Segmentation Data Pipeline")
def data_pipeline():
    extract()
    transform()
    # load(dataset_version)


if __name__ == "__main__":
    data_pipeline()