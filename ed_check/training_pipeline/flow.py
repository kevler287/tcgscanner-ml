import argparse
from pathlib import Path
import os
from dotenv import load_dotenv
from prefect import flow, task
from ed_check.config import CONFIG

load_dotenv()

@task(name="Extract")
def extract(dataset_version: str):
    """Download dataset from GCS to local dir."""
    from common.training_pipeline.extract import extract as run_extract
    run_extract(
        dataset_version=dataset_version,
        bucket_name=CONFIG.bucket.name, 
        blob_prefix=CONFIG.bucket.pf_datasets,
        dest_dir=Path(os.environ.get("LOCAL_DATA_DIR")) / CONFIG.model_prefix / dataset_version,
        creds_path="credentials/training_pipeline_credentials.json"
    )

@task(name="Train")
def train(dataset_version: str):
    """Generate synthetic YOLO training dataset from raw cards and backgrounds."""
    from ed_check.training_pipeline.train import train
    train(data_dir=Path(os.environ.get("LOCAL_DATA_DIR")) / CONFIG.model_prefix / dataset_version)

# @task(name="Evaluate")
# def evaluate():
#     """Generate synthetic YOLO training dataset from raw cards and backgrounds."""
#     from ed_check.training_pipeline.evaluate import run_crop_transform
#     run_crop_transform()

# @task(name="Load")
# def load():
#     """Zip dataset and upload back to Google Cloud Storage."""
#     from common.training_pipeline.load import load as run_load
#     run_load(
#         bucket_name=CONFIG.bucket.name, 
#         source_dir=CONFIG.transform_cfg.output_dir, 
#         dest_dir=CONFIG.bucket.pf_datasets
#     )


@flow(name="Edition Checker Training Pipeline")
def training_pipeline(dataset_version: str):
    extract(dataset_version)
    train(dataset_version)
    # evaluate()
    # load()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perfrom training on selected dataset")
    parser.add_argument("--version", required=True, help="Dataset version e.g. v1")
    args = parser.parse_args()

    training_pipeline(dataset_version=args.version)