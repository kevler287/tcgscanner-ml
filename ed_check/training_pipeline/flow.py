import argparse
from pathlib import Path
import os
from dotenv import load_dotenv
from prefect import flow, task
from ed_check.config import CONFIG
from google.oauth2 import service_account

load_dotenv()

WORK_DIR = Path(os.environ.get("LOCAL_DATA_DIR")) / CONFIG.model_prefix

def load_credentials(credentials_path: Path | None) -> service_account.Credentials | None:
    """Loads credentials from a service account key file, or None to fall back to ADC."""
    if credentials_path is None:
        return None
    return service_account.Credentials.from_service_account_file(str(credentials_path))

@task(name="Extract")
def extract(dataset_version: str, testset_version: str):
    """Download dataset from GCS to local dir."""
    from common.training_pipeline.extract import extract as run_extract
    run_extract(
        file_name=dataset_version,
        bucket_name=CONFIG.bucket.name, 
        blob_prefix=CONFIG.bucket.pf_datasets,
        dest_dir=WORK_DIR / dataset_version,
        creds_path="credentials/training_pipeline_credentials.json"
    )
    run_extract(
        file_name=testset_version,
        bucket_name=CONFIG.bucket.name, 
        blob_prefix=CONFIG.pf_test_data,
        dest_dir=WORK_DIR / testset_version,
        creds_path="credentials/training_pipeline_credentials.json"
    )

@task(name="Train")
def train(dataset_version: str):
    """Train ResNet18 on dataset"""
    from ed_check.training_pipeline.train import train
    train(data_dir=WORK_DIR / dataset_version)

@task(name="Evaluate")
def evaluate(testset_version: str):
    """Evaluate model on test set."""
    from ed_check.training_pipeline.evaluate import evaluate
    metrics = evaluate(
        data_dir=WORK_DIR / testset_version,
        checkpoint_path=CONFIG.train_cfg.results_dir + "/best.pt"
    )
    return metrics

@task(name="Load")
def load(dataset_version: str, testset_version: str, eval_metrics: dict):
    """Zip dataset and upload back to Google Cloud Storage."""
    from ed_check.training_pipeline.load import load as run_load
    run_load(
        data_version=dataset_version,
        test_version=testset_version,
        run_dir="ed_check/training_pipeline/results",
        eval_metrics=eval_metrics,
        creds=load_credentials("credentials/training_pipeline_credentials.json")
    )


@flow(name="Edition Checker Training Pipeline")
def training_pipeline(dataset_version: str, testset_version: str):
    extract(dataset_version, testset_version)
    train(dataset_version)
    eval_metrics = evaluate(testset_version)
    load(
        dataset_version=dataset_version,
        testset_version=testset_version,
        eval_metrics=eval_metrics, 
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perfrom training on selected dataset")
    parser.add_argument("--data_version", required=True, help="Dataset version e.g. v1")
    parser.add_argument("--test_version", required=True, help="Testset version e.g. t1")
    args = parser.parse_args()

    training_pipeline(dataset_version=args.data_version, testset_version=args.test_version)