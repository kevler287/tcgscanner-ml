from dataclasses import dataclass
import os
import json
import traceback
import logging
from pathlib import Path

from google.oauth2 import service_account
from google.cloud import storage

from common.training_pipeline.extract import extract
from ed_check.src.training_pipeline.tasks.train import train
from ed_check.src.training_pipeline.tasks.evaluate import evaluate
from ed_check.src.training_pipeline.tasks.load import load
from ed_check.src.config import CONFIG
from common.training_pipeline.pod_control import save_logs, stop_pod
from common.training_pipeline.load import get_model_version

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class WorkDirs:
    root:    Path = Path("/app")
    dataset: Path = Path("/app/dataset")
    runs:    Path = Path("/app/runs")
    logs:    Path = Path("/app/training.log")

def main():
    dirs = WorkDirs()
    dirs.dataset.mkdir(parents=True, exist_ok=True)
    dirs.runs.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(dirs.logs)
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logging.getLogger().addHandler(file_handler)

    creds_json = os.environ["GOOGLE_APPLICATION_CREDENTIALS_JSON"]
    creds = service_account.Credentials.from_service_account_info(json.loads(creds_json))
    os.environ["GOOGLE_CLOUD_PROJECT"] = creds.project_id

    client = storage.Client(credentials=creds)
    bucket = client.bucket(CONFIG.bucket.name)

    dataset_version = os.environ["DATASET_VERSION"]
    testset_version = os.environ["TESTSET_VERSION"]
    model_version = get_model_version(
        dataset_version=dataset_version,
        testset_version=testset_version,
        bucket_name=CONFIG.bucket.name,
        model_prefix=CONFIG.bucket.pf_models,
        creds=creds
    )
    logger.info(f"Executing Training Pipeline for {CONFIG.model_prefix}/v{model_version}")

    try:
        logger.info("=== Step 1/4: Extract ===")
        extract(
            file_name=dataset_version,
            bucket=bucket, 
            blob_prefix=CONFIG.bucket.pf_datasets,
            dest_dir=dirs.dataset / dataset_version,
        )
        extract(
            file_name=testset_version,
            bucket_name=bucket, 
            blob_prefix=CONFIG.pf_test_data,
            dest_dir=dirs.dataset / testset_version,
        )

        logger.info("=== Step 2/4: Train ===")
        weights_path, train_csv_path = train(
            data_dir=dirs.dataset / dataset_version,
            results_dir=dirs.runs
        )

        logger.info("=== Step 3/4: Evaluate ===")
        eval_metrics = evaluate(
            data_dir=dirs.dataset / testset_version,
            checkpoint_path=weights_path
        )

        logger.info("=== Step 4/4: Load ===")
        load(
            model_version=model_version,
            weights_path=weights_path,
            train_csv_path=train_csv_path,
            eval_metrics=eval_metrics,
            creds=creds
        )
        logger.info("=== Done ===")
    except Exception as e:
        logger.error("Pipeline failed: %s\n%s", e, traceback.format_exc())
    finally:
        try:
            file_handler.flush()
            file_handler.close()
            save_logs(
                bucket=bucket,
                src_path=dirs.logs,
                dest_path=CONFIG.bucket.pf_logs + model_version + ".log"
            )
        except Exception as e:
            logger.error("Failed to upload logs: %s", e)
        stop_pod()


if __name__ == "__main__":
    main()