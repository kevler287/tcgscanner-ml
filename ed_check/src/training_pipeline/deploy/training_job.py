from dataclasses import dataclass
import os
import json
import traceback
import logging
from pathlib import Path

from google.oauth2 import service_account
from google.cloud import storage

from common.cloud_computing.pod_fs import PodFileSystem
from common.tasks.extract import extract_zip
from ed_check.src.training_pipeline.tasks.train import train
from ed_check.src.training_pipeline.tasks.evaluate import evaluate
from ed_check.src.training_pipeline.tasks.load import load
from ed_check.src.config import CONFIG
from common.cloud_computing.pod_control import save_logs, stop_pod
from common.tasks.load_ml import get_model_version

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    fs = PodFileSystem()

    file_handler = logging.FileHandler(fs.log_path)
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
        extract_zip(
            bucket=bucket, 
            blob_paths=[
                CONFIG.bucket.pf_datasets / f"{dataset_version}.zip",
                CONFIG.pf_test_data / f"{testset_version}.zip",
            ],
            work_dir=fs.data_dir / dataset_version,
        )

        logger.info("=== Step 2/4: Train ===")
        weights_path, train_csv_path = train(
            data_dir=fs.data_dir / dataset_version,
            results_dir=fs.run_dir
        )

        logger.info("=== Step 3/4: Evaluate ===")
        eval_metrics = evaluate(
            data_dir=fs.data_dir / testset_version,
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
                src_path=fs.log_path,
                dest_path=CONFIG.bucket.pf_logs + model_version + ".log"
            )
        except Exception as e:
            logger.error("Failed to upload logs: %s", e)
        stop_pod()


if __name__ == "__main__":
    main()