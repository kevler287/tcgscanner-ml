from dataclasses import dataclass
import os
import json
import traceback
import logging
from pathlib import Path

from google.oauth2 import service_account
from google.cloud import storage
import requests

from card_seg.training_pipeline.tasks.extract import extract
from card_seg.training_pipeline.tasks.train import train
from card_seg.training_pipeline.tasks.evaluate import evaluate
from card_seg.training_pipeline.tasks.load import load
from card_seg.config import CONFIG

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

    dataset_version = os.environ["DATASET_VERSION"]
    model_version = os.environ["MODEL_VERSION"]

    try:
        logger.info("=== Step 1/4: Extract ===")
        data_yaml = extract(
            dataset_version=dataset_version,
            creds=creds,
            work_dir=dirs.dataset
        )
        logger.info("=== Step 2/4: Train ===")
        run_dir = train(
            data_yaml=data_yaml,
            run_name=model_version,
            work_dir=dirs.runs
        )
        logger.info("=== Step 3/4: Evaluate ===")
        eval_metrics = evaluate(
            run_dir=run_dir,
            data_yaml_path=data_yaml
        )
        logger.info("=== Step 4/4: Load ===")
        weight_paths = load(
            model_version=model_version,
            dataset_version=dataset_version,
            data_yaml=data_yaml,
            run_dir=run_dir,
            eval_metrics=eval_metrics,
            creds=creds
        )
        logger.info("=== Done ===")
        logger.info("Weights uploaded to: %s", weight_paths)
    except Exception as e:
        logger.error("Pipeline failed: %s\n%s", e, traceback.format_exc())
    finally:
        try:
            _save_logs(creds=creds, log_path=dirs.logs, model_version=model_version)
        except Exception as e:
            logger.error("Failed to upload logs: %s", e)
        _stop_pod()

def _save_logs(creds: service_account.Credentials, log_path: Path, model_version: str):
    client = storage.Client(credentials=creds)
    bucket = client.bucket(CONFIG.bucket_name)
    blob = bucket.blob(f"models/{CONFIG.model_prefix}/{model_version}/training.log")    
    blob.upload_from_filename(log_path)
    logger.info("Logs uploaded to GCP")

def _stop_pod():
    pod_id = os.getenv("RUNPOD_POD_ID")
    api_key = os.getenv("RUNPOD_API_KEY")

    if not pod_id or not api_key:
        logger.warning("RUNPOD_POD_ID or RUNPOD_API_KEY not set, skipping auto-stop")
        return

    query = """
    mutation stopPod($podId: String!) {
        podStop(input: { podId: $podId }) {
            id
        }
    }
    """

    response = requests.post(
        "https://api.runpod.io/graphql",
        json={"query": query, "variables": {"podId": pod_id}},
        headers={"Authorization": f"Bearer {api_key}"}
    )
    response.raise_for_status()
    logger.info("Pod stopped successfully")


if __name__ == "__main__":
    main()