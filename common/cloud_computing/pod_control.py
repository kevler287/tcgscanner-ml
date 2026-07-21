import os
import json
import traceback
import logging
from pathlib import Path

from google.oauth2 import service_account
from google.cloud import storage
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def save_logs(bucket, src_path: Path, dest_path: str):
    blob = bucket.blob(dest_path)    
    blob.upload_from_filename(src_path)
    logger.info("Logs uploaded to GCP")

def stop_pod():
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