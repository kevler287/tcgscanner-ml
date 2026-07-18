"""
Generic upload helpers for GCS (weights) and BigQuery (table rows).
"""

import logging
from pathlib import Path

from google.cloud import bigquery, storage
from google.oauth2 import service_account

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import re

import re

import re

def get_model_version(
    dataset_version: str,
    testset_version: str,
    bucket_name: str,
    model_prefix: str,
    creds: service_account.Credentials,
) -> str:
    """
    Build the next model_version for a given dataset_version/testset_version
    combination, in the form "{testset_num}.{dataset_num}.{config_version}"
    (e.g. dataset_version="v1", testset_version="t2" -> "2.1.0").

    Looks for existing files "{prefix}*.pt" directly under
    gs://bucket_name/model_prefix/, takes the highest existing
    config_version, and increments it by 1. Starts at 0 if none exist.

    Expects files stored as gs://bucket_name/model_prefix/{model_version}.pt
    e.g. "models/ed_check/1.1.0.pt".
    """
    dataset_num = re.sub(r"\D", "", dataset_version)
    testset_num = re.sub(r"\D", "", testset_version)
    prefix = f"{testset_num}.{dataset_num}."
    full_prefix = f"{model_prefix.rstrip('/')}/{prefix}"

    client = storage.Client(credentials=creds)
    blobs = client.list_blobs(bucket_name, prefix=full_prefix)

    config_versions = []
    for blob in blobs:
        filename = blob.name.rsplit("/", 1)[-1]
        if not filename.endswith(".pt"):
            continue
        version_str = filename[:-len(".pt")]
        suffix = version_str[len(prefix):]
        if suffix.isdigit():
            config_versions.append(int(suffix))

    next_config_version = max(config_versions, default=-1) + 1
    return f"{prefix}{next_config_version}"


def upload_weights(
    files: dict[str, Path],
    bucket_name: str,
    blob_prefix: str,
    creds: service_account.Credentials,
    overwrite: bool = False,
) -> dict[str, str]:
    """
    Upload local files to a GCS bucket under a given prefix.

    Args:
        files: mapping of {blob_name: local_path}, e.g. {"best.pt": run_dir / "weights" / "best.pt"}
        bucket_name: target GCS bucket
        blob_prefix: prefix under which files are stored, e.g. "models/v1/"
                      (trailing slash recommended, will be added if missing)
        creds: service account credentials
        overwrite: if False, raises if any target blob already exists

    Returns:
        dict of {blob_name: "gs://bucket/path"} for all uploaded files
    """
    if not blob_prefix.endswith("/"):
        blob_prefix += "/"

    client = storage.Client(credentials=creds)
    bucket = client.bucket(bucket_name)

    if not overwrite:
        for name in files:
            blob_path = blob_prefix + name
            if bucket.blob(blob_path).exists():
                raise FileExistsError(
                    f"Blob already exists at gs://{bucket_name}/{blob_path}"
                )

    uploaded = {}
    for name, local_path in files.items():
        local_path = Path(local_path)
        if not local_path.exists():
            raise FileNotFoundError(f"{local_path} not found. Aborting upload.")

        blob_path = blob_prefix + name
        logger.info("Uploading %s -> gs://%s/%s", local_path, bucket_name, blob_path)
        bucket.blob(blob_path).upload_from_filename(str(local_path))
        uploaded[name] = f"gs://{bucket_name}/{blob_path}"

    logger.info("Weights upload complete.")
    return uploaded


def upload_table_rows(
    bq_datatset: str, 
    bq_table: str,
    rows: list[dict],
    creds: service_account.Credentials,
) -> None:
    """
    Insert rows into a BigQuery table.

    Args:
        table_id: fully qualified table id, e.g. "project.dataset.table"
        rows: list of JSON-serializable dicts, one per row
        creds: service account credentials
    """
    if not rows:
        logger.warning("No rows to insert into %s, skipping.", bq_table)
        return

    client = bigquery.Client(credentials=creds)
    errors = client.insert_rows_json(f"{client.project}.{bq_datatset}.{bq_table}", rows)
    if errors:
        raise RuntimeError(f"Failed to insert rows into {bq_table}: {errors}")

    logger.info("Inserted %d rows into %s", len(rows), bq_table)


def safe_float(value):
    """Convert value to float, returning None if conversion fails."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_int(value):
    """Convert value to int, returning None if conversion fails."""
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None