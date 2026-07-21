"""
Load step for the training pipeline: uploads model weights to GCS and
writes training_epochs / model_runs rows to BigQuery.

Uses the generic helpers from common.training_pipeline.load
(upload_weights, upload_table_rows, safe_float, safe_int).
"""

import csv
import logging
from datetime import datetime, timezone
from pathlib import Path

from google.oauth2 import service_account
from ed_check.src.config import CONFIG
from common.tasks.load_ml import *

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def read_training_epochs(metrics_csv: Path, model_version: str) -> list[dict]:
    """
    Read the per-epoch metrics.csv written by train() and enrich each row
    with model_version, converting fields to their proper types.
    """
    rows = []
    with open(metrics_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "model_version": model_version,
                "epoch":         safe_int(row.get("epoch")),
                "train_loss":    safe_float(row.get("train_loss")),
                "train_acc":     safe_float(row.get("train_acc")),
                "val_loss":      safe_float(row.get("val_loss")),
                "val_acc":       safe_float(row.get("val_acc")),
                "lr":            safe_float(row.get("lr")),
                "duration_sec":  safe_float(row.get("duration_sec")),
            })

    if not rows:
        logger.warning("No rows found in %s", metrics_csv)

    return rows


def build_model_run_row(
    model_version: str,
    num_epochs: int,
    eval_metrics: dict,
) -> dict:
    """
    Flatten the dict returned by evaluate() (accuracy + per-class
    precision/recall/f1/support) into a single model_runs row.
    """
    row = {
        "model_version": model_version,
        "timestamp":     datetime.now(timezone.utc).isoformat(),
        "epochs":        num_epochs,
        "accuracy":      eval_metrics.get("accuracy"),
    }

    for metric_name in ("precision", "recall", "f1", "support"):
        per_class = eval_metrics.get(metric_name, {})
        for class_name, value in per_class.items():
            row[f"{metric_name}_{class_name}"] = value

    return row


def load(
    model_version: str,
    weights_path: Path,
    train_csv_path: Path,
    eval_metrics: dict,
    creds: service_account.Credentials,
) -> dict:
    """
    Upload model weights + write training_epochs and model_runs rows.

    Args:
        model_version: unique identifier for this training run
        weights_path: Path to best.pt,
        train_csv_path: Path to csv file with metrics per epoch,
        eval_metrics: dict returned by evaluate() (accuracy, precision, recall, f1, support)
        creds: service account credentials

    Returns:
        dict with "weight_paths" (gs:// paths) and "num_epochs"
    """

    upload_weights(
        files={f"{model_version}.pt": weights_path},
        bucket_name=CONFIG.bucket.name,
        blob_prefix=CONFIG.bucket.pf_models,
        creds=creds,
    )

    epoch_rows = read_training_epochs(train_csv_path, model_version)
    upload_table_rows(CONFIG.model_prefix, CONFIG.bq_training_epochs, epoch_rows, creds)

    model_run_row = build_model_run_row(
        model_version=model_version,
        num_epochs=len(epoch_rows),
        eval_metrics=eval_metrics,
    )
    upload_table_rows(CONFIG.model_prefix, CONFIG.bq_model_runs, [model_run_row], creds)

    logger.info("Load complete for model_version=%s", model_version)