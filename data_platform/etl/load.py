import logging
import argparse
import yaml
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from dotenv import load_dotenv
from google.cloud import storage, bigquery
from data_platform.config import CONFIG

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def get_classes() -> list:
    data_yaml = Path(CONFIG.transform.output_dir) / "data.yaml"
    with open(data_yaml) as f:
        data = yaml.safe_load(f)
    return list(data["names"].values())


def upload_model(storage_client, run_dir: Path, model_version: str):
    weights_path = run_dir / "weights" / "best.pt"
    if not weights_path.exists():
        logger.error("best.pt not found at %s", weights_path)
        return

    blob_name = f"{CONFIG.bucket.seg_models_prefix}{model_version}/best.pt"
    bucket    = storage_client.bucket(CONFIG.bucket.name)
    blob      = bucket.blob(blob_name)
    blob.upload_from_filename(str(weights_path))
    logger.info("Uploaded best.pt → gs://%s/%s", CONFIG.bucket.name, blob_name)


def parse_results(run_dir: Path):
    results_path = run_dir / "results.csv"
    if not results_path.exists():
        logger.error("results.csv not found at %s", results_path)
        return None

    df = pd.read_csv(results_path)
    df.columns = df.columns.str.strip()

    map50_col = [c for c in df.columns if "mAP50" in c and "95" not in c][0]
    best_row  = df.loc[df[map50_col].idxmax()]
    logger.info("Best epoch: %d  mAP50: %.4f", int(best_row["epoch"]), best_row[map50_col])
    return best_row


def parse_train_args(run_dir: Path) -> dict:
    args_path = run_dir / "args.yaml"
    if not args_path.exists():
        logger.warning("args.yaml not found, hyperparams will be None.")
        return {}
    with open(args_path) as f:
        return yaml.safe_load(f)


def upload_epochs_to_bigquery(bq_client, run_dir: Path, model_version: str, model_type: str):
    results_path = run_dir / "results.csv"
    if not results_path.exists():
        logger.warning("results.csv not found, skipping epoch upload.")
        return

    df = pd.read_csv(results_path)
    df.columns = df.columns.str.strip()

    table_id = f"{bq_client.project}.{CONFIG.model_results_dataset.name}.{CONFIG.model_results_dataset.training_epoch_table}"

    rows = []
    for _, row in df.iterrows():
        rows.append({
            "model_type":      model_type,
            "model_version":   model_version,
            "epoch":           int(row["epoch"]),
            "train_box_loss":  float(row.get("train/box_loss", 0) or 0),
            "train_seg_loss":  float(row.get("train/seg_loss", 0) or 0),
            "train_cls_loss":  float(row.get("train/cls_loss", 0) or 0),
            "train_dfl_loss":  float(row.get("train/dfl_loss", 0) or 0),
            "val_box_loss":    float(row.get("val/box_loss",   0) or 0),
            "val_seg_loss":    float(row.get("val/seg_loss",   0) or 0),
            "val_cls_loss":    float(row.get("val/cls_loss",   0) or 0),
            "val_dfl_loss":    float(row.get("val/dfl_loss",   0) or 0),
            "val_precision":   float(row.get("metrics/precision(B)", 0) or 0),
            "val_recall":      float(row.get("metrics/recall(B)",    0) or 0),
            "val_map50":       float(row.get("metrics/mAP50(B)",     0) or 0),
            "val_map50_95":    float(row.get("metrics/mAP50-95(B)",  0) or 0),
        })

    errors = bq_client.insert_rows_json(table_id, rows)
    if errors:
        logger.error("BigQuery epoch insert errors: %s", errors)
    else:
        logger.info("Inserted %d epochs into BigQuery.", len(rows))


def upload_run_to_bigquery(bq_client, best_row, train_args: dict,
                            model_version: str, model_type: str,
                            classes: list, eval_metrics: dict):
    table_id = f"{bq_client.project}.{CONFIG.model_results_dataset.name}.{CONFIG.model_results_dataset.model_runs_table}"
    cfg      = CONFIG.transform

    row = {
        "model_type":       model_type,
        "model_version":    model_version,
        "timestamp":        datetime.now(timezone.utc).isoformat(),
        "class_names":      classes,
        # Transform Config
        "random_seed":      CONFIG.random_seed,
        "samples_per_card": cfg.samples_per_card,
        "train_split":      1.0 - cfg.val_split - cfg.test_split - cfg.empty_split,
        "val_split":        cfg.val_split,
        "test_split":       cfg.test_split,
        "empty_split":      cfg.empty_split,
        # Training Config — from args.yaml
        "pretrained_model": train_args.get("model"),
        "epochs_planned":   train_args.get("epochs"),
        "batch_size":       train_args.get("batch"),
        "learning_rate":    train_args.get("lr0"),
        "optimizer":        train_args.get("optimizer"),
        "amp":              train_args.get("amp", False),
        # Eval metrics
        **eval_metrics,
    }

    # dataset_size from results
    if best_row is not None:
        pass  # dataset_size comes from transform, not results

    errors = bq_client.insert_rows_json(table_id, [row])
    if errors:
        logger.error("BigQuery run insert errors: %s", errors)
    else:
        logger.info("Inserted model run into BigQuery: %s/%s", model_type, model_version)


def load(model_version: str, model_type: str, run_dir: Path, eval_metrics: dict):
    classes        = get_classes()
    storage_client = storage.Client()
    bq_client      = bigquery.Client()

    upload_model(storage_client, run_dir, model_version)

    best_row   = parse_results(run_dir)
    train_args = parse_train_args(run_dir)

    upload_epochs_to_bigquery(bq_client, run_dir, model_version, model_type)
    upload_run_to_bigquery(bq_client, best_row, train_args,
                           model_version, model_type, classes, eval_metrics)

    logger.info("Load complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--model-type",    required=True)
    parser.add_argument("--run-dir",       required=True)
    args = parser.parse_args()

    # When called standalone, eval metrics are empty — run evaluate.py first
    load(
        model_version = args.model_version,
        model_type    = args.model_type,
        run_dir       = Path(args.run_dir),
        eval_metrics  = {
            "test_precision": None,
            "test_recall":    None,
            "test_map50":     None,
            "test_map50_95":  None,
            "min_iou":        None,
            "max_iou":        None,
            "avg_iou":        None,
        },
    )