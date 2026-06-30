import logging
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv
from google.cloud import bigquery, storage

from card_seg.config import CONFIG

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

load_dotenv()

BUCKET_NAME      = CONFIG.bucket.name
PF_MODELS        = CONFIG.bucket.pf_models
BQ_DATASET       = CONFIG.model_results_dataset.name
MODEL_RUNS_TABLE = CONFIG.model_results_dataset.model_runs_table
EPOCHS_TABLE     = CONFIG.model_results_dataset.training_epoch_table
MODEL_PREFIX = CONFIG.model_prefix


def upload_weights(run_dir: Path, model_version: str) -> dict:
    best_pt = run_dir / "weights" / "best.pt"
    last_pt = run_dir / "weights" / "last.pt"

    if not best_pt.exists():
        raise FileNotFoundError(f"best.pt not found. Aborting Pipeline.")
    
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    blob_prefix = PF_MODELS + f"{model_version}/"

    if bucket.blob(blob_prefix + "best.pt").exists():
        raise FileExistsError(
            f"Model {model_version} already exists at gs://{BUCKET_NAME}/{blob_prefix}"
        )

    paths = {}
    for local_path, name in [(best_pt, "best.pt"), (last_pt, "last.pt")]:
        blob_path = blob_prefix + name
        logger.info("Uploading %s → gs://%s/%s", local_path, BUCKET_NAME, blob_path)
        bucket.blob(blob_path).upload_from_filename(local_path)
        paths[name] = f"gs://{BUCKET_NAME}/{blob_path}"

    logger.info("Weights upload complete.")
    return paths


def insert_model_run(
    model_version: str,
    dataset_version: str,
    dataset_size: int,
    eval_metrics: dict,
):
    client = bigquery.Client()
    table_id = f"{client.project}.{BQ_DATASET}.{MODEL_RUNS_TABLE}"

    transform_cfg = CONFIG.transform
    yolo_cfg = CONFIG.yolo_seg

    row = {
        "model_prefix":     MODEL_PREFIX,
        "model_version":    model_version,
        "timestamp":        datetime.now(timezone.utc).isoformat(),
        "class_names":      ["ygo_card"],
        "dataset_version":  dataset_version,
        "dataset_size":     dataset_size,
        "samples_per_card": transform_cfg.samples_per_card,
        "train_split":      1 - transform_cfg.val_split - transform_cfg.test_split,
        "val_split":        transform_cfg.val_split,
        "test_split":       transform_cfg.test_split,
        "empty_split":      transform_cfg.empty_split,
        "pretrained_model": yolo_cfg.pretrained_model,
        "epochs_planned":   yolo_cfg.epochs,
        "batch_size":       yolo_cfg.batch,
        "learning_rate":    yolo_cfg.lr0,
        "optimizer":        yolo_cfg.optimizer,
        "amp":              yolo_cfg.amp,
        "test_precision":   eval_metrics.get("precision"),
        "test_recall":      eval_metrics.get("recall"),
        "test_map50":       eval_metrics.get("mAP50"),
        "test_map50_95":    eval_metrics.get("mAP50_95"),
        "min_iou":          eval_metrics.get("min_iou"),
        "max_iou":          eval_metrics.get("max_iou"),
        "avg_iou":          eval_metrics.get("avg_iou"),
    }

    errors = client.insert_rows_json(table_id, [row])
    if errors:
        raise RuntimeError(f"Failed to insert model_run row: {errors}")

    logger.info("Inserted model_run row for %s", model_version)


def insert_training_epochs(model_version: str, run_dir: Path):
    epoch_metrics = []
    # results.csv in run_dir holds per-epoch history
    csv_path = run_dir / "results.csv"
    if csv_path.exists():
        import csv as csv_module
        with open(csv_path) as f:
            reader = csv_module.DictReader(f)
            for row in reader:
                epoch_metrics.append({k.strip(): v.strip() for k, v in row.items()})
    
    if not epoch_metrics:
        logger.warning("No epoch metrics to insert, skipping.")
        return

    client = bigquery.Client()
    table_id = f"{client.project}.{BQ_DATASET}.{EPOCHS_TABLE}"

    rows = []
    for row in epoch_metrics:
        rows.append({
            "model_prefix":     MODEL_PREFIX,
            "model_version":    model_version,
            "epoch":            int(float(row.get("epoch", 0))),
            "train_box_loss":   _safe_float(row.get("train/box_loss")),
            "train_seg_loss":   _safe_float(row.get("train/seg_loss")),
            "train_cls_loss":   _safe_float(row.get("train/cls_loss")),
            "train_dfl_loss":   _safe_float(row.get("train/dfl_loss")),
            "val_box_loss":     _safe_float(row.get("val/box_loss")),
            "val_seg_loss":     _safe_float(row.get("val/seg_loss")),
            "val_cls_loss":     _safe_float(row.get("val/cls_loss")),
            "val_dfl_loss":     _safe_float(row.get("val/dfl_loss")),
            "val_precision":    _safe_float(row.get("metrics/precision(M)")),
            "val_recall":       _safe_float(row.get("metrics/recall(M)")),
            "val_map50":        _safe_float(row.get("metrics/mAP50(M)")),
            "val_map50_95":     _safe_float(row.get("metrics/mAP50-95(M)")),
        })

    errors = client.insert_rows_json(table_id, rows)
    if errors:
        raise RuntimeError(f"Failed to insert training_epochs rows: {errors}")

    logger.info("Inserted %d epoch rows for %s", len(rows), model_version)


def _safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load(
    model_version: str,
    dataset_version: str,
    dataset_size: int,
    run_dir: str,
    eval_metrics: dict,
):
    run_dir = Path(run_dir)
    weight_paths = upload_weights(run_dir, model_version)
    insert_model_run(model_version, dataset_version, dataset_size, eval_metrics)
    insert_training_epochs(model_version, run_dir)

    logger.info("Load complete. Weights at %s", weight_paths)
    return weight_paths