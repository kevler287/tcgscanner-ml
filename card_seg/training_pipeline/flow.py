from pathlib import Path
import logging
from prefect import flow, task

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


@task(name="Extract")
def extract():
    """Download cards and backgrounds from GCS to local cache."""
    from card_seg.data_pipeline.tasks.extract import main as run_extract
    run_extract()


@task(name="Transform")
def transform():
    """Generate synthetic YOLO training dataset from raw cards and backgrounds."""
    from card_seg.data_pipeline.tasks.transform import main as run_transform
    run_transform()


@task(name="Train")
def train(model_version: str, epochs: int = None) -> Path:
    """Fine-tune YOLO segmentation model on generated dataset."""
    from ultralytics import YOLO
    from data_platform.config import CONFIG

    cfg     = CONFIG.yolo_seg
    model   = YOLO(cfg.pretrained_model)
    epochs = epochs if epochs is not None else cfg.epochs
    results = model.train(
        data      = "data_platform/etl/output/data.yaml",
        epochs    = epochs,
        imgsz     = cfg.imgsz,
        batch     = cfg.batch,
        device    = cfg.device,
        amp       = cfg.amp,
        optimizer = cfg.optimizer,
        lr0       = cfg.lr0,
        patience  = epochs if epochs < 100 else int(epochs * 0.15),
        name      = model_version,
        close_mosaic = 0 if epochs <= 10 else 10
    )

    run_dir = Path(results.save_dir)
    logger.info("Training complete. Run dir: %s", run_dir)
    return run_dir


@task(name="Evaluate")
def evaluate(run_dir: Path):
    """Evaluate model on test set and compute IoU metrics."""
    from card_seg.training_pipeline.tasks.evaluate import evaluate as run_evaluate
    return run_evaluate(model_path=run_dir / "weights/best.pt")


@task(name="Load")
def load(model_version: str, model_type: str, run_dir: Path, eval_metrics: dict):
    """Upload best.pt to GCS and write model results to BigQuery."""
    from card_seg.training_pipeline.tasks.load import load as run_load
    run_load(model_version, model_type, run_dir, eval_metrics)


@flow(name="TCG Scanner Training Pipeline")
def training_pipeline(model_version: str, model_type: str, epochs: int = None):
    extract()
    transform()
    run_dir      = train(model_version, epochs=epochs)
    eval_metrics = evaluate(run_dir)
    load(model_version, model_type, run_dir, eval_metrics)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--model-type",    default="segmentation")
    args = parser.parse_args()

    training_pipeline(
        model_version = args.model_version,
        model_type    = args.model_type,
        epochs=args.epochs
    )