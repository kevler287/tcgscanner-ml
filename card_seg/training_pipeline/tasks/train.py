import logging

from ultralytics import YOLO

from card_seg.config import CONFIG

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

YOLO_CFG = CONFIG.yolo_seg


def train(data_yaml: str, run_name: str, project_dir: str = "/workspace/runs"):
    logger.info("Starting training: %s", run_name)

    model = YOLO(YOLO_CFG.pretrained_model)

    results = model.train(
        data=data_yaml,
        epochs=YOLO_CFG.epochs,
        imgsz=YOLO_CFG.imgsz,
        batch=YOLO_CFG.batch,
        device=YOLO_CFG.device,
        amp=YOLO_CFG.amp,
        optimizer=YOLO_CFG.optimizer,
        lr0=YOLO_CFG.lr0,
        project=project_dir,
        name=run_name,
        exist_ok=True,
    )

    run_dir = results.save_dir
    best_pt = run_dir / "weights" / "best.pt"
    last_pt = run_dir / "weights" / "last.pt"

    if not best_pt.exists():
        raise FileNotFoundError(f"Training finished but best.pt not found: {best_pt}")

    logger.info("Training complete. Weights at %s", run_dir / "weights")

    # Per-epoch metrics for BigQuery (training_epoch_table)
    epoch_metrics = []
    if hasattr(results, "results_dict"):
        # results.csv in run_dir holds per-epoch history
        csv_path = run_dir / "results.csv"
        if csv_path.exists():
            import csv as csv_module
            with open(csv_path) as f:
                reader = csv_module.DictReader(f)
                for row in reader:
                    epoch_metrics.append({k.strip(): v.strip() for k, v in row.items()})

    return {
        "best_pt": str(best_pt),
        "last_pt": str(last_pt),
        "run_dir": str(run_dir),
        "epoch_metrics": epoch_metrics,
    }
