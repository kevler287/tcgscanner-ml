import logging
import os
from pathlib import Path

from ultralytics import YOLO

from card_seg.config import CONFIG

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

YOLO_CFG = CONFIG.yolo_seg
WORKSPACE_DIR = os.environ.get("WORKSPACE_DIR", "/workspace")


def train(data_yaml: str, run_name: str, project_dir: str = None):
    if project_dir is None:
        project_dir = f"{WORKSPACE_DIR}/runs"

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
    logger.info("Training complete. Weights at %s", run_dir / "weights")
    return run_dir
