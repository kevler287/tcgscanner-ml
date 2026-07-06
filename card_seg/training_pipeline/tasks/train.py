from dataclasses import asdict
import json
import logging
import os
from ultralytics import YOLO
from card_seg.config import CONFIG

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

YOLO_MODEL = os.getenv("YOLO_MODEL", "yolo11n-seg.pt")
_overrides = json.loads(os.getenv("TRAIN_CONFIG", "{}"))
TRAIN_CONFIG = {**asdict(CONFIG.yolo_seg), **_overrides}

def train(data_yaml: str, run_name: str, work_dir: str = None):
    logger.info("Starting training: %s with config %s", run_name, TRAIN_CONFIG)

    model = YOLO(YOLO_MODEL)
    results = model.train(
        data=data_yaml,
        project=work_dir,
        name=run_name,
        exist_ok=True,
        **TRAIN_CONFIG
    )
    run_dir = results.save_dir
    logger.info("Training complete. Weights at %s", run_dir / "weights")
    return run_dir
