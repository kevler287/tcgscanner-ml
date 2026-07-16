import logging
from pathlib import Path
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def load_ground_truth(label_path: Path, img_w: int, img_h: int) -> list:
    masks = []
    if not label_path.exists():
        return masks
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            class_id = int(parts[0])
            coords   = list(map(float, parts[1:]))
            polygon  = [(coords[i] * img_w, coords[i+1] * img_h)
                        for i in range(0, len(coords), 2)]
            mask_img = Image.new("L", (img_w, img_h), 0)
            ImageDraw.Draw(mask_img).polygon(polygon, fill=255)
            masks.append((class_id, np.array(mask_img) > 0))
    return masks


def compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    intersection = np.logical_and(mask_a, mask_b).sum()
    union        = np.logical_or(mask_a, mask_b).sum()
    return float(intersection / union) if union > 0 else 0.0


def evaluate(run_dir: str, data_yaml_path: str) -> dict:
    data_yaml    = Path(data_yaml_path)
    test_img_dir = data_yaml.parent / "images/test"
    test_lbl_dir = data_yaml.parent / "labels/test"

    run_dir = Path(run_dir)
    best_pt = run_dir / "weights" / "best.pt"

    if not best_pt.exists():
        raise FileNotFoundError(f"Training finished but best.pt not found: {best_pt}")
    
    model       = YOLO(str(best_pt))
    image_paths = sorted(test_img_dir.glob("*.[jp][pn]g"))

    if not image_paths:
        logger.error("No images found in %s", test_img_dir)
        return {}

    logger.info("Evaluating %d images...", len(image_paths))

    all_ious   = []

    for img_path in image_paths:
        img          = Image.open(img_path).convert("RGB")
        img_w, img_h = img.size
        label_path   = test_lbl_dir / (img_path.stem + ".txt")
        gt_masks     = load_ground_truth(label_path, img_w, img_h)
        result       = model(img_path, verbose=False)[0]

        if result.masks is None or len(result.masks) == 0:
            continue

        pred_classes   = result.boxes.cls.cpu().numpy().astype(int)
        pred_masks_raw = result.masks.data.cpu().numpy()

        for cls_id, pred_mask_small in zip(pred_classes, pred_masks_raw):
            pred_mask = np.array(
                Image.fromarray((pred_mask_small * 255).astype(np.uint8)).resize(
                    (img_w, img_h), Image.NEAREST)) > 127

            best_iou = 0.0
            for gt_cls, gt_mask in gt_masks:
                if gt_cls == cls_id:
                    best_iou = max(best_iou, compute_iou(pred_mask, gt_mask))

            all_ious.append(best_iou)

    # Official YOLO val for precision/recall/mAP
    val_results = model.val(data=str(data_yaml), split="test", verbose=False)
    precision   = float(val_results.seg.p.mean()) if hasattr(val_results.seg, "p") else 0.0
    recall      = float(val_results.seg.r.mean()) if hasattr(val_results.seg, "r") else 0.0


    metrics = {
        "precision": round(precision,                        4),
        "recall":    round(recall,                           4),
        "map50":     round(float(val_results.box.map50),     4),
        "map50_95":  round(float(val_results.box.map),       4),
        "min_iou":        round(float(np.min(all_ious)),          4) if all_ious else 0.0,
        "max_iou":        round(float(np.max(all_ious)),          4) if all_ious else 0.0,
        "avg_iou":        round(float(np.mean(all_ious)),         4) if all_ious else 0.0,
    }

    logger.info("Evaluation complete: %s", metrics)
    return metrics
