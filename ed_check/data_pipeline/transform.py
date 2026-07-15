"""
Transform step: crop edition_0 / edition_1 regions from card images.

Rules:
- Subdir name ends with "_0"     -> crop using "edition_0" position
- Subdir name ends with "_1"     -> crop using "edition_1" position
- Subdir name is something else e.g. "unlimited"     -> crop using "edition_0" AND "edition_1"
  (two output files per image)
"""

import json
import os
from pathlib import Path
import random

import cv2
import numpy as np
from ed_check.config import CONFIG
from common.data_pipeline import transform_helper
from dotenv import load_dotenv

load_dotenv()

LOCAL_DATA_DIR  = Path(os.getenv("LOCAL_DATA_DIR"))
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

random.seed(CONFIG.random_seed)
np.random.seed(CONFIG.random_seed)


def load_config(config_path: Path) -> dict:
    """Load the JSON config file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def add_margin(position, margin: float):
    """
    Expand a relative box (0..1) by 'margin' * box width/height on
    every side. margin=0.10 -> 10% growth on each side.
    Result is clamped to [0, 1].
    """
    (x0, y0), (x1, y1) = position
    w = x1 - x0
    h = y1 - y0

    x0 = max(0.0, x0 - w * margin)
    y0 = max(0.0, y0 - h * margin)
    x1 = min(1.0, x1 + w * margin)
    y1 = min(1.0, y1 + h * margin)

    return [[x0, y0], [x1, y1]]
 
 
def crop_edition(image_path: Path, position) -> np.ndarray:
    """Crop a single image to the given relative position."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")
 
    h, w = image.shape[:2]
 
    position = add_margin(position, CONFIG.transform_cfg.crop_margin)
    box = transform_helper.relative_box_to_pixels(position, w, h)
    left, top, right, bottom = box
 
    cropped = image[top:bottom, left:right]
 
    # cv2 always writes 3-channel BGR (or grayscale) -> no palette/alpha
    # handling needed, unlike Pillow/JPEG
    return cropped

def random_crop_within_margin(image: np.ndarray, margin: float) -> np.ndarray:
    if margin <= 0:
        return image.copy()

    h, w = image.shape[:2]
    inner_ratio = 1.0 / (1.0 + 2.0 * margin)

    target_w = max(1, round(w * inner_ratio))
    target_h = max(1, round(h * inner_ratio))

    max_x_offset = w - target_w
    max_y_offset = h - target_h

    x0 = random.randint(0, max_x_offset) if max_x_offset > 0 else 0
    y0 = random.randint(0, max_y_offset) if max_y_offset > 0 else 0

    return image[y0:y0 + target_h, x0:x0 + target_w]


def determine_editions_for_subdir(subdir_name: str):
    """Return the list of edition keys that apply to this subdir."""
    if subdir_name.endswith("_0"):
        return [CONFIG.transform_cfg.edition_0]
    elif subdir_name.endswith("_1"):
        return [CONFIG.transform_cfg.edition_1]
    else:
        return [CONFIG.transform_cfg.edition_0, CONFIG.transform_cfg.edition_1]

def run_crop_transform() -> int:
    out_path = Path(CONFIG.transform_cfg.output_dir)

    train_path_0 = out_path / "train" / "other_ed"
    train_path_1 = out_path / "train" / "first_ed"
    val_path_0 = out_path / "val" / "other_ed"
    val_path_1 = out_path / "val" / "first_ed"

    for path in [train_path_0, train_path_1, val_path_0, val_path_1]:
        path.mkdir(parents=True, exist_ok=True)

    in_path = LOCAL_DATA_DIR / CONFIG.pf_ed_types

    total_crops = 0

    for subdir in sorted(p for p in in_path.iterdir() if p.is_dir()):
        positions = determine_editions_for_subdir(subdir.name)

        images = sorted(
            p for p in subdir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )

        for image_path in images:
            for i, position in enumerate(positions):
                for j in range(CONFIG.transform_cfg.samples_per_type):
                    img = crop_edition(image_path, position)
                    img, _ = transform_helper.random_perspective_warp(img, max_angle_deg=5)
                    img = random_crop_within_margin(img, margin=CONFIG.transform_cfg.crop_margin)
                    img = transform_helper.add_brightness_contrast(img)
                    img = transform_helper.random_glare(img, max_intensity=0.3)
                    img = transform_helper.add_blur(img, params=[1,3])

                    # For "unlimited" (2 editions per image), append a suffix
                    # to avoid overwriting. For _0/_1 dirs (1 edition), keep
                    # the original filename.
                    if len(positions) > 1:
                        out_name = f"{subdir.name}_{i}_{image_path.stem}_{j}{image_path.suffix}"
                    else:
                        out_name = f"{subdir.name}_{image_path.stem}_{j}{image_path.suffix}"

                    if j > CONFIG.transform_cfg.samples_per_type * CONFIG.transform_cfg.val_split:
                        if "first" in subdir.name:
                            dest_path = train_path_1 / out_name
                        else:
                            dest_path = train_path_0 / out_name
                    else:
                        if "first" in subdir.name:
                            dest_path = val_path_1 / out_name
                        else:
                            dest_path = val_path_0 / out_name
                    
                    cv2.imwrite(str(dest_path), img)
                    total_crops += 1

    return total_crops