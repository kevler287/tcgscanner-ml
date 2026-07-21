from pathlib import Path
import cv2
import numpy as np
import random
import logging
from tqdm import tqdm
from dotenv import load_dotenv
from card_seg.src.config import CONFIG
import os

load_dotenv()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ── Config ────────────────────────────────────────────────────────────────────

SAMPLES_PER_CARD = CONFIG.transform.samples_per_card
RANDOM_SEED      = CONFIG.random_seed
BACKGROUND_SIZE = CONFIG.transform.background_size
MAX_ANGLE_DEG   = CONFIG.transform.max_angle_deg
VAL_SPLIT       = CONFIG.transform.val_split
EMPTY_SPLIT     = CONFIG.transform.empty_split

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_background_paths(bg_dir):
    files = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        files.extend(Path(bg_dir).rglob(ext))
    return files


def load_random_background(bg_paths, bg_size):
    bg = cv2.imread(str(random.choice(bg_paths)))
    h, w = bg.shape[:2]
    target_w, target_h = bg_size
    scale = max(target_w / w, target_h / h)
    bg = cv2.resize(bg, (int(w * scale), int(h * scale)))
    h, w = bg.shape[:2]
    x = random.randint(0, w - target_w)
    y = random.randint(0, h - target_h)
    return bg[y:y+target_h, x:x+target_w]


def random_perspective_warp(image, max_angle_deg=10):
    h, w = image.shape[:2]
    max_offset = int(min(h, w) * np.tan(np.radians(max_angle_deg)))

    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([
        [random.randint(0, max_offset), random.randint(0, max_offset)],
        [w - random.randint(0, max_offset), random.randint(0, max_offset)],
        [w - random.randint(0, max_offset), h - random.randint(0, max_offset)],
        [random.randint(0, max_offset), h - random.randint(0, max_offset)],
    ])

    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, matrix, (w, h))
    return warped, dst


def add_glare(image, corners):
    h, w = image.shape[:2]
    angle = random.uniform(0, 2 * np.pi)
    intensity = random.uniform(0.1, 1)

    xs = np.linspace(-1, 1, w)
    ys = np.linspace(-1, 1, h)
    xv, yv = np.meshgrid(xs, ys)
    glare = np.cos(angle) * xv + np.sin(angle) * yv
    glare = (glare - glare.min()) / (glare.max() - glare.min())
    glare = (glare * intensity * 255).astype(np.uint8)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [corners.astype(np.int32)], 255)
    glare = cv2.bitwise_and(glare, glare, mask=mask)

    result = image.copy()
    for c in range(3):
        result[:, :, c] = np.clip(
            result[:, :, c].astype(np.int32) + glare, 0, 255
        ).astype(np.uint8)
    return result


def add_blur(image):
    k = random.choice([1, 3, 5])
    return cv2.GaussianBlur(image, (k, k), 0)

def add_brightness_contrast(image):
    alpha = random.uniform(0.6, 1.4)   # contrast
    beta  = random.randint(-40, 40)    # brightness
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


def paste_on_random_background(card, corners, bg_paths, bg_size):
    bg = load_random_background(bg_paths, bg_size)
    bg_w, bg_h = bg_size
    h, w = card.shape[:2]

    # x_min = -int(w / 2)
    # x_max = bg_w - int(w * 0.9)
    # if x_max < x_min:
    #     x = x_max
    # else:
    #     x = random.randint(x_min, x_max)

    # y_min = -int(h / 2)
    # y_max = bg_h - int(h * 0.8)
    # if y_max < y_min:
    #     y = int((y_max+y_min)/2)
    # else:
    #     y = random.randint(-int(h / 2), bg_h - int(h * 0.8))

    x = random.randint(0, bg_w-w)
    y = random.randint(0, bg_h-h)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [corners.astype(np.int32)], 255)

    x1_card = max(0, -x)
    y1_card = max(0, -y)
    x2_card = min(w, bg_w - x)
    y2_card = min(h, bg_h - y)

    x1_bg = max(0, x)
    y1_bg = max(0, y)
    x2_bg = x1_bg + (x2_card - x1_card)
    y2_bg = y1_bg + (y2_card - y1_card)

    if x2_card <= x1_card or y2_card <= y1_card:
        return bg, (x, y)

    card_crop = card[y1_card:y2_card, x1_card:x2_card]
    mask_crop = mask[y1_card:y2_card, x1_card:x2_card]
    mask_inv = cv2.bitwise_not(mask_crop)

    roi = bg[y1_bg:y2_bg, x1_bg:x2_bg]
    bg_part = cv2.bitwise_and(roi, roi, mask=mask_inv)
    card_part = cv2.bitwise_and(card_crop, card_crop, mask=mask_crop)
    bg[y1_bg:y2_bg, x1_bg:x2_bg] = cv2.add(bg_part, card_part)

    return bg, (x, y)


def generate_yolo_label(corners, offset, bg_size):
    bg_w, bg_h = bg_size
    ox, oy = offset

    pts = corners.copy()
    pts[:, 0] += ox
    pts[:, 1] += oy

    pts[:, 0] /= bg_w
    pts[:, 1] /= bg_h

    coords = " ".join(f"{v:.6f}" for pt in pts for v in pt)
    return f"0 {coords}"


# ── Main ──────────────────────────────────────────────────────────────────────

def transform(
    cards_dir: Path,
    bg_dir: Path,
    dest_dir: Path
):
    card_paths = list(cards_dir.rglob("*.jpg")) + \
                 list(cards_dir.rglob("*.jpeg")) + \
                 list(cards_dir.rglob("*.png"))

    bg_paths = load_background_paths(bg_dir)

    empty_count    = int(len(card_paths) * SAMPLES_PER_CARD * EMPTY_SPLIT)
    expected_total = len(card_paths) * SAMPLES_PER_CARD + empty_count

    if is_transform_done(dest_dir, expected_total):
        return

    for split in ("train", "val"):
        (dest_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (dest_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    (dest_dir / "classes.txt").write_text("ygo_card\n")
    (dest_dir / "data.yaml").write_text(
        f"path: \n"
        f"train: images/train/\n"
        f"val: images/val/\n"
        f"\n"
        f"nc: 1\n"
        f"names:\n"
        f"  0: ygo_card\n"
    )

    random.shuffle(card_paths)
    val_count  = int(len(card_paths) * VAL_SPLIT)
    val_cards  = set(str(p) for p in card_paths[:val_count])

    logger.info("Generating %d samples per card + %d empty → %d total",
                SAMPLES_PER_CARD, empty_count, expected_total)

    total = len(card_paths) * SAMPLES_PER_CARD
    with tqdm(total=expected_total, unit="sample") as pbar:
        for card_path in card_paths:
            card = cv2.imread(str(card_path))
            if card is None:
                logger.warning("Could not load %s, skipping.", card_path)
                continue

            if str(card_path) in val_cards:
                split = "val"
            else:
                split = "train"

            for i in range(SAMPLES_PER_CARD):
                size_ratio = min(BACKGROUND_SIZE[1]/card.shape[0], BACKGROUND_SIZE[0]/card.shape[1])
                scale = random.uniform(size_ratio*0.6, size_ratio*0.9)
                card_resized = cv2.resize(
                    card,
                    (int(card.shape[1] * scale), int(card.shape[0] * scale))
                )

                warped, corners = random_perspective_warp(card_resized, MAX_ANGLE_DEG)
                warped = add_glare(warped, corners)
                warped = add_blur(warped)

                composite, (ox, oy) = paste_on_random_background(
                    warped, corners, bg_paths, BACKGROUND_SIZE
                )
                composite = add_blur(composite)
                composite = add_brightness_contrast(composite)

                label = generate_yolo_label(corners, (ox, oy), BACKGROUND_SIZE)

                stem = f"{card_path.stem}_{i:04d}"
                cv2.imwrite(str(dest_dir / "images" / split / f"{stem}.jpg"), composite)
                (dest_dir / "labels" / split / f"{stem}.txt").write_text(label + "\n")

                pbar.update(1)

        # Generate empty samples
        for i in range(empty_count):
            bg = load_random_background(bg_paths, BACKGROUND_SIZE)
            stem  = f"empty_{i:04d}"
            split = "train"  # all empty samples go to train
            cv2.imwrite(str(dest_dir / "images" / split / f"{stem}.jpg"), bg)
            (dest_dir / "labels" / split / f"{stem}.txt").write_text("")
            pbar.update(1)

    logger.info("Done. Output: %s", str(dest_dir))


def is_transform_done(output_dir: Path, expected_total: int) -> bool:
    existing = (
        len(list((output_dir / "images/train").glob("*.jpg"))) +
        len(list((output_dir / "images/val").glob("*.jpg")))
    )
    if existing == expected_total:
        logger.info("Transform already done (%d samples found), skipping.", existing)
        return True
    return False


if __name__ == "__main__":
    transform()