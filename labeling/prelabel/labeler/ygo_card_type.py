import numpy as np
from PIL import Image

# Rough frame colors (RGB), adjust if needed
COLORS = {
    "normal": (222, 190, 110),
    "effect": (200, 120, 70),
    "ritual": (100, 140, 190),
    "fusion": (140, 100, 160),
    "synchro": (225, 225, 225),
    "xyz": (30, 30, 30),
    "link": (30, 70, 140),
    "spell": (40, 150, 125),
    "trap": (175, 70, 120),
}
NAMES = list(COLORS)
PROTOS = np.array(list(COLORS.values()), dtype=float)


def dominant(img):
    """Most common color class in the image (each pixel votes for its nearest color)."""
    pixels = np.asarray(img, dtype=float).reshape(-1, 3)
    dist = np.linalg.norm(pixels[:, None] - PROTOS[None], axis=-1)
    return NAMES[np.bincount(dist.argmin(axis=1), minlength=len(NAMES)).argmax()]


def label(image_path, label):
    img = Image.open(image_path).convert("RGB")
    img.thumbnail((200, 290))  # small is enough and keeps it fast
    w, h = img.size
    top = dominant(img.crop((0, 0, w, h // 2)))
    bottom = dominant(img.crop((0, h // 2, w, h)))

    # Pendulum: monster color in the upper half, spell green in the lower half
    label["card_type"] = "pendulum" if bottom == "spell" and top != "spell" else top
    return label