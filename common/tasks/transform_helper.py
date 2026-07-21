import random

import cv2
import numpy as np

def relative_box_to_pixels(position, width: int, height: int):
    """Convert a relative (0..1) box into absolute pixel coordinates."""
    (x0, y0), (x1, y1) = position
    left = int(x0 * width)
    top = int(y0 * height)
    right = int(x1 * width)
    bottom = int(y1 * height)
    return (left, top, right, bottom)

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
 
def random_glare(image: np.ndarray, max_intensity: float = 1.0, min_intensity: float = 0.1) -> np.ndarray:
    h, w = image.shape[:2]
 
    angle = random.uniform(0, 2 * np.pi)
    intensity = random.uniform(min_intensity, max_intensity)
 
    xs = np.linspace(-1, 1, w)
    ys = np.linspace(-1, 1, h)
    xv, yv = np.meshgrid(xs, ys)
    glare = np.cos(angle) * xv + np.sin(angle) * yv
    glare = (glare - glare.min()) / (glare.max() - glare.min())
    glare = (glare * intensity * 255).astype(np.uint8)
 
    result = image.copy()
    for c in range(3):
        result[:, :, c] = np.clip(
            result[:, :, c].astype(np.int32) + glare, 0, 255
        ).astype(np.uint8)
 
    return result


def add_blur(image, params: list[int] = [1,3,5]):
    k = random.choice(params)
    return cv2.GaussianBlur(image, (k, k), 0)

def add_brightness_contrast(image, contrast: tuple[float, float] = (0.6, 1.4), brightness: tuple[int, int] = (-40, 40)):
    alpha = random.uniform(contrast[0], contrast[1])   # contrast
    beta  = random.randint(brightness[0], brightness[1])    # brightness
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)