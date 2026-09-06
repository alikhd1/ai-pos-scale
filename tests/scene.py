"""
scene.py
--------
Synthetic tray scenes shared by the segmentation and matcher tests.

Real produce photographs are not available on a build machine, so the tests use
composited scenes with a textured tray, coloured objects with shading and a
cast shadow, camera noise, and optional gain / white-balance drift.  They are
deterministic (fixed seeds), which is what makes the thresholds in the tests
meaningful as regression gates rather than as claims about real accuracy.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

FRAME_W, FRAME_H = 640, 480
TRAY_BOX = (140, 90, 500, 400)               # x0, y0, x1, y1
TRAY_POLY = [(140, 90), (500, 90), (500, 400), (140, 400)]

# (name, BGR colour, radius, aspect) - deliberately similar pairs included
CLASSES = [
    ("apple_red", (44, 44, 196), 46, 1.00),
    ("apple_green", (60, 168, 96), 45, 1.00),
    ("lemon", (56, 214, 226), 40, 1.15),
    ("cucumber", (72, 150, 78), 30, 2.60),
    ("potato", (96, 132, 168), 44, 1.25),
    ("aubergine", (110, 52, 88), 38, 1.80),
]
UNKNOWN = ("mango", (70, 190, 240), 44, 1.30)


def make_tray(seed: int = 7, textured: bool = True) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.full((FRAME_H, FRAME_W, 3), 138, np.uint8)
    img = np.clip(img.astype(np.int16) + rng.normal(0, 7, img.shape), 0, 255).astype(np.uint8)
    x0, y0, x1, y1 = TRAY_BOX
    cv2.rectangle(img, (x0, y0), (x1, y1), (172, 168, 162), -1)
    if textured:                                   # a tray with visible grain
        for i in range(x0, x1, 13):
            cv2.line(img, (i, y0), (i, y1), (162, 158, 152), 1)
        for j in range(y0, y1, 19):
            cv2.line(img, (x0, j), (x1, j), (166, 162, 156), 1)
    cv2.rectangle(img, (x0, y0), (x1, y1), (120, 118, 114), 2)
    return img


def draw_object(img: np.ndarray, centre: Tuple[int, int], spec, angle: float = 0.0,
                scale: float = 1.0, shadow: bool = True) -> None:
    _name, colour, rad, aspect = spec
    a = int(rad * scale * aspect ** 0.5)
    b = int(rad * scale / aspect ** 0.5)
    cx, cy = centre
    if shadow:
        # the shadow is blurred on its own layer: blurring the whole frame once
        # per object would destroy the tray texture the segmenter relies on
        layer = np.zeros(img.shape[:2], np.uint8)
        cv2.ellipse(layer, (cx + 9, cy + 9), (a, b), angle, 0, 360, 255, -1, cv2.LINE_AA)
        alpha = (cv2.GaussianBlur(layer, (0, 0), 5).astype(np.float32) / 255.0 * 0.45)[..., None]
        img[:] = np.clip(img.astype(np.float32) * (1.0 - alpha), 0, 255).astype(np.uint8)
    cv2.ellipse(img, (cx, cy), (a, b), angle, 0, 360, colour, -1, cv2.LINE_AA)
    # a body shading gradient: this is what separates an object from a flat shadow
    overlay = img.copy()
    cv2.ellipse(overlay, (cx - a // 4, cy - b // 4), (max(3, a // 2), max(3, b // 2)), angle, 0, 360,
                tuple(min(255, int(c * 1.35)) for c in colour), -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.45, img, 0.55, 0, dst=img)


def scene(objects: Sequence, seed: int = 0, gain: float = 1.0, awb: Tuple[float, float, float] = (1, 1, 1),
          hand: bool = False, tray_seed: int = 7, textured: bool = True) -> np.ndarray:
    """objects: [(centre, spec, angle, scale), ...]"""
    rng = np.random.default_rng(seed)
    img = make_tray(tray_seed, textured)
    for centre, spec, angle, scale in objects:
        draw_object(img, centre, spec, angle, scale)
    if hand:                                       # an arm reaching in from outside
        cv2.ellipse(img, (330, 470), (44, 150), 22, 0, 360, (120, 150, 200), -1, cv2.LINE_AA)
    f = img.astype(np.float32) * gain * np.array(awb, np.float32)
    f += rng.normal(0, 2.2, f.shape)
    return np.clip(f, 0, 255).astype(np.uint8)


def empty_frames(n: int = 9, seed: int = 100, tray_seed: int = 7, textured: bool = True) -> List[np.ndarray]:
    return [scene([], seed=seed + i, tray_seed=tray_seed, textured=textured) for i in range(n)]


def positions(n: int, seed: int = 0, spread: bool = True) -> List[Tuple[int, int]]:
    """n well-separated positions inside the tray."""
    rng = np.random.default_rng(seed)
    slots = [(215, 165), (410, 165), (215, 330), (410, 330), (312, 245)]
    if not spread:
        return [(300 + i * 46, 240) for i in range(n)]
    idx = list(range(len(slots)))
    rng.shuffle(idx)
    return [slots[i] for i in idx[:n]]


def iou(box_a, box_b) -> float:
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    ix0, iy0 = max(ax, bx), max(ay, by)
    ix1, iy1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


def object_box(centre, spec, scale: float = 1.0) -> Tuple[int, int, int, int]:
    _n, _c, rad, aspect = spec
    a = int(rad * scale * aspect ** 0.5)
    b = int(rad * scale / aspect ** 0.5)
    return (centre[0] - a, centre[1] - b, 2 * a, 2 * b)
