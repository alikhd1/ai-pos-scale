"""
ai_engine.py
------------
Few-shot product recognition for the AI POS Scale, with multi-object support.

Per frame
    tray crop (polygon, outside masked)
      -> TraySegmenter finds up to 5 objects against the stored empty-tray image
      -> RegionTracker confirms a region only once it is seen in 2 of 3 frames
      -> each confirmed object is cropped tight, masked and embedded (MobileNetV2)
      -> two-headed matcher: rank in PCA-whitened space, accept or reject on the
         top-PC-removed score against that item's own calibrated threshold
      -> the scale, not the camera, is the authority on "the tray is empty"

Enrolling an item stores a handful of embeddings, so a new product is added at
the till in seconds and no training loop ever runs on the POS machine.

Why the matcher looks like this (all measured on this project's benchmark, on
held-out queries under changed lighting and rotation, 8 random enrolment subsets):

    shipped: mean of top-3 sample cosines, one global 0.65 threshold
        hard-set top-1 0.508, and a false-accept rate on never-enrolled produce
        of 1.000 - the "unknown item" verdict did not exist, because raw
        MobileNetV2 cosines between crops sharing a tray all sit in 0.76-0.94
    class centroid instead of top-k          0.508 -> 0.628   (and 10x cheaper)
    remove the top principal component       0.628 -> 0.674   (it encodes the
        tray and the light, not the produce; helps most when samples are few)
    rank in PCA-whitened space               0.674 -> 0.756
    per-item thresholds from impostors       false accepts 1.000 -> 0.078

Public classes
    * ``TrayRegion``        - where the tray is, and the masked square crop
    * ``FeatureExtractor``  - MobileNetV2 backbone, 1280-d L2-normalised embeddings
    * ``ItemDatabase``      - items, prices, embeddings, empty-tray reference (v3)
    * ``Recognizer``        - the two-headed matcher
    * ``RecognitionEngine`` - worker thread: segment, recognise, track
"""
from __future__ import annotations

import logging
import os
import pickle
import tempfile
import threading
import time
import uuid
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore

from tray_segment import (BUILD_FRAMES, EmptyTrayModel, ObjectRegion, RegionTracker, SegmentResult,
                          TraySegmenter, box_iou)

log = logging.getLogger("ai")

EMBEDDING_DIM = 1280
INPUT_SIZE = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
MOBILENET_V2_URL = "https://download.pytorch.org/models/mobilenet_v2-b0353104.pth"
MOBILENET_V2_FILE = "mobilenet_v2-b0353104.pth"
BACKBONE_ID = "mobilenet_v2"          # stored in the database; a mismatch means "re-enrol"
FILL_COLOR = (114, 114, 114)          # neutral grey outside the tray / outside an object
EMPTY_ID = "__empty__"                # the empty tray is scored as an ordinary class
MODE_TRAY, MODE_OBJECT = "tray", "object"
WHITEN_DIM = 32
CALIB_PERCENTILE = 99.0               # q=99: false accepts 0.078 at 0.636 false rejects
                                      # (q=95 gives 0.300 / 0.319 - a wrong price costs more than a tap)
SELF_FLOOR_FRAC = 0.35                # fallback floor (fraction of median self-similarity) when an item
                                      # has only one capture, so leave-one-capture-out is impossible
LOO_SLACK = 0.05                      # tolerance below the worst leave-one-capture-out score


# --------------------------------------------------------------------------- #
# Tray region
# --------------------------------------------------------------------------- #
class TrayRegion:
    """The area of the frame the model looks at.

    ``points`` is a normalised polygon (x, y in 0..1).  With fewer than three
    points a centred square of ``roi_ratio`` * min(w, h) is used.  ``crop()``
    returns a *square* image: the polygon's bounding box, everything outside the
    polygon painted neutral grey (when ``mask_outside``), padded to a square so
    shapes are not distorted when resized to 224x224.
    """

    def __init__(self, points: Optional[Sequence[Sequence[float]]] = None, mask_outside: bool = True, roi_ratio: float = 0.6):
        self.points: List[Tuple[float, float]] = [(float(np.clip(p[0], 0, 1)), float(np.clip(p[1], 0, 1))) for p in (points or [])]
        self.mask_outside = bool(mask_outside)
        self.roi_ratio = float(np.clip(roi_ratio, 0.2, 1.0))
        self._mask_cache: Dict[Tuple, np.ndarray] = {}

    @classmethod
    def from_settings(cls, camera_cfg: dict) -> "TrayRegion":
        return cls(camera_cfg.get("tray_points") or [], camera_cfg.get("tray_mask", True), camera_cfg.get("roi_ratio", 0.6))

    def to_settings(self) -> dict:
        return {"tray_points": [[round(x, 4), round(y, 4)] for x, y in self.points], "tray_mask": self.mask_outside}

    @property
    def is_polygon(self) -> bool:
        return len(self.points) >= 3

    def copy(self) -> "TrayRegion":
        return TrayRegion(list(self.points), self.mask_outside, self.roi_ratio)

    def pixel_points(self, w: int, h: int) -> np.ndarray:
        if self.is_polygon:
            return np.array([[int(round(x * (w - 1))), int(round(y * (h - 1)))] for x, y in self.points], dtype=np.int32)
        x0, y0, x1, y1 = self.bbox(w, h)
        return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.int32)

    def bbox(self, w: int, h: int) -> Tuple[int, int, int, int]:
        """Bounding box (x0, y0, x1, y1) in pixels, at least 32 px wide."""
        if self.is_polygon:
            pts = np.array([[x * (w - 1), y * (h - 1)] for x, y in self.points])
            x0, y0 = int(np.floor(pts[:, 0].min())), int(np.floor(pts[:, 1].min()))
            x1, y1 = int(np.ceil(pts[:, 0].max())) + 1, int(np.ceil(pts[:, 1].max())) + 1
        else:
            side = int(max(32, min(w, h) * self.roi_ratio))
            x0, y0 = (w - side) // 2, (h - side) // 2
            x1, y1 = x0 + side, y0 + side
        x0, y0 = max(0, min(x0, w - 32)), max(0, min(y0, h - 32))
        x1, y1 = max(x0 + 32, min(x1, w)), max(y0 + 32, min(y1, h))
        return x0, y0, x1, y1

    def _mask(self, w: int, h: int, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        key = (w, h, bbox, tuple(self.points))
        m = self._mask_cache.get(key)
        if m is None:
            x0, y0, x1, y1 = bbox
            m = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
            pts = self.pixel_points(w, h) - np.array([x0, y0], dtype=np.int32)
            cv2.fillPoly(m, [pts.reshape(-1, 1, 2)], 255)
            self._mask_cache = {key: m}          # keep only the latest geometry
        return m

    def crop(self, frame: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """Return (square crop, bbox).  The crop is what the network sees."""
        h, w = frame.shape[:2]
        bbox = self.bbox(w, h)
        x0, y0, x1, y1 = bbox
        sub = frame[y0:y1, x0:x1]
        if self.is_polygon and self.mask_outside and cv2 is not None:
            sub = sub.copy()
            sub[self._mask(w, h, bbox) == 0] = FILL_COLOR
        return pad_square(sub), bbox


def pad_square(sub: np.ndarray) -> np.ndarray:
    """Pad to a square with neutral grey so a resize to 224x224 keeps the shape."""
    ch, cw = sub.shape[:2]
    if ch == cw:
        return sub
    side = max(ch, cw)
    canvas = np.empty((side, side, 3), dtype=np.uint8)
    canvas[:] = FILL_COLOR
    oy, ox = (side - ch) // 2, (side - cw) // 2
    canvas[oy:oy + ch, ox:ox + cw] = sub
    return canvas


def region_crop(frame: np.ndarray, region: ObjectRegion, expand: float = 0.12, apply_mask: bool = True) -> np.ndarray:
    """Tight, background-free, square crop of one detected object.

    This is the single biggest accuracy lever available: the object fills the
    network's input instead of occupying a fifth of a tray-wide crop.
    """
    h, w = frame.shape[:2]
    x, y, bw, bh = region.box
    ex, ey = int(round(bw * expand)), int(round(bh * expand))
    x0, y0 = max(0, x - ex), max(0, y - ey)
    x1, y1 = min(w, x + bw + ex), min(h, y + bh + ey)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return pad_square(frame[max(0, y):min(h, y + max(8, bh)), max(0, x):min(w, x + max(8, bw))].copy())
    sub = frame[y0:y1, x0:x1].copy()
    if apply_mask and region.mask is not None and region.mask.size and cv2 is not None:
        m = np.zeros(sub.shape[:2], np.uint8)
        mx0, my0 = x - x0, y - y0
        mh, mw = region.mask.shape[:2]
        mh, mw = min(mh, m.shape[0] - my0), min(mw, m.shape[1] - mx0)
        if mh > 0 and mw > 0:
            m[my0:my0 + mh, mx0:mx0 + mw] = region.mask[:mh, :mw]
            # keep a thin rim of context: the object's own shaded edge is informative
            grow = max(3, int(0.04 * max(bw, bh))) | 1
            m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow, grow)))
            sub[m == 0] = FILL_COLOR
    return pad_square(sub)


def center_roi(frame: np.ndarray, ratio: float = 0.6) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Backwards compatible helper: centred square crop."""
    return TrayRegion(None, True, ratio).crop(frame)


# --------------------------------------------------------------------------- #
# Weights management (works offline once the file is bundled)
# --------------------------------------------------------------------------- #
def ensure_weights(dest: str, progress: Optional[Callable[[str], None]] = None) -> str:
    """Make sure the MobileNetV2 ImageNet checkpoint exists at ``dest``.

    Search order: ``dest`` -> torch hub cache -> download from pytorch.org.
    """
    if dest and os.path.isfile(dest):
        return dest
    import torch  # local import keeps module import cheap
    hub_dir = os.path.join(torch.hub.get_dir(), "checkpoints")
    cached = os.path.join(hub_dir, MOBILENET_V2_FILE)
    if os.path.isfile(cached):
        return cached
    target = dest or cached
    os.makedirs(os.path.dirname(os.path.abspath(target)) or ".", exist_ok=True)
    if progress:
        progress(f"downloading {MOBILENET_V2_URL}")
    log.info("downloading MobileNetV2 weights to %s", target)
    torch.hub.download_url_to_file(MOBILENET_V2_URL, target, progress=False)
    return target


# --------------------------------------------------------------------------- #
# MobileNetV2 in pure PyTorch (torchvision-compatible parameter names)
# --------------------------------------------------------------------------- #
def build_mobilenet_v2():
    """An ``nn.Module`` with exactly the layout of ``torchvision.models.mobilenet_v2()``.

    Defined locally so the frozen executable does not depend on torchvision's
    C++ extension (``torchvision._C``), which PyInstaller does not bundle
    reliably.  The official checkpoint loads with ``strict=True``.
    """
    from torch import nn

    def conv_bn_relu(inp, oup, kernel=3, stride=1, groups=1):
        return nn.Sequential(nn.Conv2d(inp, oup, kernel, stride, (kernel - 1) // 2, groups=groups, bias=False),
                             nn.BatchNorm2d(oup), nn.ReLU6(inplace=True))

    class InvertedResidual(nn.Module):
        def __init__(self, inp, oup, stride, expand):
            super().__init__()
            hidden = inp * expand
            self.use_res_connect = stride == 1 and inp == oup
            layers = []
            if expand != 1:
                layers.append(conv_bn_relu(inp, hidden, kernel=1))
            layers += [conv_bn_relu(hidden, hidden, stride=stride, groups=hidden),
                       nn.Conv2d(hidden, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup)]
            self.conv = nn.Sequential(*layers)

        def forward(self, x):
            return x + self.conv(x) if self.use_res_connect else self.conv(x)

    class MobileNetV2(nn.Module):
        SETTINGS = [(1, 16, 1, 1), (6, 24, 2, 2), (6, 32, 3, 2), (6, 64, 4, 2), (6, 96, 3, 1), (6, 160, 3, 2), (6, 320, 1, 1)]

        def __init__(self, num_classes=1000):
            super().__init__()
            features = [conv_bn_relu(3, 32, stride=2)]
            inp = 32
            for t, c, n, s in self.SETTINGS:
                for i in range(n):
                    features.append(InvertedResidual(inp, c, s if i == 0 else 1, t))
                    inp = c
            features.append(conv_bn_relu(inp, EMBEDDING_DIM, kernel=1))
            self.features = nn.Sequential(*features)
            self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(EMBEDDING_DIM, num_classes))

        def forward(self, x):
            x = self.features(x)
            x = nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)
            return self.classifier(x)

    return MobileNetV2()


# --------------------------------------------------------------------------- #
# Feature extractor
# --------------------------------------------------------------------------- #
class FeatureExtractor:
    """MobileNetV2 backbone producing normalised 1280-d embeddings (CPU friendly)."""

    def __init__(self, model_file: str = "", num_threads: int = 1, progress: Optional[Callable[[str], None]] = None):
        import torch
        self.torch = torch
        try:
            torch.set_num_threads(max(1, int(num_threads)))
        except Exception:  # pragma: no cover
            pass
        if progress:
            progress("building MobileNetV2")
        model = build_mobilenet_v2()
        path = ensure_weights(model_file, progress)
        state = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        self.backbone = model.features.eval()
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.weights_path = path
        self.device = torch.device("cpu")
        with torch.inference_mode():                       # warm-up
            self.backbone(torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE))
        log.info("MobileNetV2 ready (weights: %s)", path)

    def preprocess(self, bgr: np.ndarray) -> np.ndarray:
        if bgr.ndim == 2:
            bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
        img = cv2.resize(bgr, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
        return np.ascontiguousarray(rgb.transpose(2, 0, 1))

    def embed_batch(self, images: Sequence[np.ndarray]) -> np.ndarray:
        if len(images) == 0:
            return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
        torch = self.torch
        batch = np.stack([self.preprocess(im) for im in images])
        with torch.inference_mode():
            feats = self.backbone(torch.from_numpy(batch))
            vec = torch.nn.functional.normalize(self.pool(feats).flatten(1), dim=1)
        return vec.cpu().numpy().astype(np.float32)

    def embed(self, image: np.ndarray) -> np.ndarray:
        return self.embed_batch([image])[0]

    @staticmethod
    def augment(image: np.ndarray) -> List[np.ndarray]:
        """Orientation variants for enrolment (produce is placed at random angles).

        Flips and quarter turns only.  Brightness / white-balance augmentation was
        measured at -3 to -8 points of top-1: lighting is handled by removing the
        top principal component instead, which is where that nuisance lives.
        """
        return [image, cv2.flip(image, 1), cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE),
                cv2.rotate(image, cv2.ROTATE_180), cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)]


AUGMENT_FACTOR = 5


def make_thumbnail(image: np.ndarray, size: int = 96, quality: int = 80) -> Optional[bytes]:
    if cv2 is None or image is None or image.size == 0:
        return None
    ok, buf = cv2.imencode(".jpg", cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else None


# --------------------------------------------------------------------------- #
# Item database
# --------------------------------------------------------------------------- #
@dataclass
class Item:
    id: str
    name: str
    price_per_kg: float
    embeddings: np.ndarray = field(default_factory=lambda: np.zeros((0, EMBEDDING_DIM), dtype=np.float32))
    thumbnail: Optional[bytes] = None
    created: float = field(default_factory=time.time)
    updated: float = field(default_factory=time.time)
    sessions: int = 0
    groups: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.int32))
    threshold: float = 0.0                 # calibrated from impostors; 0 = not calibrated yet
    crop_mode: str = MODE_TRAY             # "tray" (legacy) or "object" (per-object crops)
    drift: np.ndarray = field(default_factory=lambda: np.zeros((0, EMBEDDING_DIM), dtype=np.float32))
    unit: str = "kg"                       # "kg" or "pcs"
    price_per_piece: float = 0.0
    avg_unit_weight_kg: float = 0.0

    @property
    def sample_count(self) -> int:
        return int(self.embeddings.shape[0])

    @property
    def capture_count(self) -> int:
        return int(len(np.unique(self.groups))) if self.groups.size else max(1, self.sample_count // AUGMENT_FACTOR)

    @property
    def drift_count(self) -> int:
        return int(self.drift.shape[0])

    def all_embeddings(self) -> np.ndarray:
        if self.drift_count and self.sample_count:
            return np.vstack([self.embeddings, self.drift])
        return self.drift if self.drift_count else self.embeddings


class ItemDatabase:
    """Pickle-backed store: items, the empty-tray reference and the tray image model.

    Version 3 adds ``groups`` (which capture each embedding came from),
    ``threshold`` (calibrated per item), ``crop_mode``, the bounded ``drift``
    buffer and the empty-tray image model.  Version 2 files load unchanged: the
    missing fields are synthesized and nothing needs re-enrolling.
    """

    VERSION = 3

    def __init__(self, path: str = "items_db.pkl", autoload: bool = True):
        self.path = path
        self._items: Dict[str, Item] = {}
        self._lock = threading.RLock()
        self.background: Optional[np.ndarray] = None      # (n, 1280) embeddings of the empty tray
        self.background_thumb: Optional[bytes] = None
        self.background_time: float = 0.0
        self.background_image: Optional[Dict[str, Any]] = None    # EmptyTrayModel payload
        self.load_error: Optional[str] = None
        self.backbone_mismatch: Optional[str] = None
        self.calib_percentile = CALIB_PERCENTILE
        self.use_whitened = True
        # derived, rebuilt by _rebuild()
        self._mu: Optional[np.ndarray] = None
        self._pc: Optional[np.ndarray] = None
        self._white: Optional[np.ndarray] = None
        self._gallery: Dict[str, Dict[str, Any]] = {}
        if autoload:
            self.load()

    # ------------------------------------------------------------ persistence
    def load(self) -> None:
        with self._lock:
            self._items = {}
            self.background, self.background_thumb, self.background_time = None, None, 0.0
            self.background_image = None
            self.backbone_mismatch = None
            if os.path.isfile(self.path):
                try:
                    with open(self.path, "rb") as fh:
                        payload = pickle.load(fh)
                    items = payload.get("items", []) if isinstance(payload, dict) else payload
                    backbone = payload.get("backbone", BACKBONE_ID) if isinstance(payload, dict) else BACKBONE_ID
                    if backbone != BACKBONE_ID:
                        # never let a dimension mismatch explode inside reshape()
                        self.backbone_mismatch = str(backbone)
                        log.error("items database was enrolled with backbone %r, this build uses %r", backbone, BACKBONE_ID)
                        items = []
                    for raw in items:
                        item = Item(**{k: raw[k] for k in ("id", "name", "price_per_kg")})
                        item.embeddings = np.asarray(raw.get("embeddings", []), dtype=np.float32).reshape(-1, EMBEDDING_DIM)
                        item.thumbnail = raw.get("thumbnail")
                        item.created = raw.get("created", time.time())
                        item.updated = raw.get("updated", item.created)
                        item.sessions = raw.get("sessions", 1)
                        item.threshold = float(raw.get("threshold", 0.0))
                        item.crop_mode = raw.get("crop_mode", MODE_TRAY)
                        item.unit = raw.get("unit", "kg")
                        item.price_per_piece = float(raw.get("price_per_piece", 0.0))
                        item.avg_unit_weight_kg = float(raw.get("avg_unit_weight_kg", 0.0))
                        item.drift = np.asarray(raw.get("drift", []), dtype=np.float32).reshape(-1, EMBEDDING_DIM)
                        groups = raw.get("groups")
                        item.groups = (np.asarray(groups, dtype=np.int32).reshape(-1) if groups is not None
                                       else self._synth_groups(item.sample_count))
                        if item.groups.shape[0] != item.sample_count:
                            item.groups = self._synth_groups(item.sample_count)
                        self._items[item.id] = item
                    if isinstance(payload, dict):
                        bg = payload.get("background")
                        if bg and bg.get("embeddings") is not None and len(bg["embeddings"]):
                            self.background = np.asarray(bg["embeddings"], dtype=np.float32).reshape(-1, EMBEDDING_DIM)
                            self.background_thumb = bg.get("thumbnail")
                            self.background_time = bg.get("time", 0.0)
                        self.background_image = payload.get("background_image")
                    self.load_error = None
                except Exception as exc:
                    self.load_error = f"{type(exc).__name__}: {exc}"
                    log.error("items database unreadable (%s); starting empty. Backup kept.", exc)
                    try:
                        os.replace(self.path, self.path + ".corrupt")
                    except OSError:
                        pass
            self._rebuild()

    @staticmethod
    def _synth_groups(n: int) -> np.ndarray:
        """A v2 database stored no capture index.  Enrolment appended exactly one
        block of AUGMENT_FACTOR embeddings per button press, so contiguous blocks
        reconstruct it; otherwise fall back to one capture per sample."""
        if n <= 0:
            return np.zeros((0,), dtype=np.int32)
        if n % AUGMENT_FACTOR == 0:
            return np.repeat(np.arange(n // AUGMENT_FACTOR, dtype=np.int32), AUGMENT_FACTOR)
        return np.arange(n, dtype=np.int32)

    def save(self) -> None:
        with self._lock:
            payload = {
                "version": self.VERSION, "saved": time.time(), "backbone": BACKBONE_ID,
                "items": [{"id": it.id, "name": it.name, "price_per_kg": it.price_per_kg,
                           "embeddings": it.embeddings, "groups": it.groups, "drift": it.drift,
                           "threshold": it.threshold, "crop_mode": it.crop_mode,
                           "unit": it.unit, "price_per_piece": it.price_per_piece,
                           "avg_unit_weight_kg": it.avg_unit_weight_kg,
                           "thumbnail": it.thumbnail, "created": it.created, "updated": it.updated,
                           "sessions": it.sessions}
                          for it in self._items.values()],
                "background": {"embeddings": self.background, "thumbnail": self.background_thumb,
                               "time": self.background_time} if self.background is not None else None,
                "background_image": self.background_image,
            }
            directory = os.path.dirname(os.path.abspath(self.path)) or "."
            os.makedirs(directory, exist_ok=True)
            fd, tmp = tempfile.mkstemp(prefix="items_", suffix=".tmp", dir=directory)
            try:
                with os.fdopen(fd, "wb") as fh:
                    pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
                    fh.flush()
                    os.fsync(fh.fileno())
                os.replace(tmp, self.path)
            finally:
                if os.path.exists(tmp):
                    try:
                        os.remove(tmp)
                    except OSError:
                        pass

    # ------------------------------------------------- empty tray reference
    def set_background(self, embeddings: np.ndarray, thumbnail: Optional[bytes] = None,
                       image_payload: Optional[Dict[str, Any]] = None) -> None:
        with self._lock:
            self.background = np.asarray(embeddings, dtype=np.float32).reshape(-1, EMBEDDING_DIM)
            self.background_thumb = thumbnail
            self.background_time = time.time()
            if image_payload is not None:
                self.background_image = image_payload
            self._rebuild()

    def clear_background(self) -> None:
        with self._lock:
            self.background, self.background_thumb, self.background_time = None, None, 0.0
            self.background_image = None
            self._rebuild()

    @property
    def has_background(self) -> bool:
        return self.background is not None and len(self.background) > 0

    @property
    def has_background_image(self) -> bool:
        return bool(self.background_image)

    def empty_tray_model(self) -> Optional[EmptyTrayModel]:
        return EmptyTrayModel.from_payload(self.background_image) if self.background_image else None

    # ------------------------------------------------------------------ CRUD
    def add_item(self, name: str, price_per_kg: float, embeddings: np.ndarray, thumbnail: Optional[bytes] = None,
                 groups: Optional[np.ndarray] = None, crop_mode: str = MODE_TRAY) -> Item:
        with self._lock:
            emb = np.asarray(embeddings, dtype=np.float32).reshape(-1, EMBEDDING_DIM)
            item = Item(id=uuid.uuid4().hex[:12], name=name.strip(), price_per_kg=float(price_per_kg),
                        embeddings=emb, thumbnail=thumbnail, sessions=1, crop_mode=crop_mode,
                        groups=(np.asarray(groups, np.int32).reshape(-1) if groups is not None
                                else self._synth_groups(emb.shape[0])))
            self._items[item.id] = item
            self._rebuild()
            return item

    def add_samples(self, item_id: str, embeddings: np.ndarray, thumbnail: Optional[bytes] = None,
                    groups: Optional[np.ndarray] = None, crop_mode: Optional[str] = None) -> Item:
        with self._lock:
            item = self._items[item_id]
            new = np.asarray(embeddings, dtype=np.float32).reshape(-1, EMBEDDING_DIM)
            base = int(item.groups.max()) + 1 if item.groups.size else 0
            g = (np.asarray(groups, np.int32).reshape(-1) if groups is not None
                 else self._synth_groups(new.shape[0])) + base
            item.embeddings = np.vstack([item.embeddings, new]) if item.sample_count else new
            item.groups = np.concatenate([item.groups, g]) if item.groups.size else g
            item.updated = time.time()
            item.sessions += 1
            if crop_mode:
                item.crop_mode = crop_mode
            if thumbnail and not item.thumbnail:
                item.thumbnail = thumbnail
            self._rebuild()
            return item

    def reinforce(self, item_id: str, embedding: np.ndarray, cap: int = 5) -> Optional[Item]:
        """Add one operator-confirmed view to a bounded ring buffer.

        Called ONLY when the operator picks the item by hand, never from a model
        prediction: unbounded self-training was measured to collapse accuracy by
        14 points by injecting its own mistakes.  The enrolment samples are
        immutable, so five correct taps evict a bad one.
        """
        if cap <= 0:
            return None
        with self._lock:
            item = self._items.get(item_id)
            if item is None:
                return None
            e = np.asarray(embedding, np.float32).reshape(1, EMBEDDING_DIM)
            item.drift = np.vstack([item.drift, e])[-int(cap):] if item.drift_count else e
            item.updated = time.time()
            self._rebuild()
            return item

    def forget_drift(self, item_id: str) -> None:
        with self._lock:
            item = self._items.get(item_id)
            if item is not None:
                item.drift = np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
                self._rebuild()

    def update_item(self, item_id: str, name: Optional[str] = None, price_per_kg: Optional[float] = None) -> Item:
        with self._lock:
            item = self._items[item_id]
            if name is not None and name.strip():
                item.name = name.strip()
            if price_per_kg is not None:
                item.price_per_kg = float(price_per_kg)
            item.updated = time.time()
            return item

    def delete_item(self, item_id: str) -> None:
        with self._lock:
            self._items.pop(item_id, None)
            self._rebuild()

    def clear_samples(self, item_id: str) -> None:
        with self._lock:
            item = self._items[item_id]
            item.embeddings = np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
            item.groups = np.zeros((0,), dtype=np.int32)
            item.drift = np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
            item.updated = time.time()
            self._rebuild()

    def get(self, item_id: str) -> Optional[Item]:
        with self._lock:
            return self._items.get(item_id)

    def find_by_name(self, name: str) -> Optional[Item]:
        key = name.strip().lower()
        with self._lock:
            for it in self._items.values():
                if it.name.lower() == key:
                    return it
        return None

    def all_items(self) -> List[Item]:
        with self._lock:
            return sorted(self._items.values(), key=lambda it: it.name.lower())

    def items_in_mode(self, mode: str) -> List[Item]:
        return [it for it in self.all_items() if it.crop_mode == mode and it.sample_count]

    def __len__(self) -> int:
        return len(self._items)

    # ------------------------------------------------ projections & galleries
    def _fit_projection(self, X: Optional[np.ndarray]) -> None:
        """Fit the nuisance-removal and whitening transforms on the enrolled set.

        Nothing here is persisted: an SVD of a few hundred 1280-d rows takes
        about 2 ms, and refitting keeps the transform correct as the catalogue
        grows.  Stored embeddings always stay raw and L2-normalised.
        """
        self._mu = self._pc = self._white = None
        if X is None or X.shape[0] < 4:
            return
        self._mu = X.mean(0, keepdims=True).astype(np.float32)
        try:
            _, S, Vt = np.linalg.svd(X - self._mu, full_matrices=False)
        except np.linalg.LinAlgError:  # pragma: no cover
            self._mu = None
            return
        self._pc = Vt[:1].astype(np.float32)
        d = int(min(WHITEN_DIM, len(S)))
        if d >= 2:
            scale = np.sqrt(S[:d] ** 2 / max(1, len(S)) + 1e-3)
            self._white = (Vt[:d] / scale[:, None]).astype(np.float32)

    def project(self, E: np.ndarray) -> np.ndarray:
        """Rejection head: remove the top principal component, then re-normalise.

        The top component of tray-crop embeddings encodes the tray and the light,
        not the produce, which is exactly the nuisance few-shot cosine matching
        cannot otherwise escape.
        """
        E = np.atleast_2d(np.asarray(E, dtype=np.float32))
        if self._pc is None or self._mu is None:
            return E
        Z = E - self._mu
        Z = Z - (Z @ self._pc.T) @ self._pc
        return (Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-9)).astype(np.float32)

    def project_white(self, E: np.ndarray) -> np.ndarray:
        """Ranking head: PCA-whitening to 32 dims. Best at ranking, useless for
        thresholding - which is why the accept decision uses ``project`` instead."""
        E = np.atleast_2d(np.asarray(E, dtype=np.float32))
        if self._white is None or self._mu is None:
            return self.project(E)
        Z = (E - self._mu) @ self._white.T
        return (Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-9)).astype(np.float32)

    def _rebuild(self) -> None:
        """Refit projections, centroids and per-item thresholds together.

        They must move as one: a stale trio silently mismatches ranking and
        gating, which is far worse than either being slightly out of date.
        """
        parts: List[np.ndarray] = [it.all_embeddings() for it in self._items.values() if it.sample_count or it.drift_count]
        if self.background is not None and len(self.background):
            parts.append(self.background)
        X = np.vstack(parts).astype(np.float32) if parts else None
        self._fit_projection(X)

        gallery: Dict[str, Dict[str, Any]] = {}
        for mode in (MODE_TRAY, MODE_OBJECT):
            ids, cents, wcents, raw = [], [], [], []
            members = [it for it in self._items.values() if it.crop_mode == mode and (it.sample_count or it.drift_count)]
            for it in members:
                emb = it.all_embeddings()
                p = self.project(emb)
                c = p.mean(0)
                cents.append(c / max(1e-9, float(np.linalg.norm(c))))
                wp = self.project_white(emb)
                wc = wp.mean(0)
                wcents.append(wc / max(1e-9, float(np.linalg.norm(wc))))
                ids.append(it.id)
                raw.append(emb)
            if mode == MODE_TRAY and self.background is not None and len(self.background):
                # the empty tray is scored as an ordinary class: a hand-set 0.88
                # gate sat inside the noise and missed 20-37% of empty trays
                p = self.project(self.background)
                c = p.mean(0)
                cents.append(c / max(1e-9, float(np.linalg.norm(c))))
                wp = self.project_white(self.background)
                wc = wp.mean(0)
                wcents.append(wc / max(1e-9, float(np.linalg.norm(wc))))
                ids.append(EMPTY_ID)
                raw.append(self.background)
            gallery[mode] = {
                "ids": ids,
                "centroids": np.vstack(cents).astype(np.float32) if cents else None,
                "white": np.vstack(wcents).astype(np.float32) if wcents else None,
                "raw": raw,
                "use_white": self._white is not None and len(ids) >= 3 and sum(r.shape[0] for r in raw) >= 24,
            }
        self._gallery = gallery
        self._calibrate()

    def _calibrate(self) -> None:
        """Per-item accept threshold = the q-th percentile of the impostor scores.

        Every other item's samples (and the empty tray) are impostors.  The
        obvious alternative, a per-item threshold from the item's own spread, was
        measured to be *worse* than a single global number, so it is used only as
        a floor: with two enrolled items the impostors sit near -0.9 in projected
        space and the percentile alone would accept anything positive, including
        produce that was never enrolled.
        """
        q = float(self.calib_percentile)
        for mode, g in self._gallery.items():
            ids, cents, raw = g["ids"], g["centroids"], g["raw"]
            if cents is None or not ids:
                self._empty_threshold = getattr(self, "_empty_threshold", 0.0)
                continue
            projected = [self.project(r) for r in raw]
            for i, iid in enumerate(ids):
                neg = [projected[j] for j in range(len(ids)) if j != i]
                impostor = float(np.percentile(np.vstack(neg) @ cents[i], q)) if neg else -1.0
                thr = max(impostor, self._genuine_floor(ids[i], projected[i], cents[i]), 0.0)
                if iid == EMPTY_ID:
                    self._empty_threshold = thr
                else:
                    item = self._items.get(iid)
                    if item is not None:
                        item.threshold = thr

    def _genuine_floor(self, item_id: str, projected: np.ndarray, centroid: np.ndarray) -> float:
        """How low a *genuine* held-out view of this item is expected to score.

        Each capture is scored against the centroid built without it, which is an
        honest estimate of a never-seen view, unlike in-sample self-similarity
        (every sample helped build its own centroid, so it always scores high).

        This is the floor under the impostor threshold.  The impostor statistic
        alone protects against the other enrolled items, but says nothing about
        an object unlike anything in the catalogue: with a handful of mutually
        dissimilar items its percentile sits near zero and an unrelated object
        walks straight in.
        """
        item = self._items.get(item_id)
        if item is None:                                # the empty-tray pseudo-item
            return SELF_FLOOR_FRAC * float(np.median(projected @ centroid))
        g = item.groups
        n = int(item.embeddings.shape[0])
        if g.size != n or n == 0 or len(np.unique(g)) < 2:
            return SELF_FLOOR_FRAC * float(np.median(projected @ centroid))
        own = projected[:n]                             # drift samples carry no capture index
        scores: List[float] = []
        for cap in np.unique(g):
            mask = g == cap
            rest = own[~mask]
            if rest.shape[0] == 0:
                continue
            c = rest.mean(0)
            norm = float(np.linalg.norm(c))
            if norm < 1e-9:
                continue
            scores.extend((own[mask] @ (c / norm)).tolist())
        if not scores:
            return SELF_FLOOR_FRAC * float(np.median(projected @ centroid))
        return float(np.percentile(scores, 5)) - LOO_SLACK

    def gallery(self, mode: str) -> Dict[str, Any]:
        with self._lock:
            return self._gallery.get(mode, {"ids": [], "centroids": None, "white": None, "raw": [], "use_white": False})

    def item_threshold(self, item_id: str) -> float:
        if item_id == EMPTY_ID:
            return float(getattr(self, "_empty_threshold", 0.0))
        item = self._items.get(item_id)
        return float(item.threshold) if item is not None else 0.0

    # --------------------------------------------------------- diagnostics
    def confusable_pairs(self, limit: int = 10) -> List[Tuple[str, str, float]]:
        """Items whose own samples are pulled toward another item's centroid.

        Computed from the enrolled set alone, and it does predict real confusion
        (Spearman 0.60-0.86 against measured query confusion), so it is worth
        telling the shop before they discover it at the till.
        """
        with self._lock:
            out: List[Tuple[str, str, float]] = []
            for mode in (MODE_TRAY, MODE_OBJECT):
                g = self._gallery.get(mode) or {}
                ids, cents = g.get("ids") or [], g.get("centroids")
                if cents is None or len(ids) < 2:
                    continue
                raw = g["raw"]
                for i, iid in enumerate(ids):
                    if iid == EMPTY_ID:
                        continue
                    p = self.project(raw[i])
                    own = p @ cents[i]
                    for j, jid in enumerate(ids):
                        if j == i or jid == EMPTY_ID:
                            continue
                        cross = p @ cents[j]
                        conf = float((cross > own).mean())
                        if conf > 0.0:
                            a, b = self._items.get(iid), self._items.get(jid)
                            if a is not None and b is not None:
                                out.append((a.name, b.name, conf))
            out.sort(key=lambda t: t[2], reverse=True)
            return out[:limit]


# --------------------------------------------------------------------------- #
# Matching
# --------------------------------------------------------------------------- #
@dataclass
class MatchResult:
    item_id: Optional[str]
    name: str
    score: float                       # the gate score, i.e. the number the decision used
    accepted: bool
    rank_score: float = 0.0
    threshold: float = 0.0
    margin: float = 0.0
    ranking: List[Tuple[str, str, float]] = field(default_factory=list)
    price_per_kg: float = 0.0
    empty: bool = False                # the empty-tray class won (tray mode only)
    timestamp: float = field(default_factory=time.time)


class Recognizer:
    """Two-headed matcher: rank in whitened space, accept on the rmtop1 score."""

    def __init__(self, db: ItemDatabase, threshold: float = 0.65, top_k: int = 3, empty_threshold: float = 0.88):
        self.db = db
        self.threshold = float(threshold)      # slider: an offset on the calibrated thresholds
        self.top_k = max(1, int(top_k))
        self.empty_threshold = float(empty_threshold)

    def effective_threshold(self, item_id: str) -> float:
        """Calibrated threshold, nudged by the Settings slider.

        The slider defaults to 0.65 and is then a no-op; moving it up makes every
        item stricter, down looser.  It cannot be used as a raw floor because the
        calibrated numbers live in the projected space, not in raw cosine space.
        """
        base = self.db.item_threshold(item_id)
        if base <= 0.0:
            return self.threshold          # a single item with no impostors at all
        return float(np.clip(base + (self.threshold - 0.65), 0.05, 0.995))

    def match(self, embedding: np.ndarray, mode: str = MODE_TRAY) -> MatchResult:
        g = self.db.gallery(mode)
        ids, cents = g["ids"], g["centroids"]
        if cents is None or not ids:
            return MatchResult(None, "", 0.0, False)
        e = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        gate_scores = (self.db.project(e) @ cents.T)[0]
        if g["use_white"] and g["white"] is not None:
            rank_scores = (self.db.project_white(e) @ g["white"].T)[0]
        else:
            rank_scores = gate_scores
        order = np.argsort(-rank_scores)
        best = int(order[0])
        best_id = ids[best]
        gate = float(gate_scores[best])
        margin = float(rank_scores[best] - rank_scores[int(order[1])]) if len(order) > 1 else 0.0
        ranking = [(ids[i], self._name(ids[i]), float(gate_scores[i])) for i in order[:5]]
        empty = best_id == EMPTY_ID
        thr = self.effective_threshold(best_id)
        accepted = (not empty) and gate >= thr
        item = self.db.get(best_id) if not empty else None
        return MatchResult(best_id if accepted else None, item.name if item else "", gate, accepted,
                           float(rank_scores[best]), thr, margin, ranking,
                           item.price_per_kg if item else 0.0, empty)

    def _name(self, item_id: str) -> str:
        if item_id == EMPTY_ID:
            return "-"
        it = self.db.get(item_id)
        return it.name if it else "?"


class TemporalSmoother:
    """Majority vote over the last ``window`` results to avoid flicker."""

    def __init__(self, window: int = 5, min_ratio: float = 0.6):
        self.window = max(1, int(window))
        self.min_ratio = min_ratio
        self._hist: Deque[Optional[str]] = deque(maxlen=self.window)
        self._scores: Deque[float] = deque(maxlen=self.window)

    def reset(self) -> None:
        self._hist.clear()
        self._scores.clear()

    def update(self, item_id: Optional[str], score: float) -> Tuple[Optional[str], float]:
        self._hist.append(item_id)
        self._scores.append(score)
        winner, count = Counter(self._hist).most_common(1)[0]
        if winner is None or count < max(1, int(np.ceil(self.min_ratio * len(self._hist)))):
            return None, float(np.mean(self._scores)) if self._scores else 0.0
        scores = [s for h, s in zip(self._hist, self._scores) if h == winner]
        return winner, float(np.mean(scores))


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #
@dataclass
class Detection:
    """One recognised thing on the tray."""
    box: Tuple[int, int, int, int]
    result: MatchResult
    region: Optional[ObjectRegion] = None
    item_id: Optional[str] = None          # after temporal smoothing
    name: str = ""
    score: float = 0.0
    confirmed: bool = True
    track_id: int = 0
    area_frac: float = 1.0
    split: bool = False

    @property
    def accepted(self) -> bool:
        return self.item_id is not None


@dataclass
class Recognition:
    """What the UI displays for the current frame."""
    detections: List[Detection] = field(default_factory=list)
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)      # the tray bbox
    empty: bool = False
    empty_by_weight: Optional[bool] = None
    empty_score: float = 0.0
    intrusion: bool = False
    unreliable: bool = False
    segmented: bool = False
    seg_ms: float = 0.0
    infer_ms: float = 0.0
    mode: str = MODE_TRAY

    @property
    def primary(self) -> Optional[Detection]:
        for d in self.detections:
            if d.confirmed and d.accepted:
                return d
        return self.detections[0] if self.detections else None

    @property
    def confirmed(self) -> List[Detection]:
        return [d for d in self.detections if d.confirmed]

    @property
    def accepted(self) -> List[Detection]:
        return [d for d in self.detections if d.confirmed and d.accepted]

    @property
    def distinct_items(self) -> List[str]:
        return sorted({d.item_id for d in self.accepted if d.item_id})

    # --- convenience mirrors of the primary detection (used by the GUI) ---
    @property
    def stable_item_id(self) -> Optional[str]:
        p = self.primary
        return p.item_id if p else None

    @property
    def stable_name(self) -> str:
        p = self.primary
        return p.name if p else ""

    @property
    def stable_score(self) -> float:
        p = self.primary
        return p.score if p else 0.0

    @property
    def result(self) -> MatchResult:
        p = self.primary
        return p.result if p else MatchResult(None, "", 0.0, False)


class RecognitionEngine(threading.Thread):
    """Segments the tray and recognises every object, in the background.

    ``get_frame`` returns ``(frame_id, frame)``; frames already seen are skipped.
    ``request_embedding`` computes embeddings for a captured sample inside this
    thread, so the network is never used from two threads at once.
    """

    def __init__(self, extractor: FeatureExtractor, db: ItemDatabase, get_frame: Callable[[], Tuple[int, Optional[np.ndarray]]],
                 tray: Optional[TrayRegion] = None, threshold: float = 0.65, empty_threshold: float = 0.88,
                 infer_fps: float = 5.0, smoothing: int = 5, top_k: int = 3,
                 on_result: Optional[Callable[[Recognition, np.ndarray], None]] = None,
                 motion_gate: bool = True, motion_threshold: float = 6.0, max_idle_s: float = 1.5,
                 per_object: bool = True, max_objects: int = 5, min_area_frac: float = 0.008,
                 background_refresh: bool = True):
        super().__init__(name="RecognitionEngine", daemon=True)
        self.extractor, self.db, self.get_frame = extractor, db, get_frame
        self.tray = tray or TrayRegion()
        self.recognizer = Recognizer(db, threshold, top_k, empty_threshold)
        self.infer_fps = max(0.5, float(infer_fps))
        self.on_result = on_result
        self.motion_gate, self.motion_threshold, self.max_idle_s = bool(motion_gate), float(motion_threshold), float(max_idle_s)
        self.per_object = bool(per_object)
        self.max_objects = int(max_objects)
        self.min_area_frac = float(min_area_frac)
        self.background_refresh = bool(background_refresh)
        self.smoothing = int(smoothing)
        self._stop_evt = threading.Event()
        self._requests: Deque[Tuple[np.ndarray, Callable[[np.ndarray], None], bool]] = deque()
        self._req_lock = threading.Lock()
        self._last_frame_id = -1
        self._last_small: Optional[np.ndarray] = None
        self._last_infer_t = 0.0
        self._weight_empty: Optional[bool] = None
        self._smoothers: Dict[int, TemporalSmoother] = {}
        self._cache: Dict[int, Tuple[Tuple[int, int, int, int], MatchResult, float]] = {}
        self._last_refresh = 0.0
        self.segmenter: Optional[TraySegmenter] = None
        self.tracker = RegionTracker()
        self.paused = False
        self.last: Optional[Recognition] = None
        self.stats_ms = 0.0
        self.seg_ms = 0.0
        self.reload_segmenter()

    # ------------------------------------------------------- runtime tuning
    def reload_segmenter(self) -> None:
        """(Re)build the segmenter from the stored empty-tray image model."""
        model = None
        try:
            model = self.db.empty_tray_model()
        except Exception:
            log.exception("empty-tray model unreadable")
        self.segmenter = (TraySegmenter(model, min_area_frac=self.min_area_frac, max_objects=self.max_objects)
                          if model is not None else None)
        self.tracker.reset()
        self._cache.clear()
        log.info("segmenter %s", "ready" if self.segmenter else "disabled (no empty-tray image)")

    def set_threshold(self, value: float) -> None:
        self.recognizer.threshold = float(value)

    def set_empty_threshold(self, value: float) -> None:
        self.recognizer.empty_threshold = float(value)

    def set_tray(self, tray: TrayRegion) -> None:
        self.tray = tray.copy()
        self._smoothers.clear()
        self._cache.clear()
        self.tracker.reset()

    def set_smoothing(self, window: int) -> None:
        self.smoothing = int(window)
        self._smoothers.clear()

    def set_motion_gate(self, enabled: bool, threshold: Optional[float] = None) -> None:
        self.motion_gate = bool(enabled)
        if threshold is not None:
            self.motion_threshold = float(threshold)
        self._last_small = None

    def set_per_object(self, enabled: bool) -> None:
        self.per_object = bool(enabled)
        self._cache.clear()

    def set_weight_hint(self, tray_empty: Optional[bool]) -> None:
        self._weight_empty = tray_empty

    def reset_smoothing(self) -> None:
        self._smoothers.clear()
        self._cache.clear()
        self.tracker.reset()

    @property
    def has_segmenter(self) -> bool:
        return self.segmenter is not None

    # -------------------------------------------------- on-demand embeddings
    def request_embedding(self, image: np.ndarray, callback: Callable[[np.ndarray], None], augment: bool = True) -> None:
        with self._req_lock:
            self._requests.append((image, callback, augment))

    def current_crop(self) -> Optional[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """Crop of the newest frame with the current tray region (GUI thread safe)."""
        _, frame = self.get_frame()
        if frame is None:
            return None
        return self.tray.crop(frame)

    def current_sample(self) -> Optional[Tuple[np.ndarray, str, Optional[SegmentResult]]]:
        """What a training capture should store: the single detected object when
        segmentation is available and unambiguous, otherwise the whole tray."""
        _, frame = self.get_frame()
        if frame is None:
            return None
        seg = self.segment_frame(frame)
        if seg is not None and self.per_object and len(seg.regions) == 1 and not seg.intrusion and not seg.unreliable:
            return region_crop(frame, seg.regions[0]), MODE_OBJECT, seg
        return self.tray.crop(frame)[0], MODE_TRAY, seg

    def segment_frame(self, frame: np.ndarray) -> Optional[SegmentResult]:
        seg = self.segmenter
        if seg is None:
            return None
        try:
            return seg.segment(frame)
        except Exception:
            log.exception("segmentation failed")
            return None

    # --------------------------------------------------------------- thread
    def stop(self, timeout: float = 3.0) -> None:
        self._stop_evt.set()
        if self.is_alive():
            self.join(timeout)

    def _serve_requests(self) -> bool:
        served = False
        while True:
            with self._req_lock:
                if not self._requests:
                    return served
                image, callback, augment = self._requests.popleft()
            try:
                variants = self.extractor.augment(image) if augment else [image]
                callback(self.extractor.embed_batch(variants))
            except Exception:
                log.exception("embedding request failed")
            served = True

    def _smoother(self, track_id: int) -> TemporalSmoother:
        s = self._smoothers.get(track_id)
        if s is None:
            s = TemporalSmoother(self.smoothing)
            self._smoothers[track_id] = s
        return s

    def run(self) -> None:
        interval = 1.0 / self.infer_fps
        while not self._stop_evt.is_set():
            t0 = time.time()
            try:
                if self._serve_requests():
                    continue
                frame_id, frame = self.get_frame()
                if frame is None or frame_id == self._last_frame_id or self.paused:
                    time.sleep(0.01)
                    continue
                self._last_frame_id = frame_id
                rec = self._process(frame, t0)
                if rec is not None:
                    self.last = rec
                    if self.on_result:
                        self.on_result(rec, frame)
            except Exception:
                log.exception("recognition step failed")
                time.sleep(0.2)
            dt = time.time() - t0
            if dt < interval:
                time.sleep(interval - dt)

    def _process(self, frame: np.ndarray, t0: float) -> Optional[Recognition]:
        tray_bbox = self.tray.bbox(frame.shape[1], frame.shape[0])
        # ---- motion gate: while the tray image does not change, do not re-run
        if self.motion_gate and cv2 is not None:
            crop, _ = self.tray.crop(frame)
            small = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (32, 32), interpolation=cv2.INTER_AREA).astype(np.int16)
            if (self._last_small is not None and self.last is not None
                    and time.time() - self._last_infer_t < self.max_idle_s
                    and float(np.mean(np.abs(small - self._last_small))) < self.motion_threshold):
                time.sleep(0.03)
                return None
            self._last_small = small
        self._last_infer_t = time.time()

        seg = self.segment_frame(frame)
        self.seg_ms = seg.ms if seg else 0.0
        weight_empty = self._weight_empty

        # ---- the scale is the authority on "is anything on the tray"
        if weight_empty is True:
            self.tracker.reset()
            self._cache.clear()
            if self.background_refresh and seg is not None and not seg.intrusion and not seg.unreliable \
                    and time.time() - self._last_refresh > 2.0:
                self.segmenter.refresh_background(frame)      # type: ignore[union-attr]
                self._last_refresh = time.time()
            rec = Recognition([], tray_bbox, True, weight_empty, 0.0,
                              bool(seg and seg.intrusion), bool(seg and seg.unreliable),
                              seg is not None, self.seg_ms, (time.time() - t0) * 1000.0,
                              MODE_OBJECT if seg is not None else MODE_TRAY)
            return rec

        object_gallery = self.db.gallery(MODE_OBJECT)["centroids"] is not None
        use_objects = bool(seg is not None and self.per_object and object_gallery and seg.regions)
        detections: List[Detection] = []
        empty = False
        empty_score = 0.0

        if seg is not None:
            self.tracker.update(seg.regions)

        if use_objects:
            regions = [r for r in seg.regions if r.confirmed][:self.max_objects]
            unconfirmed = [r for r in seg.regions if not r.confirmed][:self.max_objects]
            todo: List[Tuple[ObjectRegion, bool]] = []
            for r in regions:
                cached = self._cache.get(r.track_id)
                if cached is not None and box_iou(cached[0], r.box) > 0.9 and time.time() - cached[2] < 2.0:
                    detections.append(self._detection(r, cached[1], True))
                else:
                    todo.append((r, True))
            # classify at most three new regions per frame: the CPU cost is linear
            # in the number of crops (46 / 98 / 140 ms for 1 / 2 / 3), so a burst
            # of five would stall the preview
            for r, conf in todo[:3]:
                crop = region_crop(frame, r)
                res = self.recognizer.match(self.extractor.embed(crop), MODE_OBJECT)
                self._cache[r.track_id] = (r.box, res, time.time())
                detections.append(self._detection(r, res, conf))
            for r in unconfirmed:
                cached = self._cache.get(r.track_id)
                detections.append(self._detection(r, cached[1] if cached else MatchResult(None, "", 0.0, False), False))
            detections.sort(key=lambda d: d.area_frac, reverse=True)
            empty = not seg.regions and not seg.intrusion
        else:
            crop, _ = self.tray.crop(frame)
            res = self.recognizer.match(self.extractor.embed(crop), MODE_TRAY)
            empty = bool(res.empty) if weight_empty is None else bool(weight_empty)
            empty_score = float(res.score) if res.empty else 0.0
            det = self._detection(None, res, True, tray_bbox)
            if not empty:
                detections.append(det)

        infer_ms = (time.time() - t0) * 1000.0
        self.stats_ms = 0.8 * self.stats_ms + 0.2 * infer_ms if self.stats_ms else infer_ms
        return Recognition(detections, tray_bbox, empty, weight_empty, empty_score,
                           bool(seg and seg.intrusion), bool(seg and seg.unreliable),
                           seg is not None, self.seg_ms, infer_ms,
                           MODE_OBJECT if use_objects else MODE_TRAY)

    def _detection(self, region: Optional[ObjectRegion], res: MatchResult, confirmed: bool,
                   bbox: Optional[Tuple[int, int, int, int]] = None) -> Detection:
        track = region.track_id if region is not None else 0
        item_id, score = self._smoother(track).update(res.item_id, res.score)
        item = self.db.get(item_id) if item_id else None
        if region is not None:
            box = region.box
        elif bbox is not None:
            box = (bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1])
        else:
            box = (0, 0, 0, 0)
        return Detection(box=box, result=res, region=region, item_id=item_id if item else None,
                         name=item.name if item else "", score=score, confirmed=confirmed,
                         track_id=track, area_frac=region.area_frac if region is not None else 1.0,
                         split=bool(region is not None and region.split_from >= 0))


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    import sys
    t0 = time.time()
    fx = FeatureExtractor(sys.argv[1] if len(sys.argv) > 1 else "", num_threads=1, progress=print)
    print("model load: %.1fs" % (time.time() - t0))
    rng = np.random.default_rng(0)

    tray = np.full((480, 640, 3), 150, np.uint8)
    tray = np.clip(tray.astype(np.int16) + rng.normal(0, 5, tray.shape), 0, 255).astype(np.uint8)
    cv2.rectangle(tray, (140, 90), (500, 400), (170, 165, 160), -1)
    for i in range(0, 640, 17):
        cv2.line(tray, (i, 0), (i, 480), (158, 156, 153), 1)

    def scene(objs):
        img = tray.copy()
        for (cx, cy), colour, rad in objs:
            cv2.circle(img, (cx, cy), rad, colour, -1)
        return np.clip(img.astype(np.int16) + rng.normal(0, 2, img.shape), 0, 255).astype(np.uint8)

    APPLE, BANANA = (40, 40, 200), (40, 200, 220)
    db = ItemDatabase(os.path.join(tempfile.gettempdir(), "aipos_items_test.pkl"), autoload=False)
    frames = [scene([]) for _ in range(BUILD_FRAMES)]
    model = EmptyTrayModel.build(frames, (140, 90, 500, 400), [(140, 90), (500, 90), (500, 400), (140, 400)])
    db.set_background(fx.embed_batch(fx.augment(TrayRegion().crop(frames[0])[0])), None, model.to_payload())
    seg = TraySegmenter(model)

    for name, colour in (("Apple", APPLE), ("Banana", BANANA)):
        embs, groups = [], []
        for g, (cx, cy) in enumerate([(250, 200), (350, 260), (300, 320)]):
            img = scene([((cx, cy), colour, 46)])
            r = seg.segment(img).regions
            crop = region_crop(img, r[0]) if r else TrayRegion().crop(img)[0]
            e = fx.embed_batch(fx.augment(crop))
            embs.append(e)
            groups.append(np.full(len(e), g, np.int32))
        db.add_item(name, 45000, np.vstack(embs), groups=np.concatenate(groups), crop_mode=MODE_OBJECT)

    rec = Recognizer(db, 0.65)
    print("\ncalibrated thresholds:", {i.name: round(i.threshold, 3) for i in db.all_items()})
    for label, objs in (("one apple", [((300, 240), APPLE, 46)]),
                        ("apple + banana", [((230, 190), APPLE, 44), ((400, 300), BANANA, 48)]),
                        ("three objects", [((220, 170), APPLE, 40), ((380, 200), BANANA, 42), ((300, 330), APPLE, 40)])):
        img = scene(objs)
        s = seg.segment(img)
        names = []
        for r in s.regions:
            m = rec.match(fx.embed(region_crop(img, r)), MODE_OBJECT)
            names.append(f"{m.name or '?'}:{m.score:.2f}{'' if m.accepted else '(rejected)'}")
        print(f"{label:16} -> {s.count} regions in {s.ms:.0f} ms: {', '.join(names)}")
    unknown = scene([((300, 240), (40, 220, 60), 46)])          # never enrolled: bright green
    r = seg.segment(unknown).regions
    m = rec.match(fx.embed(region_crop(unknown, r[0])), MODE_OBJECT)
    print(f"{'unknown produce':16} -> best {m.ranking[0][1]} score {m.score:.3f} thr {m.threshold:.3f} accepted={m.accepted}")
    db.save()
    db2 = ItemDatabase(db.path)
    print("\nreloaded:", [(i.name, i.sample_count, i.capture_count, i.crop_mode) for i in db2.all_items()],
          "| background:", db2.has_background, "| tray image:", db2.has_background_image)
