"""
ai_engine.py
------------
Few-shot, embedding based product recognition for the AI POS Scale.

Pipeline
    frame (BGR) -> tray region crop (polygon / rectangle, outside masked)
    -> MobileNetV2 features (ImageNet weights) -> global average pool
    -> 1280-d L2-normalised embedding
    -> cosine similarity against enrolled samples (``items_db.pkl``)
    -> per-item score (mean of top-k sample similarities) -> threshold
    -> "empty tray" check against the stored empty-tray reference
    -> temporal majority vote (stable badge in the UI)

No training loop is needed: enrolling an item just stores a handful of
embeddings, so new products can be added at the till in a few seconds.

Public classes
    * ``TrayRegion``        – where the tray is in the frame + masked square crop
    * ``FeatureExtractor``  – the network + preprocessing
    * ``ItemDatabase``      – persistent store of items / prices / embeddings / empty-tray reference
    * ``Recognizer``        – similarity matcher
    * ``TemporalSmoother``  – majority vote over recent frames
    * ``RecognitionEngine`` – worker thread doing inference + on-demand embeddings
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
from typing import Callable, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore

log = logging.getLogger("ai")

EMBEDDING_DIM = 1280
INPUT_SIZE = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
MOBILENET_V2_URL = "https://download.pytorch.org/models/mobilenet_v2-b0353104.pth"
MOBILENET_V2_FILE = "mobilenet_v2-b0353104.pth"
FILL_COLOR = (114, 114, 114)          # neutral grey used outside the tray and for padding


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

    # ---- construction helpers ------------------------------------------------
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

    # ---- geometry --------------------------------------------------------------
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
        ch, cw = sub.shape[:2]
        if ch != cw:
            side = max(ch, cw)
            canvas = np.empty((side, side, 3), dtype=np.uint8)
            canvas[:] = FILL_COLOR
            oy, ox = (side - ch) // 2, (side - cw) // 2
            canvas[oy:oy + ch, ox:ox + cw] = sub
            sub = canvas
        return sub, bbox


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
    """Return an ``nn.Module`` with exactly the layout of ``torchvision.models.mobilenet_v2()``.

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
        """Orientation variants used when enrolling (produce is placed at random angles)."""
        return [image, cv2.flip(image, 1), cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE),
                cv2.rotate(image, cv2.ROTATE_180), cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)]


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

    @property
    def sample_count(self) -> int:
        return int(self.embeddings.shape[0])


class ItemDatabase:
    """Pickle-backed store (``items_db.pkl``): items + the empty-tray reference."""

    VERSION = 2

    def __init__(self, path: str = "items_db.pkl", autoload: bool = True):
        self.path = path
        self._items: Dict[str, Item] = {}
        self._lock = threading.RLock()
        self._matrix: Optional[np.ndarray] = None
        self._owners: List[str] = []
        self.background: Optional[np.ndarray] = None      # (n, 1280) embeddings of the empty tray
        self.background_thumb: Optional[bytes] = None
        self.background_time: float = 0.0
        self.load_error: Optional[str] = None
        if autoload:
            self.load()

    # persistence ------------------------------------------------------------
    def load(self) -> None:
        with self._lock:
            self._items = {}
            self.background, self.background_thumb, self.background_time = None, None, 0.0
            if os.path.isfile(self.path):
                try:
                    with open(self.path, "rb") as fh:
                        payload = pickle.load(fh)
                    items = payload.get("items", []) if isinstance(payload, dict) else payload
                    for raw in items:
                        item = Item(**{k: raw[k] for k in ("id", "name", "price_per_kg")})
                        item.embeddings = np.asarray(raw.get("embeddings", []), dtype=np.float32).reshape(-1, EMBEDDING_DIM)
                        item.thumbnail = raw.get("thumbnail")
                        item.created = raw.get("created", time.time())
                        item.updated = raw.get("updated", item.created)
                        item.sessions = raw.get("sessions", 1)
                        self._items[item.id] = item
                    bg = payload.get("background") if isinstance(payload, dict) else None
                    if bg and bg.get("embeddings") is not None and len(bg["embeddings"]):
                        self.background = np.asarray(bg["embeddings"], dtype=np.float32).reshape(-1, EMBEDDING_DIM)
                        self.background_thumb = bg.get("thumbnail")
                        self.background_time = bg.get("time", 0.0)
                    self.load_error = None
                except Exception as exc:
                    self.load_error = f"{type(exc).__name__}: {exc}"
                    log.error("items database unreadable (%s); starting empty. Backup kept.", exc)
                    try:
                        os.replace(self.path, self.path + ".corrupt")
                    except OSError:
                        pass
            self._rebuild()

    def save(self) -> None:
        with self._lock:
            payload = {
                "version": self.VERSION, "saved": time.time(),
                "items": [{"id": it.id, "name": it.name, "price_per_kg": it.price_per_kg, "embeddings": it.embeddings,
                           "thumbnail": it.thumbnail, "created": it.created, "updated": it.updated, "sessions": it.sessions}
                          for it in self._items.values()],
                "background": {"embeddings": self.background, "thumbnail": self.background_thumb, "time": self.background_time}
                if self.background is not None else None,
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

    # empty tray reference ------------------------------------------------------
    def set_background(self, embeddings: np.ndarray, thumbnail: Optional[bytes] = None) -> None:
        with self._lock:
            self.background = np.asarray(embeddings, dtype=np.float32).reshape(-1, EMBEDDING_DIM)
            self.background_thumb = thumbnail
            self.background_time = time.time()

    def clear_background(self) -> None:
        with self._lock:
            self.background, self.background_thumb, self.background_time = None, None, 0.0

    @property
    def has_background(self) -> bool:
        return self.background is not None and len(self.background) > 0

    # CRUD ---------------------------------------------------------------------
    def add_item(self, name: str, price_per_kg: float, embeddings: np.ndarray, thumbnail: Optional[bytes] = None) -> Item:
        with self._lock:
            item = Item(id=uuid.uuid4().hex[:12], name=name.strip(), price_per_kg=float(price_per_kg),
                        embeddings=np.asarray(embeddings, dtype=np.float32).reshape(-1, EMBEDDING_DIM),
                        thumbnail=thumbnail, sessions=1)
            self._items[item.id] = item
            self._rebuild()
            return item

    def add_samples(self, item_id: str, embeddings: np.ndarray, thumbnail: Optional[bytes] = None) -> Item:
        with self._lock:
            item = self._items[item_id]
            new = np.asarray(embeddings, dtype=np.float32).reshape(-1, EMBEDDING_DIM)
            item.embeddings = np.vstack([item.embeddings, new]) if item.sample_count else new
            item.updated = time.time()
            item.sessions += 1
            if thumbnail and not item.thumbnail:
                item.thumbnail = thumbnail
            self._rebuild()
            return item

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
        """Forget all samples of an item (e.g. after the tray area changed) but keep name / price."""
        with self._lock:
            item = self._items[item_id]
            item.embeddings = np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
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

    def __len__(self) -> int:
        return len(self._items)

    def _rebuild(self) -> None:
        mats, owners = [], []
        for it in self._items.values():
            if it.sample_count:
                mats.append(it.embeddings)
                owners.extend([it.id] * it.sample_count)
        self._matrix = np.vstack(mats).astype(np.float32) if mats else None
        self._owners = owners

    def matrix(self) -> Tuple[Optional[np.ndarray], List[str]]:
        with self._lock:
            return self._matrix, list(self._owners)

    def background_matrix(self) -> Optional[np.ndarray]:
        with self._lock:
            return self.background


# --------------------------------------------------------------------------- #
# Matching
# --------------------------------------------------------------------------- #
@dataclass
class MatchResult:
    item_id: Optional[str]
    name: str
    score: float
    accepted: bool
    margin: float = 0.0
    ranking: List[Tuple[str, str, float]] = field(default_factory=list)
    price_per_kg: float = 0.0
    empty_score: float = 0.0            # similarity to the stored empty-tray reference
    empty: bool = False                 # vision says: nothing on the tray
    timestamp: float = field(default_factory=time.time)


class Recognizer:
    def __init__(self, db: ItemDatabase, threshold: float = 0.65, top_k: int = 3, empty_threshold: float = 0.88):
        self.db = db
        self.threshold = float(threshold)
        self.top_k = max(1, int(top_k))
        self.empty_threshold = float(empty_threshold)

    def empty_similarity(self, embedding: np.ndarray) -> float:
        bg = self.db.background_matrix()
        if bg is None or len(bg) == 0:
            return 0.0
        return float(np.max(bg @ embedding.astype(np.float32)))

    def match(self, embedding: np.ndarray) -> MatchResult:
        empty_score = self.empty_similarity(embedding)
        matrix, owners = self.db.matrix()
        if matrix is None or matrix.shape[0] == 0:
            return MatchResult(None, "", 0.0, False, empty_score=empty_score, empty=empty_score >= self.empty_threshold)
        sims = matrix @ embedding.astype(np.float32)
        per_item: Dict[str, List[float]] = {}
        for owner, s in zip(owners, sims):
            per_item.setdefault(owner, []).append(float(s))
        scored = []
        for item_id, values in per_item.items():
            values.sort(reverse=True)
            k = min(self.top_k, len(values))
            scored.append((item_id, float(np.mean(values[:k]))))
        scored.sort(key=lambda t: t[1], reverse=True)
        best_id, best_score = scored[0]
        margin = best_score - (scored[1][1] if len(scored) > 1 else 0.0)
        item = self.db.get(best_id)
        ranking = [(iid, (self.db.get(iid).name if self.db.get(iid) else "?"), s) for iid, s in scored[:5]]
        # empty tray wins when the reference matches better than any product
        empty = empty_score >= self.empty_threshold and empty_score >= best_score
        accepted = best_score >= self.threshold and not empty
        return MatchResult(best_id if accepted else None, item.name if item else "", best_score, accepted, margin,
                           ranking, item.price_per_kg if item else 0.0, empty_score, empty)


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
# Worker thread: inference + on-demand embeddings (training / empty tray)
# --------------------------------------------------------------------------- #
@dataclass
class Recognition:
    """What the UI displays for the current frame."""
    result: MatchResult
    stable_item_id: Optional[str]
    stable_name: str
    stable_score: float
    bbox: Tuple[int, int, int, int]
    infer_ms: float
    empty: bool = False                 # final verdict (weight hint overrides vision)
    empty_by_weight: Optional[bool] = None
    empty_score: float = 0.0


class RecognitionEngine(threading.Thread):
    """Runs MobileNetV2 in the background on the newest camera frame.

    * ``get_frame`` must return ``(frame_id, frame)``; frames already seen are skipped.
    * ``request_embedding(image, callback)`` computes embeddings for a captured
      sample (training, empty-tray reference) inside this thread so the model is
      never used from two threads at once.  ``callback(embeddings)`` runs in the
      engine thread; GUI code should forward it through a Qt signal.
    * ``set_weight_hint(True/False/None)`` lets the scale decide whether the tray
      is empty (None = no hardware scale, use vision only).
    """

    def __init__(self, extractor: FeatureExtractor, db: ItemDatabase, get_frame: Callable[[], Tuple[int, Optional[np.ndarray]]],
                 tray: Optional[TrayRegion] = None, threshold: float = 0.65, empty_threshold: float = 0.88,
                 infer_fps: float = 6.0, smoothing: int = 5, top_k: int = 3,
                 on_result: Optional[Callable[[Recognition, np.ndarray], None]] = None,
                 motion_gate: bool = True, motion_threshold: float = 6.0, max_idle_s: float = 1.5):
        super().__init__(name="RecognitionEngine", daemon=True)
        # motion gate: while the tray image does not change, re-run the network only every max_idle_s
        self.motion_gate, self.motion_threshold, self.max_idle_s = bool(motion_gate), float(motion_threshold), float(max_idle_s)
        self._last_small: Optional[np.ndarray] = None
        self._last_infer_t = 0.0
        self.extractor, self.db, self.get_frame = extractor, db, get_frame
        self.tray = tray or TrayRegion()
        self.recognizer = Recognizer(db, threshold, top_k, empty_threshold)
        self.smoother = TemporalSmoother(smoothing)
        self.infer_fps = max(0.5, float(infer_fps))
        self.on_result = on_result
        self._stop_evt = threading.Event()
        self._requests: Deque[Tuple[np.ndarray, Callable[[np.ndarray], None], bool]] = deque()
        self._req_lock = threading.Lock()
        self._last_frame_id = -1
        self._weight_empty: Optional[bool] = None
        self.paused = False
        self.last: Optional[Recognition] = None
        self.stats_ms = 0.0

    # runtime tuning -----------------------------------------------------------
    def set_threshold(self, value: float) -> None:
        self.recognizer.threshold = float(value)

    def set_empty_threshold(self, value: float) -> None:
        self.recognizer.empty_threshold = float(value)

    def set_tray(self, tray: TrayRegion) -> None:
        self.tray = tray.copy()
        self.smoother.reset()

    def set_smoothing(self, window: int) -> None:
        self.smoother = TemporalSmoother(window)

    def set_motion_gate(self, enabled: bool, threshold: Optional[float] = None) -> None:
        self.motion_gate = bool(enabled)
        if threshold is not None:
            self.motion_threshold = float(threshold)
        self._last_small = None

    def set_weight_hint(self, tray_empty: Optional[bool]) -> None:
        self._weight_empty = tray_empty

    def reset_smoothing(self) -> None:
        self.smoother.reset()

    # on-demand embeddings ----------------------------------------------------------
    def request_embedding(self, image: np.ndarray, callback: Callable[[np.ndarray], None], augment: bool = True) -> None:
        with self._req_lock:
            self._requests.append((image, callback, augment))

    def current_crop(self) -> Optional[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """Crop of the newest camera frame with the current tray region (GUI thread safe)."""
        _, frame = self.get_frame()
        if frame is None:
            return None
        return self.tray.crop(frame)

    # main loop --------------------------------------------------------------------
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
                emb = self.extractor.embed_batch(variants)
                callback(emb)
            except Exception:
                log.exception("embedding request failed")
            served = True

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
                crop, bbox = self.tray.crop(frame)
                if self.motion_gate and cv2 is not None:
                    small = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (32, 32), interpolation=cv2.INTER_AREA).astype(np.int16)
                    if (self._last_small is not None and self.last is not None
                            and time.time() - self._last_infer_t < self.max_idle_s
                            and float(np.mean(np.abs(small - self._last_small))) < self.motion_threshold):
                        time.sleep(0.03)
                        continue
                    self._last_small = small
                self._last_infer_t = time.time()
                emb = self.extractor.embed(crop)
                result = self.recognizer.match(emb)
                hint = self._weight_empty
                empty = hint if hint is not None else result.empty
                if empty:
                    stable_id, stable_score = self.smoother.update(None, 0.0)
                    stable_id = None
                else:
                    stable_id, stable_score = self.smoother.update(result.item_id, result.score)
                item = self.db.get(stable_id) if stable_id else None
                infer_ms = (time.time() - t0) * 1000.0
                self.stats_ms = 0.8 * self.stats_ms + 0.2 * infer_ms if self.stats_ms else infer_ms
                rec = Recognition(result, stable_id, item.name if item else "", stable_score, bbox, infer_ms,
                                  empty=bool(empty), empty_by_weight=hint, empty_score=result.empty_score)
                self.last = rec
                if self.on_result:
                    self.on_result(rec, crop)
            except Exception:
                log.exception("recognition step failed")
                time.sleep(0.2)
            dt = time.time() - t0
            if dt < interval:
                time.sleep(interval - dt)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    import sys
    t0 = time.time()
    fx = FeatureExtractor(sys.argv[1] if len(sys.argv) > 1 else "", num_threads=1, progress=print)
    print("model load: %.1fs" % (time.time() - t0))
    rng = np.random.default_rng(0)
    tray = np.full((480, 640, 3), 120, np.uint8)
    cv2.circle(tray, (320, 240), 150, (150, 150, 150), -1)
    apple = tray.copy(); cv2.circle(apple, (320, 240), 80, (40, 40, 220), -1)
    banana = tray.copy(); cv2.ellipse(banana, (320, 240), (110, 40), 30, 0, 360, (40, 220, 240), -1)
    region = TrayRegion([[0.25, 0.15], [0.75, 0.15], [0.8, 0.85], [0.2, 0.85]], True)
    crop_fn = lambda img: region.crop(img)[0]
    db = ItemDatabase(os.path.join(tempfile.gettempdir(), "aipos_items_test.pkl"), autoload=False)
    db.set_background(fx.embed_batch(fx.augment(crop_fn(tray))))
    db.add_item("Apple", 45000, fx.embed_batch(fx.augment(crop_fn(apple))))
    db.add_item("Banana", 62000, fx.embed_batch(fx.augment(crop_fn(banana))))
    rec = Recognizer(db, 0.65, empty_threshold=0.88)
    for label, img in (("empty tray", tray), ("apple", apple), ("banana rotated", cv2.rotate(banana, cv2.ROTATE_90_CLOCKWISE))):
        noisy = np.clip(img.astype(int) + rng.integers(-15, 15, img.shape), 0, 255).astype(np.uint8)
        r = rec.match(fx.embed(crop_fn(noisy)))
        print(f"{label:16} -> item={r.name or '-':8} score={r.score:.3f} empty_score={r.empty_score:.3f} empty={r.empty}")
    db.save(); db2 = ItemDatabase(db.path)
    print("reloaded:", [(i.name, i.sample_count) for i in db2.all_items()], "background:", db2.has_background)
