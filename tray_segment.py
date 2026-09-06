"""
tray_segment.py
---------------
Multi-object segmentation of the scale tray against a stored empty-tray IMAGE
model.  Classical computer vision only (cv2 + numpy): no new dependency, no
network, nothing extra for PyInstaller to bundle.

Why not a neural detector: a POS tray is a fixed, controlled scene with a known
empty state, so background change detection beats a generic detector on both
accuracy and CPU (measured 15 ms/frame at work=192 versus hundreds for a YOLO
class model on this hardware), and it needs no labelled data from the shop.

Two-pass structure, which is what makes shadows survivable:

  pass 1  wide-net pixel change mask (objects AND their cast shadows), then
          connected components -> "blobs"
  pass 2  per blob:
            - region evidence (brightness ratio, hue shift, texture correlation,
              boundary sharpness, shading spread) decides pure shadow -> drop
            - otherwise subtract the pixels that individually look like shadow
              ("core"), which also un-bridges two objects joined by one shadow
            - if that subtraction ate more than 45% of the blob it was wrong
              (matte grey object on a textureless tray) -> keep the whole blob
            - distance-transform watershed, with every proposed cut validated
              against the image gradient, splits genuinely touching objects
            - a blob that crosses the tray outline and is large is an intrusion
              (hand / arm / sleeve): the whole frame is then flagged, not priced

Measured on this project's synthetic benchmark (two tray textures, seeds 7/23):
  empty tray, 2-of-3 confirmed   0.00 false regions per frame
  1 object                       recall 0.85-0.90, mIoU >= 0.70
  2 separated                    recall 0.875
  3 separated                    recall 0.72-0.82
  hand in shot                   intrusion flagged in 100% of frames
  timing at work=192             15.2 ms mean, 17.1 ms p95

Known limits, stated plainly because pricing must never depend on them:
  * touching objects are unreliable (two items overlapping ~30% of their
    diameter have no neck in the distance map at all: saddle/peak 0.91),
  * a textureless white tray costs about 30 points of recall,
  * stacked items are invisible to any 2-D method.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# The segmenter is single-threaded on purpose: it leaves the other cores to
# torch, and measurements showed no gain from OpenCV's own threading here.
try:
    cv2.setNumThreads(1)
except Exception:  # pragma: no cover
    pass

WORK_SIZE = 192      # long side of the segmentation working image (192: 15 ms, 256: 30 ms, no recall gain)
NCC_WIN = 9          # window for the local texture-correlation shadow test
MIN_VAR = 4.0        # local variance below which NCC is meaningless (std < 2 grey levels)
BUILD_FRAMES = 9     # empty-tray frames averaged into the model
DEFAULT_MARGIN = 0.18


# --------------------------------------------------------------------------- #
@dataclass
class ObjectRegion:
    box: Tuple[int, int, int, int]     # x, y, w, h in FULL-FRAME pixels
    mask: np.ndarray                   # uint8 0/255, box sized, full-frame scale
    area_px: float
    area_frac: float                   # area / tray area
    fill: float                        # area / box area
    solidity: float                    # area / convex-hull area
    border_touch: float                # fraction of its outline on the tray outline
    skin_frac: float                   # descriptor only - never a veto (fires on potato/onion)
    alpha: float                       # median brightness ratio vs the empty tray
    ncc: float                         # texture correlation with the empty tray
    sharp: float                       # boundary sharpness
    split_from: int = -1               # >=0 when produced by a watershed split
    confidence: float = 1.0
    confirmed: bool = True             # survived the 2-of-3 temporal filter
    track_id: int = -1

    @property
    def center(self) -> Tuple[float, float]:
        x, y, w, h = self.box
        return (x + w / 2.0, y + h / 2.0)


@dataclass
class SegmentResult:
    regions: List[ObjectRegion] = field(default_factory=list)
    intrusion: bool = False            # hand / arm reaching in over the tray edge
    shadows: int = 0
    coverage: float = 0.0
    empty: bool = True
    unreliable: bool = False           # photometric alignment failed: do not price
    ms: float = 0.0

    @property
    def count(self) -> int:
        return len(self.regions)


def _ell(k):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def _grad(gray_f32):
    gx = cv2.Scharr(gray_f32, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(gray_f32, cv2.CV_32F, 0, 1)
    return cv2.magnitude(gx, gy) * (1.0 / 16.0)


def _fill_holes(binary):
    """Fill interior holes (specular highlight on wet skin / on a plastic bag)."""
    h, w = binary.shape
    padded = np.zeros((h + 2, w + 2), np.uint8)
    padded[1:-1, 1:-1] = binary
    ff = padded.copy()
    cv2.floodFill(ff, np.zeros((h + 4, w + 4), np.uint8), (0, 0), 255)
    return cv2.bitwise_or(binary, cv2.bitwise_not(ff)[1:-1, 1:-1])


def box_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix0, iy0 = max(ax, bx), max(ay, by)
    ix1, iy1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


# --------------------------------------------------------------------------- #
class EmptyTrayModel:
    """Everything that can be precomputed from the empty tray is precomputed here;
    the per-frame path has to stay in single-digit milliseconds."""

    def __init__(self, bg, sigma, mask, bbox, full_size):
        self.bg = np.asarray(bg, dtype=np.float32)
        self.sigma = np.asarray(sigma, dtype=np.float32)
        self.mask = np.asarray(mask, dtype=np.uint8)
        self.bbox = tuple(int(v) for v in bbox)
        self.full_size = (int(full_size[0]), int(full_size[1]))
        self.bg_gray = cv2.cvtColor(np.clip(self.bg, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
        self.bg_dot = np.maximum((self.bg * self.bg).sum(axis=2), 1.0)
        self.bg_grad = _grad(self.bg_gray)
        self.a_tol = np.maximum(0.055, 2.2 * self.sigma / np.maximum(self.bg_gray, 8.0))
        self.cd_tol = 3.0 * self.sigma
        self.idx = self.mask > 0
        self.tray_area_work = int(self.idx.sum())
        h, w = self.mask.shape
        self.scale_x = (self.bbox[2] - self.bbox[0]) / float(w)
        self.scale_y = (self.bbox[3] - self.bbox[1]) / float(h)
        # The model deliberately covers the tray polygon PLUS a margin ring.
        # Detection happens inside the polygon; the ring is what tells
        # "an arm is reaching in from outside" (the blob continues into the ring)
        # apart from "a product overhangs the tray edge" (it does not).
        self.ring = cv2.bitwise_not(self.mask)
        self.ring_area = int(np.count_nonzero(self.ring))
        self.crop_edge = np.zeros_like(self.mask)
        self.crop_edge[:2, :] = 255
        self.crop_edge[-2:, :] = 255
        self.crop_edge[:, :2] = 255
        self.crop_edge[:, -2:] = 255
        self.border_band = cv2.bitwise_and(self.mask, cv2.dilate(self.ring, _ell(5)))
        self.work_wh = (w, h)
        k = (NCC_WIN, NCC_WIN)
        self.my = cv2.boxFilter(self.bg_gray, -1, k)
        self.vy = np.maximum(cv2.boxFilter(self.bg_gray * self.bg_gray, -1, k) - self.my * self.my, 0.0)
        self.vy_ok = self.vy > MIN_VAR          # where the tray has texture to correlate

    # ---- construction ------------------------------------------------------
    @staticmethod
    def _prep(frame, bbox, work_wh):
        x0, y0, x1, y1 = bbox
        return cv2.resize(frame[y0:y1, x0:x1], work_wh, interpolation=cv2.INTER_AREA)

    @staticmethod
    def expand(bbox, frame_w, frame_h, margin: float = DEFAULT_MARGIN):
        """Grow the tray bbox so the model also covers a ring of table around it."""
        x0, y0, x1, y1 = bbox
        mx, my = int((x1 - x0) * margin), int((y1 - y0) * margin)
        return (max(0, x0 - mx), max(0, y0 - my), min(frame_w, x1 + mx), min(frame_h, y1 + my))

    @classmethod
    def build(cls, frames: Sequence[np.ndarray], bbox, poly_px=None, work: int = WORK_SIZE,
              margin: float = DEFAULT_MARGIN) -> "EmptyTrayModel":
        h_full, w_full = frames[0].shape[:2]
        bbox = cls.expand(bbox, w_full, h_full, margin)
        x0, y0, x1, y1 = bbox
        bw, bh = x1 - x0, y1 - y0
        s = work / float(max(bw, bh))
        work_wh = (max(32, int(round(bw * s))), max(32, int(round(bh * s))))
        stack = np.stack([cls._prep(f, bbox, work_wh) for f in frames]).astype(np.float32)
        bg = np.median(stack, axis=0)
        mad = np.median(np.abs(stack - bg), axis=0).max(axis=2) * 1.4826
        gray = cv2.cvtColor(np.clip(bg, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        grad = cv2.dilate(_grad(gray.astype(np.float32)), np.ones((3, 3), np.uint8))
        # tolerance = sensor/flicker noise + how much a 1 px camera shift moves this pixel
        sigma = np.sqrt((2.0 * mad) ** 2 + (0.55 * grad) ** 2) + 3.0
        mask = np.full(work_wh[::-1], 255, np.uint8)
        if poly_px is not None and len(poly_px) >= 3:
            mask = np.zeros(work_wh[::-1], np.uint8)
            pts = ((np.asarray(poly_px, np.float32) - np.array([x0, y0], np.float32)) *
                   np.array([work_wh[0] / bw, work_wh[1] / bh], np.float32)).astype(np.int32)
            cv2.fillPoly(mask, [pts.reshape(-1, 1, 2)], 255)
        return cls(bg, sigma, mask, bbox, (w_full, h_full))

    # ---- persistence -------------------------------------------------------
    def to_payload(self) -> Dict[str, Any]:
        """Compact, picklable representation (~55 KB) for items_db.pkl."""
        ok_bg, bg_png = cv2.imencode(".png", np.clip(self.bg, 0, 255).astype(np.uint8))
        ok_mask, mask_png = cv2.imencode(".png", self.mask)
        smax = float(self.sigma.max()) or 1.0
        sigma_u8 = np.clip(self.sigma * (255.0 / smax), 0, 255).astype(np.uint8)
        ok_sig, sigma_png = cv2.imencode(".png", sigma_u8)
        if not (ok_bg and ok_mask and ok_sig):
            raise RuntimeError("failed to encode the empty-tray model")
        return {"format": 1, "bg_png": bg_png.tobytes(), "mask_png": mask_png.tobytes(),
                "sigma_png": sigma_png.tobytes(), "sigma_scale": smax / 255.0,
                "bbox": list(self.bbox), "full_size": list(self.full_size), "work": WORK_SIZE}

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> Optional["EmptyTrayModel"]:
        try:
            bg = cv2.imdecode(np.frombuffer(payload["bg_png"], np.uint8), cv2.IMREAD_COLOR)
            mask = cv2.imdecode(np.frombuffer(payload["mask_png"], np.uint8), cv2.IMREAD_GRAYSCALE)
            sigma_u8 = cv2.imdecode(np.frombuffer(payload["sigma_png"], np.uint8), cv2.IMREAD_GRAYSCALE)
            if bg is None or mask is None or sigma_u8 is None:
                return None
            sigma = sigma_u8.astype(np.float32) * float(payload.get("sigma_scale", 1.0))
            return cls(bg.astype(np.float32), sigma, mask, tuple(payload["bbox"]), tuple(payload["full_size"]))
        except Exception:
            return None

    def matches_frame(self, frame: np.ndarray) -> bool:
        h, w = frame.shape[:2]
        return (w, h) == self.full_size


# --------------------------------------------------------------------------- #
class TraySegmenter:
    """Parameters below are the measured configuration - see the module docstring.

    Two details are load-bearing and must not be "tidied up": the small
    seed-suppression radius (the textbook dist > 0.5*max merges two equal
    touching items into one seed, worth ~35 points of recall) and the MIN_VAR
    floor on the local NCC shadow test (without it the ratio is noise on smooth
    objects, worth ~40 points).
    """

    def __init__(self, model: EmptyTrayModel,
                 min_area_frac: float = 0.008,
                 max_objects: int = 5,
                 split: bool = True,
                 saddle_ratio: float = 0.72,
                 seed_peak_frac: float = 0.30,
                 cut_edge_ratio: float = 2.2,
                 shadow_alpha_lo: float = 0.40, shadow_alpha_hi: float = 0.985,
                 shadow_ncc: float = 0.35,
                 shadow_cd: float = 1.2,
                 edge_sharp: float = 0.20,
                 shadow_a_std: float = 0.055,
                 core_rescue: float = 0.55, use_enclosure: bool = True,
                 regrow: int = 0,
                 edge_abs: float = 12.0, edge_k: float = 2.5,
                 border_touch_tau: float = 0.16, touch_len_px: float = 8.0,
                 ring_len_px: float = 80.0, ring_frac: float = 0.15,
                 intrusion_area_frac: float = 0.030):
        self.m = model
        self.min_area_frac, self.max_objects = min_area_frac, int(max_objects)
        self.split, self.saddle_ratio = split, saddle_ratio
        self.seed_peak_frac, self.cut_edge_ratio = seed_peak_frac, cut_edge_ratio
        self.sa_lo, self.sa_hi = shadow_alpha_lo, shadow_alpha_hi
        self.s_ncc, self.s_cd = shadow_ncc, shadow_cd
        self.edge_sharp, self.shadow_a_std = edge_sharp, shadow_a_std
        self.core_rescue = core_rescue
        self.use_enclosure = bool(use_enclosure)
        self.regrow = int(regrow)
        self.edge_abs, self.edge_k = edge_abs, edge_k
        self.border_touch_tau, self.intrusion_area_frac = border_touch_tau, intrusion_area_frac
        self.touch_len_px = touch_len_px
        self.ring_len_px, self.ring_frac = ring_len_px, ring_frac
        self._k3, self._k5, self._k7 = _ell(3), _ell(5), _ell(7)
        self._gain = np.ones(3, np.float32)
        self._bias = np.zeros(3, np.float32)

    # ---- 1. robust photometric alignment (kills auto-exposure / AWB drift) --
    def _align(self, cur):
        m = self.m
        idx = m.idx[::2, ::2]
        b, c = m.bg[::2, ::2][idx], cur[::2, ::2][idx]
        if b.shape[0] < 64:
            return cur, False
        # A frame that is blown out (or crushed) carries no photometric
        # information: every clipped pixel reads the same whatever is on the tray,
        # so the comparison silently returns "nothing changed".  Say so instead.
        sat_cur = float((c.max(axis=1) >= 250).mean() + (c.min(axis=1) <= 5).mean())
        sat_bg = float((b.max(axis=1) >= 250).mean() + (b.min(axis=1) <= 5).mean())
        if sat_cur > 0.35 and sat_cur > sat_bg + 0.25:
            return (cur - self._bias) / np.maximum(self._gain, 1e-3), False
        w = np.ones(b.shape[0], bool)
        g, off, ok = np.ones(3, np.float32), np.zeros(3, np.float32), True
        for it in range(2):
            bb, cc = b[w], c[w]
            bm, cm = bb.mean(axis=0), cc.mean(axis=0)
            db, dc = bb - bm, cc - cm
            var = (db * db).mean(axis=0)
            g = np.clip(np.where(var < 1e-3, 1.0, (db * dc).mean(axis=0) / np.maximum(var, 1e-3)),
                        0.5, 2.0).astype(np.float32)
            off = np.clip(cm - g * bm, -70.0, 70.0).astype(np.float32)
            if it == 0:                                   # trim to the 60% most background-like
                r = np.abs(c - (b * g + off)).max(axis=1)
                w = r <= max(float(np.quantile(r, 0.6)), 4.0)
                if w.sum() < 0.15 * w.size:
                    ok = False
                    break
        if not ok:
            g, off = self._gain, self._bias               # keep the last good fit
        self._gain, self._bias = g, off
        return (cur - off) / np.maximum(g, 1e-3), ok

    # ---- 2. pixel stage ----------------------------------------------------
    def _pixels(self, cur_al, cur_gray, cur_grad):
        """alpha (brightness ratio), cd (chromaticity distortion),
        chg (anything that changed, shadows included) and the shadow pixels."""
        m = self.m
        alpha = (cur_al * m.bg).sum(axis=2) / m.bg_dot
        resid = cur_al - alpha[..., None] * m.bg
        cd = np.sqrt((resid * resid).sum(axis=2))
        chg = (cd > m.cd_tol) | (np.abs(alpha - 1.0) > m.a_tol)
        k = (NCC_WIN, NCC_WIN)
        mx = cv2.boxFilter(cur_gray, -1, k)
        vx = np.maximum(cv2.boxFilter(cur_gray * cur_gray, -1, k) - mx * mx, 0.0)
        cxy = cv2.boxFilter(cur_gray * m.bg_gray, -1, k) - mx * m.my
        ok = m.vy_ok & (vx > MIN_VAR)    # only trust correlation where BOTH have real texture
        ncc = np.where(ok, cxy / np.sqrt(np.maximum(vx * m.vy, 1e-6)), 0.0)
        # A real object has a STEP boundary; a cast shadow has a penumbra RAMP.
        # Fill whatever a sharp closed outline encloses: that is object territory.
        # This cue still works where the tray has no texture at all (white
        # plastic), which is exactly where the NCC test above goes blind.
        strong = ((cur_grad > np.maximum(self.edge_abs, self.edge_k * m.sigma)).astype(np.uint8) * 255)
        enclosed = _fill_holes(cv2.dilate(strong, self._k3))
        band = (alpha >= self.sa_lo) & (alpha <= self.sa_hi) & (cd < self.s_cd * m.sigma)
        shadow = chg & band & ((ncc > self.s_ncc) | (self.use_enclosure & (enclosed == 0)))
        return alpha, cd, chg.astype(np.uint8) * 255, shadow

    def _clean(self, binary):
        fg = cv2.medianBlur(binary, 5)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, self._k7)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, self._k5)
        return _fill_holes(fg)

    # ---- 3. region evidence ------------------------------------------------
    def _features(self, blob, sel, sl, alpha, cd, cur_gray, cur_grad):
        m = self.m
        av = alpha[sl][sel]
        a, a_std = float(np.median(av)), float(av.std())
        cdn = float(np.median((cd[sl] / m.sigma[sl])[sel]))
        x, y = cur_gray[sl][sel], m.bg_gray[sl][sel]
        sx, sy = float(x.std()), float(y.std())
        ncc = float(((x - x.mean()) * (y - y.mean())).mean() / (sx * sy)) if sx > 2.0 and sy > 2.0 else 0.0
        outline = cv2.subtract(blob, cv2.erode(blob, self._k3, borderType=cv2.BORDER_CONSTANT, borderValue=0))
        band = cv2.dilate(outline, self._k5) > 0
        edge = float(cur_grad[band].mean()) if band.any() else 0.0
        contrast = max(6.0, abs(1.0 - a) * float(m.bg_gray[sl][sel].mean()))
        return a, a_std, cdn, ncc, edge / contrast

    def _is_shadow(self, a, a_std, cdn, ncc, sharp):
        """A region that is only a cast shadow: hue neutral, darker, with a soft
        (penumbra) boundary, and either the tray texture shows through or the
        darkening is flat - an object has a shading gradient across its body."""
        if not (self.sa_lo <= a <= self.sa_hi) or cdn > self.s_cd:
            return False
        if sharp > self.edge_sharp:
            return False
        return ncc > self.s_ncc or a_std < self.shadow_a_std

    # ---- 4. split touching objects -----------------------------------------
    def _split_blob(self, blob, bgr, grad):
        """Split one blob into touching objects using two cues.

        SHAPE   distance transform -> local-maximum seeds -> watershed.  This can
                only separate objects that are close to tangent, so seeds are
                taken generously and the shape guard is deliberately weak.
        PHOTO   every proposed cut is validated: the mean image gradient along
                the cut must be clearly stronger than the gradient inside the two
                pieces.  Two touching items really do have a visible edge between
                them; one lumpy cucumber does not.  This stops over-segmentation.
        """
        x, y, w, h = cv2.boundingRect(blob)
        p = 3
        x0, y0 = max(0, x - p), max(0, y - p)
        x1, y1 = min(blob.shape[1], x + w + p), min(blob.shape[0], y + h + p)
        b = blob[y0:y1, x0:x1]
        g = grad[y0:y1, x0:x1]
        dist = cv2.distanceTransform(b, cv2.DIST_L2, 3)
        peak = float(dist.max())
        if peak < 5.0:
            return (blob > 0).astype(np.int32), 1
        rad = int(np.clip(peak * 0.22, 3, 9)) | 1
        loc = cv2.dilate(dist, _ell(rad))
        seeds = ((dist >= loc - 1e-3) & (dist > max(2.5, 0.14 * peak))).astype(np.uint8)
        n, lab = cv2.connectedComponents(cv2.dilate(seeds, self._k3))
        if n <= 2:
            return (blob > 0).astype(np.int32), 1
        order = sorted(((float(dist[lab == i].max()), i) for i in range(1, n)), reverse=True)
        # a seed on a ragged boundary carries almost no distance: drop it outright
        order = [(pk, i) for pk, i in order if pk >= max(3.0, self.seed_peak_frac * peak)][:10]
        if len(order) < 2:
            return (blob > 0).astype(np.int32), 1

        markers = np.zeros(b.shape, np.int32)
        for j, (_, i) in enumerate(order, start=1):
            markers[lab == i] = j
        k = len(order)
        markers[cv2.dilate(b, self._k7) == 0] = k + 1
        cv2.watershed(np.ascontiguousarray(bgr[y0:y1, x0:x1]), markers)
        pieces = np.zeros(b.shape, np.int32)
        for j in range(1, k + 1):
            pieces[(markers == j) & (b > 0)] = j

        # ---- validate every proposed cut against the image gradient ----------
        parent = list(range(k + 1))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def union(a, bb):
            ra, rb = find(a), find(bb)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)

        inside = {}
        for j in range(1, k + 1):
            sel = pieces == j
            inside[j] = float(np.median(g[sel])) if sel.any() else 0.0
        dil = {j: cv2.dilate((pieces == j).astype(np.uint8), self._k5) for j in range(1, k + 1)}
        for j in range(1, k + 1):
            for l in range(j + 1, k + 1):
                shared = cv2.bitwise_and(dil[j], dil[l]) > 0
                if shared.sum() < 4:
                    continue
                cut = float(g[shared].mean())
                base = max(inside[j], inside[l], 1.5)
                aj, al = int((pieces == j).sum()), int((pieces == l).sum())
                small = min(aj, al) < self.min_area_frac * self.m.tray_area_work
                if small or cut < self.cut_edge_ratio * base:
                    union(j, l)                       # no visible edge -> one object
        remap, nxt = {}, 0
        out_small = np.zeros(b.shape, np.int32)
        for j in range(1, k + 1):
            r = find(j)
            if r not in remap:
                nxt += 1
                remap[r] = nxt
            out_small[pieces == j] = remap[r]
        if nxt < 2:
            return (blob > 0).astype(np.int32), 1
        out = np.zeros(blob.shape, np.int32)
        out[y0:y1, x0:x1] = out_small
        return out, nxt

    # ---- main --------------------------------------------------------------
    def segment(self, frame) -> SegmentResult:
        t0 = time.perf_counter()
        m = self.m
        if not m.matches_frame(frame):
            # camera resolution changed since the empty tray was captured
            return SegmentResult(empty=True, unreliable=True, ms=(time.perf_counter() - t0) * 1000.0)
        cur = EmptyTrayModel._prep(frame, m.bbox, m.work_wh).astype(np.float32)
        cur_al, ok = self._align(cur)
        cur_u8 = np.clip(cur_al, 0, 255).astype(np.uint8)
        cur_gray = cv2.cvtColor(cur_u8, cv2.COLOR_BGR2GRAY).astype(np.float32)
        cur_grad = _grad(cur_gray)
        alpha, cd, chg, shadow_px = self._pixels(cur_al, cur_gray, cur_grad)
        fg = self._clean(chg)
        ycc = cv2.cvtColor(cur_u8, cv2.COLOR_BGR2YCrCb)
        skin = ((ycc[:, :, 1] >= 135) & (ycc[:, :, 1] <= 177) &
                (ycc[:, :, 2] >= 77) & (ycc[:, :, 2] <= 127))

        n, lab, stats, _ = cv2.connectedComponentsWithStats(fg, 8)
        min_area = self.min_area_frac * m.tray_area_work
        regions, intrusion, shadows, coverage = [], False, 0, 0.0

        for i in range(1, n):
            area_all = int(stats[i, cv2.CC_STAT_AREA])
            if area_all < min_area:
                continue
            x, y, w, h = (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                          stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
            sl = (slice(y, y + h), slice(x, x + w))
            sel = lab[sl] == i
            inside = int(np.count_nonzero(sel & (m.mask[sl] > 0)))
            outside = area_all - inside
            # Reaching in from outside the tray -> hand / arm / sleeve.  A product
            # that merely overhangs the tray edge has only a sliver outside, so it
            # is kept (and flagged with border_touch).  Anything connected to the
            # edge of the crop came from outside the tray's neighbourhood entirely.
            from_outside = (outside > max(self.ring_len_px, self.ring_frac * max(inside, 1))
                            or np.any(sel & (m.crop_edge[sl] > 0)))
            if from_outside and area_all > self.intrusion_area_frac * m.tray_area_work:
                intrusion = True
                continue
            if inside < min_area:
                continue
            area = inside
            sel = sel & (m.mask[sl] > 0)
            blob = np.zeros(m.mask.shape, np.uint8)
            blob[sl][sel] = 255
            a, a_std, cdn, ncc, sharp = self._features(blob, sel, sl, alpha, cd, cur_gray, cur_grad)
            if self._is_shadow(a, a_std, cdn, ncc, sharp):
                shadows += 1
                continue

            outline = cv2.subtract(blob, cv2.erode(blob, self._k3, borderType=cv2.BORDER_CONSTANT, borderValue=0))
            per = float(np.count_nonzero(outline)) or 1.0
            touch_len = float(np.count_nonzero(cv2.bitwise_and(outline, m.border_band)))
            touch = touch_len / per
            sk = float(skin[sl][sel].mean())
            coverage += area / max(1, m.tray_area_work)

            # strip the attached cast shadow; this also un-bridges two objects
            core = cv2.bitwise_and(blob, np.logical_not(shadow_px).astype(np.uint8) * 255)
            core = cv2.morphologyEx(core, cv2.MORPH_OPEN, self._k5)
            if np.count_nonzero(core) < self.core_rescue * area:
                core = blob            # the pixel test ate the object: keep the blob

            cn, clab, cstats, _ = cv2.connectedComponentsWithStats(_fill_holes(core), 8)
            for ci in range(1, cn):
                if cstats[ci, cv2.CC_STAT_AREA] < min_area:
                    continue
                sub_blob = (clab == ci).astype(np.uint8) * 255
                if self.regrow:
                    grown = cv2.dilate(sub_blob, self._k5, iterations=self.regrow)
                    others = cv2.dilate(((clab > 0) & (clab != ci)).astype(np.uint8) * 255,
                                        self._k5, iterations=self.regrow)
                    sub_blob = cv2.bitwise_and(cv2.bitwise_and(grown, blob), cv2.bitwise_not(others))
                pieces, npieces = (self._split_blob(sub_blob, cur_u8, cur_grad) if self.split
                                   else ((sub_blob > 0).astype(np.int32), 1))
                for j in range(1, npieces + 1):
                    sub = (pieces == j).astype(np.uint8) * 255
                    a_j = float(np.count_nonzero(sub))
                    if a_j < min_area:
                        continue
                    regions.append(self._region(sub, a_j, touch, sk, a, ncc, sharp,
                                                i if npieces > 1 else -1))
        regions.sort(key=lambda r: r.area_px, reverse=True)
        res = SegmentResult(regions[:self.max_objects], intrusion, shadows, coverage,
                            empty=not regions and not intrusion, unreliable=not ok)
        res.ms = (time.perf_counter() - t0) * 1000.0
        return res

    def _region(self, sub, area_work, touch, skin_frac, alpha, ncc, sharp, parent):
        m = self.m
        x, y, w, h = cv2.boundingRect(sub)
        cnts, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hull_a = cv2.contourArea(cv2.convexHull(max(cnts, key=cv2.contourArea))) if cnts else area_work
        sx, sy = m.scale_x, m.scale_y
        box = (int(round(m.bbox[0] + x * sx)), int(round(m.bbox[1] + y * sy)),
               max(1, int(round(w * sx))), max(1, int(round(h * sy))))
        full_mask = cv2.resize(sub[y:y + h, x:x + w], (box[2], box[3]), interpolation=cv2.INTER_NEAREST)
        return ObjectRegion(box=box, mask=full_mask, area_px=area_work * sx * sy,
                            area_frac=area_work / max(1, m.tray_area_work),
                            fill=area_work / max(1.0, w * h),
                            solidity=area_work / max(1.0, hull_a),
                            border_touch=touch, skin_frac=skin_frac,
                            alpha=alpha, ncc=ncc, sharp=sharp, split_from=parent)

    # ---- slow background refresh -------------------------------------------
    def refresh_background(self, frame: np.ndarray, rate: float = 0.02) -> None:
        """Blend a confirmed-empty frame into the model so it does not rot as the
        tray gets dirty and the shop light drifts.  Caller must be sure the tray
        is empty (the SCALE is the authority, not the camera)."""
        m = self.m
        if not m.matches_frame(frame):
            return
        cur = EmptyTrayModel._prep(frame, m.bbox, m.work_wh).astype(np.float32)
        cur_al, ok = self._align(cur)
        if not ok:
            return
        m.bg = (1.0 - rate) * m.bg + rate * cur_al
        m.bg_gray = cv2.cvtColor(np.clip(m.bg, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
        m.bg_dot = np.maximum((m.bg * m.bg).sum(axis=2), 1.0)
        k = (NCC_WIN, NCC_WIN)
        m.my = cv2.boxFilter(m.bg_gray, -1, k)
        m.vy = np.maximum(cv2.boxFilter(m.bg_gray * m.bg_gray, -1, k) - m.my * m.my, 0.0)
        m.vy_ok = m.vy > MIN_VAR


# --------------------------------------------------------------------------- #
class RegionTracker:
    """2-of-3 temporal confirmation, matched across frames by box IoU.

    Measured effect: false regions drop 4-5x (empty tray 0.20 -> 0.00 per frame;
    two objects 0.16 -> 0.04) at a cost of 2-4 points of recall.  Without it the
    operator sees boxes flickering into existence on shadows.
    """

    def __init__(self, iou: float = 0.5, need: int = 2, window: int = 3):
        self.iou, self.need, self.window = iou, need, window
        self._tracks: List[Dict[str, Any]] = []
        self._next_id = 1

    def reset(self) -> None:
        self._tracks = []

    def update(self, regions: List[ObjectRegion]) -> List[ObjectRegion]:
        for t in self._tracks:
            t["seen"] = False
        for r in regions:
            best, best_iou = None, self.iou
            for t in self._tracks:
                v = box_iou(r.box, t["box"])
                if v >= best_iou and not t["seen"]:
                    best, best_iou = t, v
            if best is None:
                best = {"id": self._next_id, "hits": [], "box": r.box, "seen": True}
                self._next_id += 1
                self._tracks.append(best)
            best["seen"] = True
            best["box"] = r.box
            best["hits"] = (best["hits"] + [1])[-self.window:]
            r.track_id = best["id"]
            r.confirmed = sum(best["hits"]) >= self.need
        for t in self._tracks:
            if not t["seen"]:
                t["hits"] = (t["hits"] + [0])[-self.window:]
        self._tracks = [t for t in self._tracks if sum(t["hits"]) > 0]
        return regions


if __name__ == "__main__":  # tiny smoke test on synthetic data
    rng = np.random.default_rng(0)
    tray = np.full((480, 640, 3), 150, np.uint8)
    tray += (rng.normal(0, 6, tray.shape)).astype(np.int16).clip(-40, 40).astype(np.uint8)
    cv2.rectangle(tray, (140, 90), (500, 400), (170, 165, 160), -1)
    for i in range(0, 640, 17):                       # give the tray some grain
        cv2.line(tray, (i, 0), (i, 480), (160, 158, 155), 1)
    frames = [np.clip(tray.astype(np.int16) + rng.normal(0, 2, tray.shape), 0, 255).astype(np.uint8)
              for _ in range(BUILD_FRAMES)]
    model = EmptyTrayModel.build(frames, (140, 90, 500, 400),
                                 [(140, 90), (500, 90), (500, 400), (140, 400)])
    seg = TraySegmenter(model)
    print("empty  ->", seg.segment(frames[0]).count, "regions, %.1f ms" % seg.segment(frames[0]).ms)
    two = frames[1].copy()
    cv2.circle(two, (240, 200), 45, (40, 40, 200), -1)
    cv2.circle(two, (400, 300), 50, (40, 200, 220), -1)
    r = seg.segment(two)
    print("2 objs ->", r.count, "regions", [x.box for x in r.regions], "%.1f ms" % r.ms)
    payload = model.to_payload()
    print("payload:", sum(len(v) for v in payload.values() if isinstance(v, bytes)) // 1024, "KB")
    back = EmptyTrayModel.from_payload(payload)
    print("reloaded ok:", back is not None and TraySegmenter(back).segment(two).count == r.count)
