"""
diagnostics.py
--------------
Collects everything needed to debug a real till into one zip file.

Recognition and segmentation problems cannot be fixed from a description: what
the camera sees on a particular shop counter, under particular lighting, with
particular produce, is the whole problem.  This module packages that evidence:

    frame.png            the current camera frame
    tray.png             the same frame with the tray outline and the detected
                         objects drawn on it
    model_input.png      exactly what the network is fed for the whole tray
    object_00.png ...    exactly what the network is fed for each object, which
                         is where "it only sees part of the item" becomes visible
    mask_00.png ...      the mask the segmenter produced for each object
    empty_tray.png       the stored empty-tray reference
    recognition.txt      every candidate and score for every object
    items.txt            the catalogue: samples, captures, thresholds, look-alikes
    config.json          the settings (no secrets are stored in it)
    camera.txt           what the camera layer tried and what Windows can see
    selftest_report.txt  the last self-test, if there is one
    aipos.log            the tail of the log

Nothing here leaves the machine on its own: it writes a file the operator can
send on.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
import platform
import sys
import zipfile
from typing import List, Optional

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore

log = logging.getLogger("diagnostics")
LOG_TAIL_BYTES = 200_000


def _png(image: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", image)
    return buf.tobytes() if ok else b""


def _draw_overlay(frame: np.ndarray, tray, seg) -> np.ndarray:
    """The frame with the tray outline and every detected object drawn on it."""
    out = frame.copy()
    h, w = out.shape[:2]
    try:
        pts = tray.pixel_points(w, h).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], True, (255, 200, 60), 2)
    except Exception:
        pass
    if seg is not None:
        for i, r in enumerate(seg.regions):
            x, y, bw, bh = r.box
            colour = (90, 220, 110) if r.confirmed else (90, 160, 255)
            cv2.rectangle(out, (x, y), (x + bw, y + bh), colour, 2)
            cv2.putText(out, f"#{i} {r.area_frac:.3f}", (x, max(14, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
            try:
                mask = np.zeros((h, w), np.uint8)
                mh, mw = r.mask.shape[:2]
                mh, mw = min(mh, h - y), min(mw, w - x)
                if mh > 0 and mw > 0:
                    mask[y:y + mh, x:x + mw] = r.mask[:mh, :mw]
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(out, contours, -1, colour, 1)
            except Exception:
                pass
        if seg.intrusion:
            cv2.putText(out, "INTRUSION", (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 60, 240), 2, cv2.LINE_AA)
        if seg.unreliable:
            cv2.putText(out, "UNRELIABLE", (10, h - 36), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 180, 240), 2, cv2.LINE_AA)
    return out


def collect(path: str, frame: Optional[np.ndarray], tray, db, engine, settings,
            recognition=None, note: str = "") -> str:
    """Write the bundle and return its path."""
    from ai_engine import MODE_OBJECT, MODE_TRAY, region_crop

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    seg = None
    if engine is not None and frame is not None:
        try:
            seg = engine.segment_frame(frame)
        except Exception:
            log.exception("segmentation failed while collecting diagnostics")

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        header = [f"AI POS Scale diagnostic bundle  {dt.datetime.now():%Y-%m-%d %H:%M:%S}",
                  f"platform: {platform.platform()}  python: {sys.version.split()[0]}",
                  f"note: {note}" if note else ""]

        if frame is not None and cv2 is not None:
            z.writestr("frame.png", _png(frame))
            z.writestr("tray.png", _png(_draw_overlay(frame, tray, seg)))
            try:
                z.writestr("model_input.png", _png(tray.crop(frame)[0]))
            except Exception:
                pass
            if seg is not None:
                for i, r in enumerate(seg.regions):
                    try:
                        z.writestr(f"object_{i:02d}.png", _png(region_crop(frame, r)))
                        z.writestr(f"mask_{i:02d}.png", _png(r.mask))
                    except Exception:
                        pass
        else:
            header.append("NO CAMERA FRAME WAS AVAILABLE")

        # the empty-tray reference, which is what segmentation compares against
        try:
            model = db.empty_tray_model() if db is not None else None
            if model is not None:
                z.writestr("empty_tray.png", _png(np.clip(model.bg, 0, 255).astype(np.uint8)))
                z.writestr("empty_tray_sigma.png", _png(np.clip(model.sigma * 4, 0, 255).astype(np.uint8)))
                z.writestr("empty_tray_mask.png", _png(model.mask))
        except Exception:
            header.append("the empty-tray model could not be read")

        # what the segmenter and the matcher actually concluded
        lines: List[str] = []
        if seg is not None:
            lines.append(f"segmentation: {seg.count} regions, intrusion={seg.intrusion}, "
                         f"unreliable={seg.unreliable}, shadows={seg.shadows}, {seg.ms:.1f} ms")
            for i, r in enumerate(seg.regions):
                lines.append(f"  #{i} box={r.box} area_frac={r.area_frac:.4f} fill={r.fill:.2f} "
                             f"solidity={r.solidity:.2f} split_from={r.split_from} confirmed={r.confirmed} "
                             f"alpha={r.alpha:.2f} ncc={r.ncc:.2f} sharp={r.sharp:.2f}")
        else:
            lines.append("segmentation: unavailable (no empty-tray image captured)")
        if recognition is not None:
            lines.append("")
            lines.append(f"recognition mode: {recognition.mode}, empty={recognition.empty}, "
                         f"{recognition.infer_ms:.0f} ms")
            for d in recognition.detections:
                lines.append(f"  track {d.track_id} box={d.box} confirmed={d.confirmed} "
                             f"-> {d.name or '(unknown)'} score={d.score:.3f} "
                             f"threshold={d.result.threshold:.3f} accepted={d.result.accepted}")
                for iid, name, score in d.result.ranking:
                    lines.append(f"      {name:<24} {score:+.3f}")
        z.writestr("recognition.txt", "\n".join(lines))

        # the catalogue, which is where look-alike pairs show up
        if db is not None:
            items = ["items: %d  (object mode: %d, whole tray: %d)" %
                     (len(db), len(db.items_in_mode(MODE_OBJECT)), len(db.items_in_mode(MODE_TRAY))),
                     f"empty-tray embeddings: {db.has_background}  image model: {db.has_background_image}", ""]
            for it in db.all_items():
                items.append(f"{it.name:<28} mode={it.crop_mode:<7} captures={it.capture_count:<3} "
                             f"samples={it.sample_count:<4} learned={it.drift_count} "
                             f"threshold={it.threshold:.3f} price={it.price_per_kg:g}")
            try:
                pairs = db.confusable_pairs(10)
                if pairs:
                    items.append("")
                    items.append("items that look alike to the camera (worst first):")
                    items += [f"  {a} ~ {b}   {c:.0%} of {a}'s samples are pulled toward {b}" for a, b, c in pairs]
            except Exception:
                pass
            z.writestr("items.txt", "\n".join(items))

        # settings, the self-test and the log tail
        try:
            z.writestr("config.json", json.dumps(settings.as_dict(), ensure_ascii=False, indent=2))
        except Exception:
            pass
        # a bundle sent because "the camera does not work" is useless without
        # what the camera layer actually tried and what Windows can see
        try:
            from rtk_camera import diagnose, windows_imaging_devices
            cam_lines = [diagnose(settings.get("camera.sdk_dll", "RTKCamSDK.dll")), "",
                         "camera settings: " + json.dumps(settings.section("camera"), ensure_ascii=False),
                         "", "devices Windows reports:"]
            cam_lines += [f"  {d}" for d in windows_imaging_devices()] or ["  none"]
            z.writestr("camera.txt", "\n".join(cam_lines))
        except Exception as exc:
            z.writestr("camera.txt", f"could not collect camera information: {exc}")

        for name in ("selftest_report.txt", "crash_history.txt"):
            src = os.path.join(os.path.dirname(os.path.abspath(path)), name)
            if os.path.isfile(src):
                try:
                    z.write(src, name)
                except Exception:
                    pass
        log_path = os.path.join(os.path.dirname(os.path.abspath(path)), "aipos.log")
        if os.path.isfile(log_path):
            try:
                with open(log_path, "rb") as fh:
                    fh.seek(max(0, os.path.getsize(log_path) - LOG_TAIL_BYTES))
                    z.writestr("aipos.log", fh.read())
            except Exception:
                pass

        if engine is not None:
            header.append(f"segmenter: {'on' if engine.has_segmenter else 'OFF (no empty-tray image)'}")
            header.append(f"per-object recognition: {engine.per_object}, max objects: {engine.max_objects}")
            header.append(f"inference: {engine.stats_ms:.0f} ms, segmentation {engine.seg_ms:.0f} ms")
        extractor = getattr(engine, "extractor", None)
        if extractor is not None:
            header.append(f"model optimized for this CPU: {getattr(extractor, 'optimized', False)}")
        z.writestr("README.txt", "\n".join(h for h in header if h) +
                   "\n\nSend this file to support. It contains images of the tray and the"
                   "\nsettings, and no personal or payment data.\n")
    return path


def default_path(folder: str) -> str:
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(folder, f"aipos_diagnostics_{stamp}.zip")
