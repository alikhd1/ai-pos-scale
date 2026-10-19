"""
app.py
------
AI POS Scale - main application (PyQt6).

Left  : live camera with the tray outline, a box and a label around every
        detected object (up to five), and the buttons Train New Item /
        Add Angle Sample / Manage Items / Tray Area / Capture Empty Tray.
Right : live weight, sales invoice, grand total, Clear, Checkout & Print, Settings.

Run from source :  python app.py            (add --lang en for English UI)
Head-less check :  python app.py --selftest (writes selftest_report.txt)
Virtual camera  :  python app.py --video some_clip.avi
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import json
import logging
import os
import subprocess
import sys
import threading
import time
import traceback
import uuid
from typing import Dict, List, Optional, Tuple

import numpy as np

from settings_manager import SettingsManager, app_dir, data_path, resource_path, is_frozen
from translations import tr, set_language, is_rtl
from scale_driver import (ScaleReader, list_serial_ports, diagnose_scale, find_scale, make_scale,
                          resolve_source, BAUDRATES, WeightReading, printable, hexdump,
                          SOURCE_AUTO, SOURCE_HID, SOURCE_SERIAL, SOURCE_SIMULATE)
from hid_scale import list_hid_scales
from remote_support import RemoteSupport, encode_frame, log_tail, ACTIONS as SUPPORT_ACTIONS
import support_config
import diagnostics

log = logging.getLogger("app")

APP_VERSION = "1.4"

#: distinguishes "the setting is None" from "there is no such setting"
_MISSING = object()


def build_id() -> Dict[str, str]:
    """Which code is actually running, for the support report.

    Read straight out of .git rather than by running git: this is called from
    a background thread on a till, and spawning a process there to answer a
    question about ourselves is not worth the risk or the wait.
    """
    info = {"version": APP_VERSION, "frozen": str(is_frozen())}
    head = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".git", "HEAD")
    try:
        with open(head, "r", encoding="utf-8") as fh:
            ref = fh.read().strip()
        if ref.startswith("ref:"):
            info["branch"] = ref.split("/")[-1]
            with open(os.path.join(os.path.dirname(head), ref[5:].strip()), "r", encoding="utf-8") as fh:
                info["commit"] = fh.read().strip()[:10]
        else:
            info["commit"] = ref[:10]            # detached head
    except OSError:
        info["commit"] = "unknown (not a git checkout)"
    return info


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def setup_logging(verbose: bool = False) -> str:
    log_path = data_path("aipos.log")
    handlers: List[logging.Handler] = [logging.FileHandler(log_path, encoding="utf-8")]
    if not is_frozen():
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:
            pass
        handlers.append(logging.StreamHandler(sys.stdout))
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", handlers=handlers)
    return log_path


# --------------------------------------------------------------------------- #
# Session heartbeat (detects Windows restarts / crashes between runs)
# --------------------------------------------------------------------------- #
SESSION_LOCK = "session.lock"
CRASH_HISTORY = "crash_history.txt"


INSTANCE_LOCK = "instance.lock"
#: a remote update starts the replacement before the old copy has finished
#: closing, so a new instance waits rather than refusing and leaving the shop
#: with no till and no way in to fix it
INSTANCE_WAIT_S = 25.0


def acquire_single_instance(wait_s: float = INSTANCE_WAIT_S):
    """Take the one-instance lock, or return None if another copy holds it.

    A real lock, not a marker file: the operating system drops it when the
    process ends, including a crash or a kill, so there is nothing to clean up
    and nothing to go stale.
    """
    path = data_path(INSTANCE_LOCK)
    try:
        handle = open(path, "a+b")
    except OSError as exc:
        log.warning("cannot open %s (%s); continuing without the single-instance lock", path, exc)
        return True                               # never block the till over this

    def take() -> bool:
        try:
            if os.name == "nt":
                import msvcrt
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError:
            return False

    deadline = time.time() + max(0.0, wait_s)
    while True:
        if take():
            handle.seek(0)
            handle.truncate()
            handle.write(f"{os.getpid()} {dt.datetime.now():%Y-%m-%d %H:%M:%S}".encode())
            handle.flush()
            return handle                         # keep it alive for the process's lifetime
        if time.time() >= deadline:
            handle.close()
            return None
        time.sleep(0.5)


def session_begin() -> Optional[str]:
    """Write the heartbeat file; return the previous heartbeat if the last session did not end cleanly."""
    lock = data_path(SESSION_LOCK)
    previous = None
    if os.path.isfile(lock):
        try:
            with open(lock, "r", encoding="utf-8") as fh:
                previous = fh.read().strip() or "?"
        except OSError:
            previous = "?"
        try:
            with open(data_path(CRASH_HISTORY), "a", encoding="utf-8") as fh:
                fh.write(f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}  previous session not closed cleanly; last heartbeat {previous}\n")
        except OSError:
            pass
    session_heartbeat()
    return previous


def session_heartbeat() -> None:
    try:
        with open(data_path(SESSION_LOCK), "w", encoding="utf-8") as fh:
            fh.write(f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}")
    except OSError:
        pass


def session_end() -> None:
    try:
        os.remove(data_path(SESSION_LOCK))
    except OSError:
        pass


def windows_shutdown_events(max_events: int = 8) -> List[str]:
    """Recent shutdown/restart related events from the Windows System log.

    41 = Kernel-Power (power loss / hard reset / thermal), 1001 = BugCheck (blue
    screen, usually a driver), 6008 = unexpected shutdown, 1074 = a program or
    user requested the restart (e.g. Windows Update).
    """
    if not sys.platform.startswith("win"):
        return []
    query = "*[System[(EventID=41 or EventID=1001 or EventID=6008 or EventID=1074)]]"
    try:
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        out = subprocess.run(["wevtutil", "qe", "System", f"/q:{query}", f"/c:{max_events}", "/rd:true", "/f:text"],
                             capture_output=True, text=True, timeout=25, creationflags=flags, encoding="utf-8", errors="replace")
    except Exception as exc:
        return [f"(event log query failed: {exc})"]
    events, cur = [], {}
    for line in (out.stdout or "").splitlines():
        s = line.strip()
        if s.startswith("Event["):
            if cur:
                events.append(cur)
            cur = {}
        elif s.startswith("Date:"):
            cur["date"] = s[5:].strip()
        elif s.startswith("Event ID:"):
            cur["id"] = s[9:].strip()
        elif s.startswith("Description:"):
            cur["desc"] = ""
        elif "desc" in cur and s and len(cur["desc"]) < 300:
            cur["desc"] += (" " if cur["desc"] else "") + s
    if cur:
        events.append(cur)
    meaning = {"41": "Kernel-Power: power lost / hard reset / thermal", "1001": "BugCheck: blue screen (driver crash)",
               "6008": "unexpected shutdown", "1074": "restart requested by a program/user (e.g. Windows Update)"}
    return [f"{e.get('date', '?')[:19]}  ID {e.get('id', '?')} ({meaning.get(e.get('id', ''), '')})  {e.get('desc', '')[:160]}" for e in events]


# --------------------------------------------------------------------------- #
# Helpers shared by GUI and self-test
# --------------------------------------------------------------------------- #
def model_file_path(settings: SettingsManager) -> str:
    """Configured weights file if present, else the copy bundled by build.bat."""
    configured = settings.resolve("ai.model_file")
    if configured and os.path.isfile(configured):
        return configured
    bundled = resource_path("models/mobilenet_v2-b0353104.pth")
    return bundled if os.path.isfile(bundled) else configured


def receipt_labels() -> dict:
    return {"invoice": tr("invoice_no"), "date": tr("date"), "item": tr("col_item"), "weight": tr("receipt_kg"),
            "price": tr("col_price"), "total": tr("col_total"), "grand_total": tr("grand_total"), "items": tr("items")}


SOURCE_CLASS = {SOURCE_HID: "HidScaleReader", SOURCE_SERIAL: "ScaleReader", SOURCE_SIMULATE: "SimulatedScale"}


def scale_mode_reason(settings: SettingsManager) -> str:
    """Single source of truth for which scale object exists and why.

    Used by the status pill's tooltip, the self-test report and the head-less
    wiring test, so "what the settings dialog tested" and "what the main window
    actually built" can never drift apart again without a test failing.
    """
    source, reason = resolve_source(settings.section("scale"))
    return f"{SOURCE_CLASS.get(source, source)}: {reason}"


def scale_is_simulated(settings: SettingsManager) -> bool:
    return resolve_source(settings.section("scale"))[0] == SOURCE_SIMULATE


# --------------------------------------------------------------------------- #
# Head-less self-test (also used by build.bat and on the target machine)
# --------------------------------------------------------------------------- #
def run_selftest(settings: SettingsManager, video_file: str = "") -> int:
    import urllib.request
    import remote_support
    from rtk_camera import diagnose, open_camera, find_camera
    from scale_driver import parse_weight, list_serial_ports, diagnose_scale
    from printer_driver import ReceiptPrinter, Invoice, InvoiceLine, list_windows_printers
    from ai_engine import FeatureExtractor, ItemDatabase, Recognizer, TrayRegion, MODE_OBJECT, MODE_TRAY
    from tray_segment import TraySegmenter

    lines: List[str] = [f"AI POS Scale self-test  {dt.datetime.now():%Y-%m-%d %H:%M:%S}",
                        f"build: {build_id()}",
                        f"frozen={is_frozen()}  app_dir={app_dir()}", f"config={settings.path}", ""]
    ok = True

    def section(title):
        lines.append(f"--- {title} ---")

    section("system")
    if sys.platform.startswith("win"):
        wv = sys.getwindowsversion()
        lines.append(f"Windows {wv.major}.{wv.minor} build {wv.build}, CPU cores: {os.cpu_count()}")
    hist = data_path(CRASH_HISTORY)
    if os.path.isfile(hist):
        with open(hist, "r", encoding="utf-8", errors="replace") as fh:
            tail = fh.read().splitlines()[-5:]
        lines.append("unclean shutdowns recorded by this app (last 5):")
        lines += [f"  {t}" for t in tail]
    else:
        lines.append("unclean shutdowns recorded by this app: none")
    events = windows_shutdown_events()
    lines.append("Windows System log, recent shutdown/restart events (newest first):")
    lines += [f"  {e}" for e in events] if events else ["  none found"]

    section("camera")
    lines.append(diagnose(settings.get("camera.sdk_dll", "RTKCamSDK.dll")))
    cam = settings.section("camera")
    res = open_camera(cam["backend"], int(cam["device_index"]), int(cam["width"]), int(cam["height"]), cam["sdk_dll"],
                      video_file or cam.get("video_file", ""))
    lines.append(f"backend={res.backend}  {res.description}")
    lines += [f"  note: {n}" for n in res.notes]
    frame = None
    deadline = time.time() + 3
    while frame is None and time.time() < deadline:
        frame = res.source.read()
    res.source.close()
    # the placeholder picture the app shows when there is no camera is still a
    # frame, so reporting "frame: OK" for it would hide exactly the fault we
    # are looking for
    live = frame is not None and res.backend != "none"
    lines.append(f"frame: {'OK ' + str(frame.shape) if live else 'NONE (no camera)'}")
    if not live:
        # the useful question on a dead camera is not "did it fail" but "what
        # was tried and is the camera on a different index"
        lines.append("no picture on the configured camera, searching every device index:")
        found = find_camera()
        lines += [f"  {line}" for line in found["report"].splitlines()]
        if found["found"]:
            lines.append(f"  >>> set camera.device_index to {found['index']} (or press "
                         f"\"{tr('find_camera')}\" on the main screen)")
        ok = False
    tray = TrayRegion.from_settings(cam)
    lines.append(f"tray area: {'polygon with ' + str(len(tray.points)) + ' points' if tray.is_polygon else 'default centre square'}, mask={tray.mask_outside}")

    section("ai")
    try:
        t0 = time.time()
        # built without the fast path so the two speeds below are measured on an
        # idle CPU; measuring while the optimizer thread runs flatters the result
        fx = FeatureExtractor(model_file_path(settings), int(settings.get("ai.num_threads", 1)), fast=False)
        lines.append(f"MobileNetV2 loaded in {time.time() - t0:.1f}s from {fx.weights_path}")
        test_img = frame if live else np.full((480, 640, 3), 90, np.uint8)
        crop, box = tray.crop(test_img)
        def median_ms(n: int = 7) -> float:
            samples = []
            for _ in range(n):
                t = time.perf_counter()
                fx.embed(crop)
                samples.append((time.perf_counter() - t) * 1000)
            return float(np.median(samples))

        emb = fx.embed(crop)
        eager_ms = median_ms()
        lines.append(f"embedding dim={emb.shape[0]} norm={np.linalg.norm(emb):.3f} bbox={box}")
        if settings.get("ai.fast_inference", True):
            t1 = time.time()
            fx._build_fast_path()
            if fx.optimized:
                fast_ms = median_ms()
                lines.append(f"speed per object: {eager_ms:.0f} ms -> {fast_ms:.0f} ms after optimizing for this "
                             f"CPU ({eager_ms / max(fast_ms, 0.01):.1f}x, built in {time.time() - t1:.0f}s)")
            else:
                lines.append(f"speed per object: {eager_ms:.0f} ms. The model could NOT be optimized for this "
                             f"CPU, so recognition is slower than it should be (see aipos.log for the reason).")
        else:
            lines.append(f"speed per object: {eager_ms:.0f} ms (optimization is switched off in the settings)")
        db = ItemDatabase(settings.resolve("ai.db_file"))
        if db.backbone_mismatch:
            ok = False
            lines.append(f"items database was enrolled with backbone {db.backbone_mismatch!r}: every item must be re-trained")
        lines.append(f"items database: {len(db)} items "
                     f"({len(db.items_in_mode(MODE_OBJECT))} per-object, {len(db.items_in_mode(MODE_TRAY))} whole-tray), "
                     f"empty-tray embeddings: {'yes' if db.has_background else 'NO'}, "
                     f"empty-tray image: {'yes' if db.has_background_image else 'NO'} ({db.path})"
                     + (f"  LOAD ERROR: {db.load_error}" if db.load_error else ""))
        model = db.empty_tray_model()
        if model is not None:
            seg = TraySegmenter(model, min_area_frac=float(settings.get("ai.min_area_frac", 0.008)),
                                max_objects=int(settings.get("ai.max_objects", 5)))
            r = seg.segment(test_img)
            lines.append(f"segmentation: {r.count} regions, intrusion={r.intrusion}, unreliable={r.unreliable}, {r.ms:.0f} ms")
        else:
            lines.append("segmentation: DISABLED (capture the empty tray to enable multi-object detection)")
        if len(db) or db.has_background:
            rec = Recognizer(db, float(settings.get("ai.confidence_threshold", 0.65)), 3,
                             float(settings.get("ai.empty_threshold", 0.88)))
            r = rec.match(emb, MODE_TRAY)
            lines.append(f"match on test frame: {r.name or '-'} score={r.score:.3f} thr={r.threshold:.3f} "
                         f"accepted={r.accepted} empty={r.empty}")
        pairs = db.confusable_pairs(5)
        if pairs:
            lines.append("items that look alike to the camera: " + ", ".join(f"{a}~{b} ({c:.0%})" for a, b, c in pairs))
    except Exception as exc:
        ok = False
        lines.append(f"AI FAILED: {exc}")
        lines.append(traceback.format_exc())

    section("scale")
    sc = settings.section("scale")
    lines.append(f"ports: {list_serial_ports() or 'none'}")
    sample = parse_weight(b"ST,GS,+  1.234kg\r\n")
    lines.append(f"parser: {'OK' if sample and abs(sample.weight_kg - 1.234) < 1e-6 else 'FAILED'}")
    lines.append(f"mode: {scale_mode_reason(settings)}")
    try:
        hid_scales = list_hid_scales()
        lines.append(f"USB HID scales: {len(hid_scales)}" + "".join(f"\n  {d}" for d in hid_scales))
    except Exception as exc:
        lines.append(f"USB HID enumeration failed: {exc}")
    source, _reason = resolve_source(sc)
    if source == SOURCE_SIMULATE:
        if (sc.get("port") or "").strip() and str(sc.get("source") or "") in ("", "simulate") and sc.get("simulate"):
            ok = False
            lines.append("  PROBLEM: a COM port is configured but simulation is ON, so the app shows a "
                         "simulated weight while the settings dialog tests the real port.")
    elif source == SOURCE_HID:
        from hid_scale import test_hid_scale
        rep = test_hid_scale(sc.get("hid_path", ""), 3.0)
        lines.append(f"USB HID scale: {'OK ' if rep['ok'] else 'PROBLEM '}{rep['message']}")
        lines += [f"  {r}" for r in rep.get("raw", [])[:4]]
        if not rep["ok"]:
            ok = False
    else:
        rep = diagnose_scale(sc, quick=True)
        lines.append(f"hardware scale on {sc['port']}: {'OK ' if rep['ok'] else 'PROBLEM '}{rep['summary']}")
        lines += [f"  {ln}" for ln in rep["report"].splitlines()]

    section("printer")
    pr = settings.section("printer")
    lines.append(f"windows printers: {list_windows_printers() or 'none'}")
    lines.append(f"configured: interface={pr['interface']} printer={pr.get('printer_name') or '-'} port={pr.get('port') or '-'} "
                 f"paper={pr['paper_width']}mm mode={pr['mode']}")
    try:
        cfg = dict(pr, interface="file", file_path=data_path("selftest_receipt.bin"))
        printer = ReceiptPrinter(cfg, settings.section("general"), receipt_labels())
        inv = Invoice(1, [InvoiceLine("Test / آزمایش", 1.234, 25000)], store_name=settings.get("general.store_name"),
                      currency=settings.get("general.currency"))
        lines.append("ESC/POS job: " + printer.print_invoice(inv))
        printer.preview_image(inv).save(data_path("selftest_receipt.png"))
    except Exception as exc:
        ok = False
        lines.append(f"printer FAILED: {exc}")

    section("remote support")
    sup = settings.section("support")
    url = (sup.get("url") or "").strip() or support_config.DEFAULT_URL
    token = (sup.get("token") or "").strip() or support_config.DEFAULT_TOKEN
    lines.append(f"enabled={bool(sup.get('enabled'))}  url={url or '(none)'}  "
                 f"key={'built into this build' if not (sup.get('token') or '').strip() else 'from config.json'}"
                 f"  every {sup.get('interval_min', 10)} min")
    bundle = remote_support.ca_bundle()
    lines.append(f"TLS trust: {bundle or 'the machine certificate store (certifi is NOT bundled)'}")
    if bundle is None:
        # the exact failure this was built to prevent, so it must be loud
        lines.append("  WARNING: without the shipped bundle this till depends on the Windows "
                     "certificate store, which is what made the support channel fail before.")
        ok = False
    if not url or not token:
        lines.append("no support service configured, nothing to check")
    else:
        # only a reachability check: posting a snapshot here would register the
        # build machine as a till every time anybody runs the self-test
        try:
            context = remote_support.ssl_context(str(sup.get("ca_file") or ""))
            t0 = time.time()
            with urllib.request.urlopen(url.rstrip("/") + "/healthz", timeout=20, context=context) as resp:
                body = resp.read(4096).decode("utf-8", errors="replace").strip()
            lines.append(f"reached {url} in {time.time() - t0:.1f}s: HTTP {resp.status} {body}")
        except Exception as exc:
            lines.append(f"CANNOT REACH {url}: {remote_support.explain_tls_error(exc, url)}")
            ok = False

    lines.append("")
    lines.append("RESULT: " + ("OK" if ok else "PROBLEMS FOUND"))
    report = "\n".join(lines)
    with open(data_path("selftest_report.txt"), "w", encoding="utf-8") as fh:
        fh.write(report)
    try:
        print(report)
    except Exception:
        pass
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
# GUI
# --------------------------------------------------------------------------- #
import cv2
from PyQt6.QtCore import Qt, QTimer, QObject, pyqtSignal, QSize, QRectF, QPointF, QBuffer, QThread
from PyQt6.QtGui import (QImage, QPixmap, QPainter, QPen, QColor, QFont, QBrush, QIcon, QKeySequence, QShortcut,
                         QFontDatabase, QGuiApplication, QPainterPath, QPolygonF)
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
                             QGridLayout, QTableWidget, QTableWidgetItem, QHeaderView, QSlider, QDoubleSpinBox,
                             QSpinBox, QComboBox, QLineEdit, QDialog, QDialogButtonBox, QFormLayout, QTabWidget,
                             QCheckBox, QMessageBox, QProgressBar, QFrame, QSizePolicy, QAbstractItemView,
                             QFileDialog, QPlainTextEdit, QStatusBar, QSplitter, QListWidget, QListWidgetItem,
                             QRadioButton, QButtonGroup)

from rtk_camera import open_camera, find_camera, CameraThread
from printer_driver import (ReceiptPrinter, Invoice, InvoiceLine, list_windows_printers, default_windows_printer,
                            fmt_money, fmt_weight)
from ai_engine import (FeatureExtractor, ItemDatabase, RecognitionEngine, Recognition, Detection, Item, TrayRegion,
                       make_thumbnail, region_crop, MODE_OBJECT, MODE_TRAY)
from tray_segment import BUILD_FRAMES, EmptyTrayModel


class Bus(QObject):
    """Thread-safe bridge: worker threads emit, GUI slots receive."""
    recognition = pyqtSignal(object, object)     # Recognition, frame
    weight = pyqtSignal(object)                  # WeightReading
    embedding_ready = pyqtSignal(str, object)    # token, embeddings (n, 1280)
    model_ready = pyqtSignal(object, object)     # FeatureExtractor | None, error | None
    camera_ready = pyqtSignal(object)            # CameraOpenResult
    camera_found = pyqtSignal(object)            # find_camera() report
    scale_found = pyqtSignal(object)             # find_scale() report
    run_on_ui = pyqtSignal(object)               # a callable to run on the GUI thread
    toast = pyqtSignal(str, bool)                # message, is_error
    print_done = pyqtSignal(bool, str)
    dialog_result = pyqtSignal(str, str, bool)   # kind, text, is_error  (settings dialog workers)


STYLE = """
QMainWindow, QDialog { background: #14171c; }
QWidget { color: #e8eaf0; font-size: 15px; }
QLabel#title { font-size: 20px; font-weight: 600; color: #9fb3ff; }
QLabel#weight { font-size: 54px; font-weight: 700; color: #7CFC9A; font-family: Consolas, 'Segoe UI', monospace; }
QLabel#total { font-size: 34px; font-weight: 700; color: #ffd166; }
QLabel#detected { font-size: 26px; font-weight: 700; color: #ffffff; }
QLabel#banner { font-size: 16px; font-weight: 700; color: #14171c; background: #ffcc4d; border-radius: 8px; padding: 8px; }
QPushButton { background: #2b3140; border: 1px solid #3b4252; border-radius: 8px; padding: 10px 14px; font-weight: 600; }
QPushButton:hover { background: #363d4f; }
QPushButton:pressed { background: #1f2430; }
QPushButton:disabled { color: #7a8194; background: #232833; }
QPushButton#primary { background: #2f6fed; border-color: #2f6fed; font-size: 18px; padding: 14px; }
QPushButton#primary:hover { background: #3b7cff; }
QPushButton#success { background: #1f9d55; border-color: #1f9d55; font-size: 18px; padding: 14px; }
QPushButton#success:hover { background: #26b463; }
QPushButton#danger { background: #a1343e; border-color: #a1343e; }
QPushButton#danger:hover { background: #c03f4b; }
QPushButton#warn { background: #8a6d1f; border-color: #8a6d1f; }
QTableWidget { background: #1b1f27; alternate-background-color: #20252f; gridline-color: #2c3340; border: 1px solid #2c3340; border-radius: 6px; }
QHeaderView::section { background: #262c38; padding: 8px; border: none; font-weight: 600; }
QTableWidget::item { padding: 6px; }
QTableWidget::item:selected { background: #2f6fed; }
QLineEdit, QDoubleSpinBox, QSpinBox, QComboBox, QPlainTextEdit, QListWidget { background: #1b1f27; border: 1px solid #3b4252; border-radius: 6px; padding: 6px; }
QComboBox QAbstractItemView { background: #1b1f27; selection-background-color: #2f6fed; }
QTabWidget::pane { border: 1px solid #2c3340; border-radius: 6px; }
QTabBar::tab { background: #232833; padding: 10px 18px; margin-right: 2px; border-top-left-radius: 6px; border-top-right-radius: 6px; }
QTabBar::tab:selected { background: #2f6fed; }
QProgressBar { background: #1b1f27; border: 1px solid #3b4252; border-radius: 6px; text-align: center; height: 20px; }
QProgressBar::chunk { background: #2f6fed; border-radius: 5px; }
QSlider::groove:horizontal { height: 8px; background: #2b3140; border-radius: 4px; }
QSlider::handle:horizontal { width: 22px; margin: -8px 0; background: #9fb3ff; border-radius: 11px; }
QStatusBar { background: #0f1115; color: #9aa3b5; font-size: 13px; }
QCheckBox::indicator, QRadioButton::indicator { width: 20px; height: 20px; }
"""
PILL = "padding:3px 10px;border-radius:10px;background:%s;font-size:13px"
PILL_NEUTRAL, PILL_OK, PILL_WARN, PILL_BAD = "#2a2f3a", "#1f9d55", "#8a6d1f", "#a1343e"
COL_OK, COL_UNKNOWN, COL_PENDING, COL_BAD, COL_EMPTY = "#38d67a", "#7f8cff", "#8a6d1f", "#a1343e", "#8a94a8"
WEIGHT_FRESH, WEIGHT_UNSTABLE, WEIGHT_STALE = "color:#7CFC9A", "color:#ffcc4d", "color:#6f7a8f"


# ------------------------------------------------------------------ helpers
def np_to_qimage(frame: np.ndarray) -> QImage:
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    frame = np.ascontiguousarray(frame)
    h, w = frame.shape[:2]
    return QImage(frame.data, w, h, frame.strides[0], QImage.Format.Format_BGR888).copy()


def np_to_pixmap(frame: np.ndarray, size: int) -> QPixmap:
    return QPixmap.fromImage(np_to_qimage(frame)).scaled(size, size, Qt.AspectRatioMode.KeepAspectRatio,
                                                         Qt.TransformationMode.SmoothTransformation)


def icon_from_jpeg(data: Optional[bytes]) -> Optional[QIcon]:
    if not data:
        return None
    img = QImage.fromData(data)
    return QIcon(QPixmap.fromImage(img)) if not img.isNull() else None


def thumb_icon(item: Item) -> Optional[QIcon]:
    return icon_from_jpeg(item.thumbnail)


def msg_info(parent, text):
    QMessageBox.information(parent, tr("info"), text)


def msg_error(parent, text):
    QMessageBox.critical(parent, tr("error"), text)


def ask(parent, text) -> bool:
    return QMessageBox.question(parent, tr("confirm"), text,
                                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes


# ------------------------------------------------------------ video view
class VideoWidget(QWidget):
    """Camera frame, tray outline, and one labelled box per detected object."""
    region_clicked = pyqtSignal(int)          # track_id, -1 to clear the selection

    def __init__(self):
        super().__init__()
        self.setMinimumSize(320, 240)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.image: Optional[QImage] = None
        self.rec: Optional[Recognition] = None
        self.tray = TrayRegion()
        self.has_items = False
        self.active_track = -1
        self.training: Optional[Tuple[int, int]] = None      # (captured, wanted)
        self.show_masks = True
        self.caption = ""
        self.banner = ""
        self.hint = ""

    def set_frame(self, frame: np.ndarray):
        self.image = np_to_qimage(frame)
        self.update()

    def set_recognition(self, rec: Optional[Recognition]):
        self.rec = rec
        self.update()

    # ---- geometry
    def _map(self) -> Tuple[float, float, float]:
        iw, ih = self.image.width(), self.image.height()
        scale = min(self.width() / iw, self.height() / ih)
        return (self.width() - iw * scale) / 2, (self.height() - ih * scale) / 2, scale

    def mousePressEvent(self, ev):
        if self.image is None or self.rec is None:
            return
        ox, oy, scale = self._map()
        fx, fy = (ev.position().x() - ox) / scale, (ev.position().y() - oy) / scale
        for d in self.rec.detections:
            x, y, w, h = d.box
            if x <= fx <= x + w and y <= fy <= y + h:
                self.region_clicked.emit(-1 if self.active_track == d.track_id else d.track_id)
                return
        self.region_clicked.emit(-1)

    # ---- paint
    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        p.fillRect(self.rect(), QColor("#0b0d11"))
        if self.image is None:
            p.setPen(QColor("#5a6275"))
            p.setFont(QFont("Segoe UI", 16))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, tr("no_camera"))
            return
        iw, ih = self.image.width(), self.image.height()
        ox, oy, scale = self._map()
        dw, dh = iw * scale, ih * scale
        p.drawImage(QRectF(ox, oy, dw, dh), self.image)

        rec = self.rec
        alert = bool(rec and (rec.intrusion or rec.unreliable))
        self._paint_tray(p, iw, ih, ox, oy, scale, rec, alert)
        if rec is not None and not self.training:
            for d in rec.detections:
                self._paint_region(p, d, ox, oy, scale, dw)
        if alert:
            p.fillRect(QRectF(ox, oy, dw, dh), QColor(255, 204, 77, 40))
        self._paint_badge(p, rec, ox, oy, dw, dh, alert)
        if self.caption:
            p.setFont(QFont("Segoe UI", 9))
            p.setPen(QColor(230, 230, 230, 200))
            p.drawText(QRectF(ox + 8, oy + 6, dw - 16, 20), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, self.caption)
        if self.hint:
            p.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
            p.setPen(QColor("#ffcc4d"))
            p.drawText(QRectF(ox + 8, oy + 24, dw - 16, 22), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, self.hint)
        p.end()

    def _paint_tray(self, p, iw, ih, ox, oy, scale, rec, alert):
        if self.training:
            colour = QColor("#ffcc4d")
        elif alert:
            colour = QColor(COL_BAD)
        elif rec is not None and rec.empty:
            colour = QColor(COL_EMPTY)
        elif rec is not None and rec.accepted:
            colour = QColor(COL_OK)
        else:
            colour = QColor(COL_UNKNOWN)
        pts = self.tray.pixel_points(iw, ih)
        poly = [QPointF(ox + x * scale, oy + y * scale) for x, y in pts]
        if self.tray.is_polygon:
            p.setPen(QPen(colour, 3))
            for i in range(len(poly)):
                p.drawLine(poly[i], poly[(i + 1) % len(poly)])
            p.setBrush(QBrush(colour))
            p.setPen(Qt.PenStyle.NoPen)
            for pt in poly:
                p.drawEllipse(pt, 4, 4)
        else:
            x0, y0, x1, y1 = self.tray.bbox(iw, ih)
            side = (x1 - x0) * scale
            rx, ry = ox + x0 * scale, oy + y0 * scale
            L = side * 0.18
            p.setPen(QPen(colour, 4))
            for (cx, cy, sx, sy) in ((rx, ry, 1, 1), (rx + side, ry, -1, 1), (rx, ry + side, 1, -1), (rx + side, ry + side, -1, -1)):
                p.drawLine(QPointF(cx, cy), QPointF(cx + sx * L, cy))
                p.drawLine(QPointF(cx, cy), QPointF(cx, cy + sy * L))
            p.setPen(QPen(colour, 1, Qt.PenStyle.DashLine))
            p.drawRect(QRectF(rx, ry, side, side))

    def _paint_region(self, p, d: Detection, ox, oy, scale, dw):
        if d.region is None:
            return                       # whole-tray fallback: the outline already shows it
        x, y, w, h = d.box
        r = QRectF(ox + x * scale, oy + y * scale, w * scale, h * scale)
        if not d.confirmed:
            colour = QColor(COL_PENDING)
        elif d.accepted:
            colour = QColor(COL_OK)
        elif d.suggested:
            colour = QColor(COL_PENDING)
        else:
            colour = QColor(COL_UNKNOWN)
        active = d.track_id == self.active_track
        style = Qt.PenStyle.DashLine if d.split else Qt.PenStyle.SolidLine
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(colour, 4 if active else 2, style))
        p.drawRoundedRect(r, 6, 6)
        if self.show_masks:
            self._paint_outline(p, d, ox, oy, scale, colour)
        if d.accepted:
            label = f"{d.name}  {d.score * 100:.0f}%"
        elif d.suggested:
            label = f"{d.suggested_name}?  {d.result.score * 100:.0f}%"
        else:
            label = tr("unknown_item")
        p.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        fm = p.fontMetrics()
        label = fm.elidedText(label, Qt.TextElideMode.ElideRight, int(max(48, r.width() + 60)))
        tw, th = fm.horizontalAdvance(label) + 12, fm.height() + 6
        ty = r.top() - th - 3
        if ty < oy:
            ty = r.top() + 3
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(QColor(colour.red(), colour.green(), colour.blue(), 235)))
        p.drawRoundedRect(QRectF(r.left(), ty, tw, th), 5, 5)
        p.setPen(QColor("#0b0d11") if d.accepted else QColor("white"))
        p.drawText(QRectF(r.left(), ty, tw, th), Qt.AlignmentFlag.AlignCenter, label)

    def _paint_outline(self, p, d: Detection, ox, oy, scale, colour):
        """Trace the segmented silhouette, so the operator can see whether the
        app has really found the whole item or only part of it."""
        region = d.region
        if region is None or region.mask is None or not region.mask.size:
            return
        try:
            contours, _ = cv2.findContours(region.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        except Exception:
            return
        x, y, _w, _h = d.box
        pen = QPen(colour, 2, Qt.PenStyle.DotLine)
        p.setPen(pen)
        for contour in contours:
            if len(contour) < 3:
                continue
            poly = QPolygonF([QPointF(ox + (x + int(pt[0][0])) * scale, oy + (y + int(pt[0][1])) * scale)
                              for pt in contour[::2]])
            p.drawPolygon(poly)

    def _paint_badge(self, p, rec, ox, oy, dw, dh, alert):
        if self.training:
            done, _total = self.training
            text, colour = tr("samples_captured", count=done), QColor(255, 204, 77, 230)
        elif self.banner:
            text, colour = self.banner, QColor(161, 52, 62, 235)
        elif rec is None:
            text, colour = tr("place_item"), QColor(40, 44, 56, 200)
        elif rec.intrusion:
            text, colour = tr("hand_detected"), QColor(161, 52, 62, 235)
        elif rec.unreliable:
            text, colour = tr("lighting_changed"), QColor(138, 109, 31, 235)
        elif rec.empty:
            text, colour = tr("tray_empty"), QColor(70, 76, 92, 220)
        elif not self.has_items:
            text, colour = tr("no_items_trained"), QColor(40, 44, 56, 200)
        else:
            acc = rec.accepted
            names = sorted({d.name for d in acc})
            if len(acc) > 1 and len(names) == 1:
                text, colour = tr("same_item_qty", count=len(acc), name=names[0]), QColor(31, 157, 85, 230)
            elif len(names) > 1:
                text, colour = " + ".join(names), QColor(138, 109, 31, 235)
            elif acc:
                d = acc[0]
                text, colour = f"{d.name}   {d.score * 100:.0f}%", QColor(31, 157, 85, 230)
            else:
                guess = next((d for d in rec.confirmed if d.suggested), None)
                if guess is not None:
                    text = tr("is_it", name=guess.suggested_name)
                    colour = QColor(138, 109, 31, 235)
                else:
                    text, colour = tr("unknown_item"), QColor(120, 52, 60, 220)
        p.setFont(QFont("Segoe UI", 15, QFont.Weight.Bold))
        fm = p.fontMetrics()
        text = fm.elidedText(text, Qt.TextElideMode.ElideRight, int(max(60, dw - 40)))
        tw, th = fm.horizontalAdvance(text) + 28, fm.height() + 16
        bx, by = ox + (dw - tw) / 2, oy + dh - th - 14
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(colour))
        p.drawRoundedRect(QRectF(bx, by, tw, th), 10, 10)
        p.setPen(QColor("white"))
        p.drawText(QRectF(bx, by, tw, th), Qt.AlignmentFlag.AlignCenter, text)


# ------------------------------------------------------------ tray calibration
class TrayCanvas(QWidget):
    """Interactive polygon / rectangle editor drawn over the live frame."""
    changed = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setMinimumSize(480, 360)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.image: Optional[QImage] = None
        self.points: List[Tuple[float, float]] = []
        self.mode = "poly"
        self.roi_ratio = 0.6
        self._drag_idx: Optional[int] = None
        self._rect_start: Optional[Tuple[float, float]] = None
        self._rect_cur: Optional[Tuple[float, float]] = None

    def set_frame(self, frame: np.ndarray):
        self.image = np_to_qimage(frame)
        self.update()

    def _image_rect(self) -> QRectF:
        if self.image is None:
            return QRectF(0, 0, self.width(), self.height())
        iw, ih = self.image.width(), self.image.height()
        scale = min(self.width() / iw, self.height() / ih)
        dw, dh = iw * scale, ih * scale
        return QRectF((self.width() - dw) / 2, (self.height() - dh) / 2, dw, dh)

    def _to_norm(self, pos: QPointF) -> Tuple[float, float]:
        r = self._image_rect()
        return (float(np.clip((pos.x() - r.x()) / r.width(), 0, 1)), float(np.clip((pos.y() - r.y()) / r.height(), 0, 1)))

    def _to_widget(self, pt: Tuple[float, float]) -> QPointF:
        r = self._image_rect()
        return QPointF(r.x() + pt[0] * r.width(), r.y() + pt[1] * r.height())

    def _hit(self, pos: QPointF) -> Optional[int]:
        for i, pt in enumerate(self.points):
            if (self._to_widget(pt) - pos).manhattanLength() <= 14:
                return i
        return None

    def mousePressEvent(self, ev):
        pos = ev.position()
        if ev.button() == Qt.MouseButton.RightButton:
            if self.points:
                idx = self._hit(pos)
                self.points.pop(idx if idx is not None else -1)
                self.changed.emit()
                self.update()
            return
        if ev.button() != Qt.MouseButton.LeftButton:
            return
        idx = self._hit(pos)
        if idx is not None:
            self._drag_idx = idx
        elif self.mode == "rect":
            self._rect_start = self._rect_cur = self._to_norm(pos)
        else:
            self.points.append(self._to_norm(pos))
            self._drag_idx = len(self.points) - 1
            self.changed.emit()
        self.update()

    def mouseMoveEvent(self, ev):
        pos = ev.position()
        if self._drag_idx is not None:
            self.points[self._drag_idx] = self._to_norm(pos)
            self.changed.emit()
            self.update()
        elif self._rect_start is not None:
            self._rect_cur = self._to_norm(pos)
            self.update()

    def mouseReleaseEvent(self, _ev):
        if self._drag_idx is not None:
            self._drag_idx = None
        elif self._rect_start is not None and self._rect_cur is not None:
            (x0, y0), (x1, y1) = self._rect_start, self._rect_cur
            if abs(x1 - x0) > 0.02 and abs(y1 - y0) > 0.02:
                xa, xb, ya, yb = min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1)
                self.points = [(xa, ya), (xb, ya), (xb, yb), (xa, yb)]
                self.changed.emit()
            self._rect_start = self._rect_cur = None
        self.update()

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor("#0b0d11"))
        r = self._image_rect()
        if self.image is not None:
            p.drawImage(r, self.image)
        else:
            p.setPen(QColor("#5a6275"))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, tr("no_frame"))
        iw = self.image.width() if self.image else 640
        ih = self.image.height() if self.image else 480
        region = TrayRegion(self.points, True, self.roi_ratio)
        pts = [self._to_widget((x / max(1, iw - 1), y / max(1, ih - 1))) for x, y in region.pixel_points(iw, ih)]
        colour = QColor(COL_OK) if region.is_polygon else QColor(COL_UNKNOWN)
        if region.is_polygon:
            outer = QPainterPath()
            outer.addRect(r)
            inner = QPainterPath()
            inner.addPolygon(QPolygonF(pts))
            inner.closeSubpath()
            p.fillPath(outer.subtracted(inner), QBrush(QColor(0, 0, 0, 110)))
        p.setPen(QPen(colour, 3, Qt.PenStyle.SolidLine if region.is_polygon else Qt.PenStyle.DashLine))
        for i in range(len(pts)):
            p.drawLine(pts[i], pts[(i + 1) % len(pts)])
        if self.points:
            p.setBrush(QBrush(QColor("#ffd166")))
            p.setPen(QPen(QColor("#14171c"), 2))
            for pt in self.points:
                p.drawEllipse(self._to_widget(pt), 7, 7)
        if self._rect_start is not None and self._rect_cur is not None:
            a, b = self._to_widget(self._rect_start), self._to_widget(self._rect_cur)
            p.setPen(QPen(QColor("#ffd166"), 2, Qt.PenStyle.DashLine))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRect(QRectF(a, b).normalized())
        p.end()


class TrayCalibrationDialog(QDialog):
    def __init__(self, parent, get_frame, tray: TrayRegion):
        super().__init__(parent)
        self.get_frame = get_frame
        self.setWindowTitle(tr("tray_calibrate_title"))
        self.resize(1060, 640)
        self.result_region: Optional[TrayRegion] = None
        self.canvas = TrayCanvas()
        self.canvas.points = list(tray.points)
        self.canvas.roi_ratio = tray.roi_ratio
        self.canvas.changed.connect(self._refresh_preview)
        side = QVBoxLayout()
        self.rb_poly = QRadioButton(tr("mode_polygon"))
        self.rb_rect = QRadioButton(tr("mode_rect"))
        self.rb_poly.setChecked(True)
        grp = QButtonGroup(self)
        grp.addButton(self.rb_poly)
        grp.addButton(self.rb_rect)
        self.rb_poly.toggled.connect(self._mode_changed)
        self.hint = QLabel(tr("tray_hint_poly"))
        self.hint.setWordWrap(True)
        self.hint.setStyleSheet("color:#9aa3b5")
        self.hint.setMaximumWidth(280)
        self.mask_cb = QCheckBox(tr("mask_outside"))
        self.mask_cb.setChecked(tray.mask_outside)
        self.mask_cb.toggled.connect(self._refresh_preview)
        self.preview = QLabel()
        self.preview.setFixedSize(260, 260)
        self.preview.setStyleSheet("background:#1b1f27;border:1px solid #2c3340;border-radius:6px")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        reset = QPushButton(tr("reset_default"))
        reset.clicked.connect(self._reset)
        side.addWidget(self.rb_poly)
        side.addWidget(self.rb_rect)
        side.addWidget(self.hint)
        side.addWidget(self.mask_cb)
        side.addSpacing(10)
        side.addWidget(QLabel(tr("model_view")))
        side.addWidget(self.preview)
        side.addWidget(reset)
        side.addStretch(1)
        box = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        box.accepted.connect(self.save)
        box.rejected.connect(self.reject)
        side.addWidget(box)
        lay = QHBoxLayout(self)
        lay.addWidget(self.canvas, 1)
        lay.addLayout(side)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(120)
        self._n = 0

    def _mode_changed(self):
        self.canvas.mode = "poly" if self.rb_poly.isChecked() else "rect"
        self.hint.setText(tr("tray_hint_poly") if self.canvas.mode == "poly" else tr("tray_hint_rect"))

    def _reset(self):
        self.canvas.points = []
        self.canvas.update()
        self._refresh_preview()

    def _tick(self):
        _, frame = self.get_frame()
        if frame is not None:
            self.canvas.set_frame(frame)
            self._n += 1
            if self._n % 3 == 0:
                self._refresh_preview()

    def _refresh_preview(self, *_args):
        _, frame = self.get_frame()
        if frame is None:
            return
        region = TrayRegion(self.canvas.points, self.mask_cb.isChecked(), self.canvas.roi_ratio)
        crop, _ = region.crop(frame)
        self.preview.setPixmap(np_to_pixmap(crop, 256))

    def save(self):
        pts = self.canvas.points
        if 0 < len(pts) < 3:
            msg_error(self, tr("tray_points_needed"))
            return
        self.result_region = TrayRegion(pts, self.mask_cb.isChecked(), self.canvas.roi_ratio)
        self.accept()

    def done(self, result: int):
        self.timer.stop()
        super().done(result)


# ------------------------------------------------------------ training dialog
class TrainDialog(QDialog):
    """Enrol a new item or add angle samples: one sample per button press.

    Every press stores its own capture group, which is what makes the per-item
    threshold calibration possible.  Capture count is the single biggest driver
    of accuracy measured (hard-set top-1 0.41 at one capture, 0.69 at six), so
    the dialog nags for six rather than three.
    """

    def __init__(self, parent: "MainWindow", existing: Optional[Item] = None):
        super().__init__(parent)
        self.win = parent
        self.engine, self.db, self.bus, self.settings = parent.engine, parent.db, parent.bus, parent.settings
        self.existing = existing
        self.setWindowTitle(tr("add_angle_sample") if existing else tr("train_new_item"))
        self.resize(920, 600)
        self.samples: List[Tuple[np.ndarray, bytes]] = []        # (embeddings, thumbnail jpeg)
        self._pending: Dict[str, bytes] = {}
        self.min_samples = int(self.settings.get("ai.min_train_samples", 3))
        self.good_samples = int(self.settings.get("ai.good_train_samples", 6))
        self.crop_mode = MODE_TRAY

        left = QVBoxLayout()
        left.addWidget(QLabel(tr("model_view")))
        self.preview = QLabel()
        self.preview.setFixedSize(300, 300)
        self.preview.setStyleSheet("background:#1b1f27;border:2px solid #2c3340;border-radius:8px")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.mode_lbl = QLabel("")
        self.mode_lbl.setWordWrap(True)
        self.mode_lbl.setMaximumWidth(300)
        self.mode_lbl.setStyleSheet("color:#9aa3b5;font-size:13px")
        self.capture_btn = QPushButton(tr("capture_sample"))
        self.capture_btn.setObjectName("primary")
        self.capture_btn.setMinimumHeight(64)
        self.auto_cb = QCheckBox(tr("auto_capture", ms=int(self.settings.get("ai.auto_capture_interval_ms", 700))))
        self.auto_cb.setChecked(bool(self.settings.get("ai.auto_capture", False)))
        hint = QLabel(tr("capture_hint"))
        hint.setWordWrap(True)
        hint.setMaximumWidth(300)
        hint.setStyleSheet("color:#9aa3b5")
        left.addWidget(self.preview)
        left.addWidget(self.mode_lbl)
        left.addWidget(self.capture_btn)
        left.addWidget(self.auto_cb)
        left.addWidget(hint)
        left.addStretch(1)

        right = QVBoxLayout()
        form = QFormLayout()
        self.item_combo = QComboBox()
        self.name_edit = QLineEdit()
        self.price_spin = QDoubleSpinBox()
        self.price_spin.setRange(0, 1e9)
        self.price_spin.setDecimals(int(self.settings.get("general.currency_decimals", 0)))
        self.price_spin.setGroupSeparatorShown(True)
        self.price_spin.setSuffix(" " + self.settings.get("general.currency", ""))
        if existing is not None:
            for it in self.db.all_items():
                self.item_combo.addItem(it.name, it.id)
                if it.id == existing.id:
                    self.item_combo.setCurrentIndex(self.item_combo.count() - 1)
            self.item_combo.currentIndexChanged.connect(self._sync_existing)
            form.addRow(tr("select_item"), self.item_combo)
            self.name_edit.setEnabled(False)
            self.price_spin.setEnabled(False)
            self._sync_existing()
        form.addRow(tr("item_name"), self.name_edit)
        form.addRow(tr("price_per_kg"), self.price_spin)
        right.addLayout(form)
        self.count_lbl = QLabel(tr("samples_captured", count=0))
        self.count_lbl.setStyleSheet("font-weight:600")
        right.addWidget(self.count_lbl)
        self.list = QListWidget()
        self.list.setViewMode(QListWidget.ViewMode.IconMode)
        self.list.setIconSize(QSize(84, 84))
        self.list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.list.setSpacing(6)
        self.list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        right.addWidget(self.list, 1)
        self.del_btn = QPushButton(tr("delete_sample"))
        self.del_btn.clicked.connect(self._delete_selected)
        right.addWidget(self.del_btn)
        self.need_lbl = QLabel(tr("need_samples", min=self.min_samples))
        self.need_lbl.setStyleSheet("color:#ffcc4d")
        self.need_lbl.setWordWrap(True)
        right.addWidget(self.need_lbl)
        btns = QHBoxLayout()
        self.save_btn = QPushButton(tr("save_item"))
        self.save_btn.setObjectName("success")
        self.save_btn.setEnabled(False)
        self.cancel_btn = QPushButton(tr("cancel"))
        btns.addWidget(self.save_btn)
        btns.addWidget(self.cancel_btn)
        right.addLayout(btns)

        lay = QHBoxLayout(self)
        lay.addLayout(left)
        lay.addLayout(right, 1)

        self.capture_btn.clicked.connect(self.capture)
        self.save_btn.clicked.connect(self.save)
        self.cancel_btn.clicked.connect(self.reject)
        self.bus.embedding_ready.connect(self.on_embedding)
        QShortcut(QKeySequence(Qt.Key.Key_Space), self, activated=self.capture)
        self.preview_timer = QTimer(self)
        self.preview_timer.timeout.connect(self._refresh_preview)
        self.preview_timer.start(150)
        self.auto_timer = QTimer(self)
        self.auto_timer.timeout.connect(self._auto_tick)
        self.auto_timer.start(int(self.settings.get("ai.auto_capture_interval_ms", 700)))
        self.win.video.training = (0, self.good_samples)

    def _sync_existing(self, *_args):
        item = self.db.get(self.item_combo.currentData())
        if item:
            self.name_edit.setText(item.name)
            self.price_spin.setValue(item.price_per_kg)

    def _refresh_preview(self):
        sample = self.engine.current_sample()
        if sample is None:
            return
        crop, mode, _seg = sample
        self.preview.setPixmap(np_to_pixmap(crop, 296))
        if not self.samples:
            self.crop_mode = mode
        self.mode_lbl.setText(tr("object_mode_note") if self.crop_mode == MODE_OBJECT else tr("tray_mode_note"))

    def _auto_tick(self):
        if self.auto_cb.isChecked() and self.isVisible():
            self.capture(auto=True)

    def capture(self, auto: bool = False):
        sample = self.engine.current_sample()
        if sample is None:
            if not auto:
                msg_error(self, tr("no_frame"))
            return
        crop, mode, seg = sample
        # a frame with a hand or two products in it teaches the model the wrong thing
        if seg is not None:
            if seg.intrusion:
                if not auto:
                    msg_error(self, tr("capture_blocked_hand"))
                return
            if len([r for r in seg.regions if r.confirmed]) > 1:
                if not auto:
                    msg_error(self, tr("capture_blocked_multi"))
                return
            if not seg.regions and self.win.tray_looks_empty():
                if auto or not ask(self, tr("training_blocked_empty")):
                    return
        elif self.win.tray_looks_empty():
            if auto or not ask(self, tr("training_blocked_empty")):
                return
        if self.samples and mode != self.crop_mode:
            mode = self.crop_mode          # never mix crop styles inside one item
        thumb = make_thumbnail(crop, 96) or b""
        token = uuid.uuid4().hex
        self._pending[token] = thumb
        self.crop_mode = mode if not self.samples else self.crop_mode
        augment = bool(self.settings.get("ai.augment_samples", True))
        self.engine.request_embedding(crop, lambda emb, t=token: self.bus.embedding_ready.emit(t, emb), augment)
        self.capture_btn.setEnabled(False)
        QTimer.singleShot(350, lambda: self.capture_btn.setEnabled(True))

    def on_embedding(self, token: str, embeddings: np.ndarray):
        thumb = self._pending.pop(token, None)
        if thumb is None:
            return                            # not ours (empty-tray or reinforcement capture)
        self.samples.append((embeddings, thumb))
        self.list.addItem(QListWidgetItem(icon_from_jpeg(thumb) or QIcon(), str(len(self.samples))))
        self._update_counts()

    def _delete_selected(self):
        row = self.list.currentRow()
        if row < 0 or row >= len(self.samples):
            return
        del self.samples[row]
        self.list.clear()
        for i, (_, thumb) in enumerate(self.samples):
            self.list.addItem(QListWidgetItem(icon_from_jpeg(thumb) or QIcon(), str(i + 1)))
        self._update_counts()

    def _update_counts(self):
        n = len(self.samples)
        self.count_lbl.setText(tr("samples_captured", count=n))
        self.save_btn.setEnabled(n > 0)
        if n < self.min_samples:
            self.need_lbl.setText(tr("need_samples", min=self.min_samples))
            self.need_lbl.setVisible(True)
        elif n < self.good_samples:
            self.need_lbl.setText(tr("capture_more_views", done=n, want=self.good_samples))
            self.need_lbl.setVisible(True)
        else:
            self.need_lbl.setVisible(False)
        self.win.video.training = (n, self.good_samples)

    def save(self):
        name = self.name_edit.text().strip()
        if not name:
            msg_error(self, tr("invalid_name"))
            return
        price = float(self.price_spin.value())
        if price <= 0:
            msg_error(self, tr("invalid_price"))
            return
        if not self.samples:
            return
        if len(self.samples) < self.min_samples and not ask(self, tr("few_samples_confirm", count=len(self.samples))):
            return
        embeddings = np.vstack([e for e, _ in self.samples])
        groups = np.concatenate([np.full(len(e), i, np.int32) for i, (e, _) in enumerate(self.samples)])
        thumb = self.samples[0][1] or None
        item_id = self.item_combo.currentData() if self.existing is not None else None
        if item_id is None:
            dup = self.db.find_by_name(name)
            if dup is not None:
                item_id = dup.id                     # same name -> add samples instead of a duplicate item
        try:
            if item_id and self.db.get(item_id):
                item = self.db.add_samples(item_id, embeddings, thumb, groups, self.crop_mode)
                if self.existing is None:
                    self.db.update_item(item_id, price_per_kg=price)
            else:
                item = self.db.add_item(name, price, embeddings, thumb, groups, self.crop_mode)
            self.db.save()
            self.engine.reset_smoothing()
        except Exception as exc:
            msg_error(self, tr("training_failed", error=str(exc)))
            return
        msg_info(self, tr("training_done", name=item.name, count=item.sample_count))
        self.accept()

    def done(self, result: int):
        self.preview_timer.stop()
        self.auto_timer.stop()
        try:
            self.bus.embedding_ready.disconnect(self.on_embedding)
        except (TypeError, RuntimeError):
            pass
        self.win.video.training = None
        super().done(result)


# ------------------------------------------------------------ items dialog
class ManageItemsDialog(QDialog):
    def __init__(self, parent, db: ItemDatabase, settings: SettingsManager):
        super().__init__(parent)
        self.db, self.settings = db, settings
        self.setWindowTitle(tr("manage_items"))
        self.resize(880, 500)
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels([tr("name"), tr("price_per_kg"), tr("samples"), tr("mode_column"),
                                              tr("learned"), tr("delete")])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        lay = QVBoxLayout(self)
        self.count_label = QLabel()
        lay.addWidget(self.count_label)
        self.warn_label = QLabel()
        self.warn_label.setWordWrap(True)
        self.warn_label.setStyleSheet("color:#ffcc4d")
        lay.addWidget(self.warn_label)
        lay.addWidget(self.table)
        box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        box.rejected.connect(self.accept)
        lay.addWidget(box)
        self.table.itemChanged.connect(self.on_changed)
        self.reload()

    def reload(self):
        self.table.blockSignals(True)
        items = self.db.all_items()
        self.table.setRowCount(len(items))
        for r, it in enumerate(items):
            name_item = QTableWidgetItem(it.name)
            name_item.setData(Qt.ItemDataRole.UserRole, it.id)
            icon = thumb_icon(it)
            if icon:
                name_item.setIcon(icon)
            self.table.setItem(r, 0, name_item)
            self.table.setItem(r, 1, QTableWidgetItem(f"{it.price_per_kg:g}"))
            cnt = QTableWidgetItem(f"{it.capture_count} ({it.sample_count})")
            cnt.setFlags(cnt.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(r, 2, cnt)
            mode = QTableWidgetItem("object" if it.crop_mode == MODE_OBJECT else "tray")
            mode.setFlags(mode.flags() & ~Qt.ItemFlag.ItemIsEditable)
            if it.crop_mode != MODE_OBJECT:
                mode.setToolTip(tr("retrain_needed"))
                mode.setForeground(QColor("#ffcc4d"))
            self.table.setItem(r, 3, mode)
            if it.drift_count:
                btn = QPushButton(f"{it.drift_count}  {tr('forget_learned')}")
                btn.clicked.connect(lambda _=False, iid=it.id: self.forget(iid))
                self.table.setCellWidget(r, 4, btn)
            else:
                empty = QTableWidgetItem("0")
                empty.setFlags(empty.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.table.setItem(r, 4, empty)
            btn = QPushButton(tr("delete"))
            btn.setObjectName("danger")
            btn.clicked.connect(lambda _=False, iid=it.id, nm=it.name: self.delete(iid, nm))
            self.table.setCellWidget(r, 5, btn)
        self.table.blockSignals(False)
        self.count_label.setText(tr("items_count", count=len(items)))
        pairs = self.db.confusable_pairs(5)
        self.warn_label.setText(tr("confusable_items", pairs=", ".join(f"{a} / {b}" for a, b, _ in pairs)) if pairs else "")
        self.warn_label.setVisible(bool(pairs))

    def on_changed(self, cell: QTableWidgetItem):
        row = cell.row()
        iid = self.table.item(row, 0).data(Qt.ItemDataRole.UserRole)
        try:
            if cell.column() == 0:
                self.db.update_item(iid, name=cell.text())
            elif cell.column() == 1:
                self.db.update_item(iid, price_per_kg=float(cell.text().replace(",", "")))
            self.db.save()
        except Exception as exc:
            msg_error(self, str(exc))
            self.reload()

    def forget(self, iid: str):
        self.db.forget_drift(iid)
        self.db.save()
        self.reload()

    def delete(self, iid: str, name: str):
        if ask(self, tr("delete_item_confirm", name=name)):
            self.db.delete_item(iid)
            self.db.save()
            self.reload()


# ------------------------------------------------------------ settings dialog
class SettingsDialog(QDialog):
    def __init__(self, parent, settings: SettingsManager, printer_factory, bus: Bus):
        super().__init__(parent)
        self.settings, self.printer_factory, self.bus = settings, printer_factory, bus
        self.setWindowTitle(tr("settings"))
        self.resize(860, 720)
        self._monitor: Optional[ScaleReader] = None
        self._suggestion: Optional[dict] = None
        self._found: Optional[dict] = None
        self._diag_running = False
        tabs = QTabWidget()
        tabs.addTab(self._general_tab(), tr("tab_general"))
        tabs.addTab(self._camera_tab(), tr("tab_camera"))
        tabs.addTab(self._scale_tab(), tr("tab_scale"))
        tabs.addTab(self._printer_tab(), tr("tab_printer"))
        tabs.addTab(self._support_tab(), tr("tab_support"))
        lay = QVBoxLayout(self)
        lay.addWidget(tabs)
        note = QLabel(tr("restart_note"))
        note.setStyleSheet("color:#9aa3b5")
        lay.addWidget(note)
        box = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        box.accepted.connect(self.save)
        box.rejected.connect(self.reject)
        lay.addWidget(box)
        self.bus.dialog_result.connect(self.on_dialog_result)

    def on_dialog_result(self, kind: str, text: str, is_error: bool):
        if kind == "support":
            self.sup_test_btn.setEnabled(True)
            (msg_error if is_error else msg_info)(self, text)
            return
        if kind == "scale_line":
            self._append_scale(text)
        elif kind == "scale_done":
            self._diag_running = False
            self.diag_btn.setEnabled(True)
            self.find_btn.setEnabled(True)
            self.apply_btn.setEnabled(self._suggestion is not None)
            self._append_scale(("PROBLEM: " if is_error else "OK: ") + text)
            # The bug this fixes: a diagnosis that succeeds returns no "suggestion",
            # so the only code that ever left simulation on was unreachable exactly
            # when the hardware worked first time.
            if not is_error and self.source.currentData() == SOURCE_SIMULATE:
                self.source.setCurrentIndex([SOURCE_AUTO, SOURCE_HID, SOURCE_SERIAL,
                                             SOURCE_SIMULATE].index(SOURCE_SERIAL))
                self._append_scale("-> " + tr("sim_unticked"))
        elif kind == "scale_found":
            self._diag_running = False
            self.find_btn.setEnabled(True)
            self.diag_btn.setEnabled(True)
            self._append_scale(("PROBLEM: " if is_error else "OK: ") + text)
            if self._found:
                self._apply_found()
                msg_info(self, tr("find_scale_found", label=self._found["label"]))
            else:
                msg_error(self, tr("find_scale_none"))
        elif kind == "print":
            (msg_error if is_error else msg_info)(self, text)

    def done(self, result: int):
        self._stop_monitor()
        try:
            self.bus.dialog_result.disconnect(self.on_dialog_result)
        except (TypeError, RuntimeError):
            pass
        super().done(result)

    # -- general
    def _general_tab(self):
        w = QWidget()
        f = QFormLayout(w)
        g, pos = self.settings.section("general"), self.settings.section("pos")
        self.lang = QComboBox()
        self.lang.addItem("فارسی (Persian)", "fa")
        self.lang.addItem("English", "en")
        self.lang.setCurrentIndex(0 if g["language"] == "fa" else 1)
        self.store = QLineEdit(g["store_name"])
        self.currency = QLineEdit(g["currency"])
        self.cur_dec = QSpinBox()
        self.cur_dec.setRange(0, 4)
        self.cur_dec.setValue(int(g["currency_decimals"]))
        self.auto_add = QCheckBox(tr("auto_add"))
        self.auto_add.setChecked(bool(g.get("auto_add", False)))
        self.multi_mode = QComboBox()
        for key in ("interlock", "quantity", "off"):
            self.multi_mode.addItem(tr(f"mode_{key}"), key)
        keys = ["interlock", "quantity", "off"]
        cur = pos.get("multi_item_mode", "interlock")
        self.multi_mode.setCurrentIndex(keys.index(cur) if cur in keys else 0)
        self.refuse_hand = QCheckBox(tr("hand_detected"))
        self.refuse_hand.setChecked(bool(pos.get("refuse_on_intrusion", True)))
        self.reinforce = QCheckBox(tr("learn_manual"))
        self.reinforce.setChecked(bool(pos.get("reinforce_on_manual_pick", True)))
        f.addRow(tr("language"), self.lang)
        f.addRow(tr("store_name"), self.store)
        f.addRow(tr("currency"), self.currency)
        f.addRow(tr("currency_decimals"), self.cur_dec)
        f.addRow("", self.auto_add)
        self.weigh_mode = QComboBox()
        for key in ("cumulative", "replace"):
            self.weigh_mode.addItem(tr(f"mode_{key}"), key)
        wm = pos.get("weighing_mode", "cumulative")
        self.weigh_mode.setCurrentIndex(0 if wm == "cumulative" else 1)
        self.tolerance = QDoubleSpinBox()
        self.tolerance.setRange(0.001, 1.0)
        self.tolerance.setDecimals(3)
        self.tolerance.setSingleStep(0.005)
        self.tolerance.setValue(float(pos.get("weight_tolerance_kg", 0.02)))
        self.block_mismatch = QCheckBox(tr("block_mismatch"))
        self.block_mismatch.setChecked(bool(pos.get("block_checkout_on_mismatch", True)))
        f.addRow(tr("weighing_mode"), self.weigh_mode)
        f.addRow(tr("weight_tolerance"), self.tolerance)
        f.addRow("", self.block_mismatch)
        f.addRow(tr("multi_item_mode"), self.multi_mode)
        f.addRow("", self.refuse_hand)
        f.addRow("", self.reinforce)
        return w

    # -- camera & AI
    def _camera_tab(self):
        w = QWidget()
        f = QFormLayout(w)
        c, a = self.settings.section("camera"), self.settings.section("ai")
        self.backend = QComboBox()
        keys = ["auto", "rtk", "opencv"]
        for key in keys:
            self.backend.addItem(tr(f"backend_{key}"), key)
        self.backend.setCurrentIndex(keys.index(c["backend"]) if c["backend"] in keys else 0)
        rtk_note = QLabel(tr("rtk_note"))
        rtk_note.setWordWrap(True)
        rtk_note.setStyleSheet("color:#9aa3b5;font-size:13px")
        self.dev_index = QSpinBox()
        self.dev_index.setRange(0, 9)
        self.dev_index.setValue(int(c["device_index"]))
        self.resolution = QComboBox()
        for res in ("640x480", "800x600", "1280x720", "1920x1080"):
            self.resolution.addItem(res)
        self.resolution.setEditable(True)
        self.resolution.setCurrentText(f"{c['width']}x{c['height']}")
        self.mirror = QCheckBox(tr("mirror"))
        self.mirror.setChecked(bool(c["mirror"]))
        self.rotate = QComboBox()
        for r in (0, 90, 180, 270):
            self.rotate.addItem(f"{r}°", r)
        rot = int(c.get("rotate", 0))
        self.rotate.setCurrentIndex((0, 90, 180, 270).index(rot) if rot in (0, 90, 180, 270) else 0)
        self.per_object = QCheckBox(tr("per_object_recognition"))
        self.per_object.setChecked(bool(a.get("per_object_recognition", True)))
        self.max_objects = QSpinBox()
        self.max_objects.setRange(1, 8)
        self.max_objects.setValue(int(a.get("max_objects", 5)))
        self.thr = QSlider(Qt.Orientation.Horizontal)
        self.thr.setRange(30, 95)
        self.thr.setValue(int(float(a["confidence_threshold"]) * 100))
        self.thr_lbl = QLabel(f"{self.thr.value() / 100:.2f}")
        self.thr.valueChanged.connect(lambda v: self.thr_lbl.setText(f"{v / 100:.2f}"))
        self.smooth = QSpinBox()
        self.smooth.setRange(1, 15)
        self.smooth.setValue(int(a["smoothing_frames"]))
        self.min_samples = QSpinBox()
        self.min_samples.setRange(1, 30)
        self.min_samples.setValue(int(a.get("min_train_samples", 3)))
        self.motion_gate = QCheckBox(tr("motion_gate"))
        self.motion_gate.setChecked(bool(a.get("motion_gate", True)))
        self.video = QLineEdit(c.get("video_file", ""))
        browse = QPushButton(tr("browse"))
        browse.clicked.connect(self._browse_video)
        vrow = QHBoxLayout()
        vrow.addWidget(self.video)
        vrow.addWidget(browse)
        f.addRow(tr("backend"), self.backend)
        f.addRow("", rtk_note)
        f.addRow(tr("device_index"), self.dev_index)
        f.addRow(tr("resolution"), self.resolution)
        f.addRow("", self.mirror)
        f.addRow(tr("rotate"), self.rotate)
        f.addRow("", self.per_object)
        f.addRow(tr("max_objects"), self.max_objects)
        row = QHBoxLayout(); row.addWidget(self.thr); row.addWidget(self.thr_lbl)
        f.addRow(tr("threshold"), row)
        f.addRow(tr("smoothing"), self.smooth)
        f.addRow(tr("train_samples"), self.min_samples)
        f.addRow("", self.motion_gate)
        f.addRow(tr("video_file"), vrow)
        return w

    def _browse_video(self):
        path, _ = QFileDialog.getOpenFileName(self, tr("video_file"), "", "Video (*.avi *.mp4 *.mkv *.mov);;All (*)")
        if path:
            self.video.setText(path)

    # -- scale
    def _scale_tab(self):
        w = QWidget()
        outer = QVBoxLayout(w)
        f = QFormLayout()
        s = self.settings.section("scale")
        self.source = QComboBox()
        for key in (SOURCE_AUTO, SOURCE_HID, SOURCE_SERIAL, SOURCE_SIMULATE):
            self.source.addItem(tr(f"source_{key}"), key)
        current = str(s.get("source") or (SOURCE_SIMULATE if s.get("simulate") else SOURCE_AUTO))
        keys = [SOURCE_AUTO, SOURCE_HID, SOURCE_SERIAL, SOURCE_SIMULATE]
        self.source.setCurrentIndex(keys.index(current) if current in keys else 0)
        self.hid = QComboBox()
        self._scan_hid()
        hid_rescan = QPushButton(tr("scan_ports"))
        hid_rescan.clicked.connect(self._scan_hid)
        hrow = QHBoxLayout(); hrow.addWidget(self.hid); hrow.addWidget(hid_rescan)
        self.port = QComboBox()
        self.port.setEditable(True)
        scan = QPushButton(tr("scan_ports"))
        scan.clicked.connect(self._scan_ports)
        self._scan_ports()
        self.port.setCurrentText(s["port"])
        prow = QHBoxLayout(); prow.addWidget(self.port); prow.addWidget(scan)
        self.baud = QComboBox()
        for b in BAUDRATES:
            self.baud.addItem(str(b), b)
        self.baud.setCurrentText(str(s["baudrate"]))
        self.bits = QComboBox()
        for b in (8, 7):
            self.bits.addItem(str(b), b)
        self.bits.setCurrentText(str(s["bytesize"]))
        self.parity = QComboBox()
        for label, val in (("None", "N"), ("Even", "E"), ("Odd", "O"), ("Mark", "M"), ("Space", "S")):
            self.parity.addItem(label, val)
        pv = str(s["parity"]).upper()[:1]
        self.parity.setCurrentIndex("NEOMS".index(pv) if pv in "NEOMS" else 0)
        self.stop = QComboBox()
        for v in (1, 1.5, 2):
            self.stop.addItem(str(v), v)
        self.stop.setCurrentText(str(s["stopbits"]))
        self.poll = QLineEdit(s.get("poll_command", ""))
        self.poll.setPlaceholderText(r"e.g. W\r\n  or  \x05  (empty = continuous stream)")
        self.unit = QComboBox()
        for u in ("auto", "kg", "g", "lb"):
            self.unit.addItem(u)
        self.unit.setCurrentText(s.get("unit", "auto"))
        self.implied = QSpinBox()
        self.implied.setRange(0, 4)
        self.implied.setValue(int(s.get("implied_decimals", 0)))
        self.hold = QDoubleSpinBox()
        self.hold.setRange(1, 60)
        self.hold.setValue(float(s.get("hold_last_s", 5.0)))
        self.stable_only = QCheckBox(tr("stable_only"))
        self.stable_only.setChecked(bool(s.get("stable_only", False)))
        f.addRow(tr("scale_source"), self.source)
        f.addRow(tr("hid_device"), hrow)
        f.addRow(tr("com_port"), prow)
        f.addRow(tr("baudrate"), self.baud)
        f.addRow(tr("data_bits"), self.bits)
        f.addRow(tr("parity"), self.parity)
        f.addRow(tr("stop_bits"), self.stop)
        f.addRow(tr("poll_command"), self.poll)
        f.addRow(tr("unit"), self.unit)
        f.addRow(tr("implied_decimals"), self.implied)
        f.addRow(tr("scale_hold"), self.hold)
        f.addRow("", self.stable_only)
        outer.addLayout(f)
        brow = QHBoxLayout()
        self.find_btn = QPushButton(tr("find_scale"))
        self.find_btn.setObjectName("primary")
        self.find_btn.clicked.connect(self._find_scale)
        self.monitor_btn = QPushButton(tr("start_monitor"))
        self.monitor_btn.clicked.connect(self._toggle_monitor)
        self.diag_btn = QPushButton(tr("diagnose_scale"))
        self.diag_btn.clicked.connect(self._diagnose)
        self.apply_btn = QPushButton(tr("apply_suggestion"))
        self.apply_btn.setEnabled(False)
        self.apply_btn.clicked.connect(self._apply_suggestion)
        copy_btn = QPushButton(tr("copy_report"))
        copy_btn.clicked.connect(self._copy_report)
        for b in (self.find_btn, self.monitor_btn, self.diag_btn, self.apply_btn, copy_btn):
            brow.addWidget(b)
        outer.addLayout(brow)
        outer.addWidget(QLabel(tr("serial_monitor")))
        self.scale_result = QPlainTextEdit()
        self.scale_result.setReadOnly(True)
        self.scale_result.setMinimumHeight(150)
        self.scale_result.setMaximumBlockCount(400)
        self.scale_result.setStyleSheet("font-family: Consolas, monospace; font-size: 13px")
        self.scale_result.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        outer.addWidget(self.scale_result, 1)
        return w

    def _scan_hid(self):
        cur = self.hid.currentData()
        self.hid.clear()
        try:
            devices = list_hid_scales()
        except Exception:
            devices = []
        if not devices:
            self.hid.addItem(tr("hid_none"), "")
        for d in devices:
            self.hid.addItem(f"{d.product or 'scale'} [{d.vid:04X}:{d.pid:04X}]", d.path)
        if cur:
            i = self.hid.findData(cur)
            if i >= 0:
                self.hid.setCurrentIndex(i)

    def _find_scale(self):
        """Search USB HID devices and every COM port, then offer what answered."""
        if self._diag_running:
            return
        self._stop_monitor()
        self._found = None
        self._diag_running = True
        self.find_btn.setEnabled(False)
        self.diag_btn.setEnabled(False)
        self._append_scale("=== " + tr("find_scale_running") + " ===")

        def work():
            try:
                res = find_scale(progress=lambda line: self.bus.dialog_result.emit("scale_line", line, False))
                self._found = res.get("best") if res.get("found") else None
                message = (tr("find_scale_found", label=res["best"]["label"]) if res.get("found")
                           else tr("find_scale_none"))
                self.bus.dialog_result.emit("scale_found", message, not res.get("found"))
            except Exception as exc:
                self.bus.dialog_result.emit("scale_found", str(exc), True)

        self._diag_thread = threading.Thread(target=work, daemon=True)
        self._diag_thread.start()

    def _apply_found(self):
        cfg = (self._found or {}).get("config") or {}
        if not cfg:
            return
        source = cfg.get("source", SOURCE_SERIAL)
        keys = [SOURCE_AUTO, SOURCE_HID, SOURCE_SERIAL, SOURCE_SIMULATE]
        self.source.setCurrentIndex(keys.index(source) if source in keys else 0)
        if cfg.get("hid_path"):
            self._scan_hid()
            i = self.hid.findData(cfg["hid_path"])
            if i >= 0:
                self.hid.setCurrentIndex(i)
        if cfg.get("port"):
            self.port.setCurrentText(cfg["port"])
            self.baud.setCurrentText(str(cfg.get("baudrate", 9600)))
            self.bits.setCurrentText(str(cfg.get("bytesize", 8)))
            self.parity.setCurrentIndex("NEOMS".index(cfg.get("parity", "N")))
            self.stop.setCurrentText(str(cfg.get("stopbits", 1)))
            self.poll.setText(cfg.get("poll_command", ""))
        self._append_scale("applied: " + json.dumps({k: v for k, v in cfg.items() if k != "hid_path"}))

    def _scan_ports(self):
        cur = self.port.currentText()
        self.port.clear()
        for dev, desc in list_serial_ports():
            self.port.addItem(dev, dev)
            self.port.setItemData(self.port.count() - 1, f"{dev} - {desc}", Qt.ItemDataRole.ToolTipRole)
        self.port.setCurrentText(cur)

    def _scale_config(self) -> dict:
        cfg = self.settings.section("scale")
        source = self.source.currentData()
        cfg.update({"source": source, "hid_path": self.hid.currentData() or "",
                    "port": self.port.currentText().strip(), "baudrate": int(self.baud.currentText()),
                    "bytesize": int(self.bits.currentText()), "parity": self.parity.currentData(),
                    "stopbits": float(self.stop.currentText()), "poll_command": self.poll.text(),
                    "unit": self.unit.currentText(), "implied_decimals": self.implied.value(),
                    "hold_last_s": float(self.hold.value()), "stable_only": self.stable_only.isChecked(),
                    "simulate": source == SOURCE_SIMULATE, "enabled": True})
        return cfg

    def _append_scale(self, text: str):
        self.scale_result.appendPlainText(text)

    def _toggle_monitor(self):
        if self._monitor is not None:
            self._stop_monitor()
            return
        if self._diag_running:
            return
        cfg = self._scale_config()
        if not cfg["port"]:
            self._append_scale("no COM port selected")
            return
        reader = ScaleReader(cfg)

        def on_frame(frame: bytes, reading):
            parsed = f"  ->  {reading.weight_kg:.3f} kg{'' if reading.stable else ' (unstable)'}" if reading else "  ->  (no weight parsed)"
            self.bus.dialog_result.emit("scale_line", f"{time.strftime('%H:%M:%S')}  {printable(frame):<40} hex: {hexdump(frame, 16)}{parsed}", False)

        reader.subscribe_frames(on_frame)
        reader.start()
        self._monitor = reader
        self.monitor_btn.setText(tr("stop_monitor"))
        self._append_scale(f"--- monitor {cfg['port']} @ {cfg['baudrate']} {cfg['bytesize']}{cfg['parity']}{cfg['stopbits']:g} ---")
        QTimer.singleShot(1500, self._monitor_status)

    def _monitor_status(self):
        if self._monitor is None:
            return
        h = self._monitor.health()
        if h["status"] == "error":
            self._append_scale(f"port error: {h['error']}")
        elif h["bytes"] == 0:
            self._append_scale("port open, waiting for data ... (nothing received yet)")
            QTimer.singleShot(3000, self._monitor_status)
        elif h.get("unstable"):
            self._append_scale(f"note: {h['unstable']} of {h['readings']} readings are flagged unstable (motion)")

    def _stop_monitor(self):
        if self._monitor is not None:
            try:
                self._monitor.stop()
            except Exception:
                pass
            self._monitor = None
            self.monitor_btn.setText(tr("start_monitor"))

    def _diagnose(self):
        if self._diag_running:
            return
        self._stop_monitor()
        cfg = self._scale_config()
        self._suggestion = None
        self.apply_btn.setEnabled(False)
        self._diag_running = True
        self.diag_btn.setEnabled(False)
        self._append_scale("=== " + tr("diag_running") + " ===")

        def work():
            try:
                rep = diagnose_scale(cfg, progress=lambda line: self.bus.dialog_result.emit("scale_line", line, False))
                self._suggestion = rep.get("suggestion")
                self.bus.dialog_result.emit("scale_done", rep["summary"], not rep["ok"])
            except Exception as exc:
                self.bus.dialog_result.emit("scale_done", str(exc), True)

        self._diag_thread = threading.Thread(target=work, daemon=True)
        self._diag_thread.start()

    def _apply_suggestion(self):
        sug = self._suggestion or {}
        if "baudrate" in sug:
            self.baud.setCurrentText(str(sug["baudrate"]))
        if "bytesize" in sug:
            self.bits.setCurrentText(str(sug["bytesize"]))
        if "parity" in sug:
            self.parity.setCurrentIndex("NEOMS".index(sug["parity"]))
        if "stopbits" in sug:
            self.stop.setCurrentText(str(sug["stopbits"]))
        if "poll_command" in sug:
            self.poll.setText(sug["poll_command"])
        self.source.setCurrentIndex([SOURCE_AUTO, SOURCE_HID, SOURCE_SERIAL, SOURCE_SIMULATE].index(SOURCE_SERIAL))
        self._append_scale("applied: " + json.dumps(sug))

    def _copy_report(self):
        QGuiApplication.clipboard().setText(self.scale_result.toPlainText())
        self._append_scale(tr("report_copied"))

    # -- printer
    def _printer_tab(self):
        w = QWidget()
        f = QFormLayout(w)
        p = self.settings.section("printer")
        self.p_enabled = QCheckBox(tr("enable_printer"))
        self.p_enabled.setChecked(bool(p["enabled"]))
        self.iface = QComboBox()
        keys = ["windows", "serial", "file", "none"]
        for key in keys:
            self.iface.addItem(tr(f"iface_{key}"), key)
        self.iface.setCurrentIndex(keys.index(p["interface"]) if p["interface"] in keys else 0)
        self.pname = QComboBox()
        self.pname.setEditable(True)
        for name in list_windows_printers():
            self.pname.addItem(name)
        self.pname.setCurrentText(p.get("printer_name") or default_windows_printer())
        self.pport = QComboBox()
        self.pport.setEditable(True)
        for dev, _d in list_serial_ports():
            self.pport.addItem(dev)
        self.pport.setCurrentText(p.get("port", ""))
        self.pbaud = QComboBox()
        for b in BAUDRATES:
            self.pbaud.addItem(str(b), b)
        self.pbaud.setCurrentText(str(p["baudrate"]))
        self.paper = QComboBox()
        self.paper.addItem("58 mm", 58)
        self.paper.addItem("80 mm", 80)
        self.paper.setCurrentIndex(1 if int(p["paper_width"]) == 80 else 0)
        self.pmode = QComboBox()
        self.pmode.addItem(tr("mode_image"), "image")
        self.pmode.addItem(tr("mode_text"), "text")
        self.pmode.setCurrentIndex(0 if p["mode"] == "image" else 1)
        self.header = QPlainTextEdit(p.get("header", ""))
        self.header.setMaximumHeight(70)
        self.footer = QPlainTextEdit(p.get("footer", ""))
        self.footer.setMaximumHeight(70)
        self.cut = QCheckBox(tr("cut_paper"))
        self.cut.setChecked(bool(p.get("cut", True)))
        test = QPushButton(tr("print_test"))
        test.clicked.connect(self._print_test)
        f.addRow("", self.p_enabled)
        f.addRow(tr("printer_interface"), self.iface)
        f.addRow(tr("printer_name"), self.pname)
        f.addRow(tr("com_port"), self.pport)
        f.addRow(tr("baudrate"), self.pbaud)
        f.addRow(tr("paper_width"), self.paper)
        f.addRow(tr("print_mode"), self.pmode)
        f.addRow(tr("header_text"), self.header)
        f.addRow(tr("footer_text"), self.footer)
        f.addRow("", self.cut)
        f.addRow("", test)
        return w

    def _printer_config(self) -> dict:
        cfg = self.settings.section("printer")
        cfg.update({"enabled": self.p_enabled.isChecked(), "interface": self.iface.currentData(),
                    "printer_name": self.pname.currentText().strip(), "port": self.pport.currentText().strip(),
                    "baudrate": int(self.pbaud.currentText()), "paper_width": self.paper.currentData(),
                    "mode": self.pmode.currentData(), "header": self.header.toPlainText(),
                    "footer": self.footer.toPlainText(), "cut": self.cut.isChecked()})
        return cfg

    def _print_test(self):
        cfg = self._printer_config()
        if cfg["interface"] == "file":
            cfg["file_path"] = data_path("test_receipt.bin")
        general = self.settings.section("general")
        general.update({"store_name": self.store.text(), "currency": self.currency.text(), "language": self.lang.currentData()})

        def work():
            try:
                msg = self.printer_factory(cfg, general).print_test()
                self.bus.dialog_result.emit("print", tr("print_test_ok") + f" ({msg})", False)
            except Exception as exc:
                self.bus.dialog_result.emit("print", tr("print_failed", error=str(exc)), True)

        threading.Thread(target=work, daemon=True).start()

    def _support_tab(self):
        """Remote support, spelled out: the owner sees exactly what leaves the till."""
        w = QWidget()
        f = QFormLayout(w)
        sup = self.settings.section("support")
        self.sup_enabled = QCheckBox(tr("support_enabled"))
        self.sup_enabled.setChecked(bool(sup.get("enabled")))
        # left blank, these fall back to what the build was made with, so the
        # placeholder has to show the value that will actually be used
        self.sup_url = QLineEdit(sup.get("url", ""))
        self.sup_url.setPlaceholderText(support_config.DEFAULT_URL or "https://example.com/pos")
        self.sup_token = QLineEdit(sup.get("token", ""))
        self.sup_token.setEchoMode(QLineEdit.EchoMode.Password)
        self.sup_token.setPlaceholderText(tr("support_builtin_token") if support_config.have_defaults() else "")
        self.sup_interval = QSpinBox()
        self.sup_interval.setRange(1, 1440)
        self.sup_interval.setValue(int(sup.get("interval_min", 10)))
        self.sup_frames = QCheckBox(tr("support_frames"))
        self.sup_frames.setChecked(bool(sup.get("send_frames", True)))
        self.sup_log = QCheckBox(tr("support_log"))
        self.sup_log.setChecked(bool(sup.get("send_log", True)))
        self.sup_actions = QCheckBox(tr("support_actions"))
        self.sup_actions.setChecked(bool(sup.get("allow_actions", True)))
        self.sup_test_btn = QPushButton(tr("support_test"))
        self.sup_test_btn.clicked.connect(self._test_support)
        note = QLabel(tr("support_builtin", url=support_config.DEFAULT_URL) + "\n\n" + tr("support_note")
                      if support_config.have_defaults() else tr("support_note"))
        note.setWordWrap(True)
        note.setStyleSheet("color:#9aa3b5")
        actions = QLabel(", ".join(sorted(SUPPORT_ACTIONS)))
        actions.setWordWrap(True)
        actions.setStyleSheet("color:#9aa3b5")
        f.addRow(self.sup_enabled)
        f.addRow(tr("support_url"), self.sup_url)
        f.addRow(tr("support_token"), self.sup_token)
        f.addRow(tr("support_interval"), self.sup_interval)
        f.addRow(self.sup_frames)
        f.addRow(self.sup_log)
        f.addRow(self.sup_actions)
        f.addRow(actions)
        f.addRow(self.sup_test_btn)
        f.addRow(note)
        return w

    def _test_support(self):
        from remote_support import RemoteSupport as _RS
        self.sup_test_btn.setEnabled(False)
        # test what is typed in the boxes, not what was last saved
        typed = {"support.enabled": True, "support.url": self.sup_url.text().strip(),
                 "support.token": self.sup_token.text().strip(),
                 "support.ca_file": self.settings.get("support.ca_file", "")}
        probe = type("TypedSettings", (), {"get": lambda _s, key, default=None: typed.get(key, default)})()
        channel = _RS(probe, lambda: {"test": True, "at": dt.datetime.now().isoformat(timespec="seconds")},
                      lambda name, args: {})

        def work():
            try:
                channel.test()
                self.bus.dialog_result.emit("support", tr("support_ok"), False)
            except Exception as exc:
                self.bus.dialog_result.emit("support", str(exc), True)

        threading.Thread(target=work, daemon=True).start()

    # -- save
    def save(self):
        s = self.settings
        s.update_section("support", {"enabled": self.sup_enabled.isChecked(),
                                     "url": self.sup_url.text().strip(),
                                     "token": self.sup_token.text().strip(),
                                     "interval_min": int(self.sup_interval.value()),
                                     "send_frames": self.sup_frames.isChecked(),
                                     "send_log": self.sup_log.isChecked(),
                                     "allow_actions": self.sup_actions.isChecked()})
        s.update_section("general", {"language": self.lang.currentData(), "store_name": self.store.text().strip(),
                                     "currency": self.currency.text().strip(), "currency_decimals": self.cur_dec.value(),
                                     "auto_add": self.auto_add.isChecked()})
        s.update_section("pos", {"multi_item_mode": self.multi_mode.currentData(),
                                 "refuse_on_intrusion": self.refuse_hand.isChecked(),
                                 "reinforce_on_manual_pick": self.reinforce.isChecked(),
                                 "weighing_mode": self.weigh_mode.currentData(),
                                 "weight_tolerance_kg": float(self.tolerance.value()),
                                 "block_checkout_on_mismatch": self.block_mismatch.isChecked()})
        try:
            w, h = [int(x) for x in self.resolution.currentText().lower().replace("×", "x").split("x")]
        except Exception:
            w, h = 640, 480
        s.update_section("camera", {"backend": self.backend.currentData(), "device_index": self.dev_index.value(),
                                    "width": w, "height": h, "mirror": self.mirror.isChecked(),
                                    "rotate": self.rotate.currentData(), "video_file": self.video.text().strip()})
        s.update_section("ai", {"confidence_threshold": self.thr.value() / 100.0,
                                "smoothing_frames": self.smooth.value(), "min_train_samples": self.min_samples.value(),
                                "motion_gate": self.motion_gate.isChecked(),
                                "per_object_recognition": self.per_object.isChecked(),
                                "max_objects": self.max_objects.value()})
        s.update_section("scale", self._scale_config())
        s.update_section("printer", self._printer_config())
        s.save()
        self.accept()


# ------------------------------------------------------------ main window
class MainWindow(QMainWindow):
    def __init__(self, settings: SettingsManager, video_override: str = ""):
        super().__init__()
        self.settings = settings
        self.video_override = video_override
        self.bus = Bus()
        self.setWindowTitle(tr("app_title"))
        self.resize(1400, 860)
        self.lines: List[InvoiceLine] = []
        self.current_rec: Optional[Recognition] = None
        self.current_frame: Optional[np.ndarray] = None
        self.current_weight: Optional[WeightReading] = None
        self.camera_thread: Optional[CameraThread] = None
        self.camera_result = None
        self.camera_ok = False
        self._bg_stability = None
        self.support = None
        self.engine: Optional[RecognitionEngine] = None
        self.extractor: Optional[FeatureExtractor] = None
        self.scale = None
        self.db = ItemDatabase(settings.resolve("ai.db_file"))
        self.tray = TrayRegion.from_settings(settings.section("camera"))
        self.printer = self._make_printer(settings.section("printer"), settings.section("general"))
        self._auto_armed = True
        self._last_frame_id = -1
        self._scale_conflict = False
        self.scale_source = SOURCE_SIMULATE
        self._override_item_id: Optional[str] = None     # operator picked from the alternatives
        self._override_track = -1
        self._optimize_notified = False
        self._bg_frames: List[np.ndarray] = []
        self._bg_payload = None
        self._bg_thumb: Optional[bytes] = None
        self._bg_pending: List[np.ndarray] = []
        self._bg_wanted = 0
        self._build_ui()
        self._wire_signals()
        self._shortcuts()
        QTimer.singleShot(50, self.start_hardware)
        self.ui_timer = QTimer(self)
        self.ui_timer.timeout.connect(self._tick)
        self.ui_timer.start(33)
        self.heartbeat = QTimer(self)
        self.heartbeat.timeout.connect(session_heartbeat)
        self.heartbeat.start(30000)

    # ---------------------------------------------------------- UI build
    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        outer = QHBoxLayout(root)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(10)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        outer.addWidget(splitter)

        # ---- left: camera
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        head = QHBoxLayout()
        t = QLabel(tr("camera_panel"))
        t.setObjectName("title")
        head.addWidget(t)
        head.addStretch(1)
        self.objects_pill = QLabel("")
        self.objects_pill.setStyleSheet(PILL % PILL_NEUTRAL)
        head.addWidget(self.objects_pill)
        self.cam_pill = QLabel(tr("no_camera"))
        self.cam_pill.setStyleSheet(PILL % PILL_NEUTRAL)
        head.addWidget(self.cam_pill)
        self.cam_retry_btn = QPushButton(tr("camera_retry"))
        self.cam_retry_btn.setVisible(False)
        self.cam_retry_btn.clicked.connect(self.retry_camera)
        head.addWidget(self.cam_retry_btn)
        ll.addLayout(head)
        self.video = VideoWidget()
        self.video.tray = self.tray
        self.video.has_items = len(self.db) > 0
        ll.addWidget(self.video, 1)
        self.detected_lbl = QLabel(tr("place_item"))
        self.detected_lbl.setObjectName("detected")
        self.detected_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.detected_lbl.setWordWrap(True)
        self.detected_lbl.setMinimumWidth(60)
        self.conf_bar = QProgressBar()
        self.conf_bar.setRange(0, 100)
        self.conf_bar.setFormat(tr("confidence") + " %p%")
        ll.addWidget(self.detected_lbl)
        ll.addWidget(self.conf_bar)
        # when two enrolled items look alike the app offers the alternatives
        # instead of silently picking one of them
        self.cand_box = QWidget()
        cl = QHBoxLayout(self.cand_box)
        cl.setContentsMargins(0, 0, 0, 0)
        self.cand_lbl = QLabel(tr("did_you_mean"))
        self.cand_lbl.setStyleSheet("color:#9aa3b5")
        cl.addWidget(self.cand_lbl)
        self.cand_buttons = []
        for _ in range(3):
            b = QPushButton("")
            b.setMinimumHeight(40)
            b.setVisible(False)
            b.clicked.connect(lambda _c=False, btn=b: self._choose_candidate(btn))
            self.cand_buttons.append(b)
            cl.addWidget(b, 1)
        self.cand_box.setVisible(False)
        ll.addWidget(self.cand_box)
        brow = QHBoxLayout()
        self.train_btn = QPushButton(tr("train_new_item"))
        self.angle_btn = QPushButton(tr("add_angle_sample"))
        self.manage_btn = QPushButton(tr("manage_items"))
        for b in (self.train_btn, self.angle_btn, self.manage_btn):
            b.setMinimumHeight(48)
            b.setMinimumWidth(60)
            brow.addWidget(b)
        ll.addLayout(brow)
        brow2 = QHBoxLayout()
        self.tray_btn = QPushButton(tr("tray_calibrate"))
        self.empty_btn = QPushButton(tr("capture_empty_tray"))
        self.diag_btn = QPushButton(tr("save_diagnostics"))
        for b in (self.tray_btn, self.empty_btn, self.diag_btn):
            b.setMinimumHeight(44)
            b.setMinimumWidth(60)
            brow2.addWidget(b)
        ll.addLayout(brow2)
        splitter.addWidget(left)

        # ---- right: invoice
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        head = QHBoxLayout()
        t = QLabel(tr("invoice_panel"))
        t.setObjectName("title")
        head.addWidget(t)
        head.addStretch(1)
        self.inv_no_lbl = QLabel("")
        self.inv_no_lbl.setStyleSheet("color:#9aa3b5")
        head.addWidget(self.inv_no_lbl)
        rl.addLayout(head)

        wbox = QFrame()
        wbox.setStyleSheet("QFrame{background:#1b1f27;border:1px solid #2c3340;border-radius:10px}")
        wl = QGridLayout(wbox)
        self.weight_lbl = QLabel("0.000")
        self.weight_lbl.setObjectName("weight")
        self.weight_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        unit = QLabel(tr("kg"))
        unit.setStyleSheet("font-size:20px;color:#9aa3b5")
        self.scale_pill = QLabel("")
        self.scale_pill.setStyleSheet(PILL % PILL_NEUTRAL)
        self.scale_pill.setWordWrap(True)
        wl.addWidget(self.weight_lbl, 0, 0, 1, 2)
        wl.addWidget(unit, 0, 2)
        wl.addWidget(self.scale_pill, 1, 0, 1, 3, Qt.AlignmentFlag.AlignCenter)
        self.sim_box = QWidget()
        sl = QVBoxLayout(self.sim_box)
        sl.setContentsMargins(0, 0, 0, 0)
        srow = QHBoxLayout()
        srow.addWidget(QLabel(tr("simulated_weight")))
        self.sim_slider = QSlider(Qt.Orientation.Horizontal)
        self.sim_slider.setRange(0, 4000)          # 0 .. 20.000 kg in 5 g steps
        self.sim_spin = QDoubleSpinBox()
        self.sim_spin.setRange(0, 20)
        self.sim_spin.setDecimals(3)
        self.sim_spin.setSingleStep(0.005)
        srow.addWidget(self.sim_slider, 1)
        srow.addWidget(self.sim_spin)
        sl.addLayout(srow)
        # one click out of "a port is configured but simulation is on", which is
        # the state that makes the app look like the scale is dead
        self.use_real_btn = QPushButton(tr("use_real_scale", port=""))
        self.use_real_btn.setObjectName("primary")
        self.use_real_btn.setVisible(False)
        self.use_real_btn.clicked.connect(self._switch_to_real_scale)
        sl.addWidget(self.use_real_btn)
        # when no port is configured at all there is nothing to switch to, so
        # offer the search itself rather than sending the operator hunting
        # through the settings tabs for it
        self.find_scale_btn = QPushButton(tr("find_scale"))
        self.find_scale_btn.setVisible(False)
        self.find_scale_btn.clicked.connect(self.find_scale_now)
        sl.addWidget(self.find_scale_btn)
        wl.addWidget(self.sim_box, 2, 0, 1, 3)
        # tray vs invoice: the line that makes a missed or extra item obvious
        self.recon_lbl = QLabel("")
        self.recon_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.recon_lbl.setStyleSheet("color:#9aa3b5;font-size:14px")
        wl.addWidget(self.recon_lbl, 3, 0, 1, 3)
        rl.addWidget(wbox)

        self.banner_lbl = QLabel("")
        self.banner_lbl.setObjectName("banner")
        self.banner_lbl.setWordWrap(True)
        self.banner_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.banner_lbl.setVisible(False)
        rl.addWidget(self.banner_lbl)

        addrow = QHBoxLayout()
        self.add_btn = QPushButton(tr("add_to_invoice"))
        self.add_btn.setObjectName("primary")
        self.add_btn.setMinimumHeight(56)
        self.add_btn.setMinimumWidth(80)
        addrow.addWidget(self.add_btn, 2)
        self.manual_combo = QComboBox()
        self.manual_combo.setMinimumHeight(44)
        self.manual_combo.setIconSize(QSize(28, 28))
        self.manual_combo.setMinimumWidth(80)
        self.manual_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.manual_btn = QPushButton(tr("add_manual"))
        self.manual_btn.setMinimumHeight(44)
        self.manual_btn.setMinimumWidth(60)
        addrow.addWidget(self.manual_combo, 2)
        addrow.addWidget(self.manual_btn, 1)
        rl.addLayout(addrow)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels([tr("col_item"), tr("col_weight"), tr("col_price"), tr("col_total"), ""])
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for c in (1, 2, 3):
            hh.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(4, QHeaderView.ResizeMode.Fixed)
        hh.setMinimumSectionSize(40)
        self.table.setColumnWidth(4, 52)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setDefaultSectionSize(44)
        rl.addWidget(self.table, 1)

        tot = QHBoxLayout()
        tl = QLabel(tr("grand_total"))
        tl.setStyleSheet("font-size:20px;color:#9aa3b5")
        self.total_lbl = QLabel("0")
        self.total_lbl.setObjectName("total")
        self.total_lbl.setMinimumWidth(60)
        self.total_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        tot.addWidget(tl)
        tot.addStretch(1)
        tot.addWidget(self.total_lbl)
        rl.addLayout(tot)

        actions = QHBoxLayout()
        self.del_btn = QPushButton(tr("delete_row"))
        self.clear_btn = QPushButton(tr("clear_invoice"))
        self.clear_btn.setObjectName("danger")
        self.checkout_btn = QPushButton(tr("checkout").replace("&", "&&"))
        self.checkout_btn.setObjectName("success")
        self.settings_btn = QPushButton(tr("settings"))
        for b in (self.del_btn, self.clear_btn, self.settings_btn):
            b.setMinimumHeight(48)
            b.setMinimumWidth(60)
        self.checkout_btn.setMinimumHeight(56)
        self.checkout_btn.setMinimumWidth(100)
        actions.addWidget(self.del_btn)
        actions.addWidget(self.clear_btn)
        actions.addWidget(self.checkout_btn, 2)
        actions.addWidget(self.settings_btn)
        rl.addLayout(actions)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 4)
        splitter.setSizes([780, 620])
        splitter.setChildrenCollapsible(False)
        left.setMinimumWidth(340)
        right.setMinimumWidth(360)

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status_ai = QLabel("")
        self.status_items = QLabel("")
        self.status_printer = QLabel("")
        for w in (self.status_ai, self.status_items, self.status_printer):
            self.status.addPermanentWidget(w)
        self._refresh_items_ui()
        self._update_total()
        self.inv_no_lbl.setText(f"{tr('invoice_no')} #{self._peek_invoice_number():06d}")

    def _wire_signals(self):
        self.bus.recognition.connect(self.on_recognition)
        self.bus.weight.connect(self.on_weight)
        self.bus.model_ready.connect(self.on_model_ready)
        self.bus.camera_ready.connect(self.on_camera_ready)
        self.bus.camera_found.connect(self.on_camera_found)
        self.bus.scale_found.connect(self.on_scale_found)
        self.bus.run_on_ui.connect(lambda fn: fn())
        self.bus.toast.connect(self.on_toast)
        self.bus.print_done.connect(self.on_print_done)
        self.bus.embedding_ready.connect(self.on_embedding)
        self.video.region_clicked.connect(self.on_region_clicked)
        self.train_btn.clicked.connect(self.train_new)
        self.angle_btn.clicked.connect(self.add_angle)
        self.manage_btn.clicked.connect(self.manage_items)
        self.tray_btn.clicked.connect(self.calibrate_tray)
        self.empty_btn.clicked.connect(self.capture_empty_tray)
        self.diag_btn.clicked.connect(self.save_diagnostics)
        self.add_btn.clicked.connect(self.add_detected)
        self.manual_btn.clicked.connect(self.add_manual)
        self.del_btn.clicked.connect(self.delete_selected)
        self.clear_btn.clicked.connect(self.clear_invoice)
        self.checkout_btn.clicked.connect(self.checkout)
        self.settings_btn.clicked.connect(self.open_settings)
        self.sim_slider.valueChanged.connect(lambda v: self.sim_spin.setValue(v / 200.0))
        self.sim_spin.valueChanged.connect(self._on_sim_spin)

    def _shortcuts(self):
        QShortcut(QKeySequence(Qt.Key.Key_Return), self, activated=self.add_detected)
        QShortcut(QKeySequence(Qt.Key.Key_Space), self, activated=self.add_detected)
        QShortcut(QKeySequence(Qt.Key.Key_Delete), self, activated=self.delete_selected)
        QShortcut(QKeySequence("F2"), self, activated=self.train_new)
        QShortcut(QKeySequence("F3"), self, activated=self.add_angle)
        QShortcut(QKeySequence("F4"), self, activated=self.calibrate_tray)
        QShortcut(QKeySequence("F5"), self, activated=self.checkout)
        QShortcut(QKeySequence("F6"), self, activated=self.capture_empty_tray)
        QShortcut(QKeySequence("F10"), self, activated=self.open_settings)

    # ------------------------------------------------------- hardware
    def _make_printer(self, cfg: dict, general: dict) -> ReceiptPrinter:
        return ReceiptPrinter(cfg, general, receipt_labels())

    def start_hardware(self):
        self.start_camera()
        self.start_scale()
        self.status_printer.setText(f"{tr('tab_printer')}: {self.settings.get('printer.interface')}")
        threading.Thread(target=self._load_model, name="ModelLoader", daemon=True).start()
        self.status.showMessage(tr("loading_model"))
        self.start_support()

    # -------------------------------------------------------- remote support
    def start_support(self):
        """Start the outbound support channel, if the owner configured one."""
        self.stop_support()
        self.support = RemoteSupport(self.settings, self.support_snapshot, self.support_action)
        # the camera and the model are still coming up; a snapshot now would
        # describe a till that has not finished starting. Ninety seconds is
        # past a slow model load, and camera/model readiness cuts it short.
        if self.support.start(first_delay=90.0):
            self.status.showMessage(tr("support_on", url=self.support.url), 8000)

    def stop_support(self):
        support = getattr(self, "support", None)
        if support:
            support.stop()
        self.support = None

    def support_snapshot(self) -> dict:
        """Everything a remote helper needs to see, in one JSON-safe dict.

        Called from the support thread, so it only reads: the one thing it asks
        the UI thread for is the window picture, and only on request.
        """
        sc = self.settings.section("scale")
        cam_res = self.camera_result
        reading = self.current_weight
        rec = self.current_rec
        snap = {
            "at": dt.datetime.now().isoformat(timespec="seconds"),
            "build": build_id(),
            "camera": {
                "ok": self.camera_ok,
                "backend": getattr(cam_res, "backend", None),
                "description": getattr(cam_res, "description", ""),
                "notes": list(getattr(cam_res, "notes", []) or []),
                "attempts": list(getattr(getattr(cam_res, "source", None), "attempts", []) or []),
                "settings": self.settings.section("camera"),
            },
            "scale": {
                "source": self.scale_source,
                "port": sc.get("port", ""),
                "conflict": bool(getattr(self, "_scale_conflict", False)),
                "health": self.scale.health() if self.scale else None,
                "weight_kg": None if reading is None else round(reading.weight_kg, 4),
                "stable": None if reading is None else bool(reading.stable),
                "pill": self.scale_pill.text(),
            },
            "screen": {
                "banner": self.banner_lbl.text(),
                "detected": self.detected_lbl.text(),
                "blocking_reason": self.blocking_reason(),
                "can_add": self.add_btn.isEnabled(),
            },
            "invoice": {
                "lines": len(self.lines),
                "invoiced_kg": round(self.invoiced_weight(), 4),
                "tray_kg": None if self.tray_weight() is None else round(self.tray_weight(), 4),
                "difference_kg": None if self.weight_difference() is None else round(self.weight_difference(), 4),
            },
            "catalogue": {"items": len(self.db), "has_empty_tray": bool(self.db.has_background_image),
                          "tray_stability": self._bg_stability or self._stability_of_loaded_model()},
            "engine": {
                "loaded": self.engine is not None,
                "paused": bool(getattr(self.engine, "paused", False)) if self.engine else None,
                "optimized": bool(getattr(self.extractor, "optimized", False)) if self.extractor else None,
                "segmenter": bool(getattr(self.engine, "has_segmenter", False)) if self.engine else False,
            },
        }
        if rec is not None:
            snap["recognition"] = {
                "mode": rec.mode, "empty": rec.empty, "intrusion": rec.intrusion,
                "unreliable": rec.unreliable, "objects": len(rec.detections),
                "seg_ms": round(rec.seg_ms, 1), "infer_ms": round(rec.infer_ms, 1),
                "items": [{"name": d.name, "score": round(d.score, 3), "accepted": d.accepted,
                           "suggested": d.suggested_name} for d in rec.detections[:5]],
            }
        if self.settings.get("support.send_frames", True):
            _fid, frame = self._camera_frame()
            snap["frame_jpeg_b64"] = encode_frame(frame)
        if self.settings.get("support.send_log", True):
            snap["log_tail"] = log_tail(data_path("aipos.log"), 200)
        return snap

    def _stability_of_loaded_model(self):
        """Stability of whatever reference is in use, even one loaded from disk."""
        model = getattr(getattr(self.engine, "segmenter", None), "m", None)   # TraySegmenter keeps it as .m
        try:
            return model.stability() if model is not None else None
        except Exception:
            return None

    def support_action(self, name: str, args: dict) -> dict:
        """Run one of the actions in remote_support.ACTIONS. Nothing else.

        Slow work (a self-test loads a second copy of the model, a camera sweep
        takes seconds per device) stays on the support thread; only the few
        steps that touch Qt are handed to the UI thread.
        """
        if name == "ping":
            return {"pong": True, "version": APP_VERSION}
        if name == "snapshot":
            return {"snapshot": self.support_snapshot()}
        if name == "logs":
            return {"log": log_tail(data_path("aipos.log"), int(args.get("lines", 1000)))}
        if name == "selftest":
            run_selftest(self.settings)
            return {"report": log_tail(data_path("selftest_report.txt"), 400)}
        if name == "find_camera":
            found = find_camera()
            if found["found"] and args.get("apply", True):
                self._on_ui(lambda: (self.settings.set("camera.device_index", int(found["index"])),
                                     self.settings.set("camera.backend", "opencv"),
                                     self.settings.save(), self.start_camera()))
            return found
        if name == "find_scale":
            found = find_scale()
            if found.get("found") and args.get("apply", True):
                self._on_ui(lambda: self.on_scale_found(found))
            return {"found": found.get("found"), "report": found.get("report", "")[-4000:],
                    "best": (found.get("best") or {}).get("label")}
        if name == "restart_camera":
            self._on_ui(self.start_camera)
            return {"restarted": True}
        if name == "restart_scale":
            self._on_ui(self.start_scale)
            return {"restarted": True, "source": self.scale_source}
        if name == "diagnostics":
            path = diagnostics.default_path(app_dir())
            _fid, frame = self._camera_frame()
            diagnostics.collect(path, frame, self.tray, self.db, self.engine, self.settings,
                                self.current_rec, note="collected remotely")
            with open(path, "rb") as fh:
                blob = fh.read()
            return {"filename": os.path.basename(path), "bytes": len(blob),
                    "zip_b64": base64.b64encode(blob).decode("ascii")}
        if name == "get_setting":
            return {"key": args.get("key"), "value": self.settings.get(str(args.get("key", "")))}
        if name == "set_setting":
            key, value = str(args.get("key", "")), args.get("value")
            # only keys that already exist: no inventing new ones remotely
            if self.settings.get(key, _MISSING) is _MISSING:
                raise ValueError(f"no such setting: {key!r}")
            self._on_ui(lambda: (self.settings.set(key, value), self.settings.save()))
            live = self._on_ui(lambda: self._apply_setting_live(key))
            return {"key": key, "value": self.settings.get(key), "applied_live": live,
                    "note": "in effect now" if live else "saved; takes effect after a restart"}
        if name == "screenshot":
            return {"png_b64": self._on_ui(self._grab_window_png)}
        if name == "update":
            return self._update_from_git(restart=bool(args.get("restart", False)))
        raise ValueError(f"unsupported action: {name!r}")

    def _update_from_git(self, restart: bool = False) -> dict:
        """Fast-forward this checkout, refuse anything that will not compile.

        A till that does not come back up cannot be fixed from here, so the
        new code is compiled before it is allowed to run and the checkout is
        reset if it does not. That covers a bad merge or a half-pushed change;
        it cannot cover code that compiles and then misbehaves, and nothing
        short of a second machine could.
        """
        folder = os.path.dirname(os.path.abspath(__file__))
        if not os.path.isdir(os.path.join(folder, ".git")):
            raise RuntimeError(f"{folder} is not a git checkout, so there is nothing to pull")

        def git(*argv, timeout=120):
            done = subprocess.run(("git",) + argv, cwd=folder, capture_output=True, text=True,
                                  timeout=timeout, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            return done.returncode, (done.stdout + done.stderr).strip()

        _rc, before = git("rev-parse", "HEAD")
        rc, output = git("pull", "--ff-only")
        if rc != 0 and "ORIG_HEAD" in output and ("reference broken" in output or "cannot lock ref" in output):
            # a pointer git rewrites on every pull: worthless, and in the way
            stale = os.path.join(folder, ".git", "ORIG_HEAD")
            try:
                os.remove(stale)
                log.warning("removed a broken %s and retrying the pull", stale)
                rc, retry = git("pull", "--ff-only")
                output = output + "\n-- removed a broken ORIG_HEAD and retried --\n" + retry
            except OSError as exc:
                output = output + f"\n-- could not remove {stale}: {exc} --"
        _rc, after = git("rev-parse", "HEAD")
        result = {"before": before[:10], "after": after[:10], "pull": output[-1500:],
                  "changed": before != after}
        if rc != 0:
            # --ff-only refuses a diverged or dirty checkout rather than guessing
            result["ok"] = False
            result["error"] = "git pull failed; the checkout was left alone"
            return result

        bad = self._files_that_do_not_compile(folder)
        if bad:
            git("reset", "--hard", before)
            result.update(ok=False, rolled_back_to=before[:10], does_not_compile=bad,
                          error="the new code does not compile, so it was rolled back")
            log.error("update rolled back: %s", bad)
            return result

        result["ok"] = True
        result["restarting"] = bool(restart and before != after)
        if result["restarting"]:
            log.warning("restarting into %s on request", after[:10])
            QTimer.singleShot(1500, self._relaunch)
        return result

    @staticmethod
    def _files_that_do_not_compile(folder: str) -> List[str]:
        """Which .py files in ``folder`` will not even parse.

        compile() on the source rather than py_compile: writing byte-code needs
        somewhere to put it, and py_compile(cfile=os.devnull) fails outright on
        Windows - which would have marked every file broken and rolled back
        every update, the exact opposite of the intent.
        """
        bad = []
        for name in sorted(os.listdir(folder)):
            if not name.endswith(".py"):
                continue
            path = os.path.join(folder, name)
            try:
                with open(path, "rb") as fh:
                    compile(fh.read(), path, "exec")
            except Exception as exc:
                bad.append(f"{name}: {exc}")
        return bad

    def _relaunch(self):
        """Start a fresh copy of ourselves and close this one."""
        try:
            argv = [sys.executable] + ([] if is_frozen() else [os.path.abspath(__file__)]) + sys.argv[1:]
            subprocess.Popen(argv, cwd=os.path.dirname(os.path.abspath(__file__)),
                             creationflags=getattr(subprocess, "DETACHED_PROCESS", 0))
        except Exception:
            log.exception("could not relaunch after update")
            return
        self.close()

    def _apply_setting_live(self, key: str) -> bool:
        """Push one changed setting into the running app, where that is possible.

        Mirrors what saving the Settings dialog does. Returns False for keys
        that only take effect on a restart, so the caller can say so instead
        of implying the change is already live.
        """
        if key == "ai.per_object_recognition" and self.engine is not None:
            self.engine.set_per_object(bool(self.settings.get(key, True)))
            return True
        if key.startswith("pos.") or key.startswith("support.") or key == "scale.stable_only":
            # read fresh from settings every time they are used
            return True
        return False

    def _grab_window_png(self) -> str:
        buf = QBuffer()
        buf.open(QBuffer.OpenModeFlag.WriteOnly)
        self.grab().save(buf, "PNG")
        return base64.b64encode(bytes(buf.data())).decode("ascii")

    def _on_ui(self, fn, timeout: float = 90.0):
        """Run ``fn`` on the UI thread and wait for its result.

        Called from the UI thread it runs straight away: posting to the event
        loop and then blocking that same loop waiting for the answer is a
        deadlock, and it is an easy one to walk into from a test or a menu.
        """
        if QThread.currentThread() is self.thread():
            return fn()
        box: dict = {}
        done = threading.Event()

        def wrapper():
            try:
                box["value"] = fn()
            except Exception as exc:
                box["error"] = exc
            finally:
                done.set()

        # NOT QTimer.singleShot: a timer started from a worker thread is
        # silently ignored by Qt, so the wait below always timed out. A queued
        # signal is the supported way to hop threads.
        self.bus.run_on_ui.emit(wrapper)
        if not done.wait(timeout):
            raise TimeoutError("the application did not answer in time")
        if "error" in box:
            raise box["error"]
        return box.get("value")

    def start_camera(self):
        if self.camera_thread:
            self.camera_thread.stop()
            self.camera_thread = None
        cam = self.settings.section("camera")

        def work():
            res = open_camera(cam["backend"], int(cam["device_index"]), int(cam["width"]), int(cam["height"]),
                              cam.get("sdk_dll", "RTKCamSDK.dll"), self.video_override or cam.get("video_file", ""),
                              lock_exposure=bool(cam.get("lock_exposure", False)))
            self.bus.camera_ready.emit(res)

        threading.Thread(target=work, name="CameraOpener", daemon=True).start()

    def on_camera_ready(self, res):
        """Adopt whatever camera we got - including none at all.

        When there is no camera the placeholder picture used to be fed to the
        segmenter, which quite reasonably decided something was moving over the
        tray and refused to sell anything.  A missing camera now stops
        recognition outright and says so, so the till still works by hand.
        """
        self.camera_result = res
        self.camera_ok = res.backend != "none"
        cam = self.settings.section("camera")
        self.camera_thread = CameraThread(res.source, float(cam.get("fps_limit", 15)), bool(cam.get("mirror")), int(cam.get("rotate", 0)))
        self.camera_thread.start()
        self.cam_pill.setText(f"{tr('camera_backend')}: {res.backend.upper()}" if self.camera_ok else tr("no_camera"))
        self.cam_pill.setStyleSheet(PILL % (PILL_NEUTRAL if self.camera_ok else PILL_BAD))
        self.cam_pill.setToolTip(res.description + ("\n" + "\n".join(res.notes) if res.notes else ""))
        self.video.caption = res.description
        self.cam_retry_btn.setVisible(not self.camera_ok)
        if self.engine:
            self.engine.paused = not self.camera_ok
        if not self.camera_ok:
            self.current_rec = None
            self.video.rec = None
        self._update_blocking()
        for n in res.notes:
            log.info("camera note: %s", n)
        if cam.get("backend") in ("auto", "rtk") and res.backend == "opencv" and any("32-bit" in n for n in res.notes):
            self.status.showMessage(tr("rtk_fallback_status"), 10000)
        self._support_push()

    def _support_push(self):
        """Report now rather than at the next tick: something worth seeing changed."""
        if self.support:
            self.support.send_now()

    def retry_camera(self):
        """Look for a camera again, trying every device index this time."""
        self.cam_retry_btn.setEnabled(False)
        self.cam_pill.setText(tr("find_camera_running"))
        self.status.showMessage(tr("find_camera_running"))

        def work():
            res = find_camera()
            if res.get("found"):
                self.settings.set("camera.device_index", int(res["index"]))
                self.settings.set("camera.backend", "opencv")
                self.settings.save()
            self.bus.camera_found.emit(res)

        threading.Thread(target=work, name="CameraFinder", daemon=True).start()

    def on_camera_found(self, res):
        self.cam_retry_btn.setEnabled(True)
        log.info("find_camera:\n%s", res.get("report", ""))
        if res.get("found"):
            self.status.showMessage(tr("find_camera_found", description=res["description"]), 8000)
            self.start_camera()
        else:
            self.cam_pill.setText(tr("no_camera"))
            msg_error(self, tr("find_camera_none") + "\n\n" + res.get("report", ""))

    def find_scale_now(self):
        """Search every port and USB device, then just use what answered."""
        self.find_scale_btn.setEnabled(False)
        self.scale_pill.setText(tr("find_scale_running"))
        self.scale_pill.setStyleSheet(PILL % PILL_NEUTRAL)

        def work():
            try:
                res = find_scale()
            except Exception as exc:
                log.exception("find_scale failed")
                res = {"found": False, "report": str(exc)}
            self.bus.scale_found.emit(res)

        threading.Thread(target=work, name="ScaleFinder", daemon=True).start()

    def on_scale_found(self, res):
        self.find_scale_btn.setEnabled(True)
        log.info("find_scale:\n%s", res.get("report", ""))
        if res.get("found"):
            for key, value in ((res.get("best") or {}).get("config") or {}).items():
                self.settings.set(f"scale.{key}", value)
            self.settings.set("scale.enabled", True)
            self.settings.save()
            self.start_scale()
            self.status.showMessage(tr("find_scale_found", label=res["best"]["label"]), 8000)
        else:
            self.start_scale()
            msg_error(self, tr("find_scale_none") + "\n\n" + str(res.get("report", ""))[-1500:])

    def _camera_frame(self):
        th = self.camera_thread
        return th.get_latest() if th else (0, None)

    def _load_model(self):
        try:
            fx = FeatureExtractor(model_file_path(self.settings), int(self.settings.get("ai.num_threads", 1)),
                                  fast=bool(self.settings.get("ai.fast_inference", True)))
            self.bus.model_ready.emit(fx, None)
        except Exception as exc:
            log.exception("model load failed")
            self.bus.model_ready.emit(None, str(exc))

    def on_model_ready(self, fx, error):
        if fx is None:
            self.status.showMessage(tr("model_failed") + f": {error}")
            msg_error(self, tr("model_failed") + f"\n{error}")
            return
        self.extractor = fx
        ai = self.settings.section("ai")
        self.engine = RecognitionEngine(fx, self.db, self._camera_frame, self.tray, float(ai["confidence_threshold"]),
                                        float(ai.get("empty_threshold", 0.88)), float(ai["infer_fps"]),
                                        int(ai["smoothing_frames"]), int(ai.get("top_k", 3)),
                                        on_result=lambda rec, frame: self.bus.recognition.emit(rec, frame),
                                        motion_gate=bool(ai.get("motion_gate", True)),
                                        motion_threshold=float(ai.get("motion_threshold", 6.0)),
                                        per_object=bool(ai.get("per_object_recognition", True)),
                                        max_objects=int(ai.get("max_objects", 5)),
                                        min_area_frac=float(ai.get("min_area_frac", 0.008)),
                                        background_refresh=bool(ai.get("background_refresh", True)))
        self.engine.paused = not self.camera_ok
        self.engine.start()
        self.video.hint = "" if self.engine.has_segmenter else tr("capture_empty_first")
        if self.db.backbone_mismatch:
            msg_error(self, tr("retrain_needed"))
        self.status.showMessage(tr("model_ready"), 5000)
        self._support_push()

    def stop_scale(self):
        if self.scale:
            try:
                self.scale.stop()
            except Exception:
                pass
            self.scale = None

    def start_scale(self):
        """Build the scale object the settings describe, and never do it silently.

        The single most confusing state is "a COM port is configured but
        simulation is still on": the settings dialog tests the port and reports
        success, while the main screen shows a simulated zero.  It is now amber,
        spelled out, and one click away from being fixed.
        """
        self.stop_scale()
        sc = self.settings.section("scale")
        port = (sc.get("port") or "").strip()
        source, _reason = resolve_source(sc)
        self.scale_source = source
        # the one state that looks exactly like a dead scale: a port is set
        # but simulation was left on, so Settings tests the port and the main
        # screen shows a fake zero
        self._scale_conflict = (source == SOURCE_SIMULATE and bool(port) and bool(sc.get("enabled", True))
                                and str(sc.get("source") or "simulate") in ("simulate", ""))
        self.scale = make_scale(sc)
        if source == SOURCE_SIMULATE:
            self.sim_box.setVisible(True)
            self.use_real_btn.setVisible(self._scale_conflict)
            self.use_real_btn.setText(tr("use_real_scale", port=port))
            self.find_scale_btn.setVisible(not port)
            note = f" ({port}: {tr('scale_sim_overrides_port')})" if self._scale_conflict else ""
            self.scale_pill.setText(f"{tr('scale_status')}: {tr('scale_simulated')}{note}")
            self.scale_pill.setStyleSheet(PILL % (PILL_WARN if self._scale_conflict else PILL_NEUTRAL))
            if self.engine:
                self.engine.set_weight_hint(None)
        else:
            self.sim_box.setVisible(False)
            self.use_real_btn.setVisible(False)
            self.scale_pill.setText(f"{tr('scale_status')}: {tr('scale_connecting')} "
                                    + tr("scale_via", source=tr(f"source_{source}")))
        self.scale_pill.setToolTip(scale_mode_reason(self.settings))
        self.scale.subscribe(lambda r: self.bus.weight.emit(r))
        self.scale.start()
        if self.scale.is_simulated:
            self.scale.set_weight(self.sim_spin.value())

    def _switch_to_real_scale(self):
        self.settings.set("scale.simulate", False)
        self.settings.set("scale.source", SOURCE_AUTO)
        self.settings.save()
        self.start_scale()

    def _on_sim_spin(self, value: float):
        self.sim_slider.blockSignals(True)
        self.sim_slider.setValue(int(round(value * 200)))
        self.sim_slider.blockSignals(False)
        if self.scale is not None and self.scale.is_simulated:
            self.scale.set_weight(value)
            if self.engine:
                self.engine.set_weight_hint(value < float(self.settings.get("scale.min_weight_kg", 0.005)))

    # ----------------------------------------------------------- events
    def _tick(self):
        fid, frame = self._camera_frame()
        if frame is not None and fid != self._last_frame_id:
            self._last_frame_id = fid
            self.video.set_frame(frame)
        if self.engine is not None:
            rec = self.current_rec
            n = len(rec.confirmed) if rec else 0
            seg = f" | seg {self.engine.seg_ms:.0f} ms" if self.engine.has_segmenter else ""
            self.status_ai.setText(f"AI {self.engine.stats_ms:.0f} ms{seg} | {n} obj")
            fx = self.extractor
            if fx is not None and not self._optimize_notified and not getattr(fx, "optimizing", False):
                self._optimize_notified = True
                if getattr(fx, "optimized", False):
                    self.status.showMessage(tr("optimized_ready"), 6000)
        if self.scale is not None and not self.scale.is_simulated:
            self._update_scale_status()

    def _update_scale_status(self):
        h = self.scale.health()
        hold = float(self.settings.get("scale.hold_last_s", 5.0))
        reading = self.scale.latest()
        fresh = reading is not None and reading.age <= hold
        if h["status"] == "error":
            state, colour = f"{tr('scale_disconnected')}: {h.get('error', '')}", PILL_BAD
        elif h["status"] == "connecting":
            state, colour = tr("scale_connecting"), PILL_NEUTRAL
        elif fresh:
            state = tr("scale_connected") + ("" if reading.stable else f" - {tr('scale_unstable')}")
            colour = PILL_OK if reading.stable else PILL_WARN
        elif h.get("bytes", 0) == 0:
            state, colour = tr("scale_no_data"), PILL_WARN
        elif h.get("readings", 0) > 0:
            # bytes arrived and parsed, they are just old: never blame the data
            state, colour = tr("scale_stale_data"), PILL_WARN
        else:
            state, colour = tr("scale_bad_data"), PILL_WARN
        self.scale_pill.setText(f"{tr('scale_status')}: {state}")
        self.scale_pill.setStyleSheet(PILL % colour)
        self.scale_pill.setToolTip(scale_mode_reason(self.settings))
        if reading is None or reading.age > hold:
            self.weight_lbl.setText("—")
            self.weight_lbl.setStyleSheet(WEIGHT_STALE)
            self.current_weight = None
            if self.engine:
                self.engine.set_weight_hint(None)
        elif reading.age > 2.0:
            self.weight_lbl.setStyleSheet(WEIGHT_STALE)

    def on_weight(self, reading: WeightReading):
        self.current_weight = reading
        self._update_reconciliation()
        self.weight_lbl.setText(fmt_weight(reading.weight_kg, int(self.settings.get("general.weight_decimals", 3))))
        self.weight_lbl.setStyleSheet(WEIGHT_FRESH if reading.stable else WEIGHT_UNSTABLE)
        min_w = float(self.settings.get("scale.min_weight_kg", 0.005))
        if reading.weight_kg < min_w:
            self._auto_armed = True
        if self.engine:
            self.engine.set_weight_hint(reading.weight_kg < min_w)
        self._maybe_auto_add()

    def on_recognition(self, rec: Recognition, frame):
        self.current_rec = rec
        self.current_frame = frame
        self.video.set_recognition(rec)
        if self.engine is not None:
            self.video.hint = "" if self.engine.has_segmenter else tr("capture_empty_first")
        n = len(rec.confirmed)
        self.objects_pill.setText(tr("objects_detected", count=n) if n != 1 else tr("one_object"))
        self.objects_pill.setStyleSheet(PILL % (PILL_OK if n == 1 else (PILL_WARN if n > 1 else PILL_NEUTRAL)))
        self._update_blocking()
        if self.video.training is not None:
            return
        active = self._active_detection()
        if active is None or active.track_id != self._override_track:
            self._override_item_id = None        # a different object: the choice no longer applies
        self._update_candidates(active)
        if self._override_item_id:
            item = self.db.get(self._override_item_id)
            if item is not None:
                self.detected_lbl.setText(item.name)
                self.detected_lbl.setStyleSheet("color:#ffd166")
                self._maybe_auto_add()
                return
        if rec.empty:
            self.detected_lbl.setText(tr("tray_empty"))
            self.detected_lbl.setStyleSheet("color:#9aa3b5")
            self.conf_bar.setValue(0)
            self._auto_armed = True
            self._override_item_id = None
        elif active is not None and active.accepted:
            self.detected_lbl.setText(active.name)
            self.detected_lbl.setStyleSheet("color:#7CFC9A")
            self.conf_bar.setValue(int(active.score * 100))
        elif active is not None and active.suggested:
            self.detected_lbl.setText(tr("is_it", name=active.suggested_name))
            self.detected_lbl.setStyleSheet("color:#ffcc4d")
            self.conf_bar.setValue(int(active.result.score * 100))
        else:
            self.detected_lbl.setText(tr("unknown_item") if len(self.db) else tr("no_items_trained"))
            self.detected_lbl.setStyleSheet("color:#ffb4b4" if len(self.db) else "color:#9aa3b5")
            self.conf_bar.setValue(int(active.result.score * 100) if active is not None else 0)
        self._maybe_auto_add()

    def on_region_clicked(self, track_id: int):
        self.video.active_track = track_id
        self.video.update()
        if self.current_rec is not None:
            self.on_recognition(self.current_rec, self.current_frame)

    def on_toast(self, text: str, is_error: bool):
        self.status.showMessage(text, 8000)
        if is_error:
            msg_error(self, text)

    # ------------------------------------------------------ POS decisions
    def _choose_candidate(self, button):
        """The operator picked one of the look-alike alternatives."""
        item_id = button.property("item_id")
        if not item_id:
            return
        active = self._active_detection()
        self._override_item_id = item_id
        self._override_track = active.track_id if active else -1
        item = self.db.get(item_id)
        if item is not None:
            self.detected_lbl.setText(item.name)
            self.detected_lbl.setStyleSheet("color:#ffd166")
            self.status.showMessage(tr("chosen", name=item.name), 4000)
        self._update_candidates(active)

    def _update_candidates(self, det: Optional[Detection]):
        """Offer the runners-up whenever the catalogue holds a look-alike."""
        limit = int(self.settings.get("ai.candidates", 3))
        margin = float(self.settings.get("ai.ambiguous_margin", 0.08))
        ranking = []
        if det is not None and det.confirmed:
            ranking = [(iid, name, score) for iid, name, score in det.result.ranking
                       if iid and not iid.startswith("__") and self.db.get(iid) is not None]
        unsure = det is not None and det.confirmed and det.suggested
        if not ranking or (len(ranking) < 2 and not unsure):
            self.cand_box.setVisible(False)
            return
        ambiguous = unsure or (len(ranking) > 1 and (ranking[0][2] - ranking[1][2]) < margin)
        for i, button in enumerate(self.cand_buttons):
            if i < min(limit, len(ranking)):
                iid, name, score = ranking[i]
                button.setText(f"{name}  {score * 100:.0f}%")
                button.setProperty("item_id", iid)
                chosen = (self._override_item_id or (det.item_id if det else None)) == iid
                button.setObjectName("primary" if chosen else "")
                button.setStyleSheet("")          # re-apply the stylesheet for the new object name
                button.setVisible(True)
            else:
                button.setVisible(False)
        self.cand_lbl.setText((tr("confirm_needed") if unsure else tr("ambiguous")) if ambiguous
                              else tr("did_you_mean"))
        self.cand_lbl.setStyleSheet("color:#ffcc4d" if ambiguous else "color:#9aa3b5")
        self.cand_box.setVisible(True)

    def _active_detection(self) -> Optional[Detection]:
        rec = self.current_rec
        if rec is None:
            return None
        if self.video.active_track >= 0:
            for d in rec.detections:
                if d.track_id == self.video.active_track:
                    return d
        return rec.primary

    def tray_looks_empty(self) -> bool:
        rec = self.current_rec
        return bool(rec and rec.empty)

    def blocking_reason(self) -> Optional[str]:
        """Why the invoice must not be priced right now.

        The scale reports ONE number for everything on the tray.  Splitting it
        between products by pixel area was measured to be more than 20% wrong on
        half of all lines, so this refuses instead of guessing.
        """
        rec = self.current_rec
        mode = self.settings.get("pos.multi_item_mode", "interlock")
        if mode == "off" or rec is None or not self.camera_ok:
            return None
        if rec.intrusion and self.settings.get("pos.refuse_on_intrusion", True):
            return tr("hand_detected")
        if rec.unreliable and self.settings.get("pos.refuse_on_unreliable", True):
            return tr("lighting_changed")
        if len(rec.distinct_items) > 1:
            return tr("multi_item_block")
        return None

    def _update_blocking(self):
        reason = self.blocking_reason()
        # a missing camera is worth saying out loud, but it must not stop a sale:
        # the operator can still pick the product from the list and weigh it
        notice = reason or (tr("camera_dead") if not self.camera_ok else "")
        self.banner_lbl.setText(notice)
        self.banner_lbl.setVisible(bool(notice))
        self.add_btn.setEnabled(reason is None)
        self.video.banner = reason if reason and reason == tr("multi_item_block") else ""

    def invoiced_weight(self) -> float:
        """Everything already on the invoice, which should also still be on the tray."""
        return float(sum(l.weight_kg for l in self.lines))

    def tray_weight(self) -> Optional[float]:
        return self.current_weight.weight_kg if self.current_weight else None

    def weight_difference(self) -> Optional[float]:
        """Tray minus invoice: what is on the scale that has not been billed."""
        tray = self.tray_weight()
        return None if tray is None else tray - self.invoiced_weight()

    def _weight_ok(self) -> Optional[float]:
        """The weight to bill for the item being added.

        In cumulative mode the products stay on the tray, so a new item weighs
        the difference between the scale and what the invoice already covers:
        put 100 g on, bill it, add 200 g, and the scale reads 300 g while the
        new line is 200 g.  That also means the tray and the invoice must agree
        at all times, which is what makes a swapped or extra item visible.
        """
        r = self.current_weight
        if r is not None and not r.stable and bool(self.settings.get("scale.stable_only", False)):
            self.status.showMessage(tr("weight_unstable"), 4000)
            return None
        tray = r.weight_kg if r else 0.0
        minimum = float(self.settings.get("scale.min_weight_kg", 0.005))
        if self.settings.get("pos.weighing_mode", "cumulative") == "cumulative":
            w = tray - self.invoiced_weight()
            if w < minimum:
                self.status.showMessage(tr("nothing_new") if tray >= minimum else tr("zero_weight"), 4000)
                return None
            return w
        if tray < minimum:
            self.status.showMessage(tr("zero_weight"), 4000)
            return None
        return tray

    def _update_reconciliation(self):
        """Show tray / invoice / new, and colour it when they disagree."""
        decimals = int(self.settings.get("general.weight_decimals", 3))
        tray, invoiced = self.tray_weight(), self.invoiced_weight()
        if tray is None:
            self.recon_lbl.setText("")
            return
        diff = tray - invoiced
        tol = float(self.settings.get("pos.weight_tolerance_kg", 0.02))
        text = (f"{tr('tray_weight')} {fmt_weight(tray, decimals)}"
                f"   |   {tr('invoiced_weight')} {fmt_weight(invoiced, decimals)}")
        colour = "#9aa3b5"
        if not self.lines:
            pass
        elif abs(diff) <= tol:
            text += f"   |   {tr('weight_matches')}"
            colour = "#7CFC9A"
        elif diff > 0:
            text += f"   |   {tr('new_weight')} {fmt_weight(diff, decimals)}"
            colour = "#ffd166"
        else:
            text += f"   |   {tr('weight_missing', diff=fmt_weight(-diff, decimals))}"
            colour = "#ff8d8d"
        self.recon_lbl.setText(text)
        self.recon_lbl.setStyleSheet(f"color:{colour};font-size:14px")

    def add_detected(self):
        reason = self.blocking_reason()
        if reason:
            self.status.showMessage(reason, 5000)
            return
        rec = self.current_rec
        active = self._active_detection()
        chosen = self._override_item_id if (active is not None and
                                            active.track_id == self._override_track) else None
        if rec is None or active is None or rec.empty or (not active.accepted and not chosen):
            self.status.showMessage(tr("no_item_detected"), 4000)
            return
        item = self.db.get(chosen or active.item_id)
        if item is None:
            return
        w = self._weight_ok()
        if w is None:
            return
        if chosen and chosen != active.item_id:
            self._reinforce(item)        # the operator corrected us: learn this view
        qty = 1
        if self.settings.get("pos.multi_item_mode", "interlock") == "quantity":
            same = [d for d in rec.accepted if d.item_id == active.item_id]
            # a count is only trustworthy for clearly separated objects: touching
            # items merge or split unpredictably (measured 28-70% recall)
            if len(same) > 1 and not any(d.split for d in same):
                qty = len(same)
        self._add_line(item, w, active.score, qty)
        self._override_item_id = None

    def add_manual(self):
        iid = self.manual_combo.currentData()
        item = self.db.get(iid) if iid else None
        if item is None:
            self.status.showMessage(tr("no_items_trained"), 4000)
            return
        w = self._weight_ok()
        if w is None:
            return
        self._reinforce(item)
        self._add_line(item, w, 0.0)

    def _reinforce(self, item: Item):
        """Teach the model today's lighting from an operator-confirmed pick.

        Only ever from a manual pick: letting the model add its own confident
        predictions was measured to collapse accuracy by 14 points.
        """
        if not self.settings.get("pos.reinforce_on_manual_pick", True) or self.engine is None:
            return
        cap = int(self.settings.get("ai.drift_cap", 5))
        if cap <= 0 or self.current_frame is None:
            return
        active = self._active_detection()
        try:
            if active is not None and active.region is not None and item.crop_mode == MODE_OBJECT:
                crop = region_crop(self.current_frame, active.region)
            elif item.crop_mode == MODE_TRAY:
                crop, _ = self.tray.crop(self.current_frame)
            else:
                return
        except Exception:
            return
        self.engine.request_embedding(crop, lambda e, iid=item.id: self.bus.embedding_ready.emit(f"reinforce:{iid}", e), False)

    def _maybe_auto_add(self):
        if not self.settings.get("general.auto_add", False) or not self._auto_armed:
            return
        if self.blocking_reason():
            return
        rec, reading = self.current_rec, self.current_weight
        if rec is None or reading is None or rec.empty or not reading.stable:
            return
        active = self._active_detection()
        if active is None or not active.accepted:
            return
        if active.score < float(self.settings.get("general.auto_add_min_confidence", 0.8)):
            return
        if reading.weight_kg < float(self.settings.get("scale.min_weight_kg", 0.005)):
            return
        item = self.db.get(active.item_id)
        if item is None:
            return
        self._auto_armed = False          # re-armed when the tray is emptied
        self._add_line(item, reading.weight_kg, active.score)

    def _add_line(self, item: Item, weight: float, conf: float, quantity: int = 1):
        self.lines.append(InvoiceLine(item.name, weight, item.price_per_kg, item.id, conf, quantity,
                                      item.unit, item.price_per_piece))
        self._refresh_table()
        self.table.selectRow(self.table.rowCount() - 1)

    def _refresh_table(self):
        cur_dec = int(self.settings.get("general.currency_decimals", 0))
        w_dec = int(self.settings.get("general.weight_decimals", 3))
        self.table.setRowCount(len(self.lines))
        for r, ln in enumerate(self.lines):
            unit_price = ln.price_per_piece if ln.unit == "pcs" else ln.price_per_kg
            cells = [ln.display_name, fmt_weight(ln.weight_kg, w_dec), fmt_money(unit_price, cur_dec),
                     fmt_money(ln.total, cur_dec)]
            for c, text in enumerate(cells):
                it = QTableWidgetItem(text)
                if c:
                    it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.table.setItem(r, c, it)
            btn = QPushButton("✕")
            btn.setObjectName("danger")
            btn.setFixedSize(40, 34)
            btn.clicked.connect(lambda _=False, idx=r: self._delete_row(idx))
            self.table.setCellWidget(r, 4, btn)
        self._update_total()
        self._update_reconciliation()

    def _update_total(self):
        total = sum(l.total for l in self.lines)
        self.total_lbl.setText(fmt_money(total, int(self.settings.get("general.currency_decimals", 0)),
                                         self.settings.get("general.currency", "")))

    def _delete_row(self, idx: int):
        if 0 <= idx < len(self.lines):
            del self.lines[idx]
            self._refresh_table()

    def delete_selected(self):
        rows = sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True)
        if not rows and self.lines:
            rows = [len(self.lines) - 1]
        for r in rows:
            self._delete_row(r)

    def clear_invoice(self):
        if self.lines and ask(self, tr("clear_confirm")):
            self.lines.clear()
            self._refresh_table()

    # invoice numbering ------------------------------------------------------
    def _counter_path(self) -> str:
        d = self.settings.resolve("general.invoice_dir") or data_path("invoices")
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, "counter.json")

    def _peek_invoice_number(self) -> int:
        try:
            with open(self._counter_path(), "r", encoding="utf-8") as fh:
                return int(json.load(fh).get("next", 1))
        except Exception:
            return 1

    def _next_invoice_number(self) -> int:
        n = self._peek_invoice_number()
        with open(self._counter_path(), "w", encoding="utf-8") as fh:
            json.dump({"next": n + 1}, fh)
        return n

    def checkout(self):
        if not self.lines:
            self.status.showMessage(tr("invoice_empty"), 4000)
            return
        # The tamper check: everything billed should still be on the tray and
        # nothing else should be.  An extra unbilled item, or one swapped for a
        # heavier one after scanning, shows up here as a weight that disagrees.
        diff = self.weight_difference()
        tol = float(self.settings.get("pos.weight_tolerance_kg", 0.02))
        if (self.settings.get("pos.weighing_mode", "cumulative") == "cumulative"
                and diff is not None and abs(diff) > tol):
            decimals = int(self.settings.get("general.weight_decimals", 3))
            text = tr("checkout_mismatch", diff=("+" if diff > 0 else "-") + fmt_weight(abs(diff), decimals))
            log.warning("checkout weight mismatch: tray %.3f, invoice %.3f", self.tray_weight() or 0.0,
                        self.invoiced_weight())
            if self.settings.get("pos.block_checkout_on_mismatch", True):
                if not ask(self, text + "\n\n" + tr("checkout_anyway")):
                    return
            else:
                self.status.showMessage(text, 8000)
        number = self._next_invoice_number()
        inv = Invoice(number, list(self.lines), store_name=self.settings.get("general.store_name", ""),
                      currency=self.settings.get("general.currency", ""))
        d = os.path.dirname(self._counter_path())
        month_dir = os.path.join(d, inv.created.strftime("%Y-%m"))
        os.makedirs(month_dir, exist_ok=True)
        with open(os.path.join(month_dir, f"invoice_{number:06d}.json"), "w", encoding="utf-8") as fh:
            json.dump(inv.to_dict(), fh, ensure_ascii=False, indent=2)
        total_text = fmt_money(inv.grand_total, int(self.settings.get("general.currency_decimals", 0)), inv.currency)
        self.status.showMessage(tr("checkout_done", number=f"{number:06d}", total=total_text), 8000)
        printer = self.printer

        def work():
            try:
                msg = printer.print_invoice(inv)
                self.bus.print_done.emit(True, msg)
            except Exception as exc:
                log.exception("print failed")
                self.bus.print_done.emit(False, str(exc))

        threading.Thread(target=work, name="Printer", daemon=True).start()
        self.lines.clear()
        self._override_item_id = None
        self._refresh_table()
        self.inv_no_lbl.setText(f"{tr('invoice_no')} #{self._peek_invoice_number():06d}")

    def on_print_done(self, ok: bool, msg: str):
        if ok:
            self.status.showMessage(tr("printed") + f" ({msg})", 6000)
        else:
            msg_error(self, tr("print_failed", error=msg))

    # ------------------------------------------------------- training UI
    def _require_engine(self) -> bool:
        if self.engine is None:
            self.status.showMessage(tr("loading_model"), 3000)
            return False
        return True

    def train_new(self):
        if not self._require_engine():
            return
        dlg = TrainDialog(self)
        dlg.exec()
        dlg.deleteLater()
        self._refresh_items_ui()

    def add_angle(self):
        if not self._require_engine():
            return
        items = self.db.all_items()
        if not items:
            msg_info(self, tr("no_items_trained"))
            return
        active = self._active_detection()
        current = self.db.get(active.item_id) if (active and active.item_id) else items[0]
        dlg = TrainDialog(self, existing=current or items[0])
        dlg.exec()
        dlg.deleteLater()
        self._refresh_items_ui()

    def manage_items(self):
        ManageItemsDialog(self, self.db, self.settings).exec()
        self._refresh_items_ui()

    def _refresh_items_ui(self):
        self.manual_combo.clear()
        for it in self.db.all_items():
            icon = thumb_icon(it)
            if icon:
                self.manual_combo.addItem(icon, it.name, it.id)
            else:
                self.manual_combo.addItem(it.name, it.id)
        self.status_items.setText(tr("items_count", count=len(self.db)))
        self.video.has_items = len(self.db) > 0

    # ------------------------------------------------------- tray / empty tray
    def calibrate_tray(self):
        dlg = TrayCalibrationDialog(self, self._camera_frame, self.tray)
        if dlg.exec() == QDialog.DialogCode.Accepted and dlg.result_region is not None:
            self.tray = dlg.result_region
            self.settings.update_section("camera", self.tray.to_settings())
            self.settings.save()
            self.video.tray = self.tray
            if self.engine:
                self.engine.set_tray(self.tray)
            self.db.clear_background()          # neither reference matches the new crop
            self.db.save()
            if self.engine:
                self.engine.reload_segmenter()
                self.video.hint = tr("capture_empty_first")
            self._refresh_items_ui()
            msg_info(self, tr("tray_saved"))
        dlg.deleteLater()

    def save_diagnostics(self):
        """Package the frame, the masks, the crops and the scores into one zip.

        Recognition faults depend entirely on what this particular camera sees
        on this particular counter, so this is what turns "it does not detect
        my goods" into something that can actually be diagnosed.
        """
        folder = app_dir()
        path = diagnostics.default_path(folder)
        try:
            _fid, frame = self._camera_frame()
            diagnostics.collect(path, frame, self.tray, self.db, self.engine, self.settings,
                                self.current_rec, note=f"scale source: {self.scale_source}")
        except Exception as exc:
            log.exception("diagnostic bundle failed")
            msg_error(self, str(exc))
            return
        size_kb = os.path.getsize(path) // 1024
        self.status.showMessage(tr("diagnostics_saved", path=os.path.basename(path)), 10000)
        msg_info(self, tr("diagnostics_saved", path=path) + f"  ({size_kb} KB)")

    def capture_empty_tray(self):
        if not self._require_engine() or self._bg_wanted or self._bg_frames:
            return
        if self.current_weight is not None and self.scale is not None and not self.scale.is_simulated \
                and self.current_weight.weight_kg >= float(self.settings.get("scale.min_weight_kg", 0.005)):
            msg_error(self, tr("empty_tray_weight_warning"))
            return
        if not ask(self, tr("empty_tray_confirm")):
            return
        self._bg_frames = []
        self._settle_prev = None
        self._settle_still = 0
        self._wait_until_still(0)

    #: consecutive quiet looks before the capture starts, and how quiet counts
    SETTLE_NEEDED = 4
    SETTLE_MOTION = 2.5
    SETTLE_GIVE_UP = 60            # x 100 ms

    def _wait_until_still(self, attempt: int):
        """Hold off until the picture stops changing.

        The operator has just taken their hands off the tray, and those hands
        take a moment to leave the frame. Averaging them into the reference
        quietly ruins it, and the damage only shows up later as recognition
        that will not work.
        """
        if not self._bg_wanted and self._bg_frames:
            return                                   # cancelled
        _fid, frame = self._camera_frame()
        if frame is not None:
            small = cv2.resize(frame, (96, 72), interpolation=cv2.INTER_AREA).astype(np.float32)
            if self._settle_prev is not None:
                motion = float(np.abs(small - self._settle_prev).mean())
                self._settle_still = self._settle_still + 1 if motion < self.SETTLE_MOTION else 0
            self._settle_prev = small
        if self._settle_still >= self.SETTLE_NEEDED:
            self.status.clearMessage()
            self._grab_bg_frame(0)
            return
        if attempt >= self.SETTLE_GIVE_UP:
            # better a reference with a note than no reference at all
            log.warning("the view never settled; capturing the empty tray anyway")
            self.status.clearMessage()
            msg_info(self, tr("tray_settle_timeout"))
            self._grab_bg_frame(0)
            return
        self.status.showMessage(tr("tray_settling"))
        QTimer.singleShot(100, lambda: self._wait_until_still(attempt + 1))

    def _grab_bg_frame(self, attempt: int):
        _, frame = self._camera_frame()
        if frame is not None:
            self._bg_frames.append(frame.copy())
        if len(self._bg_frames) < BUILD_FRAMES and attempt < 40:
            QTimer.singleShot(200, lambda: self._grab_bg_frame(attempt + 1))
        else:
            self._finish_empty_tray()

    def _finish_empty_tray(self):
        frames, self._bg_frames = self._bg_frames, []
        if len(frames) < 3:
            msg_error(self, tr("no_frame"))
            return
        h, w = frames[0].shape[:2]
        try:
            model = EmptyTrayModel.build(frames, self.tray.bbox(w, h), self.tray.pixel_points(w, h).tolist(),
                                         work=int(self.settings.get("ai.segment_work", 192)))
            self._bg_payload = model.to_payload()
        except Exception as exc:
            log.exception("empty-tray model failed")
            msg_error(self, str(exc))
            return

        # A reference can be built from a view that is far too unstable to
        # detect anything in, and until now the app said "captured" and left
        # the operator to discover that nothing works. Measure it and say so.
        self._bg_stability = model.stability()
        log.info("empty-tray stability: %s", self._bg_stability)
        if self._bg_stability.get("reason") == "overexposed":
            msg_error(self, tr("tray_overexposed", percent=100.0 * self._bg_stability["saturated"]))
        elif self._bg_stability["verdict"] == "unusable":
            msg_error(self, tr("tray_unstable", median=self._bg_stability["median"],
                               percent=100.0 * self._bg_stability["fraction_noisy"]))
        elif self._bg_stability["verdict"] == "marginal":
            msg_info(self, tr("tray_marginal", median=self._bg_stability["median"]))
        crop, _ = self.tray.crop(frames[-1])
        self._bg_thumb = make_thumbnail(crop, 96)
        self._bg_pending = []
        self._bg_wanted = 3
        for f in (frames[0], frames[len(frames) // 2], frames[-1]):
            c, _ = self.tray.crop(f)
            self.engine.request_embedding(c, lambda e: self.bus.embedding_ready.emit("bg", e), True)

    def on_embedding(self, token: str, embeddings: np.ndarray):
        if token.startswith("reinforce:"):
            item_id = token.split(":", 1)[1]
            item = self.db.reinforce(item_id, embeddings[0], int(self.settings.get("ai.drift_cap", 5)))
            if item is not None:
                self.db.save()
                self.status.showMessage(tr("reinforced", name=item.name), 4000)
            return
        if token != "bg" or not self._bg_wanted:
            return
        self._bg_pending.append(embeddings)
        if len(self._bg_pending) >= self._bg_wanted:
            emb = np.vstack(self._bg_pending)
            self.db.set_background(emb, self._bg_thumb, self._bg_payload)
            self.db.save()
            self._bg_wanted = 0
            self._bg_pending = []
            if self.engine:
                self.engine.reload_segmenter()
                self.engine.reset_smoothing()
                self.video.hint = "" if self.engine.has_segmenter else tr("capture_empty_first")
            self._refresh_items_ui()
            self.status.showMessage(tr("empty_tray_saved", count=len(emb)), 6000)

    # --------------------------------------------------------- settings
    def open_settings(self):
        before = self.settings.as_dict()
        self.stop_scale()                     # free the COM port for the monitor / diagnosis
        dlg = SettingsDialog(self, self.settings, self._make_printer, self.bus)
        accepted = dlg.exec() == QDialog.DialogCode.Accepted
        # a diagnosis worker may still own the port; wait for it before reopening
        diag = getattr(dlg, "_diag_thread", None)
        if diag is not None and diag.is_alive():
            self.status.showMessage(tr("diag_running"), 3000)
            diag.join(30.0)
        dlg.deleteLater()
        after = self.settings.as_dict()
        self.apply_settings(before, after, accepted)
        if accepted:
            self.status.showMessage(tr("settings_saved"), 4000)

    def apply_settings(self, before: dict, after: dict, accepted: bool = True):
        if accepted and before["camera"] != after["camera"]:
            keys = ("backend", "device_index", "width", "height", "video_file", "mirror", "rotate", "sdk_dll")
            if any(before["camera"].get(k) != after["camera"].get(k) for k in keys):
                self.start_camera()
        if accepted and before["ai"] != after["ai"] and self.engine:
            self.engine.set_threshold(float(after["ai"]["confidence_threshold"]))
            self.engine.set_empty_threshold(float(after["ai"].get("empty_threshold", 0.88)))
            self.engine.set_smoothing(int(after["ai"]["smoothing_frames"]))
            self.engine.set_motion_gate(bool(after["ai"].get("motion_gate", True)), float(after["ai"].get("motion_threshold", 6.0)))
            self.engine.set_per_object(bool(after["ai"].get("per_object_recognition", True)))
            self.engine.max_objects = int(after["ai"].get("max_objects", 5))
            self.engine.reload_segmenter()
        self.start_scale()                    # always: it was stopped for the dialog
        if accepted:
            self.printer = self._make_printer(after["printer"], after["general"])
            self.status_printer.setText(f"{tr('tab_printer')}: {after['printer']['interface']}")
            if before["general"].get("language") != after["general"].get("language"):
                msg_info(self, tr("restart_note"))
            self._update_total()

    # ------------------------------------------------------------ close
    def closeEvent(self, event):
        try:
            self.stop_support()
            if self.engine:
                self.engine.stop()
            if self.camera_thread:
                self.camera_thread.stop()
            self.stop_scale()
            self.db.save()
        except Exception:
            log.exception("shutdown error")
        session_end()
        super().closeEvent(event)


# ------------------------------------------------------------- run app
def build_gui(settings: SettingsManager, args) -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("AIPosScale")
    app.setStyleSheet(STYLE)
    if is_rtl():
        app.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
    # before session_begin(), which would otherwise record the copy that is
    # still running as a session that crashed
    lock = acquire_single_instance()
    if lock is None:
        log.error("another copy of the app is already running; refusing to start a second one")
        msg_error(None, tr("already_running"))
        return 2
    app._instance_lock = lock                     # held for as long as the app lives
    families = set(QFontDatabase.families())
    for fam in ("Vazirmatn", "Vazir", "IRANSans", "Segoe UI", "Tahoma"):
        if fam in families:
            f = app.font()
            f.setFamily(fam)
            f.setPointSize(11)
            app.setFont(f)
            break
    previous = session_begin()
    win = MainWindow(settings, video_override=args.video or "")
    if previous:
        log.warning("previous session did not end cleanly (last heartbeat %s)", previous)
        QTimer.singleShot(1500, lambda: win.status.showMessage(tr("unclean_shutdown"), 15000))
    if args.screenshot:
        if args.screenshot_size:
            w, h = [int(v) for v in args.screenshot_size.lower().split("x")]
            win.resize(w, h)
        win.show()
        QTimer.singleShot(int(args.screenshot_delay * 1000), lambda: (win.grab().save(args.screenshot), win.close(), app.quit()))
        return app.exec()
    if settings.get("general.window_maximized", True):
        win.showMaximized()
    else:
        win.show()
    return app.exec()


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="AI POS Scale")
    parser.add_argument("--selftest", action="store_true", help="run hardware/AI diagnostics and exit")
    parser.add_argument("--lang", choices=["fa", "en"], help="override UI language for this run")
    parser.add_argument("--video", default="", help="use a video file as a virtual camera")
    parser.add_argument("--config", default="", help="path to config.json")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--screenshot", default="", help=argparse.SUPPRESS)
    parser.add_argument("--screenshot-delay", type=float, default=6.0, help=argparse.SUPPRESS)
    parser.add_argument("--screenshot-size", default="", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    log_path = setup_logging(args.verbose)
    settings = SettingsManager(args.config or None)
    set_language(args.lang or settings.get("general.language", "fa"))
    log.info("AI POS Scale starting (frozen=%s, config=%s, log=%s)", is_frozen(), settings.path, log_path)
    if settings.load_error:
        log.warning("config.json was unreadable (%s) - defaults restored", settings.load_error)

    if args.selftest:
        return run_selftest(settings, args.video)
    try:
        return build_gui(settings, args)
    except Exception:
        log.exception("fatal error")
        if is_frozen():
            try:
                import ctypes
                ctypes.windll.user32.MessageBoxW(0, traceback.format_exc()[-1500:], "AI POS Scale - fatal error", 0x10)
            except Exception:
                pass
        return 1


if __name__ == "__main__":
    sys.exit(main())
