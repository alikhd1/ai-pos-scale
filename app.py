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

log = logging.getLogger("app")


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


def scale_mode_reason(settings: SettingsManager) -> str:
    """Single source of truth for which scale object exists and why.

    Used by the status pill's tooltip, the self-test report and the head-less
    wiring test, so "what the settings dialog tested" and "what the main window
    actually built" can never drift apart again without a test failing.
    """
    sc = settings.section("scale")
    port = (sc.get("port") or "").strip()
    if not sc.get("enabled", True):
        return "SimulatedScale: scale.enabled = false"
    if sc.get("simulate"):
        return f"SimulatedScale: scale.simulate = true (port {port or '-'} ignored)"
    if not port:
        return "SimulatedScale: no COM port configured"
    return f"ScaleReader: {port} @ {sc.get('baudrate')} {sc.get('bytesize')}{sc.get('parity')}{sc.get('stopbits')}"


def scale_is_simulated(settings: SettingsManager) -> bool:
    sc = settings.section("scale")
    return bool(sc.get("simulate") or not (sc.get("port") or "").strip() or not sc.get("enabled", True))


# --------------------------------------------------------------------------- #
# Head-less self-test (also used by build.bat and on the target machine)
# --------------------------------------------------------------------------- #
def run_selftest(settings: SettingsManager, video_file: str = "") -> int:
    from rtk_camera import diagnose, open_camera
    from scale_driver import parse_weight, list_serial_ports, diagnose_scale
    from printer_driver import ReceiptPrinter, Invoice, InvoiceLine, list_windows_printers
    from ai_engine import FeatureExtractor, ItemDatabase, Recognizer, TrayRegion, MODE_OBJECT, MODE_TRAY
    from tray_segment import TraySegmenter

    lines: List[str] = [f"AI POS Scale self-test  {dt.datetime.now():%Y-%m-%d %H:%M:%S}",
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
    lines.append(f"frame: {'OK ' + str(frame.shape) if frame is not None else 'NONE'}")
    tray = TrayRegion.from_settings(cam)
    lines.append(f"tray area: {'polygon with ' + str(len(tray.points)) + ' points' if tray.is_polygon else 'default centre square'}, mask={tray.mask_outside}")

    section("ai")
    try:
        t0 = time.time()
        fx = FeatureExtractor(model_file_path(settings), int(settings.get("ai.num_threads", 1)))
        lines.append(f"MobileNetV2 loaded in {time.time() - t0:.1f}s from {fx.weights_path}")
        test_img = frame if frame is not None else np.full((480, 640, 3), 90, np.uint8)
        crop, box = tray.crop(test_img)
        t0 = time.time()
        emb = fx.embed(crop)
        lines.append(f"embedding dim={emb.shape[0]} norm={np.linalg.norm(emb):.3f} time={1000 * (time.time() - t0):.0f} ms bbox={box}")
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
    if scale_is_simulated(settings):
        if (sc.get("port") or "").strip() and sc.get("simulate"):
            ok = False
            lines.append("  PROBLEM: a COM port is configured but simulation is ON, so the app shows a "
                         "simulated weight while the settings dialog tests the real port.")
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
from PyQt6.QtCore import Qt, QTimer, QObject, pyqtSignal, QSize, QRectF, QPointF
from PyQt6.QtGui import (QImage, QPixmap, QPainter, QPen, QColor, QFont, QBrush, QIcon, QKeySequence, QShortcut,
                         QFontDatabase, QGuiApplication, QPainterPath, QPolygonF)
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
                             QGridLayout, QTableWidget, QTableWidgetItem, QHeaderView, QSlider, QDoubleSpinBox,
                             QSpinBox, QComboBox, QLineEdit, QDialog, QDialogButtonBox, QFormLayout, QTabWidget,
                             QCheckBox, QMessageBox, QProgressBar, QFrame, QSizePolicy, QAbstractItemView,
                             QFileDialog, QPlainTextEdit, QStatusBar, QSplitter, QListWidget, QListWidgetItem,
                             QRadioButton, QButtonGroup)

from rtk_camera import open_camera, CameraThread
from scale_driver import (ScaleReader, SimulatedScale, list_serial_ports, diagnose_scale, BAUDRATES, WeightReading,
                          printable, hexdump)
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
        else:
            colour = QColor(COL_UNKNOWN)
        active = d.track_id == self.active_track
        style = Qt.PenStyle.DashLine if d.split else Qt.PenStyle.SolidLine
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(colour, 4 if active else 2, style))
        p.drawRoundedRect(r, 6, 6)
        label = f"{d.name}  {d.score * 100:.0f}%" if d.accepted else tr("unknown_item")
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
        self._diag_running = False
        tabs = QTabWidget()
        tabs.addTab(self._general_tab(), tr("tab_general"))
        tabs.addTab(self._camera_tab(), tr("tab_camera"))
        tabs.addTab(self._scale_tab(), tr("tab_scale"))
        tabs.addTab(self._printer_tab(), tr("tab_printer"))
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
        if kind == "scale_line":
            self._append_scale(text)
        elif kind == "scale_done":
            self._diag_running = False
            self.diag_btn.setEnabled(True)
            self.apply_btn.setEnabled(self._suggestion is not None)
            self._append_scale(("PROBLEM: " if is_error else "OK: ") + text)
            # The bug this fixes: a diagnosis that succeeds returns no "suggestion",
            # so the only code that ever unticked simulation was unreachable exactly
            # when the hardware worked first time.
            if not is_error and self.simulate.isChecked():
                self.simulate.setChecked(False)
                self._append_scale("-> " + tr("sim_unticked"))
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
        self.simulate = QCheckBox(tr("simulate_scale"))
        self.simulate.setChecked(bool(s["simulate"]))
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
        f.addRow("", self.simulate)
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
        self.monitor_btn = QPushButton(tr("start_monitor"))
        self.monitor_btn.clicked.connect(self._toggle_monitor)
        self.diag_btn = QPushButton(tr("diagnose_scale"))
        self.diag_btn.clicked.connect(self._diagnose)
        self.apply_btn = QPushButton(tr("apply_suggestion"))
        self.apply_btn.setEnabled(False)
        self.apply_btn.clicked.connect(self._apply_suggestion)
        copy_btn = QPushButton(tr("copy_report"))
        copy_btn.clicked.connect(self._copy_report)
        for b in (self.monitor_btn, self.diag_btn, self.apply_btn, copy_btn):
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

    def _scan_ports(self):
        cur = self.port.currentText()
        self.port.clear()
        for dev, desc in list_serial_ports():
            self.port.addItem(dev, dev)
            self.port.setItemData(self.port.count() - 1, f"{dev} - {desc}", Qt.ItemDataRole.ToolTipRole)
        self.port.setCurrentText(cur)

    def _scale_config(self) -> dict:
        cfg = self.settings.section("scale")
        cfg.update({"port": self.port.currentText().strip(), "baudrate": int(self.baud.currentText()),
                    "bytesize": int(self.bits.currentText()), "parity": self.parity.currentData(),
                    "stopbits": float(self.stop.currentText()), "poll_command": self.poll.text(),
                    "unit": self.unit.currentText(), "implied_decimals": self.implied.value(),
                    "hold_last_s": float(self.hold.value()), "stable_only": self.stable_only.isChecked(),
                    "simulate": self.simulate.isChecked(), "enabled": True})
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
        self.simulate.setChecked(False)
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

    # -- save
    def save(self):
        s = self.settings
        s.update_section("general", {"language": self.lang.currentData(), "store_name": self.store.text().strip(),
                                     "currency": self.currency.text().strip(), "currency_decimals": self.cur_dec.value(),
                                     "auto_add": self.auto_add.isChecked()})
        s.update_section("pos", {"multi_item_mode": self.multi_mode.currentData(),
                                 "refuse_on_intrusion": self.refuse_hand.isChecked(),
                                 "reinforce_on_manual_pick": self.reinforce.isChecked()})
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
        self.engine: Optional[RecognitionEngine] = None
        self.extractor: Optional[FeatureExtractor] = None
        self.scale = None
        self.db = ItemDatabase(settings.resolve("ai.db_file"))
        self.tray = TrayRegion.from_settings(settings.section("camera"))
        self.printer = self._make_printer(settings.section("printer"), settings.section("general"))
        self._auto_armed = True
        self._last_frame_id = -1
        self._scale_conflict = False
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
        for b in (self.tray_btn, self.empty_btn):
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
        wl.addWidget(self.sim_box, 2, 0, 1, 3)
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
        self.bus.toast.connect(self.on_toast)
        self.bus.print_done.connect(self.on_print_done)
        self.bus.embedding_ready.connect(self.on_embedding)
        self.video.region_clicked.connect(self.on_region_clicked)
        self.train_btn.clicked.connect(self.train_new)
        self.angle_btn.clicked.connect(self.add_angle)
        self.manage_btn.clicked.connect(self.manage_items)
        self.tray_btn.clicked.connect(self.calibrate_tray)
        self.empty_btn.clicked.connect(self.capture_empty_tray)
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

    def start_camera(self):
        if self.camera_thread:
            self.camera_thread.stop()
            self.camera_thread = None
        cam = self.settings.section("camera")

        def work():
            res = open_camera(cam["backend"], int(cam["device_index"]), int(cam["width"]), int(cam["height"]),
                              cam.get("sdk_dll", "RTKCamSDK.dll"), self.video_override or cam.get("video_file", ""))
            self.bus.camera_ready.emit(res)

        threading.Thread(target=work, name="CameraOpener", daemon=True).start()

    def on_camera_ready(self, res):
        self.camera_result = res
        cam = self.settings.section("camera")
        self.camera_thread = CameraThread(res.source, float(cam.get("fps_limit", 15)), bool(cam.get("mirror")), int(cam.get("rotate", 0)))
        self.camera_thread.start()
        self.cam_pill.setText(f"{tr('camera_backend')}: {res.backend.upper()}")
        self.cam_pill.setToolTip(res.description + ("\n" + "\n".join(res.notes) if res.notes else ""))
        self.video.caption = res.description
        for n in res.notes:
            log.info("camera note: %s", n)
        if cam.get("backend") in ("auto", "rtk") and res.backend == "opencv" and any("32-bit" in n for n in res.notes):
            self.status.showMessage(tr("rtk_fallback_status"), 10000)

    def _camera_frame(self):
        th = self.camera_thread
        return th.get_latest() if th else (0, None)

    def _load_model(self):
        try:
            fx = FeatureExtractor(model_file_path(self.settings), int(self.settings.get("ai.num_threads", 1)))
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
        self.engine.start()
        self.video.hint = "" if self.engine.has_segmenter else tr("capture_empty_first")
        if self.db.backbone_mismatch:
            msg_error(self, tr("retrain_needed"))
        self.status.showMessage(tr("model_ready"), 5000)

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
        self._scale_conflict = bool(port) and bool(sc.get("simulate")) and bool(sc.get("enabled", True))
        if sc.get("simulate") or not port or not sc.get("enabled", True):
            self.scale = SimulatedScale(self.sim_spin.value())
            self.sim_box.setVisible(True)
            self.use_real_btn.setVisible(self._scale_conflict)
            self.use_real_btn.setText(tr("use_real_scale", port=port))
            note = f" ({port}: {tr('scale_sim_overrides_port')})" if self._scale_conflict else ""
            self.scale_pill.setText(f"{tr('scale_status')}: {tr('scale_simulated')}{note}")
            self.scale_pill.setStyleSheet(PILL % (PILL_WARN if self._scale_conflict else PILL_NEUTRAL))
            if self.engine:
                self.engine.set_weight_hint(None)
        else:
            self.scale = ScaleReader(sc)
            self.sim_box.setVisible(False)
            self.use_real_btn.setVisible(False)
        self.scale_pill.setToolTip(scale_mode_reason(self.settings))
        self.scale.subscribe(lambda r: self.bus.weight.emit(r))
        self.scale.start()
        if self.scale.is_simulated:
            self.scale.set_weight(self.sim_spin.value())

    def _switch_to_real_scale(self):
        self.settings.set("scale.simulate", False)
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
        if rec.empty:
            self.detected_lbl.setText(tr("tray_empty"))
            self.detected_lbl.setStyleSheet("color:#9aa3b5")
            self.conf_bar.setValue(0)
            self._auto_armed = True
        elif active is not None and active.accepted:
            self.detected_lbl.setText(active.name)
            self.detected_lbl.setStyleSheet("color:#7CFC9A")
            self.conf_bar.setValue(int(active.score * 100))
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
        if mode == "off" or rec is None:
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
        self.banner_lbl.setText(reason or "")
        self.banner_lbl.setVisible(bool(reason))
        self.add_btn.setEnabled(reason is None)
        self.video.banner = reason if reason and reason == tr("multi_item_block") else ""

    def _weight_ok(self) -> Optional[float]:
        r = self.current_weight
        if r is not None and not r.stable and bool(self.settings.get("scale.stable_only", False)):
            self.status.showMessage(tr("weight_unstable"), 4000)
            return None
        w = r.weight_kg if r else 0.0
        if w < float(self.settings.get("scale.min_weight_kg", 0.005)):
            self.status.showMessage(tr("zero_weight"), 4000)
            return None
        return w

    def add_detected(self):
        reason = self.blocking_reason()
        if reason:
            self.status.showMessage(reason, 5000)
            return
        rec = self.current_rec
        active = self._active_detection()
        if rec is None or active is None or not active.accepted or rec.empty:
            self.status.showMessage(tr("no_item_detected"), 4000)
            return
        item = self.db.get(active.item_id)
        if item is None:
            return
        w = self._weight_ok()
        if w is None:
            return
        qty = 1
        if self.settings.get("pos.multi_item_mode", "interlock") == "quantity":
            same = [d for d in rec.accepted if d.item_id == active.item_id]
            # a count is only trustworthy for clearly separated objects: touching
            # items merge or split unpredictably (measured 28-70% recall)
            if len(same) > 1 and not any(d.split for d in same):
                qty = len(same)
        self._add_line(item, w, active.score, qty)

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
        self._grab_bg_frame(0)

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
