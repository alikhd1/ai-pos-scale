"""
settings_manager.py
-------------------
Persistent JSON configuration for the AI POS Scale application.

* All hardware / UI settings live in one ``config.json`` next to the executable
  (or next to this file when running from source).
* Missing keys are filled from ``DEFAULTS`` so old config files keep working
  after upgrades.
* Writes are atomic (temp file + replace) so a power cut during save can not
  corrupt the configuration of a POS terminal.
"""
from __future__ import annotations

import copy
import json
import os
import sys
import tempfile
import threading
from typing import Any, Dict

APP_NAME = "AIPosScale"
CONFIG_FILE_NAME = "config.json"

# --------------------------------------------------------------------------- #
# Default configuration
# --------------------------------------------------------------------------- #
DEFAULTS: Dict[str, Dict[str, Any]] = {
    "general": {
        "language": "fa",                 # "fa" (Persian, RTL) or "en"
        "store_name": "فروشگاه هوشمند",
        "currency": "تومان",
        "currency_decimals": 0,
        "weight_decimals": 3,
        "invoice_dir": "invoices",        # relative to the app directory
        "auto_add": False,                # auto-add recognised item when weight is stable
        "auto_add_min_confidence": 0.80,
        "window_maximized": True,
    },
    "camera": {
        "backend": "auto",                # "auto" | "rtk" | "opencv"
        "device_index": 0,
        "width": 640,
        "height": 480,
        "mirror": False,
        "rotate": 0,                      # 0 / 90 / 180 / 270
        "roi_ratio": 0.60,                # centre ROI size relative to min(frame w, h) (used when tray_points is empty)
        "tray_points": [],                # normalised polygon [[x, y], ...] around the tray (Tray Area dialog)
        "tray_mask": True,                # paint the area outside the polygon grey
        "video_file": "",                 # optional video file used as a virtual camera (demo/testing)
        "sdk_dll": "RTKCamSDK.dll",       # path or bare name (searched next to the exe)
        "fps_limit": 15,                  # camera grab rate (display); lower = less CPU / heat
    },
    "ai": {
        "confidence_threshold": 0.65,
        "smoothing_frames": 5,            # temporal majority vote window
        "train_samples": 12,              # frames captured per training session
        "augment_samples": True,          # add flipped / rotated variants of each sample
        "infer_fps": 5,                   # recognitions per second (CPU budget)
        "motion_gate": True,              # skip the network while the tray image is static
        "motion_threshold": 6.0,          # mean grey-level change (0-255) that counts as motion
        "empty_threshold": 0.88,          # similarity to the empty-tray reference that means "tray empty"
        "min_train_samples": 3,           # recommended minimum samples per item
        "auto_capture": False,            # training dialog: capture automatically instead of per button press
        "auto_capture_interval_ms": 700,
        "top_k": 3,                       # per item score = mean of top-k sample similarities
        "num_threads": 1,                 # torch intra-op threads (1 is fastest on small POS CPUs)
        "db_file": "items_db.pkl",
        "model_file": "models/mobilenet_v2-b0353104.pth",
    },
    "scale": {
        "enabled": True,
        "simulate": True,                 # slider instead of hardware until a port is configured
        "port": "",
        "baudrate": 9600,
        "bytesize": 8,
        "parity": "N",                    # N / E / O / M / S
        "stopbits": 1,                    # 1 / 1.5 / 2
        "poll_command": "",               # e.g. "W\\r\\n" or "\\x05" for request/response scales
        "poll_interval_ms": 300,
        "unit": "auto",                   # auto | kg | g | lb
        "implied_decimals": 0,            # for protocols that send "001234" meaning 1.234
        "number_index": 0,                # which number in the frame is the weight
        "stable_only": False,             # ignore readings flagged unstable
        "min_weight_kg": 0.005,
        "timeout_s": 1.0,
        "hold_last_s": 5.0,               # keep showing the last weight this long without new data
        "idle_gap_ms": 60,                # idle gap that terminates a frame for terminator-less protocols
        "assert_dtr": True,
        "assert_rts": True,
        "reconnect_s": 3.0,
    },
    "printer": {
        "enabled": True,
        "interface": "windows",           # "windows" (spooler RAW) | "serial" | "file" | "none"
        "printer_name": "",
        "port": "",
        "baudrate": 9600,
        "paper_width": 80,                # 58 or 80 (mm)
        "dots_per_line": 0,               # 0 = auto (58mm -> 384, 80mm -> 576)
        "mode": "image",                  # "image" (any language / fonts) | "text" (ESC/POS text)
        "codepage": "cp437",
        "codepage_id": 0,                 # ESC t n
        "header": "به فروشگاه ما خوش آمدید",
        "footer": "از خرید شما سپاسگزاریم",
        "font_file": "",                  # empty = auto (Tahoma / Segoe UI / Arial)
        "font_size": 26,
        "cut": True,
        "feed_lines": 3,
        "open_drawer": False,
    },
}


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
def is_frozen() -> bool:
    """True when running inside a PyInstaller bundle."""
    return bool(getattr(sys, "frozen", False))


def app_dir() -> str:
    """Directory holding user data (config, database, invoices).

    For a frozen exe this is the folder containing the .exe, so the data is
    editable and survives updates.  From source it is this file's folder.
    """
    if is_frozen():
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


def resource_path(relative: str) -> str:
    """Path of a read-only bundled resource (DLL, model weights...).

    PyInstaller ``--add-binary``/``--add-data`` files are extracted to
    ``sys._MEIPASS`` (onefile) or placed in the app dir (onedir).  We look in
    both places and fall back to the app dir so the same code works from source.
    """
    candidates = []
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidates.append(os.path.join(meipass, relative))
    candidates.append(os.path.join(app_dir(), relative))
    candidates.append(os.path.abspath(relative))
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[-1]


def data_path(relative: str) -> str:
    """Absolute path inside the writable application directory."""
    if os.path.isabs(relative):
        return relative
    return os.path.join(app_dir(), relative)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Return ``base`` updated with ``override`` recursively (new dict)."""
    result = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


class SettingsManager:
    """Thread-safe accessor around ``config.json``.

    Usage::

        settings = SettingsManager()          # loads (or creates) config.json
        baud = settings.get("scale.baudrate")
        settings.set("scale.port", "COM3")
        settings.save()
    """

    def __init__(self, path: str | None = None, autoload: bool = True):
        self.path = path or data_path(CONFIG_FILE_NAME)
        self._lock = threading.RLock()
        self._data: Dict[str, Any] = copy.deepcopy(DEFAULTS)
        self.load_error: str | None = None
        if autoload:
            self.load()

    # ------------------------------------------------------------------ I/O
    def load(self) -> Dict[str, Any]:
        """Load config.json (merged with defaults). Creates the file if absent."""
        with self._lock:
            loaded: Dict[str, Any] = {}
            if os.path.exists(self.path):
                try:
                    with open(self.path, "r", encoding="utf-8") as fh:
                        loaded = json.load(fh) or {}
                    if not isinstance(loaded, dict):
                        raise ValueError("config root must be a JSON object")
                    self.load_error = None
                except Exception as exc:  # corrupted file -> keep defaults, keep a backup
                    self.load_error = f"{type(exc).__name__}: {exc}"
                    try:
                        os.replace(self.path, self.path + ".corrupt")
                    except OSError:
                        pass
                    loaded = {}
            self._data = _deep_merge(DEFAULTS, loaded)
            if not os.path.exists(self.path):
                self.save()
            return copy.deepcopy(self._data)

    def save(self) -> None:
        """Atomically write the current configuration to disk."""
        with self._lock:
            directory = os.path.dirname(self.path) or "."
            os.makedirs(directory, exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(prefix="config_", suffix=".tmp", dir=directory)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    json.dump(self._data, fh, ensure_ascii=False, indent=2)
                    fh.flush()
                    os.fsync(fh.fileno())
                os.replace(tmp_path, self.path)
            finally:
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass

    def reset_to_defaults(self) -> None:
        with self._lock:
            self._data = copy.deepcopy(DEFAULTS)

    # -------------------------------------------------------------- access
    def get(self, dotted_key: str, default: Any = None) -> Any:
        """``settings.get("scale.baudrate")``"""
        with self._lock:
            node: Any = self._data
            for part in dotted_key.split("."):
                if not isinstance(node, dict) or part not in node:
                    return default
                node = node[part]
            return copy.deepcopy(node)

    def set(self, dotted_key: str, value: Any) -> None:
        with self._lock:
            parts = dotted_key.split(".")
            node = self._data
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            node[parts[-1]] = copy.deepcopy(value)

    def section(self, name: str) -> Dict[str, Any]:
        """Deep copy of a whole section, e.g. ``settings.section("printer")``."""
        with self._lock:
            return copy.deepcopy(self._data.get(name, {}))

    def update_section(self, name: str, values: Dict[str, Any]) -> None:
        with self._lock:
            self._data[name] = _deep_merge(self._data.get(name, {}), values)

    def as_dict(self) -> Dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._data)

    def __getitem__(self, section: str) -> Dict[str, Any]:
        return self.section(section)

    # ------------------------------------------------------------ helpers
    def resolve(self, dotted_key: str) -> str:
        """Return an absolute filesystem path for a path-like setting."""
        value = self.get(dotted_key, "") or ""
        return data_path(value) if value else ""


if __name__ == "__main__":  # tiny manual test
    s = SettingsManager(path=os.path.join(tempfile.gettempdir(), "aipos_test_config.json"))
    s.set("scale.port", "COM7")
    s.save()
    print(json.dumps(s.as_dict(), ensure_ascii=True, indent=2)[:400], "...")
    print("config file:", s.path)
