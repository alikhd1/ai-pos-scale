"""
rtk_camera.py
-------------
Camera access layer for the AI POS Scale.

1. ``RTKCamSDK``  – ctypes wrapper around Realtek's native ``RTKCamSDK.dll``.
2. ``RTKCameraSource`` – frame source built on the SDK.
3. ``OpenCVCameraSource`` – graceful fallback using ``cv2.VideoCapture``
   (also supports a video file as a *virtual camera* for demos / testing).
4. ``open_camera()`` – factory that applies the fallback policy.
5. ``CameraThread`` – background grabber that always keeps the latest frame.

About the SDK binding
~~~~~~~~~~~~~~~~~~~~~
The SDK ships without a header, so the binding below was derived from the DLL
export table and disassembly of ``RTKCamSDK.dll`` (v1.0, 2019-12):

* Exports are named ``RTKCam_*``.  The ``T_fRTKCam_*`` names used in Realtek's
  sample are the *function pointer typedefs* the sample obtains through
  ``GetProcAddress``.  Both names are accepted here (``sdk.T_fRTKCam_OpenDevice``
  is an alias of ``sdk.RTKCam_OpenDevice``).
* Calling convention is **cdecl** (plain ``ret``), hence ``ctypes.CDLL``.
* The DLL is **32-bit (x86)** and wraps DirectShow.  It can only be loaded by a
  32-bit Python process; a 64-bit process (required by PyTorch) automatically
  falls back to OpenCV, which talks to the very same UVC camera through
  DirectShow / Media Foundation.

Reverse-engineered prototypes (all cdecl, ``int`` = 32-bit)::

    int    RTKCam_Init(void);
    void   RTKCam_Free(void);
    WORD   RTKCam_GetDeviceList(RTKCAM_DEVICE_INFO *list, WORD maxCount);      // returns count
    int    RTKCam_OpenDevice(RTKCAM_DEVICE_INFO *dev, RTKCAM_PARAMS *opt);     // 1 = ok
    int    RTKCam_CloseDevice(void *hDev);                                     // 0
    WORD   RTKCam_GetResolutionList(void *hDev, RTKCAM_RESOLUTION *list, WORD maxCount, WORD stream);
    int    RTKCam_SetResolution(void *hDev, RTKCAM_RESOLUTION res /*by value*/, WORD stream); // 1 = ok
    int    RTKCam_CaptureControl(void *hDev, WORD action);   // 1 start preview, 2 start record, 4 stop
    BYTE*  RTKCam_GetFrame(void *hDev, int reserved, DWORD *length);           // NULL on timeout (1 s)
    void   RTKCam_ReleaseFrame(void);                                          // no-op
    int    RTKCam_SetCaptureParam(void *hDev, RTKCAM_CAPTURE_PARAM *p);
    int    RTKCam_SetRecordParam (void *hDev, RTKCAM_RECORD_PARAM *p);
    int    RTKCam_SetDispParam   (void *hDev, RTKCAM_DISP_PARAM *p);
    int    RTKCam_CapturePhoto(void);
    int    RTKCam_VideoPropGetRange/Get/Set (void *hDev, long prop, RTKCAM_PROP *p);   // IAMVideoProcAmp
    int    RTKCam_CameraPropGetRange/Get/Set(void *hDev, long prop, RTKCAM_PROP *p);   // IAMCameraControl

The frame returned by ``RTKCam_GetFrame`` is the raw sample-grabber buffer in
the negotiated format (MJPG / YUY2 / RGB24); ``RTKCameraSource`` decodes it to
a BGR ``numpy`` array.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import logging
import os
import struct
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - cv2 is a hard requirement of the app
    cv2 = None  # type: ignore

log = logging.getLogger("rtk_camera")

# --------------------------------------------------------------------------- #
# ctypes structures (layouts derived from disassembly, see module docstring)
# --------------------------------------------------------------------------- #
RTK_MAX_DEVICES = 8
RTK_MAX_RESOLUTIONS = 64

# RTKCam_GetResolutionList / RTKCam_SetResolution "stream" argument
RTK_STREAM_CAPTURE = 0      # preview / capture pin
RTK_STREAM_STILL = 1        # still-image pin
RTK_STREAM_SUB = 2          # secondary (e.g. H.264) stream

# RTKCam_CaptureControl actions
RTK_CTRL_START_PREVIEW = 1
RTK_CTRL_START_RECORD = 2
RTK_CTRL_STOP = 4

# RTKCAM_RESOLUTION.format
RTK_FMT_UNKNOWN = 0
RTK_FMT_MJPG = 1
RTK_FMT_YUY2 = 2

# IAMVideoProcAmp properties (RTKCam_VideoProp*)
VIDEOPROC_BRIGHTNESS, VIDEOPROC_CONTRAST, VIDEOPROC_HUE, VIDEOPROC_SATURATION = 0, 1, 2, 3
VIDEOPROC_SHARPNESS, VIDEOPROC_GAMMA, VIDEOPROC_COLORENABLE, VIDEOPROC_WHITEBALANCE = 4, 5, 6, 7
VIDEOPROC_BACKLIGHT, VIDEOPROC_GAIN = 8, 9
# IAMCameraControl properties (RTKCam_CameraProp*)
CAMCTRL_PAN, CAMCTRL_TILT, CAMCTRL_ROLL, CAMCTRL_ZOOM, CAMCTRL_EXPOSURE, CAMCTRL_IRIS, CAMCTRL_FOCUS = range(7)
PROP_FLAG_AUTO, PROP_FLAG_MANUAL = 1, 2


class RTKCAM_DEVICE_INFO(ctypes.Structure):
    """One entry of RTKCam_GetDeviceList (0x414 = 1044 bytes)."""
    _fields_ = [
        ("vid", ctypes.c_uint16),               # 0x000 parsed from "VID_xxxx"
        ("pid", ctypes.c_uint16),               # 0x002 parsed from "PID_xxxx"
        ("rev", ctypes.c_uint16),               # 0x004 parsed from "REV_xxxx"
        ("reserved0", ctypes.c_uint16),         # 0x006
        ("device_type", ctypes.c_uint32),       # 0x008 internal type / index
        ("reserved1", ctypes.c_uint32),         # 0x00c
        ("reserved2", ctypes.c_uint32),         # 0x010
        ("friendly_name", ctypes.c_wchar * 256),  # 0x014 e.g. "USB Video Device"
        ("device_id", ctypes.c_wchar * 256),      # 0x214 PnP instance id (used by OpenDevice)
    ]


class RTKCAM_RESOLUTION(ctypes.Structure):
    """One entry of RTKCam_GetResolutionList (0x98 = 152 bytes)."""
    _fields_ = [
        ("width", ctypes.c_uint32),             # 0x00 biWidth
        ("height", ctypes.c_uint32),            # 0x04 biHeight
        ("bit_count", ctypes.c_uint16),         # 0x08 biBitCount
        ("format", ctypes.c_uint16),            # 0x0a 1 = MJPG, 2 = YUY2
        ("reserved0", ctypes.c_uint32),         # 0x0c
        ("avg_time_per_frame", ctypes.c_int64),  # 0x10 100 ns units
        ("reserved1", ctypes.c_uint8 * 0x78),   # 0x18
        ("valid", ctypes.c_uint16),             # 0x90 set to 1 for filled entries
        ("reserved2", ctypes.c_uint8 * 6),      # 0x92
    ]

    @property
    def fps(self) -> float:
        return 1e7 / self.avg_time_per_frame if self.avg_time_per_frame else 0.0

    @property
    def format_name(self) -> str:
        return {RTK_FMT_MJPG: "MJPG", RTK_FMT_YUY2: "YUY2"}.get(self.format, "RAW")

    def __repr__(self) -> str:  # pragma: no cover
        return f"<RTKRes {self.width}x{self.height} {self.format_name} {self.fps:.1f}fps>"


class RTKCAM_PROP(ctypes.Structure):
    """Range / value block shared by the *Prop* functions (0x18 bytes)."""
    _fields_ = [
        ("min", ctypes.c_long), ("max", ctypes.c_long), ("step", ctypes.c_long),
        ("default", ctypes.c_long), ("value", ctypes.c_long), ("flags", ctypes.c_long),
    ]


class RTKCAM_CAPTURE_PARAM(ctypes.Structure):
    """Snapshot parameters (0x418 bytes): format 0 = BMP, 1 = JPG."""
    _fields_ = [
        ("reserved0", ctypes.c_uint32), ("reserved1", ctypes.c_uint32),
        ("format", ctypes.c_uint16), ("reserved2", ctypes.c_uint16),
        ("path", ctypes.c_wchar * 512), ("extra", ctypes.c_uint32 * 3),
    ]


class RTKCAM_RECORD_PARAM(ctypes.Structure):
    """AVI recording parameters (0x40c bytes)."""
    _fields_ = [
        ("reserved0", ctypes.c_uint32), ("reserved1", ctypes.c_uint32),
        ("format", ctypes.c_uint16), ("reserved2", ctypes.c_uint16),
        ("path", ctypes.c_wchar * 512),
    ]


class RTKCAM_DISP_PARAM(ctypes.Structure):
    """Preview window parameters (0x1c bytes). renderer 0..2, mode 0..4."""
    _fields_ = [
        ("hwnd", ctypes.c_uint32), ("left", ctypes.c_long), ("top", ctypes.c_long),
        ("right", ctypes.c_long), ("bottom", ctypes.c_long), ("reserved", ctypes.c_uint32),
        ("renderer", ctypes.c_uint16), ("mode", ctypes.c_uint16),
    ]


class RTKCAM_PARAMS(ctypes.Structure):
    """Optional block accepted by RTKCam_OpenDevice (0x840 bytes)."""
    _fields_ = [("capture", RTKCAM_CAPTURE_PARAM), ("record", RTKCAM_RECORD_PARAM), ("disp", RTKCAM_DISP_PARAM)]


assert ctypes.sizeof(RTKCAM_RESOLUTION) == 0x98
assert ctypes.sizeof(RTKCAM_DEVICE_INFO) == 0x414
assert ctypes.sizeof(RTKCAM_PROP) == 0x18


class RTKCamError(RuntimeError):
    """Raised when the native SDK is unusable or a call fails."""


# --------------------------------------------------------------------------- #
# DLL discovery / validation helpers
# --------------------------------------------------------------------------- #
def _candidate_dll_paths(dll: str) -> List[str]:
    paths = []
    if os.path.isabs(dll):
        paths.append(dll)
    else:
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            paths.append(os.path.join(meipass, dll))
        exe_dir = os.path.dirname(os.path.abspath(sys.executable if getattr(sys, "frozen", False) else __file__))
        paths.append(os.path.join(exe_dir, dll))
        paths.append(os.path.join(os.getcwd(), dll))
        found = ctypes.util.find_library(os.path.splitext(dll)[0])
        if found:
            paths.append(found)
    seen, unique = set(), []
    for p in paths:
        if p and p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


def dll_machine(path: str) -> Optional[int]:
    """Return the PE machine type (0x14c = x86, 0x8664 = x64) or None."""
    try:
        with open(path, "rb") as fh:
            head = fh.read(0x40)
            if head[:2] != b"MZ":
                return None
            pe_off = struct.unpack_from("<I", head, 0x3C)[0]
            fh.seek(pe_off)
            sig = fh.read(6)
            if sig[:4] != b"PE\0\0":
                return None
            return struct.unpack_from("<H", sig, 4)[0]
    except OSError:
        return None


def process_is_32bit() -> bool:
    return ctypes.sizeof(ctypes.c_void_p) == 4


# --------------------------------------------------------------------------- #
# The SDK wrapper
# --------------------------------------------------------------------------- #
class RTKCamSDK:
    """Thin, explicit ctypes binding of RTKCamSDK.dll.

    >>> sdk = RTKCamSDK()              # raises RTKCamError if unusable
    >>> devices = sdk.get_device_list()
    >>> sdk.open_device(devices[0])
    >>> res = sdk.get_resolution_list()
    >>> sdk.set_resolution(res[0])
    >>> sdk.capture_control(RTK_CTRL_START_PREVIEW)
    >>> data, length = sdk.get_frame()
    """

    EXPORTS = (
        "RTKCam_Init", "RTKCam_Free", "RTKCam_GetDeviceList", "RTKCam_OpenDevice",
        "RTKCam_CloseDevice", "RTKCam_GetResolutionList", "RTKCam_SetResolution",
        "RTKCam_CaptureControl", "RTKCam_GetFrame", "RTKCam_ReleaseFrame",
        "RTKCam_SetCaptureParam", "RTKCam_SetRecordParam", "RTKCam_SetDispParam",
        "RTKCam_CapturePhoto", "RTKCam_VideoPropGetRange", "RTKCam_VideoPropGet",
        "RTKCam_VideoPropSet", "RTKCam_CameraPropGetRange", "RTKCam_CameraPropGet",
        "RTKCam_CameraPropSet", "RTKCam_GetVersion", "RTKCam_GetLastErr",
    )

    def __init__(self, dll: str = "RTKCamSDK.dll"):
        self.path = self._locate(dll)
        machine = dll_machine(self.path)
        if machine == 0x14C and not process_is_32bit():
            raise RTKCamError(
                f"{os.path.basename(self.path)} is a 32-bit (x86) DLL but this Python process is 64-bit. "
                "A 32-bit interpreter is required to load it in-process.")
        if machine == 0x8664 and process_is_32bit():
            raise RTKCamError("DLL is 64-bit but this Python process is 32-bit.")
        try:
            self._lib = ctypes.CDLL(self.path)          # cdecl
        except OSError as exc:
            raise RTKCamError(f"LoadLibrary failed for {self.path}: {exc}") from exc
        self._bind()
        self._device: Optional[RTKCAM_DEVICE_INFO] = None
        self._current_res: Optional[RTKCAM_RESOLUTION] = None
        self._streaming = False
        self._lock = threading.Lock()
        rc = self.RTKCam_Init()
        if rc not in (0, 1):
            raise RTKCamError(f"RTKCam_Init returned {rc}")

    # ------------------------------------------------------------ binding
    @staticmethod
    def _locate(dll: str) -> str:
        for path in _candidate_dll_paths(dll):
            if os.path.isfile(path):
                return path
        raise RTKCamError(f"{dll} not found (searched: {', '.join(_candidate_dll_paths(dll))})")

    def _proto(self, name: str, restype, argtypes, optional: bool = False):
        try:
            fn = getattr(self._lib, name)
        except AttributeError:
            if optional:
                return None
            raise RTKCamError(f"export {name} missing from {self.path}")
        fn.restype = restype
        fn.argtypes = argtypes
        setattr(self, name, fn)
        setattr(self, "T_f" + name, fn)     # Realtek's typedef naming (T_fRTKCam_OpenDevice ...)
        return fn

    def _bind(self) -> None:
        P = ctypes.POINTER
        vp, i32, u16 = ctypes.c_void_p, ctypes.c_int, ctypes.c_uint16
        self._proto("RTKCam_Init", i32, [])
        self._proto("RTKCam_Free", None, [])
        self._proto("RTKCam_GetDeviceList", i32, [P(RTKCAM_DEVICE_INFO), u16])
        self._proto("RTKCam_OpenDevice", i32, [P(RTKCAM_DEVICE_INFO), vp])
        self._proto("RTKCam_CloseDevice", i32, [vp])
        self._proto("RTKCam_GetResolutionList", i32, [vp, P(RTKCAM_RESOLUTION), u16, u16])
        self._proto("RTKCam_SetResolution", i32, [vp, RTKCAM_RESOLUTION, u16])   # struct by value
        self._proto("RTKCam_CaptureControl", i32, [vp, u16])
        self._proto("RTKCam_GetFrame", vp, [vp, i32, P(ctypes.c_uint32)])
        self._proto("RTKCam_ReleaseFrame", None, [])
        self._proto("RTKCam_SetCaptureParam", i32, [vp, P(RTKCAM_CAPTURE_PARAM)])
        self._proto("RTKCam_SetRecordParam", i32, [vp, P(RTKCAM_RECORD_PARAM)])
        self._proto("RTKCam_SetDispParam", i32, [vp, P(RTKCAM_DISP_PARAM)])
        self._proto("RTKCam_CapturePhoto", i32, [])
        for grp in ("VideoProp", "CameraProp"):
            for op in ("GetRange", "Get", "Set"):
                self._proto(f"RTKCam_{grp}{op}", i32, [vp, ctypes.c_long, P(RTKCAM_PROP)])
        # exports that exist but are stubs in v1.0 (return 0)
        self._proto("RTKCam_GetVersion", i32, [], optional=True)
        self._proto("RTKCam_GetLastErr", i32, [], optional=True)

    # ------------------------------------------------------------- helpers
    @property
    def handle(self) -> Optional[ctypes.c_void_p]:
        """The DLL keeps one global device; we pass the info pointer as 'handle'."""
        return ctypes.cast(ctypes.pointer(self._device), ctypes.c_void_p) if self._device else None

    @property
    def is_open(self) -> bool:
        return self._device is not None

    @property
    def current_resolution(self) -> Optional[RTKCAM_RESOLUTION]:
        return self._current_res

    # ---------------------------------------------------------------- API
    def get_device_list(self) -> List[RTKCAM_DEVICE_INFO]:
        arr = (RTKCAM_DEVICE_INFO * RTK_MAX_DEVICES)()
        count = self.RTKCam_GetDeviceList(arr, RTK_MAX_DEVICES) & 0xFFFF
        return [arr[i] for i in range(min(count, RTK_MAX_DEVICES))]

    def open_device(self, device: RTKCAM_DEVICE_INFO | int = 0, params: Optional[RTKCAM_PARAMS] = None) -> None:
        if isinstance(device, int):
            devices = self.get_device_list()
            if not devices:
                raise RTKCamError("RTKCam_GetDeviceList returned no devices")
            if device >= len(devices):
                raise RTKCamError(f"device index {device} out of range (found {len(devices)})")
            device = devices[device]
        # keep our own copy so the pointer stays valid for the DLL lifetime
        dev_copy = RTKCAM_DEVICE_INFO()
        ctypes.memmove(ctypes.byref(dev_copy), ctypes.byref(device), ctypes.sizeof(RTKCAM_DEVICE_INFO))
        pparams = ctypes.cast(ctypes.pointer(params), ctypes.c_void_p) if params is not None else None
        rc = self.RTKCam_OpenDevice(ctypes.byref(dev_copy), pparams)
        if rc != 1:
            raise RTKCamError(f"RTKCam_OpenDevice failed (rc={rc}) for '{device.friendly_name}'")
        self._device = dev_copy
        log.info("RTKCam device opened: %s [%04X:%04X]", dev_copy.friendly_name, dev_copy.vid, dev_copy.pid)

    def close_device(self) -> None:
        with self._lock:
            if self._streaming:
                try:
                    self.RTKCam_CaptureControl(self.handle, RTK_CTRL_STOP)
                except Exception:  # pragma: no cover
                    pass
                self._streaming = False
            if self._device is not None:
                self.RTKCam_CloseDevice(self.handle)
                self._device = None
            self._current_res = None

    def get_resolution_list(self, stream: int = RTK_STREAM_CAPTURE) -> List[RTKCAM_RESOLUTION]:
        self._require_open()
        arr = (RTKCAM_RESOLUTION * RTK_MAX_RESOLUTIONS)()
        n = self.RTKCam_GetResolutionList(self.handle, arr, RTK_MAX_RESOLUTIONS, stream) & 0xFFFF
        return [arr[i] for i in range(min(n, RTK_MAX_RESOLUTIONS)) if arr[i].valid and arr[i].width]

    def set_resolution(self, res: RTKCAM_RESOLUTION, stream: int = RTK_STREAM_CAPTURE) -> None:
        self._require_open()
        rc = self.RTKCam_SetResolution(self.handle, res, stream)
        if rc != 1:
            raise RTKCamError(f"RTKCam_SetResolution({res.width}x{res.height} {res.format_name}) failed rc={rc}")
        self._current_res = res

    def choose_resolution(self, width: int, height: int, prefer: Tuple[int, ...] = (RTK_FMT_MJPG, RTK_FMT_YUY2)) -> RTKCAM_RESOLUTION:
        """Pick the list entry closest to width x height, preferring MJPG then YUY2 and highest fps."""
        entries = self.get_resolution_list()
        if not entries:
            raise RTKCamError("camera reports no resolutions")

        def rank(e: RTKCAM_RESOLUTION):
            exact = 0 if (e.width == width and e.height == height) else 1
            area_diff = abs(e.width * e.height - width * height)
            fmt_rank = prefer.index(e.format) if e.format in prefer else len(prefer)
            return (exact, area_diff, fmt_rank, -e.fps)

        return sorted(entries, key=rank)[0]

    def capture_control(self, action: int) -> None:
        self._require_open()
        rc = self.RTKCam_CaptureControl(self.handle, action)
        if action in (RTK_CTRL_START_PREVIEW, RTK_CTRL_START_RECORD):
            if rc != 1:
                raise RTKCamError(f"RTKCam_CaptureControl({action}) failed rc={rc}")
            self._streaming = True
        elif action == RTK_CTRL_STOP:
            self._streaming = False

    def get_frame(self) -> Tuple[Optional[bytes], int]:
        """Block up to ~1 s for the next frame. Returns (raw_bytes, length) or (None, 0)."""
        self._require_open()
        length = ctypes.c_uint32(0)
        with self._lock:
            ptr = self.RTKCam_GetFrame(self.handle, 0, ctypes.byref(length))
            if not ptr or length.value == 0:
                return None, 0
            data = ctypes.string_at(ptr, length.value)   # copy: the DLL reuses its buffer
        return data, length.value

    def capture_photo(self, path: str, jpeg: bool = True) -> bool:
        self._require_open()
        p = RTKCAM_CAPTURE_PARAM()
        p.format = 1 if jpeg else 0
        p.path = path
        self.RTKCam_SetCaptureParam(self.handle, ctypes.byref(p))
        return bool(self.RTKCam_CapturePhoto())

    def video_prop(self, prop: int, group: str = "Video") -> Optional[RTKCAM_PROP]:
        """Read a VideoProcAmp ('Video') or CameraControl ('Camera') property incl. its range."""
        self._require_open()
        blk = RTKCAM_PROP()
        fn_range = getattr(self, f"RTKCam_{group}PropGetRange")
        fn_get = getattr(self, f"RTKCam_{group}PropGet")
        if not fn_range(self.handle, prop, ctypes.byref(blk)):
            return None
        fn_get(self.handle, prop, ctypes.byref(blk))
        return blk

    def set_video_prop(self, prop: int, value: int, auto: bool = False, group: str = "Video") -> bool:
        self._require_open()
        blk = RTKCAM_PROP()
        blk.value = int(value)
        blk.flags = PROP_FLAG_AUTO if auto else PROP_FLAG_MANUAL
        fn_set = getattr(self, f"RTKCam_{group}PropSet")
        return bool(fn_set(self.handle, prop, ctypes.byref(blk)))

    def free(self) -> None:
        try:
            self.close_device()
        finally:
            try:
                self.RTKCam_Free()
            except Exception:  # pragma: no cover
                pass

    def _require_open(self) -> None:
        if self._device is None:
            raise RTKCamError("no device open")

    # frame decoding -------------------------------------------------------
    @staticmethod
    def decode_frame(data: bytes, width: int, height: int, fmt: int = RTK_FMT_UNKNOWN) -> Optional[np.ndarray]:
        """Convert a raw sample-grabber buffer to a BGR image."""
        if cv2 is None or not data:
            return None
        n = len(data)
        arr = np.frombuffer(data, dtype=np.uint8)
        if fmt == RTK_FMT_MJPG or data[:2] == b"\xff\xd8":
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if width <= 0 or height <= 0:
            return None
        if n == width * height * 2:                      # YUY2
            return cv2.cvtColor(arr.reshape(height, width, 2), cv2.COLOR_YUV2BGR_YUY2)
        if n == width * height * 3:                      # RGB24 DIB (bottom-up)
            return np.ascontiguousarray(np.flipud(arr.reshape(height, width, 3)))
        if n == width * height * 4:                      # RGB32 DIB (bottom-up)
            return cv2.cvtColor(np.ascontiguousarray(np.flipud(arr.reshape(height, width, 4))), cv2.COLOR_BGRA2BGR)
        if n == width * height * 3 // 2:                 # NV12 / I420 (assume NV12)
            return cv2.cvtColor(arr.reshape(height * 3 // 2, width), cv2.COLOR_YUV2BGR_NV12)
        log.debug("unrecognised frame size %d for %dx%d", n, width, height)
        return None


# --------------------------------------------------------------------------- #
# Frame sources
# --------------------------------------------------------------------------- #
class CameraSource:
    """Common interface implemented by RTKCameraSource and OpenCVCameraSource."""
    name = "none"
    description = ""

    def open(self) -> None: ...
    def read(self) -> Optional[np.ndarray]: ...
    def close(self) -> None: ...
    def is_open(self) -> bool: return False
    def resolution(self) -> Tuple[int, int]: return (0, 0)


class RTKCameraSource(CameraSource):
    name = "rtk"

    def __init__(self, dll: str = "RTKCamSDK.dll", device_index: int = 0, width: int = 640, height: int = 480):
        self.dll, self.device_index, self.width, self.height = dll, device_index, width, height
        self.sdk: Optional[RTKCamSDK] = None
        self._res: Optional[RTKCAM_RESOLUTION] = None

    def open(self) -> None:
        self.sdk = RTKCamSDK(self.dll)
        self.sdk.open_device(self.device_index)
        try:
            self._res = self.sdk.choose_resolution(self.width, self.height)
            self.sdk.set_resolution(self._res)
        except RTKCamError as exc:
            log.warning("RTK set_resolution skipped: %s", exc)
        self.sdk.capture_control(RTK_CTRL_START_PREVIEW)
        dev = self.sdk._device
        res = self._res
        self.description = (f"RTKCam SDK – {dev.friendly_name} " if dev else "RTKCam SDK ") + \
                           (f"{res.width}x{res.height} {res.format_name}" if res else "")
        # wait for the first frame so failures surface immediately
        deadline = time.time() + 3.0
        while time.time() < deadline:
            if self.read() is not None:
                return
        raise RTKCamError("RTKCam stream started but no frame arrived within 3 s")

    def read(self) -> Optional[np.ndarray]:
        if not self.sdk:
            return None
        data, _ = self.sdk.get_frame()
        if data is None:
            return None
        w, h, fmt = (self._res.width, self._res.height, self._res.format) if self._res else (self.width, self.height, RTK_FMT_UNKNOWN)
        return RTKCamSDK.decode_frame(data, w, h, fmt)

    def close(self) -> None:
        if self.sdk:
            self.sdk.free()
            self.sdk = None

    def is_open(self) -> bool:
        return self.sdk is not None and self.sdk.is_open

    def resolution(self) -> Tuple[int, int]:
        return (self._res.width, self._res.height) if self._res else (self.width, self.height)


class OpenCVCameraSource(CameraSource):
    """cv2.VideoCapture based source. ``video_file`` turns it into a looping virtual camera."""
    name = "opencv"

    def __init__(self, device_index: int = 0, width: int = 640, height: int = 480, video_file: str = ""):
        self.device_index, self.width, self.height, self.video_file = device_index, width, height, video_file
        self.cap: Optional["cv2.VideoCapture"] = None
        self._is_file = bool(video_file)
        self._lock = threading.Lock()

    def _backends(self) -> List[Tuple[str, int]]:
        if self._is_file:
            return [("FFMPEG", cv2.CAP_FFMPEG), ("ANY", cv2.CAP_ANY)]
        major = int(cv2.__version__.split(".")[0])
        if sys.platform.startswith("win"):
            # OpenCV 5 can no longer open devices by index through DirectShow
            order = [("DSHOW", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF)] if major < 5 else [("MSMF", cv2.CAP_MSMF)]
            return order + [("ANY", cv2.CAP_ANY)]
        return [("ANY", cv2.CAP_ANY)]

    def open(self) -> None:
        if cv2 is None:
            raise RuntimeError("opencv-python is not installed")
        errors = []
        for label, backend in self._backends():
            cap = cv2.VideoCapture(self.video_file if self._is_file else int(self.device_index), backend)
            if not cap.isOpened():
                cap.release()
                errors.append(f"{label}: not opened")
                continue
            if not self._is_file:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            ok, frame = False, None
            deadline = time.time() + 4.0
            while time.time() < deadline and not ok:
                ok, frame = cap.read()
            if ok and frame is not None:
                self.cap = cap
                w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                src = os.path.basename(self.video_file) if self._is_file else f"device {self.device_index}"
                self.description = f"OpenCV/{label} – {src} {w}x{h}"
                log.info("camera opened: %s", self.description)
                return
            cap.release()
            errors.append(f"{label}: opened but no frames")
        raise RuntimeError("OpenCV could not open camera (" + "; ".join(errors) + ")")

    def read(self) -> Optional[np.ndarray]:
        with self._lock:
            if self.cap is None:
                return None
            ok, frame = self.cap.read()
            if not ok and self._is_file:              # loop the demo video
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self.cap.read()
            return frame if ok else None

    def close(self) -> None:
        with self._lock:
            if self.cap is not None:
                self.cap.release()
                self.cap = None

    def is_open(self) -> bool:
        return self.cap is not None and self.cap.isOpened()

    def resolution(self) -> Tuple[int, int]:
        if self.cap is None:
            return (self.width, self.height)
        return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


class NullCameraSource(CameraSource):
    """Used when no camera could be opened: produces a dark placeholder frame."""
    name = "none"

    def __init__(self, width: int = 640, height: int = 480, reason: str = ""):
        self.width, self.height = width, height
        self.description = f"no camera ({reason})" if reason else "no camera"
        self._open = False

    def open(self) -> None:
        self._open = True

    def read(self) -> Optional[np.ndarray]:
        time.sleep(0.05)
        frame = np.full((self.height, self.width, 3), 24, dtype=np.uint8)
        if cv2 is not None:
            cv2.putText(frame, "NO CAMERA", (self.width // 2 - 110, self.height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (90, 90, 200), 2, cv2.LINE_AA)
        return frame

    def close(self) -> None:
        self._open = False

    def is_open(self) -> bool:
        return self._open

    def resolution(self) -> Tuple[int, int]:
        return (self.width, self.height)


@dataclass
class CameraOpenResult:
    source: CameraSource
    backend: str                      # "rtk" | "opencv" | "none"
    description: str
    notes: List[str] = field(default_factory=list)


def open_camera(backend: str = "auto", device_index: int = 0, width: int = 640, height: int = 480,
                dll: str = "RTKCamSDK.dll", video_file: str = "") -> CameraOpenResult:
    """Apply the fallback policy: RTK SDK -> OpenCV -> placeholder.

    ``backend``: "auto" (default), "rtk" (SDK only, still falls back with a note) or "opencv".
    """
    notes: List[str] = []
    if video_file:
        src = OpenCVCameraSource(device_index, width, height, video_file=video_file)
        try:
            src.open()
            return CameraOpenResult(src, "opencv", src.description, ["virtual camera from video file"])
        except Exception as exc:
            notes.append(f"video file failed: {exc}")

    if backend in ("auto", "rtk"):
        rtk = RTKCameraSource(dll, device_index, width, height)
        try:
            rtk.open()
            return CameraOpenResult(rtk, "rtk", rtk.description, notes)
        except Exception as exc:
            rtk.close()
            notes.append(f"RTK SDK unavailable: {exc}")
            log.warning("RTK SDK unavailable, falling back to OpenCV: %s", exc)

    if backend in ("auto", "rtk", "opencv"):
        cv = OpenCVCameraSource(device_index, width, height)
        try:
            cv.open()
            return CameraOpenResult(cv, "opencv", cv.description, notes)
        except Exception as exc:
            cv.close()
            notes.append(f"OpenCV failed: {exc}")
            log.error("OpenCV camera failed: %s", exc)

    null = NullCameraSource(width, height, reason=notes[-1] if notes else "disabled")
    null.open()
    return CameraOpenResult(null, "none", null.description, notes)


# --------------------------------------------------------------------------- #
# Background grabber
# --------------------------------------------------------------------------- #
class CameraThread(threading.Thread):
    """Continuously reads frames so consumers can grab the newest one without blocking.

    ``get_latest()`` returns ``(frame_id, frame)``; ``frame_id`` increments for
    every new frame so consumers can skip duplicates.
    """

    def __init__(self, source: CameraSource, fps_limit: float = 30.0, mirror: bool = False, rotate: int = 0):
        super().__init__(name="CameraThread", daemon=True)
        self.source = source
        self.fps_limit = max(1.0, float(fps_limit or 30))
        self.mirror, self.rotate = mirror, int(rotate) % 360
        self._latest: Optional[np.ndarray] = None
        self._frame_id = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._callbacks: List[Callable[[np.ndarray], None]] = []
        self.fps = 0.0
        self.error: Optional[str] = None

    def add_callback(self, fn: Callable[[np.ndarray], None]) -> None:
        self._callbacks.append(fn)

    def run(self) -> None:
        interval = 1.0 / self.fps_limit
        fps_t0, fps_n, misses = time.time(), 0, 0
        while not self._stop.is_set():
            t0 = time.time()
            try:
                frame = self.source.read()
            except Exception as exc:  # keep the thread alive on transient errors
                self.error = str(exc)
                frame = None
            if frame is None:
                misses += 1
                if misses % 100 == 0:
                    log.warning("camera: %d consecutive empty reads", misses)
                time.sleep(0.02)
                continue
            misses = 0
            if self.rotate == 90:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif self.rotate == 180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            elif self.rotate == 270:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            if self.mirror:
                frame = cv2.flip(frame, 1)
            with self._lock:
                self._latest = frame
                self._frame_id += 1
            for cb in self._callbacks:
                try:
                    cb(frame)
                except Exception:  # pragma: no cover
                    log.exception("camera callback failed")
            fps_n += 1
            if time.time() - fps_t0 >= 1.0:
                self.fps = fps_n / (time.time() - fps_t0)
                fps_t0, fps_n = time.time(), 0
            dt = time.time() - t0
            if dt < interval:
                time.sleep(interval - dt)

    def get_latest(self) -> Tuple[int, Optional[np.ndarray]]:
        with self._lock:
            return self._frame_id, self._latest

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        if self.is_alive():
            self.join(timeout)
        try:
            self.source.close()
        except Exception:  # pragma: no cover
            pass


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #
def diagnose(dll: str = "RTKCamSDK.dll") -> str:
    """Human readable summary of what the camera layer can do on this machine."""
    lines = [f"Python: {sys.version.split()[0]} ({'32' if process_is_32bit() else '64'}-bit)"]
    paths = [p for p in _candidate_dll_paths(dll) if os.path.isfile(p)]
    if not paths:
        lines.append(f"RTKCamSDK.dll: NOT FOUND (searched {_candidate_dll_paths(dll)})")
    else:
        m = dll_machine(paths[0])
        arch = {0x14C: "x86 (32-bit)", 0x8664: "x64 (64-bit)"}.get(m, hex(m) if m else "?")
        lines.append(f"RTKCamSDK.dll: {paths[0]}  [{arch}]")
        try:
            sdk = RTKCamSDK(paths[0])
            devs = sdk.get_device_list()
            lines.append(f"RTKCam devices: {len(devs)} " + ", ".join(f"{d.friendly_name} [{d.vid:04X}:{d.pid:04X}]" for d in devs))
            sdk.free()
        except RTKCamError as exc:
            lines.append(f"RTKCam SDK unusable in this process: {exc}")
    lines.append(f"OpenCV: {cv2.__version__ if cv2 else 'missing'}")
    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    print(diagnose())
    result = open_camera("auto", 0, 640, 480, video_file=sys.argv[1] if len(sys.argv) > 1 else "")
    print("backend:", result.backend, "|", result.description)
    for n in result.notes:
        print("  note:", n)
    th = CameraThread(result.source)
    th.start()
    time.sleep(2.0)
    fid, frame = th.get_latest()
    print("frames:", fid, "shape:", None if frame is None else frame.shape, "fps: %.1f" % th.fps)
    th.stop()
