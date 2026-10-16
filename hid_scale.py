"""
hid_scale.py
------------
USB **HID** weighing-scale support for Windows, with no new dependency.

Why this exists: an integrated POS scale does not always speak RS-232.  Many of
them (and every Dymo/Mettler/Adam USB scale) present themselves as a standard
"HID Point of Sale - Scale" device, usage page 0x8D.  Windows then exposes them
as an HID device, not as a COM port, so a serial-only driver sees nothing at all
no matter which baud rate it tries.  That is a silent, total failure which looks
exactly like "the scale is connected but the app shows no weight".

The protocol is a short input report, defined by the USB HID Point of Sale
Usage Tables:

    byte 0   report id            (usually 3)
    byte 1   scale status         1 fault, 2 stable at zero, 3 in motion,
                                  4 stable, 5 under zero, 6 over weight,
                                  7 needs calibration, 8 needs re-zeroing
    byte 2   unit                 1 mg, 2 g, 3 kg, 11 ounce, 12 pound
    byte 3   exponent             signed power of ten applied to the value
    byte 4-5 weight               unsigned 16-bit, little endian

Everything here goes through ctypes against setupapi.dll / hid.dll, so there is
nothing extra for PyInstaller to bundle.  On a non-Windows machine every entry
point degrades to "no devices".
"""
from __future__ import annotations

import ctypes
import logging
import struct
import threading
import time
from ctypes import wintypes
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

log = logging.getLogger("hid_scale")

IS_WINDOWS = hasattr(ctypes, "windll")

# HID Point of Sale usage page and the weighing-device usage
USAGE_PAGE_SCALE = 0x8D
USAGE_WEIGHING_DEVICE = 0x01

STATUS_NAMES = {1: "fault", 2: "stable at zero", 3: "in motion", 4: "stable", 5: "under zero",
                6: "over weight", 7: "requires calibration", 8: "requires re-zeroing"}
STATUS_STABLE = (2, 4)
STATUS_FAULT = (1, 7, 8)
# HID unit code -> kilograms
UNIT_TO_KG = {1: 1e-6, 2: 1e-3, 3: 1.0, 11: 0.028349523125, 12: 0.45359237}
UNIT_NAMES = {1: "mg", 2: "g", 3: "kg", 11: "oz", 12: "lb"}


# --------------------------------------------------------------------------- #
# Win32 plumbing
# --------------------------------------------------------------------------- #
if IS_WINDOWS:
    setupapi = ctypes.WinDLL("setupapi")
    hid = ctypes.WinDLL("hid")
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    class GUID(ctypes.Structure):
        _fields_ = [("Data1", wintypes.DWORD), ("Data2", wintypes.WORD),
                    ("Data3", wintypes.WORD), ("Data4", ctypes.c_ubyte * 8)]

    class SP_DEVICE_INTERFACE_DATA(ctypes.Structure):
        _fields_ = [("cbSize", wintypes.DWORD), ("InterfaceClassGuid", GUID),
                    ("Flags", wintypes.DWORD), ("Reserved", ctypes.POINTER(ctypes.c_ulong))]

    class HIDD_ATTRIBUTES(ctypes.Structure):
        _fields_ = [("Size", wintypes.ULONG), ("VendorID", wintypes.USHORT),
                    ("ProductID", wintypes.USHORT), ("VersionNumber", wintypes.USHORT)]

    class HIDP_CAPS(ctypes.Structure):
        _fields_ = [("Usage", wintypes.USHORT), ("UsagePage", wintypes.USHORT),
                    ("InputReportByteLength", wintypes.USHORT), ("OutputReportByteLength", wintypes.USHORT),
                    ("FeatureReportByteLength", wintypes.USHORT), ("Reserved", wintypes.USHORT * 17),
                    ("NumberLinkCollectionNodes", wintypes.USHORT), ("NumberInputButtonCaps", wintypes.USHORT),
                    ("NumberInputValueCaps", wintypes.USHORT), ("NumberInputDataIndices", wintypes.USHORT),
                    ("NumberOutputButtonCaps", wintypes.USHORT), ("NumberOutputValueCaps", wintypes.USHORT),
                    ("NumberOutputDataIndices", wintypes.USHORT), ("NumberFeatureButtonCaps", wintypes.USHORT),
                    ("NumberFeatureValueCaps", wintypes.USHORT), ("NumberFeatureDataIndices", wintypes.USHORT)]

    class OVERLAPPED(ctypes.Structure):
        _fields_ = [("Internal", ctypes.POINTER(ctypes.c_ulong)), ("InternalHigh", ctypes.POINTER(ctypes.c_ulong)),
                    ("Offset", wintypes.DWORD), ("OffsetHigh", wintypes.DWORD), ("hEvent", wintypes.HANDLE)]

    DIGCF_PRESENT, DIGCF_DEVICEINTERFACE = 0x02, 0x10
    GENERIC_READ, GENERIC_WRITE = 0x80000000, 0x40000000
    FILE_SHARE_READ, FILE_SHARE_WRITE = 1, 2
    OPEN_EXISTING, FILE_FLAG_OVERLAPPED = 3, 0x40000000
    INVALID_HANDLE_VALUE = wintypes.HANDLE(-1).value
    WAIT_TIMEOUT, WAIT_OBJECT_0 = 0x102, 0
    HIDP_STATUS_SUCCESS = 0x00110000

    # Declaring these is not optional: ctypes defaults every return value to a
    # 32-bit int, which silently truncates every 64-bit HANDLE and makes the
    # whole enumeration return nothing on a 64-bit build.
    setupapi.SetupDiGetClassDevsW.restype = wintypes.HANDLE
    setupapi.SetupDiGetClassDevsW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR, wintypes.HWND, wintypes.DWORD]
    setupapi.SetupDiEnumDeviceInterfaces.restype = wintypes.BOOL
    setupapi.SetupDiEnumDeviceInterfaces.argtypes = [wintypes.HANDLE, ctypes.c_void_p, ctypes.c_void_p,
                                                     wintypes.DWORD, ctypes.c_void_p]
    setupapi.SetupDiGetDeviceInterfaceDetailW.restype = wintypes.BOOL
    setupapi.SetupDiGetDeviceInterfaceDetailW.argtypes = [wintypes.HANDLE, ctypes.c_void_p, ctypes.c_void_p,
                                                          wintypes.DWORD, ctypes.c_void_p, ctypes.c_void_p]
    setupapi.SetupDiDestroyDeviceInfoList.restype = wintypes.BOOL
    setupapi.SetupDiDestroyDeviceInfoList.argtypes = [wintypes.HANDLE]

    kernel32.CreateFileW.restype = wintypes.HANDLE
    kernel32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
                                     wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    kernel32.CreateEventW.restype = wintypes.HANDLE
    kernel32.CreateEventW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.BOOL, wintypes.LPCWSTR]
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.ReadFile.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD, ctypes.c_void_p, ctypes.c_void_p]
    kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel32.WaitForSingleObject.restype = wintypes.DWORD
    kernel32.GetOverlappedResult.argtypes = [wintypes.HANDLE, ctypes.c_void_p, ctypes.c_void_p, wintypes.BOOL]
    kernel32.CancelIoEx.argtypes = [wintypes.HANDLE, ctypes.c_void_p]
    kernel32.ResetEvent.argtypes = [wintypes.HANDLE]

    hid.HidD_GetHidGuid.argtypes = [ctypes.c_void_p]
    hid.HidD_GetAttributes.restype = wintypes.BOOLEAN
    hid.HidD_GetAttributes.argtypes = [wintypes.HANDLE, ctypes.c_void_p]
    hid.HidD_GetPreparsedData.restype = wintypes.BOOLEAN
    hid.HidD_GetPreparsedData.argtypes = [wintypes.HANDLE, ctypes.POINTER(ctypes.c_void_p)]
    hid.HidD_FreePreparsedData.restype = wintypes.BOOLEAN
    hid.HidD_FreePreparsedData.argtypes = [ctypes.c_void_p]
    hid.HidP_GetCaps.restype = ctypes.c_long
    hid.HidP_GetCaps.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    for _fn in (hid.HidD_GetProductString, hid.HidD_GetManufacturerString):
        _fn.restype = wintypes.BOOLEAN
        _fn.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.ULONG]


def _hid_guid() -> "GUID":
    g = GUID()
    hid.HidD_GetHidGuid(ctypes.byref(g))
    return g


@dataclass
class HidDeviceInfo:
    path: str
    vid: int
    pid: int
    usage_page: int
    usage: int
    input_len: int
    product: str = ""
    manufacturer: str = ""

    @property
    def is_scale(self) -> bool:
        return self.usage_page == USAGE_PAGE_SCALE

    def __str__(self) -> str:
        label = (self.product or "").strip() or "(no product string)"
        return (f"{label} [{self.vid:04X}:{self.pid:04X}] usage {self.usage_page:#04x}/{self.usage:#04x} "
                f"in={self.input_len}B")


def _device_string(handle, fn) -> str:
    buf = ctypes.create_unicode_buffer(256)
    try:
        if fn(handle, buf, ctypes.sizeof(buf)):
            return buf.value
    except Exception:
        pass
    return ""


def list_hid_devices(only_scales: bool = False) -> List[HidDeviceInfo]:
    """Every present HID device, or only those on the scale usage page."""
    if not IS_WINDOWS:
        return []
    out: List[HidDeviceInfo] = []
    guid = _hid_guid()
    dev_info = setupapi.SetupDiGetClassDevsW(ctypes.byref(guid), None, None,
                                             DIGCF_PRESENT | DIGCF_DEVICEINTERFACE)
    if dev_info == INVALID_HANDLE_VALUE:
        return out
    try:
        iface = SP_DEVICE_INTERFACE_DATA()
        iface.cbSize = ctypes.sizeof(SP_DEVICE_INTERFACE_DATA)
        index = 0
        while setupapi.SetupDiEnumDeviceInterfaces(dev_info, None, ctypes.byref(guid), index, ctypes.byref(iface)):
            index += 1
            need = wintypes.DWORD(0)
            setupapi.SetupDiGetDeviceInterfaceDetailW(dev_info, ctypes.byref(iface), None, 0,
                                                      ctypes.byref(need), None)
            if not need.value:
                continue
            buf = ctypes.create_string_buffer(need.value)
            # cbSize is the size of the fixed part only: 8 on 64-bit, 6 on 32-bit
            ctypes.memmove(buf, struct.pack("I", 8 if ctypes.sizeof(ctypes.c_void_p) == 8 else 6), 4)
            if not setupapi.SetupDiGetDeviceInterfaceDetailW(dev_info, ctypes.byref(iface), buf, need.value,
                                                             ctypes.byref(need), None):
                continue
            path = ctypes.wstring_at(ctypes.addressof(buf) + 4)
            info = _probe(path)
            if info is not None and (info.is_scale or not only_scales):
                out.append(info)
    finally:
        setupapi.SetupDiDestroyDeviceInfoList(dev_info)
    return out


def _probe(path: str) -> Optional[HidDeviceInfo]:
    """Open a device read-only just long enough to read its capabilities."""
    handle = kernel32.CreateFileW(path, 0, FILE_SHARE_READ | FILE_SHARE_WRITE, None, OPEN_EXISTING, 0, None)
    if handle == INVALID_HANDLE_VALUE:
        return None
    try:
        attrs = HIDD_ATTRIBUTES()
        attrs.Size = ctypes.sizeof(HIDD_ATTRIBUTES)
        if not hid.HidD_GetAttributes(handle, ctypes.byref(attrs)):
            return None
        pre = ctypes.c_void_p()
        if not hid.HidD_GetPreparsedData(handle, ctypes.byref(pre)):
            return None
        try:
            caps = HIDP_CAPS()
            if hid.HidP_GetCaps(pre, ctypes.byref(caps)) != HIDP_STATUS_SUCCESS:
                return None
            return HidDeviceInfo(path=path, vid=attrs.VendorID, pid=attrs.ProductID,
                                 usage_page=caps.UsagePage, usage=caps.Usage,
                                 input_len=caps.InputReportByteLength,
                                 product=_device_string(handle, hid.HidD_GetProductString),
                                 manufacturer=_device_string(handle, hid.HidD_GetManufacturerString))
        finally:
            hid.HidD_FreePreparsedData(pre)
    finally:
        kernel32.CloseHandle(handle)


def list_hid_scales() -> List[HidDeviceInfo]:
    return list_hid_devices(only_scales=True)


# --------------------------------------------------------------------------- #
# Report parsing (pure, so it is testable without hardware)
# --------------------------------------------------------------------------- #
@dataclass
class HidReading:
    weight_kg: float
    stable: bool
    overload: bool
    status: int
    unit: int
    raw: bytes

    @property
    def status_name(self) -> str:
        return STATUS_NAMES.get(self.status, f"status {self.status}")


def parse_hid_report(data: bytes) -> Optional[HidReading]:
    """Decode a HID Point of Sale scale input report.

    Tolerates a leading report id of 0 (some stacks strip it) by trying both
    alignments and keeping the one whose unit code is valid.
    """
    if not data or len(data) < 5:
        return None
    for offset in (1, 0):                       # with report id, then without
        if len(data) < offset + 5:
            continue
        status, unit, exponent = data[offset], data[offset + 1], data[offset + 2]
        if unit not in UNIT_TO_KG or status not in STATUS_NAMES:
            continue
        value = int.from_bytes(data[offset + 3:offset + 5], "little")
        exp = exponent - 256 if exponent > 127 else exponent        # signed byte
        try:
            weight = value * (10.0 ** exp) * UNIT_TO_KG[unit]
        except OverflowError:
            return None
        if status == 5:                          # under zero reads as a negative load
            weight = -weight
        return HidReading(weight_kg=weight, stable=status in STATUS_STABLE, overload=status == 6,
                          status=status, unit=unit, raw=bytes(data))
    return None


# --------------------------------------------------------------------------- #
# The reader
# --------------------------------------------------------------------------- #
class HidScaleReader(threading.Thread):
    """Reads a USB HID scale.  Mirrors ScaleReader's interface exactly, so the
    rest of the application cannot tell which kind of scale it is talking to."""

    def __init__(self, config: Dict):
        super().__init__(name="HidScaleReader", daemon=True)
        self.config = dict(config)
        self.path = str(config.get("hid_path") or "")
        self._stop_evt = threading.Event()
        self._lock = threading.Lock()
        self._latest = None
        self._subs: List[Callable] = []
        self._frame_subs: List[Callable] = []
        self.status = "disconnected"
        self.last_error = ""
        self.bytes_received = 0
        self.frames_received = 0
        self.readings = 0
        self.parse_failures = 0
        self.unstable_readings = 0
        self.last_data_time = 0.0
        self.device: Optional[HidDeviceInfo] = None

    # ---- the BaseScale interface
    @property
    def is_simulated(self) -> bool:
        return False

    def subscribe(self, fn) -> None:
        self._subs.append(fn)

    def subscribe_frames(self, fn) -> None:
        self._frame_subs.append(fn)

    def latest(self):
        with self._lock:
            return self._latest

    def health(self) -> Dict:
        now = time.time()
        return {"status": self.status, "error": self.last_error, "bytes": self.bytes_received,
                "frames": self.frames_received, "readings": self.readings,
                "parse_failures": self.parse_failures, "unstable": self.unstable_readings,
                "data_age": (now - self.last_data_time) if self.last_data_time else None,
                "port": self.device.product if self.device else self.path[:40], "baudrate": "USB HID"}

    def stop(self, timeout: float = 2.0) -> None:
        self._stop_evt.set()
        if self.is_alive():
            self.join(timeout)

    # ---- the thread
    def _resolve(self) -> Optional[HidDeviceInfo]:
        scales = list_hid_scales()
        if self.path:
            for s in scales:
                if s.path == self.path:
                    return s
            # the path changes when the device is replugged into another port
            for s in list_hid_devices():
                if s.path == self.path:
                    return s
        return scales[0] if scales else None

    def run(self) -> None:
        if not IS_WINDOWS:
            self.status, self.last_error = "error", "HID scales are only supported on Windows"
            return
        reconnect = float(self.config.get("reconnect_s", 3.0))
        while not self._stop_evt.is_set():
            self.status = "connecting"
            device = self._resolve()
            if device is None:
                self.status, self.last_error = "error", "no HID scale found"
                self._stop_evt.wait(reconnect)
                continue
            self.device = device
            handle = kernel32.CreateFileW(device.path, GENERIC_READ,
                                          FILE_SHARE_READ | FILE_SHARE_WRITE, None, OPEN_EXISTING,
                                          FILE_FLAG_OVERLAPPED, None)
            if handle == INVALID_HANDLE_VALUE:
                err = ctypes.get_last_error()
                self.status = "error"
                self.last_error = f"cannot open the HID device (error {err}; another program may hold it)"
                self._stop_evt.wait(reconnect)
                continue
            self.status, self.last_error = "connected", ""
            log.info("HID scale opened: %s", device)
            try:
                self._pump(handle, max(8, device.input_len or 8))
            finally:
                kernel32.CloseHandle(handle)
        self.status = "disconnected"

    def _pump(self, handle, length: int) -> None:
        event = kernel32.CreateEventW(None, True, False, None)
        buf = ctypes.create_string_buffer(length)
        read = wintypes.DWORD(0)
        try:
            while not self._stop_evt.is_set():
                ov = OVERLAPPED()
                ov.hEvent = event
                kernel32.ResetEvent(event)
                ok = kernel32.ReadFile(handle, buf, length, ctypes.byref(read), ctypes.byref(ov))
                if not ok:
                    err = ctypes.get_last_error()
                    if err != 997:                       # ERROR_IO_PENDING
                        self.status, self.last_error = "error", f"read failed (error {err})"
                        return
                    if kernel32.WaitForSingleObject(event, 500) != WAIT_OBJECT_0:
                        kernel32.CancelIoEx(handle, ctypes.byref(ov))
                        kernel32.WaitForSingleObject(event, 200)
                        continue                          # nothing arrived; a quiet scale is normal
                    if not kernel32.GetOverlappedResult(handle, ctypes.byref(ov), ctypes.byref(read), False):
                        continue
                n = int(read.value)
                if n <= 0:
                    continue
                self._handle_report(bytes(buf.raw[:n]))
        finally:
            kernel32.CloseHandle(event)

    def _handle_report(self, data: bytes) -> None:
        from scale_driver import WeightReading          # local import avoids a cycle
        self.bytes_received += len(data)
        self.frames_received += 1
        self.last_data_time = time.time()
        parsed = parse_hid_report(data)
        reading = None
        if parsed is not None:
            reading = WeightReading(weight_kg=parsed.weight_kg, stable=parsed.stable,
                                    overload=parsed.overload, unit="kg",
                                    raw=f"HID {parsed.status_name} {parsed.weight_kg:.3f} kg")
        for cb in self._frame_subs:
            try:
                cb(data, reading)
            except Exception:
                pass
        if reading is None:
            self.parse_failures += 1
            return
        self.readings += 1
        if not reading.stable:
            self.unstable_readings += 1
        with self._lock:
            self._latest = reading
        for cb in self._subs:
            try:
                cb(reading)
            except Exception:
                log.exception("HID scale subscriber failed")


def test_hid_scale(path: str = "", seconds: float = 3.0) -> Dict:
    """Listen to one HID scale and report what arrived."""
    reader = HidScaleReader({"hid_path": path})
    frames: List[str] = []
    reader.subscribe_frames(lambda data, r: frames.append(
        " ".join(f"{b:02X}" for b in data) + (f"  -> {r.weight_kg:.3f} kg" if r else "  -> (not a scale report)")))
    reader.start()
    deadline = time.time() + seconds
    while time.time() < deadline and reader.latest() is None:
        time.sleep(0.05)
    latest, health = reader.latest(), reader.health()
    reader.stop()
    if latest is not None:
        return {"ok": True, "message": f"{latest.weight_kg:.3f} kg", "raw": frames[:6], "reading": latest}
    if health["status"] == "error":
        return {"ok": False, "message": health["error"], "raw": frames[:6], "reading": None}
    if frames:
        return {"ok": False, "message": "the device sends reports that are not scale reports",
                "raw": frames[:6], "reading": None}
    return {"ok": False, "message": "the HID device is open but sent nothing (put a load on the pan and retry)",
            "raw": [], "reading": None}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    samples = [
        (bytes([3, 4, 2, 255, 0xE8, 0x03]), 1.0),        # 1000 g with exponent -1 -> 100.0 g? see below
        (bytes([3, 4, 3, 0, 0x02, 0x00]), 2.0),          # 2 kg, stable
        (bytes([3, 4, 2, 0, 0xF4, 0x01]), 0.5),          # 500 g, stable
        (bytes([3, 3, 2, 0, 0x64, 0x00]), 0.1),          # 100 g, in motion
        (bytes([3, 6, 2, 0, 0x00, 0x00]), 0.0),          # over weight
    ]
    print("report parsing:")
    for data, _ in samples:
        r = parse_hid_report(data)
        print(f"  {' '.join(f'{b:02X}' for b in data)} -> "
              + (f"{r.weight_kg:.4f} kg  {r.status_name:<16} stable={r.stable} overload={r.overload}" if r else "unparsed"))
    print("\nHID devices on this machine:")
    devices = list_hid_devices()
    if not devices:
        print("  none (or not Windows)")
    for d in devices:
        print(f"  {'SCALE ' if d.is_scale else '      '}{d}")
    print(f"\nscale-class devices: {len(list_hid_scales())}")
