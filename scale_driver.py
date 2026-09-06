"""
scale_driver.py
---------------
RS-232 weighing-scale reader for the AI POS Scale.

* ``ScaleReader``     – background thread reading a serial port (pyserial),
                        splitting frames (CR/LF/ETX, STX or idle gap), parsing
                        weights, counting bytes/frames/readings for diagnostics
                        and reconnecting automatically.
* ``SimulatedScale``  – drop-in replacement driven by a slider in the UI.
* ``parse_weight()``  – protocol-agnostic parser for the common indicator
                        formats, e.g.::

        ST,GS,+  1.234kg\\r\\n      (CAS / A&D / Ohaus style continuous output)
        US,NT,   0.000kg
        +001.234 kg
           1.234 kg
        W  1.234
        \\x02  1.234\\r            (STX framed)
        001234 (implied decimals = 3)

* ``diagnose_scale()`` – opens the port, listens, and if nothing usable arrives
                        tries the other baud rates / parities and the common
                        request commands.  Produces a text report the operator
                        can paste into a support request.

Every reading is normalised to kilograms.
"""
from __future__ import annotations

import logging
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, List, Optional, Tuple

try:
    import serial
    import serial.tools.list_ports
except ImportError:  # pragma: no cover
    serial = None  # type: ignore

log = logging.getLogger("scale")

_NUMBER_RE = re.compile(r"[-+]?\d+(?:[.,]\d+)?|[-+]?[.,]\d+")
_UNIT_RE = re.compile(r"\b(kg|kgs|g|gr|lb|lbs|oz)\b", re.IGNORECASE)
_UNIT_FACTORS = {"kg": 1.0, "kgs": 1.0, "g": 0.001, "gr": 0.001, "lb": 0.45359237, "lbs": 0.45359237, "oz": 0.028349523}
_STABLE_TOKENS = ("ST", "STABLE")
_UNSTABLE_TOKENS = ("US", "UNSTABLE", "MOTION")
_OVERLOAD_TOKENS = ("OL", "OVER", "ERR")

PARITY_MAP = {"N": "N", "E": "E", "O": "O", "M": "M", "S": "S",
              "NONE": "N", "EVEN": "E", "ODD": "O", "MARK": "M", "SPACE": "S"}
BAUDRATES = [1200, 2400, 4800, 9600, 19200, 38400, 57600, 115200]
# request commands used by common indicators (no zero / tare commands here!)
POLL_CANDIDATES = [("W\\r\\n", "W<CR><LF>"), ("\\x05", "ENQ (0x05)"), ("P\\r\\n", "P<CR><LF>"), ("S\\r\\n", "S<CR><LF>"),
                   ("R\\r\\n", "R<CR><LF>"), ("SI\\r\\n", "SI<CR><LF> (Mettler)"), ("Q\\r\\n", "Q<CR><LF>"),
                   ("\\x1bP", "ESC P"), ("RW\\r\\n", "RW<CR><LF>")]


@dataclass
class WeightReading:
    weight_kg: float
    stable: bool = True
    overload: bool = False
    raw: str = ""
    unit: str = "kg"
    timestamp: float = field(default_factory=time.time)

    @property
    def age(self) -> float:
        return time.time() - self.timestamp


def parse_weight(frame: bytes | str, default_unit: str = "auto", implied_decimals: int = 0,
                 number_index: int = 0) -> Optional[WeightReading]:
    """Parse one frame (line) from a scale into a ``WeightReading`` (kg) or None."""
    if isinstance(frame, (bytes, bytearray)):
        text = bytes(frame).decode("latin-1", errors="replace")
    else:
        text = str(frame)
    raw = text
    text = "".join(ch for ch in text if 32 <= ord(ch) < 127)      # strip STX/ETX/CR/LF ...
    if not text.strip():
        return None
    upper = text.upper()
    tokens = re.split(r"[,\s;]+", upper)
    stable = not any(tok in _UNSTABLE_TOKENS for tok in tokens)
    if any(tok in _STABLE_TOKENS for tok in tokens):
        stable = True
    overload = any(tok in _OVERLOAD_TOKENS for tok in tokens)

    cleaned = re.sub(r"\b(ST|US|GS|NT|TR|OL|WT|W|N|G|T|B)\b", " ", upper)
    numbers = _NUMBER_RE.findall(cleaned)
    if not numbers:
        if overload:
            return WeightReading(0.0, stable=False, overload=True, raw=raw.strip())
        return None
    idx = number_index if 0 <= number_index < len(numbers) else 0
    num_text = numbers[idx].replace(",", ".")
    try:
        value = float(num_text)
    except ValueError:
        return None
    if implied_decimals > 0 and "." not in num_text:
        value = value / (10 ** implied_decimals)

    unit_match = _UNIT_RE.search(upper)
    unit = unit_match.group(1).lower() if unit_match else ""
    if default_unit and default_unit != "auto" and not unit:
        unit = default_unit.lower()
    if not unit:
        unit = "kg"
    factor = _UNIT_FACTORS.get(unit, 1.0)
    return WeightReading(weight_kg=value * factor, stable=stable, overload=overload, raw=raw.strip(), unit=unit)


def decode_escapes(text: str) -> bytes:
    r"""Turn a user typed command like ``W\r\n`` or ``\x05`` into bytes."""
    if not text:
        return b""
    try:
        return text.encode("latin-1", errors="ignore").decode("unicode_escape").encode("latin-1", errors="ignore")
    except Exception:
        return text.encode("latin-1", errors="ignore")


def hexdump(data: bytes, limit: int = 32) -> str:
    part = bytes(data[:limit])
    text = " ".join(f"{b:02X}" for b in part)
    return text + (" ..." if len(data) > limit else "")


def printable(data: bytes, limit: int = 64) -> str:
    out = []
    for b in bytes(data[:limit]):
        if 32 <= b < 127:
            out.append(chr(b))
        elif b == 2:
            out.append("<STX>")
        elif b == 3:
            out.append("<ETX>")
        elif b == 13:
            out.append("<CR>")
        elif b == 10:
            out.append("<LF>")
        else:
            out.append(f"<{b:02X}>")
    return "".join(out) + (" ..." if len(data) > limit else "")


def list_serial_ports() -> List[Tuple[str, str]]:
    """[(device, description), ...] sorted by COM number."""
    if serial is None:
        return []
    ports = [(p.device, p.description or "") for p in serial.tools.list_ports.comports()]

    def key(item):
        m = re.search(r"(\d+)", item[0])
        return int(m.group(1)) if m else 0

    return sorted(ports, key=key)


class FrameSplitter:
    """Splits the incoming byte stream into protocol frames.

    Frames end with CR / LF / ETX; an STX starts a new frame.  ``flush()`` is
    called by the reader when the line has been idle for a moment, which frames
    protocols that have no terminator at all.  A safety limit flushes over-long
    buffers.
    """
    TERMINATORS = b"\r\n\x03"
    STX = 0x02
    MAX_LEN = 256

    def __init__(self):
        self.buffer = bytearray()

    def feed(self, data: bytes) -> List[bytes]:
        frames: List[bytes] = []
        for b in data:
            if b in self.TERMINATORS or b == self.STX:
                if self.buffer:
                    frames.append(bytes(self.buffer))
                self.buffer.clear()
            else:
                self.buffer.append(b)
                if len(self.buffer) >= self.MAX_LEN:
                    frames.append(bytes(self.buffer))
                    self.buffer.clear()
        return frames

    def flush(self) -> Optional[bytes]:
        if self.buffer:
            frame = bytes(self.buffer)
            self.buffer.clear()
            return frame
        return None


class BaseScale:
    """Interface shared by the hardware reader and the simulator."""
    status: str = "disconnected"

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def latest(self) -> Optional[WeightReading]: return None
    def subscribe(self, fn: Callable[[WeightReading], None]) -> None: ...
    def health(self) -> Dict: return {"status": self.status}
    @property
    def is_simulated(self) -> bool: return False


def open_serial(cfg: Dict, timeout: float = 0.2):
    """Open a pyserial port from a scale/printer config dict."""
    if serial is None:
        raise RuntimeError("pyserial is not installed")
    port = cfg.get("port") or ""
    if not port:
        raise RuntimeError("no COM port configured")
    bytesize = {5: serial.FIVEBITS, 6: serial.SIXBITS, 7: serial.SEVENBITS, 8: serial.EIGHTBITS}[int(cfg.get("bytesize", 8))]
    parity = PARITY_MAP.get(str(cfg.get("parity", "N")).upper(), "N")
    stop_val = float(cfg.get("stopbits", 1))
    stopbits = {1.0: serial.STOPBITS_ONE, 1.5: serial.STOPBITS_ONE_POINT_FIVE, 2.0: serial.STOPBITS_TWO}.get(stop_val, serial.STOPBITS_ONE)
    ser = serial.Serial(port=port, baudrate=int(cfg.get("baudrate", 9600)), bytesize=bytesize, parity=parity,
                        stopbits=stopbits, timeout=timeout, write_timeout=1.0)
    try:
        # many indicators only transmit when the PC asserts DTR / RTS
        ser.dtr = bool(cfg.get("assert_dtr", True))
        ser.rts = bool(cfg.get("assert_rts", True))
        ser.reset_input_buffer()
    except Exception:
        pass
    return ser


class ScaleReader(threading.Thread, BaseScale):
    """Background serial reader with diagnostics counters.

    ``config`` keys (see settings_manager.DEFAULTS["scale"]): port, baudrate,
    bytesize, parity, stopbits, poll_command, poll_interval_ms, unit,
    implied_decimals, number_index, stable_only, idle_gap_ms, reconnect_s.
    """

    def __init__(self, config: Dict):
        threading.Thread.__init__(self, name="ScaleReader", daemon=True)
        self.config = dict(config)
        self._stop_evt = threading.Event()
        self._lock = threading.Lock()
        self._latest: Optional[WeightReading] = None
        self._subs: List[Callable[[WeightReading], None]] = []
        self._raw_subs: List[Callable[[bytes], None]] = []
        self._frame_subs: List[Callable[[bytes, Optional[WeightReading]], None]] = []
        self.status = "disconnected"          # disconnected | connecting | connected | error
        self.last_error: str = ""
        self.bytes_received = 0
        self.frames_received = 0
        self.readings = 0
        self.parse_failures = 0
        self.last_data_time = 0.0
        self.recent_frames: Deque[Tuple[float, bytes, Optional[WeightReading]]] = deque(maxlen=40)
        self._ser = None

    # ----------------------------------------------------------- public
    def subscribe(self, fn: Callable[[WeightReading], None]) -> None:
        self._subs.append(fn)

    def subscribe_raw(self, fn: Callable[[bytes], None]) -> None:
        self._raw_subs.append(fn)

    def subscribe_frames(self, fn: Callable[[bytes, Optional[WeightReading]], None]) -> None:
        self._frame_subs.append(fn)

    def latest(self) -> Optional[WeightReading]:
        with self._lock:
            return self._latest

    def health(self) -> Dict:
        now = time.time()
        return {"status": self.status, "error": self.last_error, "bytes": self.bytes_received,
                "frames": self.frames_received, "readings": self.readings, "parse_failures": self.parse_failures,
                "data_age": (now - self.last_data_time) if self.last_data_time else None,
                "port": self.config.get("port", ""), "baudrate": self.config.get("baudrate", 9600)}

    def stop(self, timeout: float = 2.0) -> None:
        self._stop_evt.set()
        if self.is_alive():
            self.join(timeout)
        self._close()

    # ---------------------------------------------------------- internals
    def _close(self):
        if self._ser is not None:
            try:
                self._ser.close()
            except Exception:
                pass
            self._ser = None

    def run(self) -> None:
        cfg = self.config
        splitter = FrameSplitter()
        poll_cmd = decode_escapes(str(cfg.get("poll_command", "") or ""))
        poll_interval = max(0.05, float(cfg.get("poll_interval_ms", 300)) / 1000.0)
        idle_gap = max(0.01, float(cfg.get("idle_gap_ms", 60)) / 1000.0)
        reconnect_s = float(cfg.get("reconnect_s", 3.0))
        last_poll = 0.0
        last_byte_t = 0.0
        while not self._stop_evt.is_set():
            if self._ser is None:
                self.status = "connecting"
                try:
                    self._ser = open_serial(cfg, timeout=0.05)
                    self.status = "connected"
                    self.last_error = ""
                    splitter = FrameSplitter()
                    log.info("scale connected on %s @ %s", cfg.get("port"), cfg.get("baudrate"))
                except Exception as exc:
                    self.status = "error"
                    self.last_error = str(exc)
                    self._stop_evt.wait(reconnect_s)
                    continue
            try:
                if poll_cmd and time.time() - last_poll >= poll_interval:
                    self._ser.write(poll_cmd)
                    last_poll = time.time()
                waiting = self._ser.in_waiting
                data = self._ser.read(waiting if waiting > 0 else 1)
                if data:
                    self.bytes_received += len(data)
                    self.last_data_time = last_byte_t = time.time()
                    for cb in self._raw_subs:
                        try:
                            cb(data)
                        except Exception:
                            pass
                    for frame in splitter.feed(data):
                        self._handle_frame(frame)
                elif splitter.buffer and last_byte_t and time.time() - last_byte_t > idle_gap:
                    frame = splitter.flush()          # terminator-less protocol: idle gap ends the frame
                    if frame:
                        self._handle_frame(frame)
            except Exception as exc:
                log.warning("scale read error: %s", exc)
                self.status = "error"
                self.last_error = str(exc)
                self._close()
                self._stop_evt.wait(reconnect_s)
        self._close()
        self.status = "disconnected"

    def _handle_frame(self, frame: bytes) -> None:
        cfg = self.config
        self.frames_received += 1
        reading = parse_weight(frame, cfg.get("unit", "auto"), int(cfg.get("implied_decimals", 0)), int(cfg.get("number_index", 0)))
        self.recent_frames.append((time.time(), frame, reading))
        for cb in self._frame_subs:
            try:
                cb(frame, reading)
            except Exception:
                pass
        if reading is None:
            self.parse_failures += 1
            return
        self.readings += 1
        if cfg.get("stable_only") and not reading.stable:
            return
        with self._lock:
            self._latest = reading
        for cb in self._subs:
            try:
                cb(reading)
            except Exception:  # pragma: no cover
                log.exception("scale subscriber failed")


class SimulatedScale(BaseScale):
    """Software scale for development / demo. The UI sets the weight with a slider."""

    def __init__(self, initial_kg: float = 0.0):
        self._weight = float(initial_kg)
        self._subs: List[Callable[[WeightReading], None]] = []
        self.status = "simulated"
        self._lock = threading.Lock()

    @property
    def is_simulated(self) -> bool:
        return True

    def start(self) -> None:
        self.status = "simulated"

    def stop(self) -> None:
        pass

    def subscribe(self, fn: Callable[[WeightReading], None]) -> None:
        self._subs.append(fn)

    def set_weight(self, kg: float) -> None:
        with self._lock:
            self._weight = max(0.0, float(kg))
        reading = self.latest()
        for cb in self._subs:
            try:
                cb(reading)
            except Exception:  # pragma: no cover
                pass

    def latest(self) -> Optional[WeightReading]:
        with self._lock:
            return WeightReading(self._weight, stable=True, raw=f"SIM {self._weight:.3f} kg")

    def health(self) -> Dict:
        return {"status": "simulated"}


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #
def _listen(cfg: Dict, seconds: float, poll: bytes = b"", poll_every: float = 0.5) -> Tuple[bytes, List[bytes], List[WeightReading], str]:
    """Open the port with ``cfg``, optionally send ``poll`` repeatedly, collect bytes for ``seconds``."""
    raw = bytearray()
    frames: List[bytes] = []
    readings: List[WeightReading] = []
    splitter = FrameSplitter()
    try:
        ser = open_serial(cfg, timeout=0.05)
    except Exception as exc:
        return b"", [], [], str(exc)
    try:
        deadline = time.time() + seconds
        last_poll, last_byte = 0.0, 0.0
        while time.time() < deadline:
            if poll and time.time() - last_poll >= poll_every:
                ser.write(poll)
                last_poll = time.time()
            waiting = ser.in_waiting
            data = ser.read(waiting if waiting > 0 else 1)
            new_frames: List[bytes] = []
            if data:
                raw += data
                last_byte = time.time()
                new_frames = splitter.feed(data)
            elif splitter.buffer and last_byte and time.time() - last_byte > 0.06:
                f = splitter.flush()
                new_frames = [f] if f else []
            for f in new_frames:
                frames.append(f)
                r = parse_weight(f, cfg.get("unit", "auto"), int(cfg.get("implied_decimals", 0)), int(cfg.get("number_index", 0)))
                if r:
                    readings.append(r)
            if len(raw) > 4096:
                break
        f = splitter.flush()
        if f:
            frames.append(f)
    finally:
        try:
            ser.close()
        except Exception:
            pass
    return bytes(raw), frames, readings, ""


def _looks_like_text(data: bytes) -> bool:
    if not data:
        return False
    ok = sum(1 for b in data if 32 <= b < 127 or b in (2, 3, 10, 13))
    return ok / len(data) > 0.85 and any(48 <= b <= 57 for b in data)


def test_connection(config: Dict, seconds: float = 2.5) -> Dict:
    """Quick check used by the self-test: {"ok", "message", "raw", "reading"}."""
    raw, frames, readings, err = _listen(config, seconds, decode_escapes(str(config.get("poll_command", "") or "")),
                                         max(0.1, float(config.get("poll_interval_ms", 300)) / 1000.0))
    if err:
        return {"ok": False, "message": err, "raw": [], "reading": None}
    samples = [printable(f) for f in frames[:6]]
    if readings:
        r = readings[-1]
        return {"ok": True, "message": f"{r.weight_kg:.3f} kg", "raw": samples, "reading": r}
    if raw:
        return {"ok": False, "message": "data received but no weight could be parsed: " + " | ".join(samples[:3]), "raw": samples, "reading": None}
    return {"ok": False, "message": "port opened but no data received (check baud rate, cable, poll command)", "raw": [], "reading": None}


def diagnose_scale(config: Dict, progress: Optional[Callable[[str], None]] = None, quick: bool = False) -> Dict:
    """Full diagnosis. Returns {"ok", "summary", "report", "suggestion": dict|None}.

    Steps: 1) configured settings  2) other baud rates (8N1)  3) 7E1 / 7O1 at a
    baud that produced garbage  4) common request commands.
    """
    lines: List[str] = []

    def say(text: str):
        lines.append(text)
        if progress:
            progress(text)

    cfg = dict(config)
    ports = list_serial_ports()
    say("Ports: " + (", ".join(f"{d} ({desc})" for d, desc in ports) if ports else "none found"))
    port = cfg.get("port") or ""
    if not port:
        say("No COM port selected.")
        return {"ok": False, "summary": "no port selected", "report": "\n".join(lines), "suggestion": None}
    if ports and port not in [d for d, _ in ports]:
        say(f"WARNING: {port} is not in the list of present ports.")

    say(f"[1] Listening on {port} @ {cfg.get('baudrate')} {cfg.get('bytesize', 8)}{cfg.get('parity', 'N')}{cfg.get('stopbits', 1)} "
        f"poll={cfg.get('poll_command') or '-'} for 2.5 s ...")
    raw, frames, readings, err = _listen(cfg, 2.5, decode_escapes(str(cfg.get("poll_command", "") or "")),
                                         max(0.1, float(cfg.get("poll_interval_ms", 300)) / 1000.0))
    if err:
        say(f"    cannot open port: {err}")
        say("    -> Is another program (or a second copy of this app) using the port? Is the cable/adapter present?")
        return {"ok": False, "summary": f"cannot open {port}: {err}", "report": "\n".join(lines), "suggestion": None}
    say(f"    {len(raw)} bytes, {len(frames)} frames, {len(readings)} weight readings")
    for f in frames[:5]:
        say(f"    frame: {printable(f)}    hex: {hexdump(f)}")
    if readings:
        r = readings[-1]
        say(f"    OK: last weight {r.weight_kg:.3f} kg (stable={r.stable}) raw='{r.raw}'")
        return {"ok": True, "summary": f"{r.weight_kg:.3f} kg", "report": "\n".join(lines), "suggestion": None}
    if raw:
        if _looks_like_text(raw):
            say("    Text data arrives but no number could be extracted. Send this report to support; the protocol needs a parser rule.")
            return {"ok": False, "summary": "data without recognisable weight", "report": "\n".join(lines), "suggestion": None}
        say("    Bytes arrive but look like garbage -> wrong baud rate / parity / data bits. Trying alternatives ...")
    else:
        say("    Nothing received." + ("" if quick else " Trying other baud rates ..."))
    if quick:
        return {"ok": False, "summary": "no usable data", "report": "\n".join(lines), "suggestion": None}

    # [2] baud sweep at 8N1
    say("[2] Baud rate sweep (8N1, 1.2 s each)")
    best: Optional[Dict] = None
    garbage_bauds: List[int] = []
    for baud in BAUDRATES:
        trial = dict(cfg, baudrate=baud, bytesize=8, parity="N", stopbits=1)
        if baud == int(cfg.get("baudrate", 0)) and int(cfg.get("bytesize", 8)) == 8 and str(cfg.get("parity", "N")).upper().startswith("N"):
            continue
        raw2, frames2, readings2, err2 = _listen(trial, 1.2, decode_escapes(str(cfg.get("poll_command", "") or "")))
        if err2:
            say(f"    {baud}: {err2}")
            continue
        status = f"{len(raw2)} bytes, {len(readings2)} readings"
        if readings2:
            status += f"  -> WEIGHT {readings2[-1].weight_kg:.3f} kg  '{readings2[-1].raw}'"
            if best is None:
                best = {"baudrate": baud, "bytesize": 8, "parity": "N", "stopbits": 1}
        elif raw2:
            status += "  (" + ("text: " + printable(raw2, 30) if _looks_like_text(raw2) else "garbage: " + hexdump(raw2, 12)) + ")"
            if not _looks_like_text(raw2):
                garbage_bauds.append(baud)
        say(f"    {baud}: {status}")
    if best:
        say(f"    -> Suggested settings: {best['baudrate']} 8N1")
        return {"ok": True, "summary": f"use {best['baudrate']} baud 8N1", "report": "\n".join(lines), "suggestion": best}

    # [3] parity variants where bytes arrived
    if garbage_bauds or raw:
        bauds = garbage_bauds or [int(cfg.get("baudrate", 9600))]
        say("[3] Parity / data-bit variants: " + ", ".join(str(b) for b in bauds[:3]))
        for baud in bauds[:3]:
            for bits, par in ((7, "E"), (7, "O"), (8, "E"), (8, "O")):
                trial = dict(cfg, baudrate=baud, bytesize=bits, parity=par, stopbits=1)
                raw3, frames3, readings3, err3 = _listen(trial, 1.2, decode_escapes(str(cfg.get("poll_command", "") or "")))
                status = f"{len(raw3)} bytes, {len(readings3)} readings"
                if readings3:
                    say(f"    {baud} {bits}{par}1: {status} -> WEIGHT {readings3[-1].weight_kg:.3f} kg")
                    sug = {"baudrate": baud, "bytesize": bits, "parity": par, "stopbits": 1}
                    return {"ok": True, "summary": f"use {baud} {bits}{par}1", "report": "\n".join(lines), "suggestion": sug}
                say(f"    {baud} {bits}{par}1: {status}" + (f" text: {printable(raw3, 24)}" if raw3 and _looks_like_text(raw3) else ""))

    # [4] request commands (the scale may only answer when asked)
    if not raw:
        say("[4] Request commands at configured settings (0.9 s each)")
        for cmd_esc, label in POLL_CANDIDATES:
            raw4, frames4, readings4, err4 = _listen(cfg, 0.9, decode_escapes(cmd_esc), 0.3)
            if err4:
                say(f"    {label}: {err4}")
                break
            status = f"{len(raw4)} bytes, {len(readings4)} readings"
            if readings4:
                say(f"    {label}: {status} -> WEIGHT {readings4[-1].weight_kg:.3f} kg  '{readings4[-1].raw}'")
                say(f"    -> Suggested poll command: {cmd_esc}")
                return {"ok": True, "summary": f"use poll command {cmd_esc}", "report": "\n".join(lines),
                        "suggestion": {"poll_command": cmd_esc}}
            say(f"    {label}: {status}" + (f"  {printable(raw4, 24)}" if raw4 else ""))
        say("    No response to any request command.")
        say("    -> Check: scale in 'continuous'/'stream' output mode? RX/TX crossed (null-modem) cable? correct COM port?")
    return {"ok": False, "summary": "no usable data on any setting", "report": "\n".join(lines), "suggestion": None}


if __name__ == "__main__":
    samples = [b"ST,GS,+  1.234kg\r\n", b"US,GS,   0.000kg\r\n", b"+001.234 kg\r", b"   2.500 kg\r\n",
               b"W  0.750\r\n", b"\x02  1.234\r", b"001234", b"ST,NT,  12.5 g\r\n", b"OL\r\n", b"1,250 kg"]
    for s in samples:
        r = parse_weight(s, implied_decimals=3 if s == b"001234" else 0)
        print(f"{s!r:32} -> {r.weight_kg if r else None} kg  stable={r.stable if r else None}")
    sp = FrameSplitter()
    print("idle-gap framing:", sp.feed(b"  1.234"), sp.flush())
    print("ports:", list_serial_ports())
