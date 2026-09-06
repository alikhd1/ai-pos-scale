"""
printer_driver.py
-----------------
Thermal receipt printing (ESC/POS) for the AI POS Scale.

Transports
    * ``SerialTransport``          – COM port (pyserial)
    * ``WindowsPrinterTransport``  – any printer installed in Windows, RAW
                                     data through the spooler (pywin32)
    * ``FileTransport``            – writes the ESC/POS stream to a file (debug)

Rendering modes
    * ``image`` – the receipt is rendered with Pillow into a 1-bit bitmap and
                  sent with ``GS v 0``.  Works with **any language** (Persian,
                  Arabic, ...) and any font.  Recommended.
    * ``text``  – plain ESC/POS text commands (Latin code pages only).

``Invoice`` / ``InvoiceLine`` are the data contract used by the GUI.
"""
from __future__ import annotations

import datetime as _dt
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Iterable, List, Optional

from PIL import Image, ImageDraw, ImageFont

try:
    import serial
except ImportError:  # pragma: no cover
    serial = None  # type: ignore

try:
    import win32print  # type: ignore
except ImportError:  # pragma: no cover
    win32print = None  # type: ignore

try:  # Persian / Arabic shaping for the image renderer (optional)
    import arabic_reshaper
    from bidi.algorithm import get_display as _bidi_display
except ImportError:  # pragma: no cover
    arabic_reshaper = None
    _bidi_display = None

log = logging.getLogger("printer")

# --------------------------------------------------------------------------- #
# ESC/POS constants
# --------------------------------------------------------------------------- #
ESC, GS = b"\x1b", b"\x1d"
CMD_INIT = ESC + b"@"
CMD_CUT_FULL = GS + b"V\x00"
CMD_CUT_PARTIAL = GS + b"V\x01"
CMD_CUT_FEED = GS + b"V\x41\x03"          # feed then partial cut (most printers)
CMD_DRAWER = ESC + b"p\x00\x19\xfa"        # kick cash drawer pin 2

PAPER_DOTS = {58: 384, 80: 576}            # printable dots at 203 dpi
PAPER_CHARS = {58: 32, 80: 48}             # Font A (12x24) characters per line

_RTL_RE = re.compile(r"[\u0590-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]")


# --------------------------------------------------------------------------- #
# Invoice model
# --------------------------------------------------------------------------- #
@dataclass
class InvoiceLine:
    name: str
    weight_kg: float
    price_per_kg: float
    item_id: str = ""
    confidence: float = 0.0
    quantity: int = 1                 # pieces recognised on the tray (display, or pricing when unit == "pcs")
    unit: str = "kg"                  # "kg" = priced by weight, "pcs" = priced per piece
    price_per_piece: float = 0.0

    @property
    def total(self) -> float:
        if self.unit == "pcs":
            return round(self.quantity * self.price_per_piece, 4)
        return round(self.weight_kg * self.price_per_kg, 4)

    @property
    def display_name(self) -> str:
        return f"{self.name} x{self.quantity}" if self.quantity > 1 else self.name


@dataclass
class Invoice:
    number: int
    lines: List[InvoiceLine] = field(default_factory=list)
    created: _dt.datetime = field(default_factory=_dt.datetime.now)
    store_name: str = ""
    currency: str = ""

    @property
    def grand_total(self) -> float:
        return sum(l.total for l in self.lines)

    def to_dict(self) -> dict:
        return {
            "number": self.number,
            "created": self.created.isoformat(timespec="seconds"),
            "store_name": self.store_name,
            "currency": self.currency,
            "lines": [{"item_id": l.item_id, "name": l.name, "weight_kg": round(l.weight_kg, 4),
                       "price_per_kg": l.price_per_kg, "quantity": l.quantity, "unit": l.unit,
                       "price_per_piece": l.price_per_piece,
                       "total": l.total, "confidence": round(l.confidence, 3)}
                      for l in self.lines],
            "grand_total": self.grand_total,
        }


def fmt_money(value: float, decimals: int = 0, currency: str = "") -> str:
    text = f"{value:,.{max(0, int(decimals))}f}"
    return f"{text} {currency}".strip()


def fmt_weight(kg: float, decimals: int = 3) -> str:
    return f"{kg:.{max(0, int(decimals))}f}"


# --------------------------------------------------------------------------- #
# ESC/POS byte builder
# --------------------------------------------------------------------------- #
class EscPosBuilder:
    """Accumulates ESC/POS commands. ``bytes(builder)`` gives the payload."""

    def __init__(self, codepage: str = "cp437", codepage_id: int = 0):
        self.buf = bytearray()
        self.codepage = codepage
        self.codepage_id = codepage_id

    def raw(self, data: bytes) -> "EscPosBuilder":
        self.buf += data
        return self

    def init(self) -> "EscPosBuilder":
        self.buf += CMD_INIT
        self.buf += ESC + b"t" + bytes([self.codepage_id & 0xFF])
        return self

    def align(self, mode: str = "left") -> "EscPosBuilder":
        self.buf += ESC + b"a" + bytes([{"left": 0, "center": 1, "right": 2}.get(mode, 0)])
        return self

    def bold(self, on: bool = True) -> "EscPosBuilder":
        self.buf += ESC + b"E" + (b"\x01" if on else b"\x00")
        return self

    def size(self, width: int = 1, height: int = 1) -> "EscPosBuilder":
        w, h = max(1, min(8, width)) - 1, max(1, min(8, height)) - 1
        self.buf += GS + b"!" + bytes([(w << 4) | h])
        return self

    def text(self, text: str, newline: bool = True) -> "EscPosBuilder":
        data = text.encode(self.codepage, errors="replace")
        self.buf += data + (b"\n" if newline else b"")
        return self

    def feed(self, lines: int = 1) -> "EscPosBuilder":
        self.buf += ESC + b"d" + bytes([max(0, min(255, lines))])
        return self

    def cut(self, partial: bool = True) -> "EscPosBuilder":
        self.buf += CMD_CUT_FEED if partial else CMD_CUT_FULL
        return self

    def drawer(self) -> "EscPosBuilder":
        self.buf += CMD_DRAWER
        return self

    def image(self, img: Image.Image, max_chunk_rows: int = 256) -> "EscPosBuilder":
        """Print a PIL image with GS v 0 (raster bit image), split in chunks."""
        if img.mode != "1":
            img = img.convert("L").point(lambda p: 0 if p < 160 else 255, mode="1")
        width, height = img.size
        # width must be a multiple of 8
        if width % 8:
            padded = Image.new("1", (width + 8 - width % 8, height), 1)
            padded.paste(img, (0, 0))
            img, width = padded, padded.size[0]
        bytes_per_row = width // 8
        # PIL "1" mode: 1 = white. ESC/POS: 1 = black -> invert
        inverted = img.point(lambda p: 255 - p)
        data = inverted.tobytes()
        for y0 in range(0, height, max_chunk_rows):
            rows = min(max_chunk_rows, height - y0)
            chunk = data[y0 * bytes_per_row:(y0 + rows) * bytes_per_row]
            self.buf += GS + b"v0\x00" + bytes([bytes_per_row & 0xFF, bytes_per_row >> 8, rows & 0xFF, rows >> 8]) + chunk
        return self

    def __bytes__(self) -> bytes:
        return bytes(self.buf)


# --------------------------------------------------------------------------- #
# Transports
# --------------------------------------------------------------------------- #
class Transport:
    name = "none"

    def send(self, data: bytes) -> None:
        raise NotImplementedError


class NullTransport(Transport):
    def send(self, data: bytes) -> None:
        log.info("printer disabled – %d bytes discarded", len(data))


class FileTransport(Transport):
    name = "file"

    def __init__(self, path: str):
        self.path = path

    def send(self, data: bytes) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self.path)) or ".", exist_ok=True)
        with open(self.path, "wb") as fh:
            fh.write(data)
        log.info("receipt written to %s (%d bytes)", self.path, len(data))


class SerialTransport(Transport):
    name = "serial"

    def __init__(self, port: str, baudrate: int = 9600, timeout: float = 3.0):
        if serial is None:
            raise RuntimeError("pyserial is not installed")
        if not port:
            raise RuntimeError("no printer COM port configured")
        self.port, self.baudrate, self.timeout = port, int(baudrate), timeout

    def send(self, data: bytes) -> None:
        with serial.Serial(self.port, self.baudrate, timeout=self.timeout, write_timeout=self.timeout) as ser:
            chunk = 4096
            for i in range(0, len(data), chunk):
                ser.write(data[i:i + chunk])
            ser.flush()


class WindowsPrinterTransport(Transport):
    """RAW job through the Windows spooler (works for USB / network / shared ESC/POS printers)."""
    name = "windows"

    def __init__(self, printer_name: str = ""):
        if win32print is None:
            raise RuntimeError("pywin32 is not installed")
        self.printer_name = printer_name or win32print.GetDefaultPrinter()
        if not self.printer_name:
            raise RuntimeError("no Windows printer selected and no default printer")

    def send(self, data: bytes) -> None:
        handle = win32print.OpenPrinter(self.printer_name)
        try:
            job = win32print.StartDocPrinter(handle, 1, ("AI POS Receipt", None, "RAW"))
            try:
                win32print.StartPagePrinter(handle)
                win32print.WritePrinter(handle, data)
                win32print.EndPagePrinter(handle)
            finally:
                win32print.EndDocPrinter(handle)
            log.info("receipt spooled to '%s' as job %s (%d bytes)", self.printer_name, job, len(data))
        finally:
            win32print.ClosePrinter(handle)


def list_windows_printers() -> List[str]:
    if win32print is None:
        return []
    try:
        flags = win32print.PRINTER_ENUM_LOCAL | win32print.PRINTER_ENUM_CONNECTIONS
        return sorted({p[2] for p in win32print.EnumPrinters(flags)})
    except Exception as exc:  # pragma: no cover
        log.warning("EnumPrinters failed: %s", exc)
        return []


def default_windows_printer() -> str:
    if win32print is None:
        return ""
    try:
        return win32print.GetDefaultPrinter()
    except Exception:
        return ""


# --------------------------------------------------------------------------- #
# Receipt rendering
# --------------------------------------------------------------------------- #
_FONT_CANDIDATES = [
    "Vazirmatn-Regular.ttf", "Vazir.ttf", "IRANSans.ttf", "Sahel.ttf",
    "tahoma.ttf", "segoeui.ttf", "arial.ttf", "DejaVuSans.ttf",
]
_BOLD_CANDIDATES = ["Vazirmatn-Bold.ttf", "Vazir-Bold.ttf", "tahomabd.ttf", "segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf"]


def _find_font(candidates: Iterable[str], explicit: str = "") -> Optional[str]:
    paths = []
    if explicit:
        paths.append(explicit)
    font_dirs = [os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts"),
                 os.path.join(os.environ.get("LOCALAPPDATA", ""), "Microsoft", "Windows", "Fonts"),
                 os.path.dirname(os.path.abspath(__file__)), "/usr/share/fonts/truetype/dejavu"]
    for name in candidates:
        for d in font_dirs:
            paths.append(os.path.join(d, name))
    for p in paths:
        if p and os.path.isfile(p):
            return p
    return None


def shape_text(text: str) -> str:
    """Apply Arabic/Persian letter shaping + bidi reordering for bitmap rendering."""
    if not text or not _RTL_RE.search(text):
        return text
    if arabic_reshaper is None or _bidi_display is None:
        return text
    try:
        return _bidi_display(arabic_reshaper.reshape(text))
    except Exception:  # pragma: no cover
        return text


class ReceiptRenderer:
    """Renders an ``Invoice`` to a 1-bit PIL image sized for the printer head."""

    def __init__(self, dots: int = 576, font_file: str = "", font_size: int = 26, rtl: bool = False):
        self.dots = int(dots)
        self.rtl = rtl
        regular = _find_font(_FONT_CANDIDATES, font_file)
        bold = _find_font(_BOLD_CANDIDATES, "") or regular
        size = max(14, int(font_size))
        try:
            self.font = ImageFont.truetype(regular, size) if regular else ImageFont.load_default()
            self.font_small = ImageFont.truetype(regular, int(size * 0.8)) if regular else self.font
            self.font_bold = ImageFont.truetype(bold, size) if bold else self.font
            self.font_title = ImageFont.truetype(bold, int(size * 1.35)) if bold else self.font
            self.font_total = ImageFont.truetype(bold, int(size * 1.25)) if bold else self.font
        except OSError:  # pragma: no cover
            self.font = self.font_small = self.font_bold = self.font_title = self.font_total = ImageFont.load_default()
        self.margin = 8
        self.line_gap = 6

    # helpers --------------------------------------------------------------
    def _text_w(self, draw: ImageDraw.ImageDraw, text: str, font) -> int:
        return int(draw.textlength(text, font=font))

    def _wrap(self, draw, text: str, font, max_w: int) -> List[str]:
        words, lines, cur = text.split(), [], ""
        for w in words:
            trial = (cur + " " + w).strip()
            if self._text_w(draw, shape_text(trial), font) <= max_w or not cur:
                cur = trial
            else:
                lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)
        return lines or [""]

    def render(self, inv: Invoice, header: str = "", footer: str = "", currency_decimals: int = 0,
               weight_decimals: int = 3, labels: Optional[dict] = None) -> Image.Image:
        labels = labels or {}
        L = {"invoice": labels.get("invoice", "Invoice"), "date": labels.get("date", "Date"),
             "item": labels.get("item", "Item"), "weight": labels.get("weight", "Kg"),
             "price": labels.get("price", "Price/Kg"), "total": labels.get("total", "Total"),
             "grand_total": labels.get("grand_total", "TOTAL"), "items": labels.get("items", "Items")}
        # draw on a tall canvas, crop at the end
        W = self.dots
        img = Image.new("L", (W, 4000), 255)
        draw = ImageDraw.Draw(img)
        y = self.margin
        usable = W - 2 * self.margin

        def line(text: str, font, align: str = "auto", gap: int = None):
            nonlocal y
            shaped = shape_text(text)
            tw = self._text_w(draw, shaped, font)
            if align == "auto":
                align = "right" if (self.rtl or _RTL_RE.search(text)) else "left"
            x = {"left": self.margin, "center": (W - tw) // 2, "right": W - self.margin - tw}[align]
            draw.text((x, y), shaped, font=font, fill=0)
            bbox = font.getbbox("Ag")
            y += (bbox[3] - bbox[1]) + (self.line_gap if gap is None else gap) + 4

        def rule(char_h: int = 2):
            nonlocal y
            y += 4
            draw.line((self.margin, y, W - self.margin, y), fill=0, width=char_h)
            y += 8

        def row(cells: List[str], fonts, widths: List[float], aligns: List[str]):
            """Table row: widths are fractions of the usable width; RTL flips the column order."""
            nonlocal y
            xs, x = [], self.margin
            for frac in widths:
                xs.append((x, int(usable * frac)))
                x += int(usable * frac)
            order = list(range(len(cells)))
            if self.rtl:
                order = order[::-1]
            row_h = 0
            for col_idx, cell_idx in enumerate(order):
                cell, font, align = cells[cell_idx], fonts[cell_idx] if isinstance(fonts, list) else fonts, aligns[cell_idx]
                cx, cw = xs[col_idx]
                shaped = shape_text(cell)
                tw = self._text_w(draw, shaped, font)
                # shrink long cells (item names) instead of overflowing
                f = font
                while tw > cw - 6 and f.size > 12:
                    f = ImageFont.truetype(f.path, f.size - 2) if getattr(f, "path", None) else f
                    tw = self._text_w(draw, shaped, f)
                    if not getattr(f, "path", None):
                        break
                if self.rtl:
                    align = {"left": "right", "right": "left"}.get(align, align)
                tx = {"left": cx + 3, "center": cx + (cw - tw) // 2, "right": cx + cw - tw - 3}[align]
                draw.text((tx, y), shaped, font=f, fill=0)
                bbox = f.getbbox("Ag")
                row_h = max(row_h, bbox[3] - bbox[1])
            y += row_h + self.line_gap + 4

        # ---- header
        if inv.store_name:
            for part in self._wrap(draw, inv.store_name, self.font_title, usable):
                line(part, self.font_title, "center")
        for hl in (header or "").splitlines():
            for part in self._wrap(draw, hl, self.font, usable):
                line(part, self.font, "center")
        rule()
        line(f"{L['invoice']}: {inv.number:06d}", self.font_small)
        line(f"{L['date']}: {inv.created.strftime('%Y-%m-%d  %H:%M')}", self.font_small)
        rule(1)
        # ---- table
        widths = [0.40, 0.18, 0.20, 0.22]
        aligns = ["left", "right", "right", "right"]
        row([L["item"], L["weight"], L["price"], L["total"]], self.font_bold, widths, aligns)
        rule(1)
        for ln in inv.lines:
            unit_price = ln.price_per_piece if ln.unit == "pcs" else ln.price_per_kg
            row([ln.display_name, fmt_weight(ln.weight_kg, weight_decimals), fmt_money(unit_price, currency_decimals),
                 fmt_money(ln.total, currency_decimals)], self.font, widths, aligns)
        rule()
        line(f"{L['items']}: {len(inv.lines)}", self.font_small)
        total_text = f"{L['grand_total']}: {fmt_money(inv.grand_total, currency_decimals, inv.currency)}"
        line(total_text, self.font_total, "center")
        rule()
        for fl in (footer or "").splitlines():
            for part in self._wrap(draw, fl, self.font, usable):
                line(part, self.font, "center")
        y += self.margin
        img = img.crop((0, 0, W, min(y, img.size[1])))
        return img.point(lambda p: 0 if p < 160 else 255, mode="1")


# --------------------------------------------------------------------------- #
# Facade used by the application
# --------------------------------------------------------------------------- #
class ReceiptPrinter:
    """Builds the ESC/POS job for an invoice and pushes it through the configured transport.

    ``config`` = settings["printer"]; ``general`` = settings["general"].
    """

    def __init__(self, config: dict, general: Optional[dict] = None, labels: Optional[dict] = None):
        self.config = dict(config or {})
        self.general = dict(general or {})
        self.labels = labels or {}

    # configuration helpers -------------------------------------------------
    @property
    def enabled(self) -> bool:
        return bool(self.config.get("enabled", True)) and self.config.get("interface", "windows") != "none"

    def dots(self) -> int:
        explicit = int(self.config.get("dots_per_line", 0) or 0)
        return explicit if explicit > 0 else PAPER_DOTS.get(int(self.config.get("paper_width", 80)), 576)

    def chars(self) -> int:
        return PAPER_CHARS.get(int(self.config.get("paper_width", 80)), 48)

    def transport(self) -> Transport:
        iface = self.config.get("interface", "windows")
        if iface == "serial":
            return SerialTransport(self.config.get("port", ""), int(self.config.get("baudrate", 9600)))
        if iface == "windows":
            return WindowsPrinterTransport(self.config.get("printer_name", ""))
        if iface == "file":
            return FileTransport(self.config.get("file_path") or "last_receipt.bin")
        return NullTransport()

    # job builders ----------------------------------------------------------
    def build_job(self, inv: Invoice) -> bytes:
        b = EscPosBuilder(self.config.get("codepage", "cp437"), int(self.config.get("codepage_id", 0)))
        b.init()
        header, footer = self.config.get("header", ""), self.config.get("footer", "")
        cur_dec = int(self.general.get("currency_decimals", 0))
        w_dec = int(self.general.get("weight_decimals", 3))
        if self.config.get("mode", "image") == "image":
            renderer = ReceiptRenderer(self.dots(), self.config.get("font_file", ""), int(self.config.get("font_size", 26)),
                                       rtl=str(self.general.get("language", "en")).startswith("fa"))
            b.align("left").image(renderer.render(inv, header, footer, cur_dec, w_dec, self.labels))
        else:
            self._build_text(b, inv, header, footer, cur_dec, w_dec)
        b.feed(int(self.config.get("feed_lines", 3)))
        if self.config.get("cut", True):
            b.cut(partial=True)
        if self.config.get("open_drawer", False):
            b.drawer()
        return bytes(b)

    def _build_text(self, b: EscPosBuilder, inv: Invoice, header: str, footer: str, cur_dec: int, w_dec: int) -> None:
        n = self.chars()
        L = self.labels
        b.align("center").bold(True).size(2, 2).text(inv.store_name or "").size(1, 1).bold(False)
        for hl in (header or "").splitlines():
            b.text(hl)
        b.text("-" * n).align("left")
        b.text(f"{L.get('invoice', 'Invoice')}: {inv.number:06d}")
        b.text(f"{L.get('date', 'Date')}: {inv.created.strftime('%Y-%m-%d %H:%M')}")
        b.text("-" * n)
        name_w = n - 30 if n >= 42 else n - 24
        wcol, pcol, tcol = (9, 10, 11) if n >= 42 else (7, 8, 9)
        b.bold(True).text(f"{L.get('item', 'Item')[:name_w]:<{name_w}}{L.get('weight', 'Kg')[:wcol]:>{wcol}}"
                          f"{L.get('price', 'Price')[:pcol]:>{pcol}}{L.get('total', 'Total')[:tcol]:>{tcol}}").bold(False)
        for ln in inv.lines:
            unit_price = ln.price_per_piece if ln.unit == "pcs" else ln.price_per_kg
            b.text(f"{ln.display_name[:name_w]:<{name_w}}{fmt_weight(ln.weight_kg, w_dec):>{wcol}}"
                   f"{fmt_money(unit_price, cur_dec):>{pcol}}{fmt_money(ln.total, cur_dec):>{tcol}}")
        b.text("-" * n)
        b.align("right").bold(True).size(2, 2)
        b.text(f"{L.get('grand_total', 'TOTAL')}: {fmt_money(inv.grand_total, cur_dec, inv.currency)}")
        b.size(1, 1).bold(False).align("center")
        for fl in (footer or "").splitlines():
            b.text(fl)

    # public API --------------------------------------------------------------
    def print_invoice(self, inv: Invoice) -> str:
        """Send the receipt. Returns a short status message; raises on transport error."""
        if not self.enabled:
            return "printer disabled"
        data = self.build_job(inv)
        transport = self.transport()
        transport.send(data)
        return f"{len(data)} bytes via {transport.name}"

    def print_test(self) -> str:
        inv = Invoice(number=0, store_name=self.general.get("store_name", "AI POS Scale"),
                      currency=self.general.get("currency", ""))
        inv.lines = [InvoiceLine("Test item / کالای آزمایشی", 1.234, 25000.0), InvoiceLine("Apple / سیب", 0.500, 60000.0)]
        return self.print_invoice(inv)

    def preview_image(self, inv: Invoice) -> Image.Image:
        renderer = ReceiptRenderer(self.dots(), self.config.get("font_file", ""), int(self.config.get("font_size", 26)),
                                   rtl=str(self.general.get("language", "en")).startswith("fa"))
        return renderer.render(inv, self.config.get("header", ""), self.config.get("footer", ""),
                               int(self.general.get("currency_decimals", 0)), int(self.general.get("weight_decimals", 3)),
                               self.labels)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    inv = Invoice(number=42, store_name="فروشگاه هوشمند", currency="تومان")
    inv.lines = [InvoiceLine("سیب قرمز", 1.234, 45000), InvoiceLine("Banana", 0.750, 62000), InvoiceLine("گوجه فرنگی گلخانه‌ای درجه یک", 2.05, 30000)]
    cfg = {"interface": "file", "file_path": "last_receipt.bin", "paper_width": 80, "mode": "image",
           "header": "به فروشگاه ما خوش آمدید", "footer": "از خرید شما سپاسگزاریم"}
    gen = {"language": "fa", "currency_decimals": 0, "weight_decimals": 3}
    p = ReceiptPrinter(cfg, gen, labels={"invoice": "فاکتور", "date": "تاریخ", "item": "کالا", "weight": "کیلو",
                                        "price": "قیمت", "total": "جمع", "grand_total": "جمع کل", "items": "اقلام"})
    print(p.print_invoice(inv))
    p.preview_image(inv).save("receipt_preview.png")
    print("preview saved: receipt_preview.png; printers:", list_windows_printers())
