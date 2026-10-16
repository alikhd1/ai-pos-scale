"""
test_scale_wiring.py
--------------------
Guards the class of bug that produced "the scale is connected, the settings test
says OK, but the main screen shows no weight".

Two independent defects caused it and both are covered here:

  1. ``scale.simulate`` shipped as True and won over a configured COM port, so
     the app ran the simulator while the settings dialog tested the real port.
  2. ``stable_only`` filtered every reading out inside the driver, so ``latest()``
     stayed None forever on an indicator that always reports motion - while the
     diagnosis, which ignores that flag, happily reported a weight.

No hardware needed: ``scale_driver.open_serial`` is replaced by a fake port.

Run:  python tests/test_scale_wiring.py
"""
from __future__ import annotations

import os
import sys
import tempfile
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scale_driver
from scale_driver import ScaleReader, parse_weight

CHECKS = []


def check(name, condition, detail=""):
    CHECKS.append((name, bool(condition), detail))
    print(f"  {'PASS' if condition else 'FAIL'}  {name}" + (f"   [{detail}]" if detail and not condition else ""))


_QAPP = None


def qt_app():
    """Return the one QApplication, keeping a reference so Qt is not torn down
    underneath its own widgets."""
    global _QAPP
    from PyQt6.QtWidgets import QApplication
    _QAPP = QApplication.instance() or QApplication([])
    return _QAPP


class FakePort:
    """A serial port that endlessly replays one protocol frame."""

    def __init__(self, script: bytes):
        self.script = script
        self.buf = bytearray(script)
        self.written = bytearray()
        self.is_open = True
        self.dtr = self.rts = False

    @property
    def in_waiting(self):
        if not self.buf:
            self.buf += self.script
        return len(self.buf)

    def read(self, n=1):
        if not self.buf:
            self.buf += self.script
        out = bytes(self.buf[:n])
        del self.buf[:n]
        return out

    def write(self, data):
        self.written += data
        return len(data)

    def reset_input_buffer(self):
        pass

    def close(self):
        self.is_open = False


def install_fake(script: bytes):
    scale_driver.open_serial = lambda cfg, timeout=0.2: FakePort(script)


# --------------------------------------------------------------------------- #
def test_defaults_and_parser():
    print("\n[1] defaults and parser")
    from settings_manager import SettingsManager
    with tempfile.TemporaryDirectory() as d:
        s = SettingsManager(os.path.join(d, "config.json"))
        check("a fresh install does not simulate", s.get("scale.simulate") is False,
              f"simulate={s.get('scale.simulate')}")
        check("a fresh install has no port", s.get("scale.port") == "")
    r = parse_weight(b"ST,GS,+  1.234kg\r\n")
    check("parses a stable frame", r is not None and abs(r.weight_kg - 1.234) < 1e-6 and r.stable)
    r = parse_weight(b"US,GS,+  1.234kg\r\n")
    check("parses an unstable frame and flags it", r is not None and abs(r.weight_kg - 1.234) < 1e-6 and not r.stable)


def test_driver_never_drops_a_weight():
    print("\n[2] the driver keeps every parsed weight (stable_only is a POS rule, not a driver filter)")
    got = []
    reader = ScaleReader({"port": "COM_TEST", "stable_only": True, "unit": "auto", "implied_decimals": 0,
                          "number_index": 0})
    reader.subscribe(got.append)
    reader._handle_frame(b"US,GS,+  1.234kg\r\n")
    latest = reader.latest()
    check("latest() holds the weight even when it is unstable",
          latest is not None and abs(latest.weight_kg - 1.234) < 1e-6, f"latest={latest}")
    check("the reading is still flagged unstable", latest is not None and not latest.stable)
    check("subscribers were notified", len(got) == 1, f"got {len(got)}")
    h = reader.health()
    check("health counts the unstable reading", h.get("unstable") == 1 and h.get("readings") == 1, str(h))
    check("nothing was counted as a parse failure", h.get("parse_failures") == 0)


def test_reader_reads_a_fake_scale():
    print("\n[3] ScaleReader against a fake indicator")
    install_fake(b"ST,GS,+  2.500kg\r\n")
    reader = ScaleReader({"port": "COM_TEST", "baudrate": 9600, "bytesize": 8, "parity": "N", "stopbits": 1,
                          "unit": "auto", "reconnect_s": 0.2, "idle_gap_ms": 60})
    reader.start()
    deadline = time.time() + 3
    while time.time() < deadline and reader.latest() is None:
        time.sleep(0.02)
    latest = reader.latest()
    h = reader.health()
    reader.stop()
    check("a streaming scale produces a weight", latest is not None and abs(latest.weight_kg - 2.5) < 1e-6, str(latest))
    check("status is connected", h["status"] == "connected", h["status"])
    check("bytes were counted", h["bytes"] > 0)


def test_mode_matrix():
    print("\n[4] every scale source builds the object it claims to")
    from settings_manager import SettingsManager
    from scale_driver import SimulatedScale, SOURCE_AUTO, SOURCE_SERIAL, SOURCE_SIMULATE
    import app as A

    install_fake(b"ST,GS,+  1.234kg\r\n")
    qt_app()
    with tempfile.TemporaryDirectory() as d:
        s = SettingsManager(os.path.join(d, "config.json"))
        s.set("ai.db_file", os.path.join(d, "items.pkl"))
        s.set("general.invoice_dir", os.path.join(d, "invoices"))
        s.save()
        win = A.MainWindow(s)
        try:
            # (source, port, enabled) -> is it the simulator?
            cases = [
                (SOURCE_SIMULATE, "COM_TEST", True, True, "simulation wins over a configured port"),
                (SOURCE_SIMULATE, "", True, True, "simulation with no port"),
                (SOURCE_SERIAL, "COM_TEST", True, False, "serial with a port"),
                (SOURCE_SERIAL, "", True, True, "serial without a port falls back"),
                (SOURCE_AUTO, "COM_TEST", True, False, "auto picks the configured port"),
                (SOURCE_AUTO, "", True, True, "auto with nothing attached"),
                (SOURCE_SERIAL, "COM_TEST", False, True, "disabled always simulates"),
                (SOURCE_AUTO, "COM_TEST", False, True, "disabled beats auto"),
            ]
            for source, port, enabled, expect_sim, label in cases:
                s.set("scale.source", source)
                s.set("scale.port", port)
                s.set("scale.enabled", enabled)
                win.start_scale()
                actual_sim = isinstance(win.scale, SimulatedScale)
                reason = A.scale_mode_reason(s)
                check(f"object matches settings ({label})", actual_sim == expect_sim,
                      f"expected {'Simulated' if expect_sim else 'hardware'}, got {type(win.scale).__name__}")
                check(f"the reason string agrees ({label})",
                      reason.startswith("SimulatedScale") == expect_sim, reason)
                check(f"helper agrees ({label})", A.scale_is_simulated(s) == expect_sim)

            # a configuration written before "source" existed still behaves
            s.set("scale.source", "")
            s.set("scale.simulate", True)
            s.set("scale.port", "COM_TEST")
            s.set("scale.enabled", True)
            win.start_scale()
            check("a legacy simulate=true config still simulates", isinstance(win.scale, SimulatedScale))
            from translations import tr
            pill = win.scale_pill.text()
            check("the conflict is spelled out on the pill",
                  tr("scale_sim_overrides_port") in pill and "COM_TEST" in pill, pill)
            check("no untranslated key leaks into the pill", "scale_sim_overrides" not in pill, pill)
            win._switch_to_real_scale()
            check("'use the real scale' switches to the hardware reader",
                  not isinstance(win.scale, SimulatedScale), type(win.scale).__name__)
        finally:
            win.stop_scale()
            win.close()


def test_hid_support():
    print("\n[4b] USB HID scales")
    from hid_scale import parse_hid_report, list_hid_devices, list_hid_scales
    r = parse_hid_report(bytes([3, 4, 3, 0, 0x02, 0x00]))
    check("parses 2 kg, stable", r is not None and abs(r.weight_kg - 2.0) < 1e-9 and r.stable)
    r = parse_hid_report(bytes([3, 3, 2, 0, 0xF4, 0x01]))
    check("parses 500 g, in motion", r is not None and abs(r.weight_kg - 0.5) < 1e-9 and not r.stable)
    r = parse_hid_report(bytes([3, 4, 2, 0xFF, 0xE8, 0x03]))
    check("applies the negative exponent", r is not None and abs(r.weight_kg - 0.1) < 1e-9, str(r))
    r = parse_hid_report(bytes([3, 6, 2, 0, 0, 0]))
    check("flags over weight", r is not None and r.overload)
    check("rejects a non-scale report", parse_hid_report(bytes([1, 0, 0, 0])) is None)
    try:
        devices = list_hid_devices()
        check("HID enumeration runs without error", True, f"{len(devices)} devices, {len(list_hid_scales())} scales")
    except Exception as exc:
        check("HID enumeration runs without error", False, str(exc))


def test_reconciliation():
    print("\n[4c] the invoice weight must match the tray")
    from settings_manager import SettingsManager
    from scale_driver import WeightReading
    import app as A
    from translations import tr

    qt_app()
    with tempfile.TemporaryDirectory() as d:
        st = SettingsManager(os.path.join(d, "config.json"))
        st.set("ai.db_file", os.path.join(d, "items.pkl"))
        st.set("general.invoice_dir", os.path.join(d, "invoices"))
        st.set("pos.weighing_mode", "cumulative")
        st.save()
        win = A.MainWindow(st)
        try:
            from ai_engine import Item
            a = Item(id="a", name="A", price_per_kg=1000.0)
            b = Item(id="b", name="B", price_per_kg=2000.0)

            # 100 g of A on an empty tray
            win.current_weight = WeightReading(0.100, stable=True)
            w = win._weight_ok()
            check("the first item weighs the whole tray", w is not None and abs(w - 0.100) < 1e-9, str(w))
            win._add_line(a, w, 1.0)
            check("the invoice now holds 100 g", abs(win.invoiced_weight() - 0.100) < 1e-9)

            # B is added on top: the tray reads 300 g, so B must be 200 g
            win.current_weight = WeightReading(0.300, stable=True)
            w = win._weight_ok()
            check("the second item weighs the difference", w is not None and abs(w - 0.200) < 1e-9, str(w))
            win._add_line(b, w, 1.0)
            check("the invoice now holds 300 g", abs(win.invoiced_weight() - 0.300) < 1e-9)
            check("tray and invoice agree", abs(win.weight_difference()) < 1e-9)

            # nothing new on the tray
            check("adding again is refused", win._weight_ok() is None)

            # an unbilled item is slipped on: the difference exposes it
            win.current_weight = WeightReading(0.450, stable=True)
            check("an extra item shows as a difference", abs(win.weight_difference() - 0.150) < 1e-9)
            win._update_reconciliation()
            check("the reconciliation line reports it", tr("new_weight") in win.recon_lbl.text(), win.recon_lbl.text())

            # an item is taken off after being billed
            win.current_weight = WeightReading(0.100, stable=True)
            win._update_reconciliation()
            check("a removed item is reported too", tr("weight_missing", diff="0.200") in win.recon_lbl.text(),
                  win.recon_lbl.text())

            # replace mode bills the whole tray every time
            st.set("pos.weighing_mode", "replace")
            win.current_weight = WeightReading(0.250, stable=True)
            check("replace mode bills the whole tray", abs(win._weight_ok() - 0.250) < 1e-9)
        finally:
            win.close()


def test_weight_gate_and_status():
    print("\n[5] the POS gate and the status text")
    from settings_manager import SettingsManager
    from scale_driver import WeightReading
    import app as A
    from translations import tr

    qt_app()
    with tempfile.TemporaryDirectory() as d:
        s = SettingsManager(os.path.join(d, "config.json"))
        s.set("ai.db_file", os.path.join(d, "items.pkl"))
        s.set("general.invoice_dir", os.path.join(d, "invoices"))
        s.save()
        win = A.MainWindow(s)
        try:
            win.current_weight = WeightReading(1.234, stable=False)
            s.set("scale.stable_only", True)
            check("an unstable weight is refused when 'stable only' is on", win._weight_ok() is None)
            s.set("scale.stable_only", False)
            check("the same weight is accepted when it is off", win._weight_ok() == 1.234)
            win.current_weight = WeightReading(0.0, stable=True)
            check("a zero weight is always refused", win._weight_ok() is None)

            class FakeScale:
                is_simulated = False

                def __init__(self, h, r):
                    self._h, self._r = h, r

                def health(self):
                    return self._h

                def latest(self):
                    return self._r

            old = WeightReading(1.0, stable=True)
            old.timestamp = time.time() - 30
            win.scale = FakeScale({"status": "connected", "bytes": 400, "readings": 12, "parse_failures": 0,
                                   "data_age": 1.0, "unstable": 0}, old)
            win._update_scale_status()
            check("parsed but old data reads as stale, not as unparseable",
                  tr("scale_stale_data") in win.scale_pill.text(), win.scale_pill.text())
            win.scale = FakeScale({"status": "connected", "bytes": 0, "readings": 0, "parse_failures": 0,
                                   "data_age": None, "unstable": 0}, None)
            win._update_scale_status()
            check("silence reads as 'no data'", tr("scale_no_data") in win.scale_pill.text(), win.scale_pill.text())
            win.scale = FakeScale({"status": "connected", "bytes": 400, "readings": 0, "parse_failures": 30,
                                   "data_age": 0.2, "unstable": 0}, None)
            win._update_scale_status()
            check("unparseable bytes read as 'data not understood'",
                  tr("scale_bad_data") in win.scale_pill.text(), win.scale_pill.text())
            win.scale = None
        finally:
            win.close()


def test_successful_diagnosis_unticks_simulation():
    print("\n[6] a successful diagnosis turns simulation off")
    from settings_manager import SettingsManager
    from scale_driver import SOURCE_SERIAL, SOURCE_SIMULATE
    import app as A

    qt_app()
    with tempfile.TemporaryDirectory() as d:
        s = SettingsManager(os.path.join(d, "config.json"))
        s.set("scale.source", SOURCE_SIMULATE)
        s.set("scale.port", "COM_TEST")
        s.save()
        dlg = A.SettingsDialog(None, s, lambda c, g: None, A.Bus())
        try:
            check("the dialog starts in simulation", dlg.source.currentData() == SOURCE_SIMULATE)
            dlg.on_dialog_result("scale_done", "1.234 kg", False)
            check("a successful diagnosis selects the serial source",
                  dlg.source.currentData() == SOURCE_SERIAL, str(dlg.source.currentData()))
            dlg.source.setCurrentIndex([*(x for x in range(dlg.source.count()))
                                        ][[dlg.source.itemData(i) for i in range(dlg.source.count())
                                           ].index(SOURCE_SIMULATE)])
            dlg.on_dialog_result("scale_done", "no usable data", True)
            check("a failed diagnosis leaves it alone", dlg.source.currentData() == SOURCE_SIMULATE)
            cfg = dlg._scale_config()
            check("the saved config carries the source", cfg["source"] == SOURCE_SIMULATE and cfg["simulate"] is True,
                  str({k: cfg[k] for k in ("source", "simulate")}))
        finally:
            dlg.close()


def main() -> int:
    print("=" * 70)
    print("scale wiring tests")
    print("=" * 70)
    test_defaults_and_parser()
    test_driver_never_drops_a_weight()
    test_reader_reads_a_fake_scale()
    test_mode_matrix()
    test_hid_support()
    test_reconciliation()
    test_weight_gate_and_status()
    test_successful_diagnosis_unticks_simulation()
    failed = [c for c in CHECKS if not c[1]]
    print("\n" + "=" * 70)
    print(f"{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
    for name, _, detail in failed:
        print(f"  FAILED: {name}   {detail}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
