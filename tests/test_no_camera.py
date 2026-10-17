"""
test_no_camera.py
-----------------
A till with no camera must look like a till with no camera.

This exists because of a real failure on the shop floor: with no camera the app
fed its own "NO CAMERA" placeholder picture to the segmenter, which decided
something was moving over the tray, and the screen showed "take your hand off
the tray" with the Add button greyed out.  The operator could not sell anything,
and nothing on the screen said the camera was the problem.

So: no camera means recognition is stopped, the banner names the real fault, and
the operator can still pick a product from the list and weigh it by hand.

Run:  python tests/test_no_camera.py
"""
import os, sys, tempfile, shutil, traceback
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["QT_QPA_FONTDIR"] = r"C:\Windows\Fonts"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tests"))
os.chdir(ROOT)

import app as A
import rtk_camera as RC
from ai_engine import Recognition
from PyQt6.QtWidgets import QApplication
from settings_manager import SettingsManager
from translations import set_language, tr

TMP = os.path.join(tempfile.gettempdir(), "aipos_no_camera")
shutil.rmtree(TMP, ignore_errors=True); os.makedirs(TMP, exist_ok=True)

ok = {"all": True}
FAILED = []
def check(name, cond, detail=""):
    if not cond:
        ok["all"] = False
        FAILED.append((name, detail))
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))

A.msg_info = lambda p, t: None
A.msg_error = lambda p, t: None

settings = SettingsManager(os.path.join(TMP, "config.json"))
settings.set("ai.db_file", os.path.join(TMP, "items.pkl"))
settings.set("general.invoice_dir", os.path.join(TMP, "invoices"))
settings.set("ai.fast_inference", False)
settings.set("support.enabled", False)      # never report a test run as a real till
settings.save()
set_language("en")

# hold the application alive: letting it be collected while widgets live
# crashes Qt with 0xC0000409, which has bitten this project before
_QAPP = QApplication([])
win = A.MainWindow(settings)

try:
    print("\n[1] no camera at all")
    dead = RC.CameraOpenResult(RC.NullCameraSource(640, 480), "none", "no camera", ["nothing answered"])
    win.on_camera_ready(dead)
    check("camera_ok is false", win.camera_ok is False)
    check("pill says no camera", win.cam_pill.text() == tr("no_camera"), win.cam_pill.text())
    check("retry button offered", win.cam_retry_btn.isVisible() or not win.isVisible())
    check("banner names the camera", win.banner_lbl.text() == tr("camera_dead"), win.banner_lbl.text())
    check("selling is still possible", win.add_btn.isEnabled())

    print("\n[2] the placeholder picture cannot fake a hand")
    win.current_rec = Recognition(intrusion=True, unreliable=True)
    check("no hand warning without a camera", win.blocking_reason() is None, str(win.blocking_reason()))
    win._update_blocking()
    check("Add stays enabled", win.add_btn.isEnabled())

    print("\n[3] with a camera the interlock works again")
    live = RC.CameraOpenResult(RC.NullCameraSource(640, 480), "opencv", "fake live camera", [])
    win.on_camera_ready(live)
    check("camera_ok is true", win.camera_ok is True)
    win.current_rec = Recognition(intrusion=True)
    check("hand is reported", win.blocking_reason() == tr("hand_detected"), str(win.blocking_reason()))
    win._update_blocking()
    check("Add is blocked by the hand", not win.add_btn.isEnabled())

    print("\n[4] the camera search gives up quickly when there is nothing there")
    import time
    t0 = time.time()
    try:
        RC.OpenCVCameraSource(0, 640, 480).open(budget=6.0)
        print("  note: a camera actually opened on this machine")
    except RuntimeError as exc:
        elapsed = time.time() - t0
        check("bounded by the budget", elapsed < 9.0, f"{elapsed:.1f}s")
        check("says what it tried", "device 0" in str(exc) or "no frames" in str(exc), str(exc)[:90])

except Exception:
    ok["all"] = False
    traceback.print_exc()

print("\n" + ("ALL CHECKS PASSED" if ok["all"] else f"{len(FAILED)} CHECK(S) FAILED"))
for name, detail in FAILED:
    print(f"  - {name} {detail}")
sys.exit(0 if ok["all"] else 1)
