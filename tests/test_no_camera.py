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

import time
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

    print("\n[4] a remote action works from a worker thread, not just the main one")
    # the bug this covers shipped and was only caught on a real till: _on_ui
    # used QTimer.singleShot, which Qt silently ignores when it is called from
    # a thread that does not own the object, so every UI-touching remote action
    # timed out after 90s. The tests ran support_action on the main thread,
    # where the inline fast path hides it entirely.
    import threading as _th
    box = {}

    def from_worker():
        try:
            box["setting"] = win.support_action("set_setting", {"key": "ai.top_k", "value": 5})
            box["ping"] = win.support_action("ping", {})
        except Exception as exc:
            box["error"] = exc

    worker = _th.Thread(target=from_worker, daemon=True)
    worker.start()
    # the GUI thread has to keep pumping or the queued call never lands
    deadline = time.time() + 30
    while worker.is_alive() and time.time() < deadline:
        _QAPP.processEvents()
        time.sleep(0.01)
    worker.join(timeout=5)
    check("the worker finished instead of hanging", not worker.is_alive(), "still blocked in _on_ui")
    check("no timeout off the main thread", "error" not in box, str(box.get("error")))
    check("the action actually ran", box.get("ping", {}).get("pong") is True, str(box.get("ping")))
    check("and it reached the settings", int(win.settings.get("ai.top_k")) == 5,
          str(win.settings.get("ai.top_k")))

    print("\n[5] the report says which code produced it")
    # a till was updated, restarted, and still reported without two new
    # fields; working out that the checkout was stale cost a round trip
    b = A.build_id()
    check("the snapshot carries a build", isinstance(win.support_snapshot().get("build"), dict))
    check("it names a version", b.get("version") == A.APP_VERSION, str(b))
    check("and a commit", bool(b.get("commit")), str(b))
    check("it is always a dict, checkout or not", isinstance(b, dict))

    print("\n[6] the backend is chosen for what the camera is asked to do")
    # a real till reported it could not lock exposure or white balance: Media
    # Foundation barely implements camera controls. That matters because a
    # camera that re-exposes itself when a dark product lands on a white tray
    # brightens the whole picture, so nothing matches the reference any more.
    plain = [b[0] for b in RC.OpenCVCameraSource(0, 640, 480)._backends()]
    locking = [b[0] for b in RC.OpenCVCameraSource(0, 640, 480, lock_exposure=True)._backends()]
    check("normally Media Foundation leads", plain[0] == "MSMF", str(plain))
    check("but DirectShow leads when the exposure must hold", locking[0] == "DSHOW", str(locking))
    check("and nothing is dropped either way", sorted(plain) == sorted(locking),
          f"{plain} vs {locking}")

    print("\n[7] the camera search gives up quickly when there is nothing there")
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
