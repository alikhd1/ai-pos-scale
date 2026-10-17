"""
test_pos_flow.py
----------------
The whole till, head-less: enrol two look-alike products, recognise them, bill
them with the tray filling up, catch an unbilled item, and collect diagnostics.

It covers the behaviour that is hard to get right and easy to break:

  * cumulative weighing - put 100 g on and bill it, add 200 g and the scale
    reads 300 g while the new line is 200 g,
  * the tamper check - an item added to the tray without being billed makes the
    tray and the invoice disagree, and checkout says so,
  * the look-alike chooser - when two enrolled products score close together the
    operator is offered both instead of being given a silent guess,
  * the diagnostic bundle actually contains the images needed to debug a shop.

Needs no hardware: the camera and the scale are both scripted.

Run:  python tests/test_pos_flow.py
"""
import os, sys, time, tempfile, shutil, traceback, zipfile
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["QT_QPA_FONTDIR"] = r"C:\Windows\Fonts"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tests"))
os.chdir(ROOT)

import numpy as np
import scene as S
import app as A
from PyQt6.QtWidgets import QApplication
from settings_manager import SettingsManager
from translations import set_language, tr
from ai_engine import MODE_OBJECT, region_crop

TMP = os.path.join(tempfile.gettempdir(), "aipos_pos_flow")
shutil.rmtree(TMP, ignore_errors=True); os.makedirs(TMP, exist_ok=True)
SHOT = os.path.join(tempfile.gettempdir(), "aipos_pos_flow.png")

results = {"ok": True}
FAILED = []
def check(name, cond, detail=""):
    if not cond:
        results["ok"] = False
        FAILED.append((name, detail))
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))

messages = []
A.msg_info = lambda p, t: messages.append(("info", t))
A.msg_error = lambda p, t: messages.append(("error", t))
A.ask = lambda p, t: messages.append(("ask", t)) or True


class FakeCamera:
    def __init__(self):
        self.objects, self.hand, self.n = [], False, 0
    def set(self, objects, hand=False):
        self.objects, self.hand = objects, hand
    def get(self):
        self.n += 1
        return self.n, S.scene(self.objects, seed=2000 + self.n, hand=self.hand)


cam = FakeCamera()
settings = SettingsManager(os.path.join(TMP, "config.json"))
settings.set("ai.db_file", os.path.join(TMP, "items.pkl"))
settings.set("general.invoice_dir", os.path.join(TMP, "invoices"))
settings.set("printer.interface", "file"); settings.set("printer.file_path", os.path.join(TMP, "receipt.bin"))
settings.set("ai.motion_gate", False)
settings.set("ai.fast_inference", False)          # keep the test quick and deterministic
settings.set("pos.weighing_mode", "cumulative")
settings.set("camera.tray_points", [[x / S.FRAME_W, y / S.FRAME_H] for x, y in S.TRAY_POLY])
settings.save()
set_language("en")

qapp = QApplication([]); qapp.setStyleSheet(A.STYLE)
win = A.MainWindow(settings)
win._camera_frame = cam.get
win.show()


def pump(sec):
    end = time.time() + sec
    while time.time() < end:
        qapp.processEvents(); time.sleep(0.005)


def wait(cond, timeout, what):
    end = time.time() + timeout
    while time.time() < end:
        qapp.processEvents()
        if cond(): return True
        time.sleep(0.02)
    check(what, False, "timeout"); return False


try:
    print("\n[1] setup")
    wait(lambda: win.engine is not None, 240, "model loads")
    win.engine.get_frame = cam.get
    # this test scripts its own frames, so the window is told it has a camera:
    # a window with no camera correctly stops recognising (see test_no_camera.py)
    wait(lambda: win.camera_result is not None, 30, "camera opener finishes")
    win.camera_ok = True
    win.engine.paused = False
    win.capture_empty_tray()
    wait(lambda: win.db.has_background_image, 60, "empty tray captured")
    check("segmenter live", win.engine.has_segmenter)

    # two deliberately similar items: the same red, slightly different shape
    LOOKALIKE_A = ("apple_red", (44, 44, 196), 46, 1.00)
    LOOKALIKE_B = ("tomato_red", (40, 46, 200), 44, 1.06)
    for spec in (LOOKALIKE_A, LOOKALIKE_B):
        embs, groups = [], []
        for g, (angle, centre) in enumerate([(0, (300, 240)), (40, (250, 200)), (90, (360, 280))]):
            img = S.scene([(centre, spec, angle, 1.0)], seed=g)
            regs = win.engine.segment_frame(img).regions
            crop = region_crop(img, regs[0]) if regs else win.tray.crop(img)[0]
            e = win.extractor.embed_batch(win.extractor.augment(crop))
            embs.append(e); groups.append(np.full(len(e), g, np.int32))
        win.db.add_item(spec[0], 50000 if spec is LOOKALIKE_A else 30000,
                        np.vstack(embs), None, np.concatenate(groups), MODE_OBJECT)
    win.db.save(); win._refresh_items_ui()
    check("two look-alike items enrolled", len(win.db.items_in_mode(MODE_OBJECT)) == 2)

    print("\n[2] the look-alike chooser")
    cam.set([((300, 240), LOOKALIKE_A, 15, 1.0)])
    win.sim_spin.setValue(0.100)
    wait(lambda: win.current_rec is not None and win.current_rec.mode == MODE_OBJECT
         and win.current_rec.confirmed, 60, "object seen in per-object mode")
    pump(0.6)
    rec = win.current_rec
    det = win._active_detection()
    print(f"     mode={rec.mode} confirmed={len(rec.confirmed)} accepted={len(rec.accepted)} "
          f"suggested={det.suggested if det else None} "
          f"ranking={[(n, round(sc, 2)) for _, n, sc in (det.result.ranking if det else [])]}")
    check("the object is found", det is not None and det.confirmed)
    check("it is either billed or offered", det is not None and (det.accepted or det.suggested),
          "neither accepted nor suggested")
    check("alternatives are offered", win.cand_box.isVisible(), "candidate bar hidden")
    visible = [b for b in win.cand_buttons if b.isVisible()]
    check("both look-alikes are listed", len(visible) >= 2, str([b.text() for b in visible]))
    win.grab().save(SHOT)

    print("\n[3] cumulative weighing: A then B")
    if not det.accepted:
        # the model is unsure, so the operator taps the right one - which is the
        # whole point of the chooser
        btn = next((b for b in visible if win.db.get(b.property("item_id")).name == "apple_red"), visible[0])
        win._choose_candidate(btn)
        check("tapping a candidate sets the choice", win._override_item_id is not None)
    win.add_detected()
    check("first line added", win.table.rowCount() == 1, str(win.table.rowCount()))
    check("invoice holds 100 g", abs(win.invoiced_weight() - 0.100) < 1e-9, str(win.invoiced_weight()))

    # second product goes on top of the first: the tray reads 300 g
    cam.set([((230, 190), LOOKALIKE_A, 0, 1.0), ((400, 300), LOOKALIKE_B, 30, 1.0)])
    settings.set("pos.multi_item_mode", "off")        # allow pricing with two on the tray for this step
    win.sim_spin.setValue(0.300)
    wait(lambda: win.current_rec is not None and len(win.current_rec.confirmed) >= 2, 60, "two objects")
    pump(0.6)
    check("the new item weighs the difference", abs((win.weight_difference() or 0) - 0.200) < 1e-9,
          str(win.weight_difference()))
    det2 = win._active_detection()
    if det2 is not None and not det2.accepted:
        vis = [b for b in win.cand_buttons if b.isVisible()]
        if vis:
            win._choose_candidate(vis[0])
    win.add_detected()
    check("second line added", win.table.rowCount() == 2, str(win.table.rowCount()))
    check("second line is 200 g", abs(win.lines[1].weight_kg - 0.200) < 1e-9, str(win.lines[1].weight_kg))
    check("invoice total is 300 g", abs(win.invoiced_weight() - 0.300) < 1e-9)
    check("tray and invoice agree", abs(win.weight_difference()) < 1e-9)
    check("reconciliation says so", tr("weight_matches") in win.recon_lbl.text(), win.recon_lbl.text())

    print("\n[4] tamper: an unbilled item appears on the tray")
    win.sim_spin.setValue(0.480)
    pump(0.4)
    win._update_reconciliation()
    check("the extra 180 g is reported", abs((win.weight_difference() or 0) - 0.180) < 1e-9,
          str(win.weight_difference()))
    check("the line turns to a warning", tr("new_weight") in win.recon_lbl.text(), win.recon_lbl.text())
    messages.clear()
    before_rows = win.table.rowCount()
    win.checkout()
    pump(1.0)
    asked = [m for m in messages if m[0] == "ask" and tr("checkout_anyway") in m[1]]
    check("checkout warns about the mismatch", bool(asked), str(messages[-2:]))

    print("\n[5] matching tray finishes cleanly")
    win.lines.clear(); win._refresh_table()
    win.sim_spin.setValue(0.0)
    pump(0.3)
    win.sim_spin.setValue(0.250)
    pump(0.3)
    wait(lambda: win.current_rec is not None and win.current_rec.confirmed, 40, "object seen again")
    pump(0.4)
    d5 = win._active_detection()
    if d5 is not None and not d5.accepted:
        vis = [b for b in win.cand_buttons if b.isVisible()]
        if vis:
            win._choose_candidate(vis[0])
    win.add_detected()
    check("one line", win.table.rowCount() == 1)
    messages.clear()
    win.checkout(); pump(1.2)
    asked = [m for m in messages if m[0] == "ask" and tr("checkout_anyway") in m[1]]
    check("a matching tray checks out without a warning", not asked, str(messages[-2:]))
    files = [f for dp, _, fs in os.walk(os.path.join(TMP, "invoices")) for f in fs if f.startswith("invoice_")]
    check("invoices written", len(files) >= 1, str(files))

    print("\n[6] diagnostic bundle")
    messages.clear()
    win.save_diagnostics()
    zips = [f for f in os.listdir(A.app_dir()) if f.startswith("aipos_diagnostics_") and f.endswith(".zip")]
    check("a bundle was written", bool(zips), str(zips))
    if zips:
        path = os.path.join(A.app_dir(), sorted(zips)[-1])
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
        print("     contents:", ", ".join(sorted(names)))
        for required in ("frame.png", "tray.png", "recognition.txt", "items.txt", "config.json", "README.txt"):
            check(f"bundle contains {required}", required in names)
        check("bundle contains an object crop", any(n.startswith("object_") for n in names), str(names))
        check("bundle contains the empty-tray reference", "empty_tray.png" in names)
        os.remove(path)

    print("\n[7] object outlines are drawn")
    check("mask outlines are on by default", win.video.show_masks)
except Exception:
    traceback.print_exc(); results["ok"] = False
finally:
    if FAILED:
        print("\nfailed checks:")
        for name, detail in FAILED:
            print(f"  - {name}   {detail}")
    print("\nRESULT:", "PASS" if results["ok"] else "FAIL")
    try:
        win.close(); pump(0.4)
    except Exception:
        pass
    sys.exit(0 if results["ok"] else 1)
