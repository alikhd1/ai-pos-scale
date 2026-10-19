"""
test_segment.py
---------------
Regression gates for multi-object tray segmentation.

The thresholds are deliberately below the measured values: they exist to catch a
regression (a parameter "cleaned up", an OpenCV behaviour change), not to assert
a level of accuracy on real produce.  Two parameters in tray_segment.py are
load-bearing and this file is what protects them:

  * the small seed-suppression radius in _split_blob (the textbook
    dist > 0.5 * dist.max() merges two touching items into one seed),
  * the MIN_VAR floor on the local NCC shadow test (without it the correlation
    is noise on smooth objects).

Run:  python tests/test_segment.py
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import scene as S
from tray_segment import EmptyTrayModel, RegionTracker, TraySegmenter
from ai_engine import TrayRegion

CHECKS = []


def check(name, condition, detail=""):
    CHECKS.append((name, bool(condition), detail))
    print(f"  {'PASS' if condition else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))


def build(tray_seed=7, textured=True):
    frames = S.empty_frames(9, tray_seed=tray_seed, textured=textured)
    model = EmptyTrayModel.build(frames, S.TRAY_BOX, S.TRAY_POLY, work=192)
    return model, TraySegmenter(model)


def confirmed_count(seg, img, rounds: int = 3) -> int:
    """Run the 2-of-3 temporal filter the app uses before it believes a region."""
    tracker = RegionTracker()
    result = None
    for _ in range(rounds):
        result = seg.segment(img)
        tracker.update(result.regions)
    return len([r for r in result.regions if r.confirmed]) if result else 0


def test_empty_tray():
    print("\n[1] an empty tray produces no objects")
    model, seg = build()
    false_regions = 0
    for i in range(6):
        img = S.scene([], seed=500 + i)
        false_regions += confirmed_count(seg, img)
    check("no confirmed region on an empty tray", false_regions == 0, f"{false_regions} false regions in 6 frames")
    r = seg.segment(S.scene([], seed=777))
    check("the result says empty", r.empty and not r.intrusion)


def test_counts_and_boxes():
    print("\n[2] one to five separated objects")
    model, seg = build()
    for n in (1, 2, 3, 4, 5):
        pos = S.positions(n, seed=n)
        specs = [S.CLASSES[i % len(S.CLASSES)] for i in range(n)]
        img = S.scene([(p, sp, 0.0, 1.0) for p, sp in zip(pos, specs)], seed=n)
        found = seg.segment(img)
        floor = {1: 1, 2: 2, 3: 2, 4: 3, 5: 3}[n]
        check(f"{n} objects -> at least {floor} regions", found.count >= floor,
              f"found {found.count} in {found.ms:.0f} ms")
        check(f"{n} objects -> never more than {n}", found.count <= n, f"found {found.count}")
        if n == 1:
            truth = S.object_box(pos[0], specs[0])
            best = max((S.iou(truth, r.box) for r in found.regions), default=0.0)
            check("the single box lands on the object", best >= 0.55, f"IoU {best:.2f}")


def test_hand_intrusion():
    print("\n[3] a hand over the tray is always flagged")
    model, seg = build()
    hits = 0
    for i in range(5):
        img = S.scene([((300, 240), S.CLASSES[0], 0.0, 1.0)], seed=900 + i, hand=True)
        if seg.segment(img).intrusion:
            hits += 1
    check("intrusion detected in every frame", hits == 5, f"{hits}/5")


def test_lighting_drift():
    print("\n[4] gain and white-balance drift never produce a confident wrong answer")
    model, seg = build()
    for gain, awb, label in ((0.62, (1, 1, 1), "dim"), (1.35, (1, 1, 1), "bright"),
                             (1.8, (1, 1, 1), "blown out"), (1.0, (1.12, 1.0, 0.9), "warm")):
        img = S.scene([((300, 240), S.CLASSES[0], 0.0, 1.0)], seed=11, gain=gain, awb=awb)
        r = seg.segment(img)
        ok = r.unreliable or r.count >= 1
        check(f"{label} light: either found or flagged unreliable", ok,
              f"count={r.count} unreliable={r.unreliable}")
        if not r.unreliable:
            check(f"{label} light: no phantom objects", r.count <= 2, f"count={r.count}")
        if label == "blown out":
            check("a saturated frame is reported as unreliable, not as an empty tray", r.unreliable,
                  f"unreliable={r.unreliable} count={r.count}")


#: segmentation cost divided by the calibration cost, measured on an idle
#: reference machine. A ratio rather than milliseconds, because the two are
#: timed alternately and a machine that slows down moves both together.
REFERENCE_RATIO = 3.0
#: how far past that is still allowed. Deliberately loose: this exists to catch
#: segmentation becoming fundamentally slower, not to police a few per cent.
RATIO_CEILING = 4.0


def calibration_ms() -> float:
    """Time a fixed lump of array work: a stand-in for 'how fast is this box now'.

    The same shape of work segmentation does - float maths over a few hundred
    thousand pixels - so load and CPU speed move this and the measurement
    together.
    """
    rng = np.random.default_rng(0)
    a = rng.random((480, 640), dtype=np.float32)
    b = rng.random((480, 640), dtype=np.float32)
    t = time.perf_counter()
    for _ in range(6):
        np.sqrt(np.abs(a - b) + 1.0).sum()
    return (time.perf_counter() - t) * 1000


def test_timing():
    """Report the per-frame cost, and fail only on a real regression.

    An absolute millisecond budget was tried first and had to go: on this
    project's VM the same code measured 22 ms on one run and 100 ms on the
    next, so the check went red at random and the honest reading of a red
    result became "probably just load" - which is exactly how a genuine
    regression gets waved through. Calibration and segmentation are now timed
    alternately and compared as a ratio, and the threshold is loose enough that
    only something structural trips it.
    """
    print("\n[5] per-frame cost")
    model, seg = build()
    img = S.scene([(p, S.CLASSES[i], 0.0, 1.0) for i, p in enumerate(S.positions(3, seed=3))], seed=3)
    seg.segment(img)
    times, ratios = [], []
    for _ in range(12):
        calib = calibration_ms()
        t = time.perf_counter()
        seg.segment(img)
        frame_ms = (time.perf_counter() - t) * 1000
        times.append(frame_ms)
        ratios.append(frame_ms / max(calib, 0.01))
    median, p95 = float(np.median(times)), float(np.percentile(times, 95))
    ratio = float(np.median(ratios))
    print(f"     {median:.1f} ms median, {p95:.1f} ms p95 on this machine "
          f"(work ratio {ratio:.1f}, reference {REFERENCE_RATIO})")
    check("segmentation has not become structurally slower",
          ratio <= REFERENCE_RATIO * RATIO_CEILING,
          f"ratio {ratio:.1f} vs reference {REFERENCE_RATIO} "
          f"(limit {REFERENCE_RATIO * RATIO_CEILING:.0f}), {median:.1f} ms median")


def test_stale_background_is_not_a_hand():
    """Light that moved since the capture must not be reported as an arm.

    From a real till: one product was placed and the app said "take your hand
    off the tray", finding nothing. Measuring its own frame against its own
    reference showed plain paper away from the product reading 23 to 38 levels
    brighter, against a tolerance near 16. The light had changed since the
    reference was captured, and the shift was a gradient across the frame that
    a single gain and offset cannot undo. Nine tenths of the tray counted as
    changed, that one blob reached past the tray edge, and "reaching in from
    outside" is what the code calls an arm.

    What is pinned here is the rule, not the optics: when nearly the whole tray
    reads as changed, the reference is stale - no arrangement of products
    covers that much - so say so instead of blaming a hand. Reproducing the
    gradient synthetically does not work, because this fixture's background is
    textured and the photometric alignment copes with it; the evidence that
    the rule matches reality is the till's own frame, which went from
    intrusion=True to unreliable=True with no regions either way.
    """
    print("\n[8] a reference the tray no longer matches reads as stale, not as a hand")
    model, seg = build()
    print(f"     stale above {100 * seg.stale_background_frac:.0f}% of the tray changing at once")

    # a view the reference cannot explain at all
    nothing_alike = (255 - S.scene([], seed=11)).astype(np.uint8)
    res = seg.segment(nothing_alike)
    check("it is not called an intrusion", res.intrusion is False, str(res.intrusion))
    check("it is flagged unreliable", res.unreliable is True, str(res.unreliable))
    check("not reported as a clean empty tray", res.empty is False, str(res.empty))
    check("and offers no phantom products", len(res.regions) == 0, str(len(res.regions)))

    # the two things it must still tell apart
    model2, seg2 = build()
    res_hand = seg2.segment(S.scene([((300, 240), S.CLASSES[0], 0.0, 1.0)], hand=True, seed=12))
    check("a real hand is still an intrusion", res_hand.intrusion is True,
          f"intrusion={res_hand.intrusion} unreliable={res_hand.unreliable}")
    check("and a hand is not mistaken for a stale reference", res_hand.unreliable is False,
          str(res_hand.unreliable))

    model3, seg3 = build()
    res_full = seg3.segment(S.scene([(p, S.CLASSES[i], 0.0, 1.0)
                                     for i, p in enumerate(S.positions(5, seed=13))], seed=13))
    check("a tray full of products is not called stale", res_full.unreliable is False,
          f"regions={len(res_full.regions)} unreliable={res_full.unreliable}")
    check("and they are all still found", len(res_full.regions) >= 3, str(len(res_full.regions)))


def test_stability_report():
    """A reference built from an unsteady view has to admit it.

    A real till built one whose noise estimate was saturated across a third of
    the tray and the app reported success; everything after that was
    meaningless, and from the shop it looked like "the camera does not detect
    my goods". The numbers here bracket that case: a steady synthetic view must
    read "good", and a view that jitters in brightness between captures - which
    is what a glossy surface plus a hunting auto-exposure produces - must not.
    """
    print("\n[9] an unsteady view is reported, not silently accepted")
    steady, _ = build()
    good = steady.stability()
    print(f"     steady view: median {good['median']:.1f}, {100 * good['fraction_noisy']:.0f}% noisy "
          f"-> {good['verdict']}")
    check("a steady view reads good", good["verdict"] == "good", str(good))

    # the same tray, but each capture at a different exposure and tint, which
    # is exactly what an auto-exposing camera on a shiny surface delivers
    frames = []
    for i in range(6):
        f = S.scene([], seed=i).astype(np.float32)
        gain = 1.0 + 0.30 * ((i % 3) - 1)
        f[:, :, 0] *= gain * 1.08
        f[:, :, 2] *= gain * 0.92
        frames.append(np.clip(f * gain, 0, 255).astype(np.uint8))
    h, w = frames[0].shape[:2]
    tray = TrayRegion([(x / S.FRAME_W, y / S.FRAME_H) for x, y in S.TRAY_POLY], True, 0.8)
    jittery = EmptyTrayModel.build(frames, tray.bbox(w, h), tray.pixel_points(w, h).tolist(), work=192)
    bad = jittery.stability()
    print(f"     hunting exposure: median {bad['median']:.1f}, {100 * bad['fraction_noisy']:.0f}% noisy "
          f"-> {bad['verdict']}")
    check("a jittering view is not called good", bad["verdict"] != "good", str(bad))
    check("and it is noisier than the steady one", bad["median"] > good["median"],
          f"{bad['median']:.1f} vs {good['median']:.1f}")
    for key in ("median", "p95", "verdict", "fraction_noisy", "saturated", "reason"):
        check(f"the report carries {key}", key in bad)

    # The trap a real till fell into, and I with it: its reference scored the
    # lowest noise yet and was called good. It was 99.7% clipped white. A
    # clipped pixel reads 255 whatever happens, so it has no variance, and a
    # picture with no information in it looks perfectly steady.
    blown = [np.clip(S.scene([], seed=i).astype(np.float32) * 3.0 + 120, 0, 255).astype(np.uint8)
             for i in range(6)]
    hb, wb = blown[0].shape[:2]
    over = EmptyTrayModel.build(blown, tray.bbox(wb, hb), tray.pixel_points(wb, hb).tolist(), work=192)
    rep = over.stability()
    print(f"     blown-out view: median {rep['median']:.1f}, {100 * rep['saturated']:.0f}% clipped "
          f"-> {rep['verdict']} ({rep['reason']})")
    check("a blown-out view is NOT called good, however steady it looks", rep["verdict"] != "good", str(rep))
    check("it is named as overexposed", rep["reason"] == "overexposed", str(rep))
    check("even though its noise really is low", rep["median"] < 8.0, f"{rep['median']:.1f}")


def test_no_model_is_off_not_wrong():
    print("\n[6] a resolution change is detected instead of guessed at")
    model, seg = build()
    small = S.scene([((300, 240), S.CLASSES[0], 0.0, 1.0)], seed=5)[:240, :320]
    r = seg.segment(small)
    check("a frame of the wrong size is unreliable and empty", r.unreliable and r.count == 0,
          f"count={r.count} unreliable={r.unreliable}")


def test_persistence():
    print("\n[7] the empty-tray model survives a save/load cycle")
    model, seg = build()
    img = S.scene([(p, S.CLASSES[i], 0.0, 1.0) for i, p in enumerate(S.positions(2, seed=2))], seed=2)
    before = seg.segment(img).count
    payload = model.to_payload()
    size_kb = sum(len(v) for v in payload.values() if isinstance(v, bytes)) / 1024
    back = EmptyTrayModel.from_payload(payload)
    after = TraySegmenter(back).segment(img).count if back else -1
    check("the reloaded model finds the same objects", after == before, f"{before} -> {after}")
    check("the stored model stays small", size_kb < 400, f"{size_kb:.0f} KB")


def main() -> int:
    print("=" * 70)
    print("tray segmentation tests")
    print("=" * 70)
    test_empty_tray()
    test_counts_and_boxes()
    test_hand_intrusion()
    test_lighting_drift()
    test_timing()
    test_stale_background_is_not_a_hand()
    test_stability_report()
    test_no_model_is_off_not_wrong()
    test_persistence()
    failed = [c for c in CHECKS if not c[1]]
    print("\n" + "=" * 70)
    print(f"{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
    for name, _, detail in failed:
        print(f"  FAILED: {name}   {detail}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
