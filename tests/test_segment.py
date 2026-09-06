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


def test_timing():
    print("\n[5] per-frame cost")
    model, seg = build()
    img = S.scene([(p, S.CLASSES[i], 0.0, 1.0) for i, p in enumerate(S.positions(3, seed=3))], seed=3)
    seg.segment(img)
    times = []
    for _ in range(12):
        t = time.perf_counter()
        seg.segment(img)
        times.append((time.perf_counter() - t) * 1000)
    mean, p95 = float(np.mean(times)), float(np.percentile(times, 95))
    check("mean frame cost stays in budget", mean <= 60.0, f"{mean:.1f} ms mean, {p95:.1f} ms p95")


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
