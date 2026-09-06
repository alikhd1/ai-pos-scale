"""
test_matcher.py
---------------
Gates on the recognition matcher and on database migration.

The two properties that matter commercially:
  * top-1 accuracy on enrolled produce, and
  * the false-accept rate on produce that was never enrolled - which was 1.000
    before per-item calibrated thresholds existed, i.e. an un-enrolled mango was
    always billed as something else, silently.

Also proves that a version-2 items_db.pkl (no capture groups, no thresholds, no
tray image) still loads and keeps every sample.

Run:  python tests/test_matcher.py
"""
from __future__ import annotations

import os
import pickle
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import scene as S
from ai_engine import (EMBEDDING_DIM, FeatureExtractor, ItemDatabase, MODE_OBJECT, Recognizer, TrayRegion,
                       region_crop)
from tray_segment import EmptyTrayModel, TraySegmenter

CHECKS = []


def check(name, condition, detail=""):
    CHECKS.append((name, bool(condition), detail))
    print(f"  {'PASS' if condition else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))


def crop_for(fx, seg, spec, centre, angle, scale, seed, gain=1.0, awb=(1, 1, 1)):
    img = S.scene([(centre, spec, angle, scale)], seed=seed, gain=gain, awb=awb)
    regions = seg.segment(img).regions
    return region_crop(img, regions[0]) if regions else TrayRegion().crop(img)[0]


def build_world(fx):
    frames = S.empty_frames(9)
    model = EmptyTrayModel.build(frames, S.TRAY_BOX, S.TRAY_POLY, work=192)
    seg = TraySegmenter(model)
    db = ItemDatabase(os.path.join(tempfile.gettempdir(), "aipos_test_matcher.pkl"), autoload=False)
    db.set_background(fx.embed_batch(fx.augment(TrayRegion().crop(frames[0])[0])), None, model.to_payload())
    # four captures per item, each from a different angle and position
    for spec in S.CLASSES:
        embs, groups = [], []
        for g, (angle, centre) in enumerate([(0, (300, 240)), (35, (250, 200)), (75, (360, 280)), (120, (300, 300))]):
            e = fx.embed_batch(fx.augment(crop_for(fx, seg, spec, centre, angle, 1.0, seed=g)))
            embs.append(e)
            groups.append(np.full(len(e), g, np.int32))
        db.add_item(spec[0], 40000, np.vstack(embs), None, np.concatenate(groups), MODE_OBJECT)
    return db, seg


def test_accuracy_and_rejection(fx):
    print("\n[1] top-1 accuracy and unknown rejection")
    db, seg = build_world(fx)
    rec = Recognizer(db, 0.65)
    thresholds = {i.name: round(i.threshold, 3) for i in db.all_items()}
    print(f"    calibrated thresholds: {thresholds}")
    check("every item got a positive calibrated threshold", all(v > 0 for v in thresholds.values()), str(thresholds))

    correct = total = 0
    for spec in S.CLASSES:
        for angle, centre, gain in ((20, (280, 220), 1.0), (95, (380, 300), 0.85), (150, (240, 300), 1.15)):
            crop = crop_for(fx, seg, spec, centre, angle, 1.05, seed=50, gain=gain)
            m = rec.match(fx.embed(crop), MODE_OBJECT)
            total += 1
            if m.accepted and m.name == spec[0]:
                correct += 1
    acc = correct / total
    check("top-1 accuracy on enrolled produce", acc >= 0.80, f"{acc:.0%} ({correct}/{total})")

    false_accepts = 0
    trials = 6
    for i in range(trials):
        crop = crop_for(fx, seg, S.UNKNOWN, (300, 240), i * 30, 1.0, seed=60 + i)
        if rec.match(fx.embed(crop), MODE_OBJECT).accepted:
            false_accepts += 1
    far = false_accepts / trials
    check("never-enrolled produce is rejected", far <= 0.35, f"false-accept rate {far:.0%} ({false_accepts}/{trials})")
    return db


def test_matcher_variants(fx):
    print("\n[2] the shipped matcher beats plain cosine on the same data")
    db, seg = build_world(fx)
    rec = Recognizer(db, 0.65)
    raw_hits = new_hits = total = 0
    gallery = db.gallery(MODE_OBJECT)
    ids, raw = gallery["ids"], gallery["raw"]
    for spec in S.CLASSES:
        for angle, centre in ((20, (280, 220)), (95, (380, 300))):
            e = fx.embed(crop_for(fx, seg, spec, centre, angle, 1.05, seed=70))
            # plain cosine against every stored sample, the pre-existing rule
            best, best_score = None, -2.0
            for iid, block in zip(ids, raw):
                s = float(np.max(block @ e))
                if s > best_score:
                    best, best_score = iid, s
            item = db.get(best)
            raw_hits += int(item is not None and item.name == spec[0])
            m = rec.match(e, MODE_OBJECT)
            new_hits += int(m.name == spec[0])
            total += 1
    check("the two-headed matcher is at least as accurate as raw cosine",
          new_hits >= raw_hits, f"new {new_hits}/{total} vs raw {raw_hits}/{total}")
    print(f"    raw nearest-sample cosine {raw_hits}/{total}, shipped matcher {new_hits}/{total}")


def test_cost(fx):
    print("\n[3] matching cost")
    db, seg = build_world(fx)
    rec = Recognizer(db, 0.65)
    e = fx.embed(crop_for(fx, seg, S.CLASSES[0], (300, 240), 0, 1.0, seed=80))
    rec.match(e, MODE_OBJECT)
    t = time.perf_counter()
    for _ in range(200):
        rec.match(e, MODE_OBJECT)
    ms = (time.perf_counter() - t) * 1000 / 200
    check("matching is negligible beside the network", ms <= 2.0, f"{ms:.3f} ms per match")


def test_v2_migration():
    print("\n[4] a version-2 database still loads")
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "items_db.pkl")
        rng = np.random.default_rng(0)
        emb = rng.standard_normal((15, EMBEDDING_DIM)).astype(np.float32)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        legacy = {"version": 2, "saved": 0.0,
                  "items": [{"id": "abc123", "name": "سیب قرمز", "price_per_kg": 45000.0,
                             "embeddings": emb, "thumbnail": None, "created": 0.0, "updated": 0.0, "sessions": 3}],
                  "background": None}
        with open(path, "wb") as fh:
            pickle.dump(legacy, fh)
        db = ItemDatabase(path)
        check("it loads without an exception", db.load_error is None, str(db.load_error))
        items = db.all_items()
        check("the item survived", len(items) == 1 and items[0].name == "سیب قرمز")
        it = items[0]
        check("every sample survived", it.sample_count == 15, f"{it.sample_count}")
        check("capture groups were reconstructed", it.capture_count == 3, f"{it.capture_count} captures")
        check("it is treated as a whole-tray item", it.crop_mode == "tray", it.crop_mode)
        check("no tray image, so segmentation stays off", db.empty_tray_model() is None)
        db.save()
        again = ItemDatabase(path)
        check("it round-trips through version 3", again.all_items()[0].sample_count == 15)


def test_backbone_guard():
    print("\n[5] a database from a different backbone refuses to load its items")
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "items_db.pkl")
        legacy = {"version": 3, "backbone": "resnet18", "items": [
            {"id": "x", "name": "x", "price_per_kg": 1.0, "embeddings": np.zeros((5, 512), np.float32)}], "background": None}
        with open(path, "wb") as fh:
            pickle.dump(legacy, fh)
        db = ItemDatabase(path)
        check("the mismatch is reported, not raised", db.backbone_mismatch == "resnet18", str(db.backbone_mismatch))
        check("no item is loaded with the wrong dimensions", len(db) == 0)


def main() -> int:
    print("=" * 70)
    print("matcher tests")
    print("=" * 70)
    t0 = time.time()
    fx = FeatureExtractor(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                       "models", "mobilenet_v2-b0353104.pth"), num_threads=1)
    print(f"  model loaded in {time.time() - t0:.1f}s")
    test_accuracy_and_rejection(fx)
    test_matcher_variants(fx)
    test_cost(fx)
    test_v2_migration()
    test_backbone_guard()
    failed = [c for c in CHECKS if not c[1]]
    print("\n" + "=" * 70)
    print(f"{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
    for name, _, detail in failed:
        print(f"  FAILED: {name}   {detail}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
