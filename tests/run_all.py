"""
run_all.py
----------
Runs every head-less regression suite.  No hardware, no pytest, no network.

    python tests/run_all.py

Exit code 0 means every check passed.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SUITES = ["test_scale_wiring.py", "test_segment.py", "test_matcher.py"]


def main() -> int:
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", PYTHONIOENCODING="utf-8")
    env.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")
    results = []
    for suite in SUITES:
        print("\n" + "#" * 70)
        print("# " + suite)
        print("#" * 70)
        t0 = time.time()
        proc = subprocess.run([sys.executable, "-X", "utf8", os.path.join(HERE, suite)], env=env)
        results.append((suite, proc.returncode, time.time() - t0))
    print("\n" + "=" * 70)
    print("SUMMARY")
    for suite, code, secs in results:
        print(f"  {'PASS' if code == 0 else 'FAIL'}  {suite:24} {secs:5.1f}s")
    failed = [s for s, c, _ in results if c != 0]
    print("=" * 70)
    print("ALL SUITES PASSED" if not failed else f"FAILED: {', '.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
