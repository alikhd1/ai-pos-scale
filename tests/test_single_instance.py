"""
test_single_instance.py
-----------------------
Only one till per machine.

Someone opened the app a second time by hand while a copy was running. Both
took the camera, both posted support snapshots, and both had the items
database open to write. F6 was pressed in one window while the other was the
one reporting, so the capture looked like it did nothing - and the obvious
reading, that the new code was broken, was wrong. Two tills writing one
catalogue is also a way to lose a shop's products.

What is checked here, with real processes rather than mocks, because the
whole point is that the operating system enforces it:

  * a second process cannot take the lock while the first holds it,
  * the lock is released when that process dies, including when it is killed,
    so a crash cannot leave a till that will not start,
  * a new instance waits rather than refusing instantly - a remote update
    starts the replacement before the old copy has finished closing, and
    refusing would leave the shop with no till and no way in to fix it.

Run:  python tests/test_single_instance.py
"""
import os, sys, time, textwrap, tempfile, shutil, subprocess, traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

ok = {"all": True}
FAILED = []
def check(name, cond, detail=""):
    if not cond:
        ok["all"] = False
        FAILED.append((name, detail))
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))


TMP = tempfile.mkdtemp(prefix="aipos_instance_")
HOLDER = os.path.join(TMP, "holder.py")
with open(HOLDER, "w", encoding="utf-8") as fh:
    fh.write(textwrap.dedent(f"""
        import os, sys, time
        os.environ["AIPOS_DATA_DIR"] = r"{TMP}"
        sys.path.insert(0, r"{ROOT}")
        import app
        app.data_path = lambda name: os.path.join(r"{TMP}", name)
        held = app.acquire_single_instance(wait_s=0)
        print("HELD" if held else "BUSY", flush=True)
        if held:
            time.sleep(float(sys.argv[1]) if len(sys.argv) > 1 else 30)
    """).strip())


def start_holder(seconds=30):
    p = subprocess.Popen([sys.executable, HOLDER, str(seconds)],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    first = p.stdout.readline().strip()
    return p, first


try:
    import app
    app.data_path = lambda name: os.path.join(TMP, name)

    print("\n[1] one process holds it, a second is refused")
    holder, said = start_holder(30)
    check("the first process took the lock", said == "HELD", said)
    second = app.acquire_single_instance(wait_s=0)
    check("a second cannot take it", second is None, str(second))

    print("\n[2] and it is released the moment that process dies")
    holder.kill()                       # the harshest ending: no cleanup runs
    holder.wait(timeout=10)
    deadline = time.time() + 10
    got = None
    while time.time() < deadline and got is None:
        got = app.acquire_single_instance(wait_s=0)
        if got is None:
            time.sleep(0.2)
    check("a killed process does not leave it stuck", got not in (None, False), str(got))
    if hasattr(got, "close"):
        got.close()

    print("\n[3] a new instance waits instead of refusing outright")
    # this is what makes remote updates survivable: the replacement starts
    # before the old copy has finished shutting down
    holder, said = start_holder(2)
    check("the holder has it again", said == "HELD", said)
    t0 = time.time()
    waited = app.acquire_single_instance(wait_s=12)
    elapsed = time.time() - t0
    check("it eventually got in", waited not in (None, False), str(waited))
    check("after waiting, not instantly", elapsed > 1.0, f"{elapsed:.1f}s")
    check("and not for the whole timeout", elapsed < 11.0, f"{elapsed:.1f}s")
    if hasattr(waited, "close"):
        waited.close()
    holder.wait(timeout=10)

    print("\n[4] it never blocks the till over a file problem")
    app.data_path = lambda name: os.path.join(TMP, "no", "such", "place", name)
    fallback = app.acquire_single_instance(wait_s=0)
    check("an unusable path is not treated as 'already running'", fallback is not None, str(fallback))

except Exception:
    ok["all"] = False
    traceback.print_exc()
finally:
    try:
        holder.kill()
    except Exception:
        pass
    shutil.rmtree(TMP, ignore_errors=True)

print("\n" + ("ALL CHECKS PASSED" if ok["all"] else
              (f"{len(FAILED)} CHECK(S) FAILED" if FAILED else "CRASHED - the run stopped part way; see the traceback above")))
for name, detail in FAILED:
    print(f"  - {name} {detail}")
sys.exit(0 if ok["all"] else 1)
