"""
test_diagnostics.py
-------------------
The crash evidence a diagnostic bundle carries, and the limits on it.

A shop's till blue-screened repeatedly and the only way to find out why was a
person at the till pasting console output back. The bundle now carries the
Windows bug-check events, the third-party driver list and the crash dumps
itself. Because a remote request can name which dumps it wants, this also pins
down that it can only ever take files that are really in the Minidump folder:
a name that tries to climb out of it, or one that does not exist, is reported
and refused, never followed.

Run:  python tests/test_diagnostics.py
"""
import os, sys, io, zipfile, tempfile, shutil, traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

ok = {"all": True}
FAILED = []
def check(name, cond, detail=""):
    if not cond:
        ok["all"] = False
        FAILED.append((name, detail))
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))


root = tempfile.mkdtemp(prefix="aipos_dumps_")
saved = os.environ.get("SystemRoot")
try:
    import diagnostics as D

    def make(*named):
        folder = os.path.join(root, "Minidump")
        shutil.rmtree(folder, ignore_errors=True)
        os.makedirs(folder)
        for i, (name, kb) in enumerate(named):
            p = os.path.join(folder, name)
            with open(p, "wb") as fh:
                fh.write(b"\0" * kb * 1024)
            os.utime(p, (1_700_000_000 + i * 60, 1_700_000_000 + i * 60))   # later = newer

    def bundle(wanted=None):
        os.environ["SystemRoot"] = root
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            D._add_crash_evidence(z, wanted)
        z = zipfile.ZipFile(buf)
        dumps = sorted(n[len("minidump/"):] for n in z.namelist() if n.startswith("minidump/"))
        return dumps, z.read("dumps.txt").decode("utf-8"), z.namelist()

    print("\n[1] by default the newest dumps go, within the size cap")
    make(("old1.dmp", 800), ("old2.dmp", 800), ("mid.dmp", 900), ("new1.dmp", 1000), ("new2.dmp", 1100))
    dumps, notes, names = bundle()
    check("the newest are taken first", "new2.dmp" in dumps and "new1.dmp" in dumps, str(dumps))
    check("the cap is respected", len(dumps) <= D.MINIDUMP_MAX_FILES, str(dumps))
    check("the ones left out are still listed", "old1.dmp" in notes and "listed only" in notes, notes)
    for f in ("windows_crashes.txt", "drivers.txt", "dumps.txt"):
        check(f"the bundle carries {f}", f in names, str(names))

    print("\n[2] a request can name the dumps it wants - the decisive one may be old")
    dumps, notes, _ = bundle(["old1.dmp", "mid.dmp"])
    check("exactly the named dumps are taken", dumps == ["mid.dmp", "old1.dmp"], str(dumps))

    print("\n[3] it can only ever take files that are really in the Minidump folder")
    with open(os.path.join(root, "secret.txt"), "w") as fh:
        fh.write("not a crash dump")
    dumps, notes, _ = bundle(["../secret.txt", "..\\secret.txt", "nope.dmp", "old2.dmp"])
    check("nothing outside the folder is read", dumps == ["old2.dmp"], str(dumps))
    check("and what was refused is reported", "secret.txt" in notes and "nope.dmp" in notes, notes)

    print("\n[4] a missing Minidump folder is explained, not an error")
    shutil.rmtree(os.path.join(root, "Minidump"))
    dumps, notes, _ = bundle()
    check("no dumps", dumps == [], str(dumps))
    check("and it says why", "does not exist" in notes, notes)

except Exception:
    ok["all"] = False
    traceback.print_exc()
finally:
    if saved is None:
        os.environ.pop("SystemRoot", None)
    else:
        os.environ["SystemRoot"] = saved
    shutil.rmtree(root, ignore_errors=True)

print("\n" + ("ALL CHECKS PASSED" if ok["all"] else
              (f"{len(FAILED)} CHECK(S) FAILED" if FAILED else
               "CRASHED - the run stopped part way; see the traceback above")))
for name, detail in FAILED:
    print(f"  - {name} {detail}")
sys.exit(0 if ok["all"] else 1)
