"""
test_update.py
--------------
The remote update, and the way back from a bad one.

A till in a shop has no shell on it - remote access was declined - so if an
update leaves the app unable to start, nobody can fix it from here and the
shop has a dead till. The guard is therefore the feature: the new code is
compiled before it is allowed to run, and the checkout is reset if it is not
sound.

These checks run against a throwaway git repository, so nothing here can
touch the real one:

  * a fast-forward is applied and reported with before/after commits,
  * code that does not compile is rolled back and said so, not restarted,
  * a checkout that is not a git repository is refused rather than guessed at,
  * nothing restarts unless it was asked for AND the pull actually moved.

Run:  python tests/test_update.py
"""
import os, sys, shutil, tempfile, subprocess, traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

ok = {"all": True}
FAILED = []
def check(name, cond, detail=""):
    if not cond:
        ok["all"] = False
        FAILED.append((name, detail))
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))


# a fresh directory per run. git marks its object files read-only, and
# shutil.rmtree(ignore_errors=True) then leaves them behind on Windows, so a
# fixed path made the next run fail on the last run's leftovers.
TMP = tempfile.mkdtemp(prefix="aipos_update_")


def remove_tree(path):
    """Delete a git checkout on Windows, read-only object files and all."""
    def force(func, target, _exc):
        try:
            os.chmod(target, 0o700)
            func(target)
        except OSError:
            pass
    shutil.rmtree(path, onerror=force)


def git(folder, *argv):
    return subprocess.run(("git",) + argv, cwd=folder, capture_output=True, text=True)


def make_repo():
    """An upstream repo and a clone of it, standing in for the till."""
    up = os.path.join(TMP, "upstream")
    os.makedirs(up)
    git(up, "init", "-q", "-b", "main")
    git(up, "config", "user.email", "t@t"); git(up, "config", "user.name", "t")
    with open(os.path.join(up, "thing.py"), "w") as fh:
        fh.write("VALUE = 1\n")
    git(up, "add", "-A"); git(up, "commit", "-qm", "one")
    clone = os.path.join(TMP, "till")
    subprocess.run(("git", "clone", "-q", up, clone), capture_output=True)
    git(clone, "config", "user.email", "t@t"); git(clone, "config", "user.name", "t")
    return up, clone


def push(up, text, message):
    with open(os.path.join(up, "thing.py"), "w") as fh:
        fh.write(text)
    git(up, "add", "-A"); git(up, "commit", "-qm", message)


try:
    # the two pieces under test, lifted off MainWindow so no GUI is needed
    import app as A
    update = A.MainWindow._update_from_git
    compiles = A.MainWindow._files_that_do_not_compile

    class Fake:
        """Just enough of a window: the update only needs its own folder."""
        _files_that_do_not_compile = staticmethod(A.MainWindow._files_that_do_not_compile)

        def __init__(self, folder):
            self.folder = folder
            self.closed = False
            self.restart_scheduled = False
        def close(self):
            self.closed = True

        def _relaunch(self):
            self.restart_scheduled = True      # never actually start a second app here

    print("\n[1] a good update is applied and reported")
    up, till = make_repo()
    before = git(till, "rev-parse", "HEAD").stdout.strip()
    push(up, "VALUE = 2\n", "two")

    # point the method at the throwaway checkout instead of the real one
    import types
    def run(folder, restart=False):
        fake = Fake(folder)
        src = update.__code__
        g = dict(update.__globals__)
        g["__file__"] = os.path.join(folder, "app.py")
        scheduled = {}
        g["QTimer"] = types.SimpleNamespace(singleShot=lambda ms, fn: scheduled.setdefault("fn", fn))
        bound = types.FunctionType(src, g, update.__name__, update.__defaults__, update.__closure__)
        res = bound(fake, restart=restart)
        res["_scheduled_restart"] = "fn" in scheduled
        return res

    # the clone must stay a plain fast-forwardable mirror: committing into it
    # here diverged the branches and made every later pull fail, which is a
    # bug in this test rather than in the thing it is testing
    res = run(till)
    check("the pull succeeded", res.get("ok") is True, str(res.get("error")))
    check("it moved", res.get("changed") is True, str(res))
    check("it reports both commits", res["before"][:7] == before[:7] and res["after"] != res["before"],
          f"{res['before']} -> {res['after']}")
    with open(os.path.join(till, "thing.py")) as fh:
        check("the file really changed", "VALUE = 2" in fh.read())
    check("nothing restarted unasked", res.get("restarting") is False, str(res.get("restarting")))

    print("\n[2] code that does not compile is rolled back, not run")
    good = git(till, "rev-parse", "HEAD").stdout.strip()
    push(up, "def broken(:\n", "a half-pushed change")
    res = run(till, restart=True)
    check("the update refused", res.get("ok") is False, str(res))
    check("it rolled back", res.get("rolled_back_to", "")[:7] == good[:7],
          f"{res.get('rolled_back_to')} vs {good[:7]}")
    check("it named the broken file", any("thing.py" in b for b in res.get("does_not_compile", [])),
          str(res.get("does_not_compile")))
    check("and it did NOT restart", not res.get("_scheduled_restart"), "a restart was scheduled")
    now = git(till, "rev-parse", "HEAD").stdout.strip()
    check("the checkout is back where it was", now == good, f"{now[:7]} vs {good[:7]}")
    with open(os.path.join(till, "thing.py")) as fh:
        check("the bad file is gone", "broken" not in fh.read())

    print("\n[3] a restart only happens when asked and when something moved")
    push(up, "VALUE = 3\n", "three")
    res = run(till, restart=True)
    check("it updated", res.get("ok") is True, str(res))
    check("and scheduled the restart", res.get("_scheduled_restart") is True, str(res))
    res = run(till, restart=True)
    check("an update with nothing to pull does not restart", res.get("restarting") is False, str(res))

    print("\n[4] a broken ORIG_HEAD is cleared instead of blocking the update")
    # a real till fetched the new commit and then stopped on
    #   fatal: update_ref failed for ref 'ORIG_HEAD': ... reference broken
    # ORIG_HEAD is a pointer git rewrites on every pull, so it carries nothing
    # worth keeping - but git will not pull past it, and that shop had no other
    # way in, which is exactly what this action exists to prevent.
    push(up, "VALUE = 4\n", "four")
    with open(os.path.join(till, ".git", "ORIG_HEAD"), "w") as fh:
        fh.write("not a sha at all\n")
    res = run(till)
    check("the update still went through", res.get("ok") is True, str(res.get("error")))
    check("it moved on", res.get("changed") is True, str(res))
    check("and it said what it cleared", "ORIG_HEAD" in res.get("pull", ""),
          res.get("pull", "")[:200])
    with open(os.path.join(till, "thing.py")) as fh:
        check("the new code really landed", "VALUE = 4" in fh.read())

    print("\n[5] somewhere that is not a checkout is refused")
    plain = os.path.join(TMP, "plain")
    os.makedirs(plain)
    try:
        run(plain)
        check("it refuses a non-checkout", False, "it tried anyway")
    except RuntimeError as exc:
        check("it refuses a non-checkout", "not a git checkout" in str(exc), str(exc))

    print("\n[6] the compile check itself")
    check("a clean folder reports nothing", compiles(os.path.dirname(os.path.abspath(A.__file__))) == [])
    with open(os.path.join(plain, "bad.py"), "w") as fh:
        fh.write("def nope(:\n")
    check("a broken file is reported", any("bad.py" in b for b in compiles(plain)), str(compiles(plain)))

except Exception:
    ok["all"] = False
    traceback.print_exc()
finally:
    remove_tree(TMP)

if ok["all"]:
    print("\nALL CHECKS PASSED")
elif FAILED:
    print(f"\n{len(FAILED)} CHECK(S) FAILED")
else:
    # a crash before the first check used to print "0 CHECK(S) FAILED",
    # which reads like a pass
    print("\nCRASHED - the run stopped part way; see the traceback above")
for name, detail in FAILED:
    print(f"  - {name} {detail}")
sys.exit(0 if ok["all"] else 1)
