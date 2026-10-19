"""
test_support_server.py
----------------------
The till and the support service talking to each other for real.

test_support.py checks the till side against a stub.  This runs the actual
service from support_server/pos_support.py in this process, points a real
RemoteSupport at it over the loopback, and checks the round trip: a snapshot
arrives, an action queued on the dashboard reaches the till, and the result
comes back.

It also checks the service's own fence: the device token cannot reach the
dashboard, the admin token cannot post snapshots, an unknown action is refused
before it is ever queued, and a till name from the network cannot escape the
data directory.

Run:  python tests/test_support_server.py
"""
import os, sys, json, time, base64, shutil, tempfile, threading, traceback
import urllib.request, urllib.error

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "support_server"))

DATA = os.path.join(tempfile.gettempdir(), "pos_support_test")
shutil.rmtree(DATA, ignore_errors=True)
os.makedirs(DATA, exist_ok=True)
os.environ.update(POS_SUPPORT_DATA=DATA, POS_SUPPORT_DEVICE_TOKEN="device-secret",
                  POS_SUPPORT_ADMIN_TOKEN="admin-secret")

import pos_support as S
from remote_support import RemoteSupport
from http.server import ThreadingHTTPServer

ok = {"all": True}
FAILED = []
def check(name, cond, detail=""):
    if not cond:
        ok["all"] = False
        FAILED.append((name, detail))
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))


server = ThreadingHTTPServer(("127.0.0.1", 0), S.Handler)
threading.Thread(target=server.serve_forever, daemon=True).start()
URL = f"http://127.0.0.1:{server.server_address[1]}"


def call(path, token=None, data=None, basic=None):
    req = urllib.request.Request(URL + path, method="POST" if data is not None else "GET",
                                 data=json.dumps(data).encode() if data is not None else None)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    if basic:
        req.add_header("Authorization", "Basic " + base64.b64encode(basic.encode()).decode())
    with urllib.request.urlopen(req, timeout=10) as resp:
        body = resp.read()
        return resp.status, body


class Settings:
    def __init__(self, **kw):
        self.data = {"support.enabled": True, "support.url": URL, "support.token": "device-secret",
                     "support.interval_min": 10, "support.allow_actions": True}
        self.data.update({f"support.{k}": v for k, v in kw.items()})
    def get(self, key, default=None):
        return self.data.get(key, default)


RAN = []
def run_action(name, args):
    RAN.append(name)
    return {"did": name, "args": args}


TILL = "shop-till-1"
def snapshot():
    return {"camera": {"ok": False, "attempts": ["MSMF: opened but no frames"]},
            "scale": {"source": "simulate", "weight_kg": 0.0},
            "screen": {"banner": "No camera", "blocking_reason": None},
            "invoice": {"lines": 0}, "catalogue": {"items": 3},
            "frame_jpeg_b64": base64.b64encode(b"\xff\xd8\xff\xdbnot-a-real-jpeg").decode()}


try:
    print("\n[1] health needs no secret")
    status, body = call("/healthz")
    check("healthz answers", status == 200 and json.loads(body)["ok"] is True)

    print("\n[2] the two tokens do different jobs")
    for label, path, token in (("device token cannot see the dashboard", "/", "device-secret"),
                               ("device token cannot list tills", "/api/tills", "device-secret")):
        try:
            call(path, token=token)
            check(label, False, "it was allowed")
        except urllib.error.HTTPError as exc:
            check(label, exc.code == 401, str(exc.code))
    try:
        call("/snapshot", token="admin-secret", data={"machine": {"host": TILL}})
        check("admin token cannot post snapshots", False, "it was allowed")
    except urllib.error.HTTPError as exc:
        check("admin token cannot post snapshots", exc.code == 401, str(exc.code))

    print("\n[3] a till reports in")
    channel = RemoteSupport(Settings(), snapshot, run_action)
    import remote_support
    remote_support.machine_identity = lambda: {"host": TILL, "platform": "test", "user_app": "test"}
    check("it starts", channel.start() is True)
    deadline = time.time() + 10
    while not os.path.exists(os.path.join(DATA, TILL, "latest.json")) and time.time() < deadline:
        time.sleep(0.05)
    check("the snapshot was stored", os.path.exists(os.path.join(DATA, TILL, "latest.json")))
    check("the picture was saved", os.path.exists(os.path.join(DATA, TILL, "frame.jpg")))

    print("\n[4] the dashboard shows it")
    status, page = call("/", basic="admin:admin-secret")
    text = page.decode("utf-8")
    check("the page renders", status == 200 and "<html" in text)
    check("it names the till", TILL in text)
    check("it shows the camera is down", "no camera" in text)
    check("it shows what the camera tried", "opened but no frames" in text)
    check("it shows the scale is simulated", "scale: simulate" in text)
    status, listing = call("/api/tills", token="admin-secret")
    check("the api lists the till", status == 200 and json.loads(listing)[0]["till"] == TILL)

    print("\n[5] an action queued on the dashboard reaches the till")
    status, queued = call("/api/queue", token="admin-secret", data={"till": TILL, "name": "ping"})
    check("it was queued", status == 200 and json.loads(queued)["ok"] is True)
    channel.send_now()
    deadline = time.time() + 10
    while "ping" not in RAN and time.time() < deadline:
        time.sleep(0.05)
    check("the till ran it", "ping" in RAN, str(RAN))
    deadline = time.time() + 10
    while not os.path.exists(os.path.join(DATA, TILL, "results.json")) and time.time() < deadline:
        time.sleep(0.05)
    results = json.load(open(os.path.join(DATA, TILL, "results.json"), encoding="utf-8"))
    check("the result came back", any(r["name"] == "ping" and r["ok"] for r in results), str(results))

    print("\n[6] the queue is handed over once, not forever")
    channel.send_now(); time.sleep(1.0)
    count = sum(1 for r in json.load(open(os.path.join(DATA, TILL, "results.json"), encoding="utf-8"))
                if r["name"] == "ping")
    check("the action did not repeat", count == 1, str(count))
    channel.stop()

    print("\n[7] the service refuses an action that is not on the list")
    try:
        call("/api/queue", token="admin-secret", data={"till": TILL, "name": "shell"})
        check("an unlisted action is refused", False, "it was queued")
    except urllib.error.HTTPError as exc:
        check("an unlisted action is refused", exc.code == 400, str(exc.code))
    check("the till's list and the service's list agree",
          sorted(S.ACTIONS) == sorted(remote_support.ACTIONS),
          f"{sorted(set(S.ACTIONS) ^ set(remote_support.ACTIONS))}")

    print("\n[8] a till name off the network cannot escape the data directory")
    evil = "../../etc/cron.d/x"
    folder = S.till_dir(evil)
    check("the path stays inside the data directory",
          os.path.realpath(folder).startswith(os.path.realpath(DATA)), folder)
    # "../../etc" becomes one flat name like ".._.._etc": the dots survive as
    # characters, which is fine - what matters is that it is a single component
    # directly under the data directory and no separator got through
    leaf = os.path.relpath(folder, DATA)
    check("it is one flat name under the data directory",
          os.sep not in leaf and "/" not in leaf and leaf != "..", leaf)

except Exception:
    ok["all"] = False
    traceback.print_exc()
finally:
    server.shutdown()
    shutil.rmtree(DATA, ignore_errors=True)

# a crash before any check failed used to print "0 CHECK(S) FAILED", which
# reads like a pass
print("\n" + ("ALL CHECKS PASSED" if ok["all"] else
              (f"{len(FAILED)} CHECK(S) FAILED" if FAILED else
               "CRASHED - the run stopped part way; see the traceback above")))
for name, detail in FAILED:
    print(f"  - {name} {detail}")
sys.exit(0 if ok["all"] else 1)
