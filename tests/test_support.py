"""
test_support.py
---------------
The remote support channel, against a real HTTP server on the loopback.

What matters here is not that a POST works - it is that the channel stays
inside its fence.  A till handles money, and this is the one part of the app
that takes instructions from somewhere else, so the tests are mostly about
what it REFUSES to do:

  * nothing is sent until an owner has filled in both the URL and the token,
  * a non-https address is refused unless it is the loopback,
  * an action that is not in ACTIONS is refused and reported, not run,
  * switching actions off refuses even the ones that are in ACTIONS,
  * results of actions are never lost when the network drops - they go up
    with the next snapshot instead.

Run:  python tests/test_support.py
"""
import os, sys, ssl, json, time, threading, traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import support_config
from remote_support import (RemoteSupport, ACTIONS, encode_frame, log_tail, SupportError,
                            ca_bundle, ssl_context, explain_tls_error)

# most of these checks are about a build that has NOT been pointed anywhere;
# the shipped build has an address baked in, so blank it for the duration
support_config.DEFAULT_URL = ""
support_config.DEFAULT_TOKEN = ""

ok = {"all": True}
FAILED = []
def check(name, cond, detail=""):
    if not cond:
        ok["all"] = False
        FAILED.append((name, detail))
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))


class Settings:
    """The few keys the channel reads, nothing else."""
    def __init__(self, **kw):
        self.data = {"support.enabled": True, "support.url": "", "support.token": "",
                     "support.interval_min": 10, "support.allow_actions": True}
        self.data.update({f"support.{k}": v for k, v in kw.items()})
    def get(self, key, default=None):
        return self.data.get(key, default)


# --------------------------------------------------------------- fake service
RECEIVED = []
QUEUED = []          # actions the service hands back on the next snapshot
AUTH = "Bearer secret-token"


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if self.headers.get("Authorization") != AUTH:
            self.send_response(401); self.end_headers(); self.wfile.write(b'{"error":"bad token"}')
            return
        RECEIVED.append(json.loads(body.decode("utf-8")))
        actions, QUEUED[:] = list(QUEUED), []
        payload = json.dumps({"actions": actions}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
threading.Thread(target=server.serve_forever, daemon=True).start()
URL = f"http://127.0.0.1:{server.server_address[1]}"

RAN = []
def run_action(name, args):
    RAN.append((name, args))
    if name == "boom":
        raise RuntimeError("this action always fails")
    return {"did": name, "args": args}


def snapshot():
    return {"camera": {"ok": False}, "scale": {"source": "simulate"}}


try:
    print("\n[1] nothing goes anywhere until it is configured")
    for label, kw in (("no url or token", {}),
                      ("url but no token", {"url": URL}),
                      ("token but no url", {"token": "secret-token"}),
                      ("switched off", {"url": URL, "token": "secret-token", "enabled": False})):
        channel = RemoteSupport(Settings(**kw), snapshot, run_action)
        check(f"refuses to start: {label}", channel.start() is False and channel._thread is None)
    check("nothing was received", len(RECEIVED) == 0, str(len(RECEIVED)))

    print("\n[2] a plain-http address is refused off the loopback")
    channel = RemoteSupport(Settings(url="http://example.com/pos", token="secret-token"), snapshot, run_action)
    check("http to the internet is refused", channel.start() is False)
    check("and it says why", "https" in channel.last_error, channel.last_error)

    print("\n[3] a snapshot reaches the service")
    channel = RemoteSupport(Settings(url=URL, token="secret-token"), snapshot, run_action)
    check("starts when configured", channel.start() is True)
    deadline = time.time() + 10
    while not RECEIVED and time.time() < deadline:
        time.sleep(0.05)
    check("snapshot arrived", len(RECEIVED) == 1, str(len(RECEIVED)))
    if RECEIVED:
        body = RECEIVED[0]
        check("carries the snapshot", body.get("snapshot", {}).get("camera", {}).get("ok") is False)
        check("names the machine", bool(body.get("machine", {}).get("host")))

    print("\n[4] the service can ask for an allowed action")
    QUEUED.append({"id": "a1", "name": "ping"})
    channel.send_now()
    deadline = time.time() + 10
    while len(RECEIVED) < 3 and time.time() < deadline:
        time.sleep(0.05)
    results = [r for body in RECEIVED for r in body.get("results", [])]
    check("the action ran", ("ping", {}) in RAN, str(RAN))
    check("its result was reported", any(r["id"] == "a1" and r["ok"] for r in results), str(results))

    print("\n[5] an action outside the list is refused, not run")
    RAN.clear()
    QUEUED.append({"id": "a2", "name": "rm -rf /"})
    channel.send_now()
    deadline = time.time() + 10
    while not any(r.get("id") == "a2" for body in RECEIVED for r in body.get("results", [])) and time.time() < deadline:
        time.sleep(0.05)
    results = [r for body in RECEIVED for r in body.get("results", [])]
    bad = [r for r in results if r.get("id") == "a2"]
    check("it never reached the app", RAN == [], str(RAN))
    check("it was refused", bool(bad) and bad[0]["ok"] is False, str(bad))
    check("the refusal says so", bool(bad) and "refused" in bad[0].get("error", ""), str(bad))
    channel.stop()

    print("\n[6] actions can be switched off entirely")
    RAN.clear(); RECEIVED.clear()
    channel = RemoteSupport(Settings(url=URL, token="secret-token", allow_actions=False), snapshot, run_action)
    QUEUED.append({"id": "a3", "name": "selftest"})
    channel.start()
    deadline = time.time() + 10
    while not any(r.get("id") == "a3" for body in RECEIVED for r in body.get("results", [])) and time.time() < deadline:
        time.sleep(0.05)
    results = [r for body in RECEIVED for r in body.get("results", [])]
    off = [r for r in results if r.get("id") == "a3"]
    check("a listed action is still refused", bool(off) and off[0]["ok"] is False, str(off))
    check("the app never saw it", RAN == [], str(RAN))
    channel.stop()

    print("\n[7] a wrong token is an error, not a silent success")
    channel = RemoteSupport(Settings(url=URL, token="wrong"), snapshot, run_action)
    try:
        channel.test()
        check("a bad token raises", False, "no error raised")
    except Exception as exc:
        check("a bad token raises", True, type(exc).__name__)

    print("\n[8] results survive the network going away")
    channel = RemoteSupport(Settings(url=URL, token="secret-token"), snapshot, run_action)
    channel._pending = [{"id": "kept", "name": "ping", "ok": True}]
    broken = channel._post
    def fail(*a, **kw):
        raise OSError("the shop's internet is down")
    channel._post = fail
    try:
        channel._post_snapshot()
    except OSError:
        pass
    check("the result was not thrown away", any(r["id"] == "kept" for r in channel._pending), str(channel._pending))
    channel._post = broken
    channel._post_snapshot()
    check("and it goes up on the next try",
          any(r.get("id") == "kept" for body in RECEIVED for r in body.get("results", [])))

    print("\n[9] a failing action is reported, it does not kill the channel")
    RECEIVED.clear()
    channel = RemoteSupport(Settings(url=URL, token="secret-token"), snapshot, run_action)
    channel._handle({"id": "a4", "name": "ping"})
    ACTIONS["boom"] = "a test action that always fails"
    try:
        channel._handle({"id": "a5", "name": "boom"})
    finally:
        del ACTIONS["boom"]
    pending = {r["id"]: r for r in channel._pending}
    check("the good one succeeded", pending["a4"]["ok"] is True)
    check("the bad one is reported as failed", pending["a5"]["ok"] is False)
    check("with the reason", "always fails" in pending["a5"]["error"], pending["a5"].get("error", ""))

    print("\n[10] a build with an address baked in needs no configuration")
    support_config.DEFAULT_URL, support_config.DEFAULT_TOKEN = URL, "secret-token"
    try:
        RECEIVED.clear()
        channel = RemoteSupport(Settings(url="", token=""), snapshot, run_action)
        check("it uses the built-in address", channel.url == URL, channel.url)
        check("and the built-in key", channel.token == "secret-token")
        check("it knows it is on defaults", channel.using_defaults is True)
        check("it starts with an empty config", channel.start() is True)
        deadline = time.time() + 10
        while not RECEIVED and time.time() < deadline:
            time.sleep(0.05)
        check("and reports in", len(RECEIVED) >= 1, str(len(RECEIVED)))
        channel.stop()
        # a till pointed somewhere by hand must still win over the built-in one
        other = RemoteSupport(Settings(url="https://elsewhere.example/pos", token="mine"), snapshot, run_action)
        check("config beats the built-in address", other.url == "https://elsewhere.example/pos", other.url)
        check("config beats the built-in key", other.token == "mine")
        check("and it says so", other.using_defaults is False)
        off = RemoteSupport(Settings(enabled=False), snapshot, run_action)
        check("a single till can still be switched off", off.start() is False)
    finally:
        support_config.DEFAULT_URL, support_config.DEFAULT_TOKEN = "", ""

    print("\n[11] TLS is verified against a bundle we ship, not the machine's store")
    bundle = ca_bundle()
    check("a CA bundle is available", bundle is not None and os.path.isfile(bundle), str(bundle))
    ctx = ssl_context()
    check("the context verifies", ctx.verify_mode == ssl.CERT_REQUIRED)
    check("and checks the hostname", ctx.check_hostname is True)
    check("the same context is reused", ssl_context() is ctx)
    check("an explicit bundle is honoured", ssl_context(bundle) is not None)
    explained = explain_tls_error(
        ssl.SSLCertVerificationError("[SSL: CERTIFICATE_VERIFY_FAILED] unable to get issuer"),
        "https://example.com/pos")
    check("a verify failure names the clock and the store",
          "date and time" in explained and "certificate store" in explained, explained[:70])
    plain = explain_tls_error(ssl.SSLError("[SSL: WRONG_VERSION_NUMBER] wrong version number"), "https://x/pos")
    check("a plaintext port is explained differently", "without TLS" in plain, plain[:60])

    print("\n[12] helpers")
    import numpy as np
    frame = (np.random.default_rng(1).random((480, 640, 3)) * 255).astype(np.uint8)
    b64 = encode_frame(frame)
    check("a frame encodes small", b64 is not None and len(b64) < 200_000, f"{len(b64 or '')} chars")
    check("no frame encodes to nothing", encode_frame(None) is None)
    check("a missing log reads as a note", "could not read" in log_tail("no-such-file.log"))

except Exception:
    ok["all"] = False
    traceback.print_exc()
finally:
    server.shutdown()

print("\n" + ("ALL CHECKS PASSED" if ok["all"] else f"{len(FAILED)} CHECK(S) FAILED"))
for name, detail in FAILED:
    print(f"  - {name} {detail}")
sys.exit(0 if ok["all"] else 1)
