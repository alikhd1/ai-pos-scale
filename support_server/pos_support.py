#!/usr/bin/env python3
"""
pos_support.py
--------------
The other end of the till's support channel.

A POS terminal somewhere in a shop posts a snapshot of itself here every few
minutes; this keeps the last of them, shows them on a page, and hands back the
short list of support actions somebody has queued for that till.

It is the standard library only - no pip, no virtualenv - so it can be dropped
on a server and run under systemd with nothing else installed.

Two separate secrets, because they protect different things:

  POS_SUPPORT_DEVICE_TOKEN  what the tills send.  Leaks give someone the ability
                            to post junk snapshots, which is annoying.
  POS_SUPPORT_ADMIN_TOKEN   what the dashboard needs, and the only thing that can
                            queue an action on a till.  Leaks are serious.

Endpoints
  POST /snapshot        a till reporting in            (device token)
  GET  /                the dashboard                  (admin, HTTP basic)
  GET  /api/tills       every till, newest state       (admin)
  GET  /api/latest      one till's last snapshot       (admin)
  GET  /api/history     a till's recent snapshots      (admin)
  POST /api/queue       queue an action for a till     (admin)
  GET  /frame.jpg       the last picture from a till   (admin)
  GET  /healthz         liveness, no secret needed

Run:  POS_SUPPORT_DEVICE_TOKEN=... POS_SUPPORT_ADMIN_TOKEN=... python3 pos_support.py
"""
from __future__ import annotations

import base64
import binascii
import hmac
import html
import json
import logging
import os
import re
import secrets
import sys
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

LISTEN_HOST = os.environ.get("POS_SUPPORT_HOST", "127.0.0.1")
LISTEN_PORT = int(os.environ.get("POS_SUPPORT_PORT", "8900"))
DATA_DIR = os.environ.get("POS_SUPPORT_DATA", "/srv/projects/pos-support/data")
DEVICE_TOKEN = os.environ.get("POS_SUPPORT_DEVICE_TOKEN", "")
ADMIN_TOKEN = os.environ.get("POS_SUPPORT_ADMIN_TOKEN", "")
KEEP_SNAPSHOTS = int(os.environ.get("POS_SUPPORT_KEEP", "200"))
MAX_BODY = 8 * 1024 * 1024

#: the only actions a till will accept; kept in step with remote_support.ACTIONS
ACTIONS = ["ping", "snapshot", "selftest", "find_camera", "find_scale", "restart_camera",
           "restart_scale", "diagnostics", "logs", "get_setting", "set_setting", "screenshot"]

SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]")

log = logging.getLogger("pos-support")
_lock = threading.Lock()


# --------------------------------------------------------------------------- #
# Storage
# --------------------------------------------------------------------------- #
def till_dir(till: str) -> str:
    """A till's folder.  The name is sanitised: it arrives over the network."""
    safe = SAFE_NAME.sub("_", till)[:64] or "unknown"
    path = os.path.join(DATA_DIR, safe)
    os.makedirs(path, exist_ok=True)
    return path


def store_snapshot(till: str, body: dict) -> None:
    folder = till_dir(till)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    with _lock:
        with open(os.path.join(folder, f"snap_{stamp}.json"), "w", encoding="utf-8") as fh:
            json.dump(body, fh, ensure_ascii=False)
        with open(os.path.join(folder, "latest.json"), "w", encoding="utf-8") as fh:
            json.dump(body, fh, ensure_ascii=False)
        # a picture is handy to fetch on its own
        frame = (body.get("snapshot") or {}).get("frame_jpeg_b64")
        if frame:
            try:
                with open(os.path.join(folder, "frame.jpg"), "wb") as fh:
                    fh.write(base64.b64decode(frame))
            except (binascii.Error, ValueError):
                log.warning("%s sent a frame that is not base64", till)
        snaps = sorted(n for n in os.listdir(folder) if n.startswith("snap_"))
        for old in snaps[:-KEEP_SNAPSHOTS]:
            try:
                os.remove(os.path.join(folder, old))
            except OSError:
                pass


def read_json(path: str, default=None):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return default


def list_tills() -> list:
    try:
        names = sorted(os.listdir(DATA_DIR))
    except OSError:
        return []
    out = []
    for name in names:
        latest = read_json(os.path.join(DATA_DIR, name, "latest.json"))
        if latest:
            out.append({"till": name, "latest": latest,
                        "queued": len(read_json(os.path.join(DATA_DIR, name, "queue.json"), []) or [])})
    return out


def take_queue(till: str) -> list:
    """Hand over the queued actions and clear them in one step."""
    path = os.path.join(till_dir(till), "queue.json")
    with _lock:
        queue = read_json(path, []) or []
        if queue:
            try:
                os.remove(path)
            except OSError:
                pass
    return queue


def add_to_queue(till: str, action: dict) -> dict:
    path = os.path.join(till_dir(till), "queue.json")
    action.setdefault("id", secrets.token_hex(6))
    with _lock:
        queue = read_json(path, []) or []
        queue.append(action)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(queue, fh)
    return action


def store_results(till: str, results: list) -> None:
    if not results:
        return
    path = os.path.join(till_dir(till), "results.json")
    with _lock:
        kept = (read_json(path, []) or []) + results
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(kept[-100:], fh, ensure_ascii=False)
    # a diagnostic bundle is a file, not a log line: write it out
    for result in results:
        blob = (result.get("result") or {}).get("zip_b64") if isinstance(result.get("result"), dict) else None
        if not blob:
            continue
        name = SAFE_NAME.sub("_", str(result["result"].get("filename", "bundle.zip")))
        try:
            with open(os.path.join(till_dir(till), name), "wb") as fh:
                fh.write(base64.b64decode(blob))
            result["result"]["zip_b64"] = f"(saved as {name})"
        except (binascii.Error, ValueError, OSError) as exc:
            log.warning("could not save a bundle from %s: %s", till, exc)


# --------------------------------------------------------------------------- #
# The dashboard
# --------------------------------------------------------------------------- #
PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>POS tills</title>
<style>
 :root {{ color-scheme: dark; --bg:#0b0d11; --card:#171b22; --line:#2a3340; --dim:#9aa3b5;
          --ok:#1f9d55; --warn:#8a6d1f; --bad:#a1343e; }}
 body {{ margin:0; padding:16px; background:var(--bg); color:#e7ebf2;
         font:14px/1.5 ui-sans-serif,system-ui,Segoe UI,sans-serif; }}
 h1 {{ font-size:18px; margin:0 0 12px; }}
 .till {{ background:var(--card); border:1px solid var(--line); border-radius:12px;
          padding:14px; margin-bottom:14px; }}
 .row {{ display:flex; flex-wrap:wrap; gap:14px; align-items:flex-start; }}
 .pill {{ display:inline-block; padding:2px 9px; border-radius:9px; font-size:12px; }}
 .ok {{ background:var(--ok); }} .warn {{ background:var(--warn); }} .bad {{ background:var(--bad); }}
 .dim {{ color:var(--dim); }}
 img {{ max-width:320px; border-radius:8px; border:1px solid var(--line); }}
 pre {{ background:#0e1116; border:1px solid var(--line); border-radius:8px; padding:10px;
        overflow:auto; max-height:260px; font-size:12px; }}
 table {{ border-collapse:collapse; }} td {{ padding:2px 12px 2px 0; vertical-align:top; }}
 button, select {{ background:#222835; color:#e7ebf2; border:1px solid var(--line);
                   border-radius:7px; padding:5px 10px; font-size:13px; }}
 button:hover {{ background:#2c3444; cursor:pointer; }}
 @media (max-width:640px) {{ img {{ max-width:100%; }} }}
</style></head><body>
<h1>POS tills <span class="dim">&middot; refreshed {now}</span></h1>
{body}
<script>
 async function queue(till, name) {{
   const args = {{}};
   if (name === 'set_setting') {{
     const key = prompt('setting key, e.g. camera.device_index'); if (!key) return;
     let value = prompt('value'); if (value === null) return;
     try {{ value = JSON.parse(value); }} catch (e) {{}}
     args.key = key; args.value = value;
   }}
   const r = await fetch('api/queue', {{ method:'POST', headers:{{'Content-Type':'application/json'}},
                                         body: JSON.stringify({{till, name, args}}) }});
   alert(r.ok ? name + ' queued; it runs at the next check-in' : 'failed: ' + await r.text());
   location.reload();
 }}
 setTimeout(() => location.reload(), 60000);
</script>
</body></html>"""


def age_pill(seconds: float, interval_hint: float = 600.0) -> str:
    if seconds < interval_hint * 1.5:
        cls, text = "ok", f"{int(seconds)}s ago"
    elif seconds < interval_hint * 4:
        cls, text = "warn", f"{int(seconds / 60)} min ago"
    else:
        cls, text = "bad", f"{int(seconds / 3600)} h ago - not reporting"
    return f'<span class="pill {cls}">{text}</span>'


def render_till(entry: dict) -> str:
    till, body = entry["till"], entry["latest"]
    snap = body.get("snapshot") or {}
    cam, scale = snap.get("camera") or {}, snap.get("scale") or {}
    screen, inv = snap.get("screen") or {}, snap.get("invoice") or {}
    age = max(0.0, time.time() - float(body.get("sent_at") or 0))
    e = html.escape

    cam_pill = ('<span class="pill ok">camera ok</span>' if cam.get("ok")
                else '<span class="pill bad">no camera</span>')
    scale_src = str(scale.get("source", "?"))
    scale_pill = (f'<span class="pill warn">scale: {e(scale_src)}</span>' if scale_src == "simulate"
                  else f'<span class="pill ok">scale: {e(scale_src)}</span>')
    rows = [
        ("camera", e(str(cam.get("description") or "-"))),
        ("tried", "<br>".join(e(a) for a in (cam.get("attempts") or [])[:6]) or '<span class="dim">-</span>'),
        ("weight", f'{scale.get("weight_kg")} kg  <span class="dim">{e(str(scale.get("pill") or ""))}</span>'),
        ("screen", e(str(screen.get("banner") or screen.get("detected") or "-"))),
        ("blocked by", e(str(screen.get("blocking_reason") or "nothing"))),
        ("invoice", f'{inv.get("lines")} lines, tray {inv.get("tray_kg")} kg vs invoice {inv.get("invoiced_kg")} kg'),
        ("catalogue", e(str((snap.get("catalogue") or {}).get("items", "?"))) + " items"),
        ("engine", e(json.dumps(snap.get("engine") or {}))),
    ]
    table = "".join(f"<tr><td class='dim'>{k}</td><td>{v}</td></tr>" for k, v in rows)
    buttons = "".join(f'<button onclick="queue({json.dumps(till)}, {json.dumps(a)})">{a}</button> '
                      for a in ACTIONS)
    frame = (f'<img src="frame.jpg?till={e(till)}&t={int(time.time())}" alt="camera view">'
             if snap.get("frame_jpeg_b64") else '<span class="dim">no picture</span>')
    results = read_json(os.path.join(DATA_DIR, SAFE_NAME.sub("_", till), "results.json"), []) or []
    recent = "\n".join(json.dumps(r, ensure_ascii=False)[:600] for r in results[-6:])
    return f"""<div class="till">
      <div class="row" style="justify-content:space-between">
        <div><b>{e(till)}</b> {age_pill(age)} {cam_pill} {scale_pill}</div>
        <div class="dim">{entry['queued']} queued</div>
      </div>
      <div class="row" style="margin-top:10px">
        <div>{frame}</div>
        <div style="flex:1; min-width:280px"><table>{table}</table></div>
      </div>
      <div style="margin-top:10px">{buttons}</div>
      <details style="margin-top:10px"><summary class="dim">action results</summary>
        <pre>{e(recent) or "none yet"}</pre></details>
      <details style="margin-top:6px"><summary class="dim">log tail</summary>
        <pre>{e(str(snap.get("log_tail") or "not sent"))}</pre></details>
    </div>"""


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #
class Handler(BaseHTTPRequestHandler):
    server_version = "pos-support/1"

    def log_message(self, fmt, *args):
        log.info("%s %s", self.address_string(), fmt % args)

    # ---- helpers
    def _send(self, code: int, body: bytes, content_type: str, extra: dict = None):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, code: int, obj):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def _bearer_ok(self, expected: str) -> bool:
        header = self.headers.get("Authorization", "")
        if not expected or not header.startswith("Bearer "):
            return False
        return hmac.compare_digest(header[7:].strip(), expected)

    def _admin_ok(self) -> bool:
        """Bearer for scripts, HTTP basic so a browser can just ask."""
        if self._bearer_ok(ADMIN_TOKEN):
            return True
        header = self.headers.get("Authorization", "")
        if header.startswith("Basic ") and ADMIN_TOKEN:
            try:
                raw = base64.b64decode(header[6:]).decode("utf-8", "replace")
            except (binascii.Error, ValueError):
                return False
            return hmac.compare_digest(raw.split(":", 1)[-1], ADMIN_TOKEN)
        return False

    def _need_admin(self) -> bool:
        if self._admin_ok():
            return True
        self._send(401, b"authentication required", "text/plain; charset=utf-8",
                   {"WWW-Authenticate": 'Basic realm="POS support"'})
        return False

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY:
            raise ValueError(f"body too large ({length} bytes)")
        return json.loads(self.rfile.read(length).decode("utf-8")) if length else {}

    def _query(self) -> dict:
        return {k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()}

    # ---- routes
    def do_GET(self):
        route = urlparse(self.path).path.rstrip("/") or "/"
        if route == "/healthz":
            return self._json(200, {"ok": True, "tills": len(list_tills())})
        if not self._need_admin():
            return
        if route == "/":
            tills = list_tills()
            body = "".join(render_till(t) for t in tills) or '<p class="dim">No till has reported yet.</p>'
            page = PAGE.format(now=datetime.now().strftime("%H:%M:%S"), body=body)
            return self._send(200, page.encode("utf-8"), "text/html; charset=utf-8")
        if route == "/api/tills":
            return self._json(200, list_tills())
        if route == "/api/latest":
            till = self._query().get("till", "")
            data = read_json(os.path.join(DATA_DIR, SAFE_NAME.sub("_", till), "latest.json"))
            return self._json(200 if data else 404, data or {"error": "no such till"})
        if route == "/api/history":
            till = SAFE_NAME.sub("_", self._query().get("till", ""))
            folder = os.path.join(DATA_DIR, till)
            limit = int(self._query().get("limit", 10))
            try:
                names = sorted(n for n in os.listdir(folder) if n.startswith("snap_"))[-limit:]
            except OSError:
                return self._json(404, {"error": "no such till"})
            return self._json(200, [read_json(os.path.join(folder, n)) for n in names])
        if route == "/frame.jpg":
            till = SAFE_NAME.sub("_", self._query().get("till", ""))
            try:
                with open(os.path.join(DATA_DIR, till, "frame.jpg"), "rb") as fh:
                    return self._send(200, fh.read(), "image/jpeg")
            except OSError:
                return self._json(404, {"error": "no picture"})
        self._json(404, {"error": "not found"})

    def do_HEAD(self):
        self.do_GET()

    def do_POST(self):
        route = urlparse(self.path).path.rstrip("/") or "/"
        try:
            if route == "/snapshot":
                if not self._bearer_ok(DEVICE_TOKEN):
                    return self._json(401, {"error": "bad device token"})
                body = self._body()
                till = str((body.get("machine") or {}).get("host") or "unknown")
                store_results(till, body.get("results") or [])
                store_snapshot(till, body)
                actions = [] if body.get("test") else take_queue(till)
                if actions:
                    log.info("handing %d action(s) to %s", len(actions), till)
                return self._json(200, {"ok": True, "actions": actions})

            if route == "/api/queue":
                if not self._need_admin():
                    return
                body = self._body()
                name = str(body.get("name", ""))
                if name not in ACTIONS:
                    return self._json(400, {"error": f"unknown action {name!r}", "supported": ACTIONS})
                queued = add_to_queue(str(body.get("till", "")), {"name": name, "args": body.get("args") or {}})
                return self._json(200, {"ok": True, "queued": queued})
        except ValueError as exc:
            return self._json(400, {"error": str(exc)})
        except Exception as exc:                       # never take the service down for one bad request
            log.exception("request failed")
            return self._json(500, {"error": str(exc)})
        self._json(404, {"error": "not found"})


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not DEVICE_TOKEN or not ADMIN_TOKEN:
        print("POS_SUPPORT_DEVICE_TOKEN and POS_SUPPORT_ADMIN_TOKEN must both be set", file=sys.stderr)
        return 2
    if hmac.compare_digest(DEVICE_TOKEN, ADMIN_TOKEN):
        print("the device and admin tokens must be different: the tills hold the device one", file=sys.stderr)
        return 2
    os.makedirs(DATA_DIR, exist_ok=True)
    server = ThreadingHTTPServer((LISTEN_HOST, LISTEN_PORT), Handler)
    log.info("listening on %s:%d, data in %s", LISTEN_HOST, LISTEN_PORT, DATA_DIR)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
