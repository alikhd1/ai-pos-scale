"""
remote_support.py
-----------------
An outbound support channel for a till that nobody can walk up to.

The POS terminal this app runs on sits in a shop, often behind a router nobody
controls, and the person who has to fix it is somewhere else entirely.  Asking
the shopkeeper to run a self-test and e-mail a zip works once; it does not work
while you are chasing a fault.

So, when it is switched on, the app posts a snapshot of itself to a URL every
few minutes: what the camera is doing, what the scale reads, what the screen is
currently refusing to do and why, the tail of the log, and a small picture of
what the camera sees.  The reply may carry a short list of actions to run, and
their results go up with the next snapshot.

What this deliberately is NOT
=============================
It is not a remote shell, and it must never become one.  There is no "run this
command", no "download and execute", no file write outside the app's own
folder.  The server can only ask for things from ``ACTIONS`` below - run a
self-test, search for the camera, change one known setting - and anything else
is logged and refused.  A till handles money; a general-purpose backdoor on one
is a liability whoever installs it, and a bounded set of support actions buys
nearly all of the usefulness for a small fraction of the risk.

Everything is outbound HTTPS to one address, and nothing ever listens on a
port.  That address and the device key may be baked into the build - see
support_config.py, and read the note there about what a secret in a shipped
binary is and is not worth - because these tills are installed by someone who
will not be typing a 32-character token into a machine they cannot reach.
Anything in config.json overrides the baked-in values, ``support.enabled``
turns the channel off for a particular till, and a build whose
support_config.py is blank sends nothing until somebody fills the settings in.

Dependencies: the standard library only, so it survives PyInstaller untouched.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import platform
import socket
import ssl
import sys
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Optional

import support_config

log = logging.getLogger("support")

PROTOCOL = 1
MAX_FRAME_WIDTH = 480          # a snapshot is for looking at, not for archiving
FRAME_QUALITY = 55
MAX_LOG_LINES = 300
MAX_BODY_BYTES = 3 * 1024 * 1024


# --------------------------------------------------------------------------- #
# Actions the server is allowed to ask for
# --------------------------------------------------------------------------- #
#: Every supported action, and what it means.  The server can ask for nothing
#: else; see the module docstring for why this list is closed.
ACTIONS = {
    "ping":           "answer, and nothing else",
    "snapshot":       "send a fresh snapshot immediately",
    "selftest":       "run the head-less self-test and send the report",
    "find_camera":    "try every camera on the machine and send what answered",
    "find_scale":     "try every port and USB scale and send what answered",
    "restart_camera": "close and reopen the camera",
    "restart_scale":  "close and reopen the scale",
    "diagnostics":    "collect the full diagnostic bundle and send it",
    "logs":           "send a longer tail of the log",
    "get_setting":    "read one setting",
    "set_setting":    "change one setting (key must already exist)",
    "screenshot":     "send a picture of the application window",
}


class SupportError(Exception):
    pass


_ssl_lock = threading.Lock()
_ssl_cache: Dict[str, "ssl.SSLContext"] = {}


def ca_bundle() -> Optional[str]:
    """The certificate bundle to trust, or None to use the machine's own.

    This is the fix for a real failure on a shop's till: Python on Windows
    reads whatever root certificates happen to be cached in the Windows store,
    and - unlike a browser - it never triggers Windows' fetch-this-root-on-
    demand update.  A till that has never had the relevant root pulled down
    therefore fails to verify a perfectly good certificate, while the same URL
    opens fine in Edge on the same machine.  Let's Encrypt's 2025 hierarchy
    (Root YE under ISRG Root X2 under ISRG Root X1) is exactly the sort of
    chain a stale store has not seen.

    Shipping our own bundle takes the machine's store out of the picture.
    """
    try:
        import certifi
    except ImportError:
        return None
    path = certifi.where()
    return path if os.path.isfile(path) else None


def ssl_context(ca_file: str = "") -> "ssl.SSLContext":
    """A verifying TLS context, preferring a bundle we ship over the machine's."""
    key = ca_file or ca_bundle() or ""
    with _ssl_lock:
        context = _ssl_cache.get(key)
        if context is None:
            if key:
                context = ssl.create_default_context(cafile=key)
            else:
                context = ssl.create_default_context()
            _ssl_cache[key] = context
        return context


def explain_tls_error(exc: BaseException, url: str) -> str:
    """Turn an SSL failure into something a shopkeeper can act on."""
    detail = str(exc)
    if isinstance(exc, ssl.SSLCertVerificationError) or "CERTIFICATE_VERIFY_FAILED" in detail:
        return (f"the certificate of {url} could not be verified ({detail}). "
                "This is usually an out-of-date Windows certificate store or a clock that is "
                "wrong by more than a day: check the date and time on this machine first, then "
                "Windows Update. The address itself is fine if it opens in a browser.")
    if "WRONG_VERSION_NUMBER" in detail or "record layer" in detail:
        return f"{url} answered without TLS - the address is probably http, or the port is wrong ({detail})"
    return detail


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def encode_frame(frame, max_width: int = MAX_FRAME_WIDTH, quality: int = FRAME_QUALITY) -> Optional[str]:
    """A camera frame as a small base64 JPEG, or None if it cannot be encoded."""
    if frame is None:
        return None
    try:
        import cv2
        import numpy as np
        if not isinstance(frame, np.ndarray) or frame.size == 0:
            return None
        h, w = frame.shape[:2]
        if w > max_width:
            scale = max_width / float(w)
            frame = cv2.resize(frame, (max_width, max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        return base64.b64encode(buf.tobytes()).decode("ascii") if ok else None
    except Exception:
        log.debug("could not encode a frame for the snapshot", exc_info=True)
        return None


def log_tail(path: str, lines: int = MAX_LOG_LINES) -> str:
    """The last few hundred lines of a log file, read from the end."""
    try:
        size = os.path.getsize(path)
        want = min(size, 256 * 1024)
        with open(path, "rb") as fh:
            fh.seek(size - want)
            text = fh.read().decode("utf-8", errors="replace")
        return "\n".join(text.splitlines()[-lines:])
    except Exception as exc:
        return f"(could not read {path}: {exc})"


def machine_identity() -> Dict[str, str]:
    """Enough to tell two tills apart, and nothing that identifies a person."""
    return {"host": socket.gethostname(), "platform": platform.platform(), "user_app": "AI POS Scale"}


# --------------------------------------------------------------------------- #
# The channel
# --------------------------------------------------------------------------- #
class RemoteSupport:
    """Posts snapshots to ``url`` and runs the actions that come back.

    The app supplies two callbacks so that this module knows nothing about Qt
    and can be tested on its own:

      ``snapshot()``            -> a JSON-safe dict describing the till now
      ``run_action(name, args)`` -> a JSON-safe dict, called on the UI thread
    """

    def __init__(self, settings, snapshot: Callable[[], Dict[str, Any]],
                 run_action: Callable[[str, Dict[str, Any]], Dict[str, Any]],
                 opener: Optional[Callable[..., Any]] = None):
        self.settings = settings
        self._snapshot = snapshot
        self._run_action = run_action
        self._opener = opener or urllib.request.urlopen
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._pending: List[Dict[str, Any]] = []       # results waiting to go up
        self._lock = threading.Lock()
        self.last_error = ""
        self.last_sent = 0.0
        self.sent_count = 0

    # ---- configuration
    @property
    def url(self) -> str:
        """The configured address, or the one this build was made with."""
        return (str(self.settings.get("support.url", "") or "").strip() or support_config.DEFAULT_URL).rstrip("/")

    @property
    def token(self) -> str:
        return str(self.settings.get("support.token", "") or "").strip() or support_config.DEFAULT_TOKEN

    @property
    def using_defaults(self) -> bool:
        return not str(self.settings.get("support.url", "") or "").strip()

    @property
    def interval(self) -> float:
        minutes = float(self.settings.get("support.interval_min", 10) or 10)
        return max(60.0, minutes * 60.0)

    def configured(self) -> bool:
        return bool(self.settings.get("support.enabled", False)) and bool(self.url) and bool(self.token)

    # ---- lifecycle
    def start(self) -> bool:
        """Start the channel.  Returns False when it is not configured."""
        if self._thread is not None:
            return True
        if not self.configured():
            log.info("remote support is off (no URL or token, or not enabled)")
            return False
        if not self.url.lower().startswith("https://") and not self._allow_plain_http():
            log.error("remote support refused: %s is not https", self.url)
            self.last_error = "the support URL must be https"
            return False
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="RemoteSupport", daemon=True)
        self._thread.start()
        log.info("remote support on, posting to %s every %.0f min", self.url, self.interval / 60)
        return True

    def _allow_plain_http(self) -> bool:
        """Plain HTTP is only tolerated on the loopback, for tests."""
        host = self.url.split("://", 1)[-1].split("/", 1)[0].split(":", 1)[0]
        return host in ("127.0.0.1", "localhost", "::1")

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        thread, self._thread = self._thread, None
        if thread:
            thread.join(timeout=5)

    def send_now(self) -> None:
        """Push a snapshot at the next opportunity instead of waiting."""
        self._wake.set()

    # ---- the loop
    def _loop(self) -> None:
        backoff = 0.0
        while not self._stop.is_set():
            try:
                actions = self._post_snapshot()
                self.last_error = ""
                backoff = 0.0
                for action in actions:
                    if self._stop.is_set():
                        break
                    self._handle(action)
                if actions:
                    self._wake.set()               # report the results straight away
            except Exception as exc:
                self.last_error = str(exc)
                # a shop's internet comes and goes; back off rather than hammer
                backoff = min(max(backoff * 2, 30.0), self.interval)
                log.warning("remote support: %s (retrying in %.0fs)", exc, backoff)
            self._wake.wait(backoff or self.interval)
            self._wake.clear()

    def _post_snapshot(self) -> List[Dict[str, Any]]:
        with self._lock:
            results, self._pending = self._pending, []
        body = {"protocol": PROTOCOL, "sent_at": time.time(), "machine": machine_identity(),
                "snapshot": self._snapshot(), "results": results}
        try:
            reply = self._post("/snapshot", body)
        except Exception:
            with self._lock:                        # do not lose the results
                self._pending = results + self._pending
            raise
        self.last_sent = time.time()
        self.sent_count += 1
        actions = reply.get("actions") or []
        return [a for a in actions if isinstance(a, dict)]

    def _post(self, path: str, body: Dict[str, Any]) -> Dict[str, Any]:
        data = json.dumps(body, ensure_ascii=False, default=str).encode("utf-8")
        if len(data) > MAX_BODY_BYTES:
            raise SupportError(f"snapshot too large ({len(data) // 1024} KB)")
        req = urllib.request.Request(self.url + path, data=data, method="POST",
                                     headers={"Content-Type": "application/json",
                                              "Authorization": "Bearer " + self.token,
                                              "User-Agent": "AIPosScale-support/1"})
        context = ssl_context(str(self.settings.get("support.ca_file", "") or ""))
        try:
            with self._opener(req, timeout=30, context=context) as resp:
                raw = resp.read(MAX_BODY_BYTES).decode("utf-8", errors="replace")
        except ssl.SSLError as exc:
            raise SupportError(explain_tls_error(exc, self.url)) from exc
        except urllib.error.URLError as exc:
            # urllib wraps the SSL failure, so the useful part is inside
            if isinstance(exc.reason, ssl.SSLError):
                raise SupportError(explain_tls_error(exc.reason, self.url)) from exc
            raise
        if not raw.strip():
            return {}
        try:
            reply = json.loads(raw)
        except ValueError as exc:
            raise SupportError(f"the server did not answer with JSON: {exc}") from exc
        return reply if isinstance(reply, dict) else {}

    # ---- actions
    def _handle(self, action: Dict[str, Any]) -> None:
        name = str(action.get("name", ""))
        args = action.get("args") if isinstance(action.get("args"), dict) else {}
        entry: Dict[str, Any] = {"id": action.get("id"), "name": name, "at": time.time()}
        if name not in ACTIONS:
            # an unknown action is either a newer server or something trying it on
            entry.update(ok=False, error=f"refused: {name!r} is not a supported action",
                         supported=sorted(ACTIONS))
            log.warning("remote support refused unknown action %r", name)
        elif not self.settings.get("support.allow_actions", True):
            entry.update(ok=False, error="actions are switched off on this till")
        else:
            t0 = time.time()
            try:
                entry.update(ok=True, result=self._run_action(name, args))
            except Exception as exc:
                log.exception("remote action %s failed", name)
                entry.update(ok=False, error=str(exc))
            entry["took_s"] = round(time.time() - t0, 2)
        with self._lock:
            self._pending.append(entry)

    # ---- one-shot check, for the settings dialog
    def test(self) -> Dict[str, Any]:
        """Post one snapshot now and report what happened, without the thread."""
        if not self.url or not self.token:
            raise SupportError("set the support URL and the token first")
        reply = self._post("/snapshot", {"protocol": PROTOCOL, "sent_at": time.time(),
                                         "machine": machine_identity(), "snapshot": self._snapshot(),
                                         "results": [], "test": True})
        return reply
