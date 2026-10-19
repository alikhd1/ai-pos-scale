#!/usr/bin/env python3
"""
check_support.py
----------------
Why is this till not reporting?

The support channel fails quietly on purpose - a shop must not get a dialog box
every ten minutes because the internet is flaky - but that makes a till which
never reports completely opaque from either end.  This answers it in one
command, on the machine itself, with no GUI and no hardware:

    python check_support.py              look only, change nothing
    python check_support.py --send       also post one real snapshot
    python check_support.py --use-builtin   clear a stale hand-typed address

It checks each layer separately so the answer is a specific layer and not
"it doesn't work": the settings actually in force and where each value came
from, whether certifi is installed, DNS, TCP, the TLS handshake and chain,
then HTTP.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import ssl
import sys
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OK, BAD, WARN = "  OK   ", "  FAIL ", "  WARN "
problems: list = []


def say(mark: str, text: str) -> None:
    print(mark + text)
    if mark is BAD:
        problems.append(text)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--send", action="store_true", help="post one real snapshot, not just a health check")
    ap.add_argument("--use-builtin", action="store_true",
                    help="clear the address and key from config.json so the built-in ones are used")
    args = ap.parse_args()

    print("=" * 70)
    print(f"AI POS Scale support check   {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # ---------------------------------------------------------------- imports
    try:
        import support_config
        from settings_manager import SettingsManager, data_path
        import remote_support
    except Exception as exc:
        print(BAD + f"cannot import the app: {exc}")
        print("\nRun this from the folder that holds app.py, with the same Python the app uses.")
        return 2

    print("\n--- this build ---")
    print(f"       python {sys.version.split()[0]} ({'64' if sys.maxsize > 2**32 else '32'}-bit)")
    print(f"       folder {os.path.dirname(os.path.abspath(__file__))}")
    built_in = support_config.DEFAULT_URL or "(none)"
    print(f"       built-in address {built_in}")
    if not support_config.have_defaults():
        say(WARN, "this build has no address baked in - it must be configured by hand")

    # ------------------------------------------------------------- the config
    print("\n--- settings in force ---")
    settings = SettingsManager(data_path("config.json"))
    print(f"       config file {settings.path}")
    if settings.load_error:
        say(WARN, f"config.json could not be read and was replaced: {settings.load_error}")
    sup = settings.section("support")

    if args.use_builtin:
        settings.set("support.url", "")
        settings.set("support.token", "")
        settings.set("support.enabled", True)
        settings.save()
        sup = settings.section("support")
        print("       cleared the saved address and key: the built-in ones now apply")

    typed_url = (sup.get("url") or "").strip()
    typed_token = (sup.get("token") or "").strip()
    url = (typed_url or support_config.DEFAULT_URL).rstrip("/")
    token = typed_token or support_config.DEFAULT_TOKEN

    print(f"       enabled  {bool(sup.get('enabled'))}")
    print(f"       address  {url or '(none)'}   <- {'config.json' if typed_url else 'built into this build'}")
    print(f"       key      {'set' if token else 'MISSING'} ({len(token)} chars)"
          f"   <- {'config.json' if typed_token else 'built into this build'}")
    print(f"       every    {sup.get('interval_min', 10)} min   actions={sup.get('allow_actions', True)}")

    if not sup.get("enabled"):
        say(BAD, "the channel is switched OFF in config.json - nothing will ever be sent")
    # the trap this script exists for: a value typed in before the address was
    # baked in still wins, so hardcoding it did not take effect on this machine
    if typed_url and support_config.DEFAULT_URL and typed_url.rstrip("/") != support_config.DEFAULT_URL.rstrip("/"):
        say(BAD, f"config.json points somewhere else: {typed_url}\n"
                 f"         the built-in address is {support_config.DEFAULT_URL}\n"
                 f"         run this script with --use-builtin to clear it")
    if not url or not token:
        say(BAD, "no address or no key, so there is nothing to test")
        return report()

    # ---------------------------------------------------------------- certifi
    print("\n--- certificates ---")
    bundle = remote_support.ca_bundle()
    shipped = remote_support.shipped_roots()
    if bundle and shipped and os.path.abspath(bundle) == os.path.abspath(shipped):
        say(OK, f"certifi is not installed; using the roots shipped with the app: {bundle}")
    elif bundle:
        say(OK, f"verifying against certifi: {bundle}")
    else:
        say(BAD, "no certificate bundle at all, so TLS falls back to the Windows "
                 "certificate store - that is what caused the original SSL error.\n"
                 "         certs/support-roots.pem is missing, so run:  git pull\n"
                 "         or install certifi:  python -m pip install -r requirements.txt")

    drift = abs(time.time() - time.mktime(time.gmtime()) - (time.altzone if time.daylight else time.timezone) * -1)
    print(f"       machine clock {time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print("       (a clock wrong by more than a day fails every certificate)")
    del drift

    # ----------------------------------------------------------- the layers
    parsed = urlparse(url)
    host = parsed.hostname or ""
    port = parsed.port or (443 if parsed.scheme == "https" else 80)

    print(f"\n--- reaching {host}:{port} ---")
    try:
        addrs = sorted({ai[4][0] for ai in socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)})
        say(OK, f"DNS resolves to {', '.join(addrs)}")
    except Exception as exc:
        say(BAD, f"DNS cannot resolve {host}: {exc}")
        return report()

    try:
        t0 = time.time()
        with socket.create_connection((host, port), timeout=15):
            say(OK, f"TCP connects in {time.time() - t0:.2f}s")
    except Exception as exc:
        say(BAD, f"TCP cannot connect to {host}:{port}: {exc}\n"
                 "         a firewall, a proxy, or no internet on this machine")
        return report()

    if parsed.scheme == "https":
        try:
            context = remote_support.ssl_context(str(sup.get("ca_file") or ""))
            with socket.create_connection((host, port), timeout=15) as raw:
                with context.wrap_socket(raw, server_hostname=host) as tls:
                    cert = tls.getpeercert()
                    issuer = dict(x[0] for x in cert.get("issuer", ())).get("commonName", "?")
                    say(OK, f"TLS handshake fine, {tls.version()}, issued by {issuer}, "
                            f"valid until {cert.get('notAfter', '?')}")
        except Exception as exc:
            say(BAD, "TLS failed: " + remote_support.explain_tls_error(exc, url))
            return report()

    try:
        t0 = time.time()
        with urllib.request.urlopen(url + "/healthz", timeout=20,
                                    context=remote_support.ssl_context(str(sup.get("ca_file") or ""))) as resp:
            body = resp.read(2048).decode("utf-8", "replace").strip()
        say(OK, f"the service answered in {time.time() - t0:.2f}s: HTTP {resp.status} {body}")
    except urllib.error.HTTPError as exc:
        say(BAD, f"the service answered HTTP {exc.code} on /healthz - wrong path? the address "
                 f"should end at /pos, with no trailing slash")
        return report()
    except Exception as exc:
        say(BAD, f"could not reach {url}/healthz: {exc}")
        return report()

    # ------------------------------------------------------------ the real thing
    print("\n--- posting as this till ---")
    identity = remote_support.machine_identity()
    print(f"       this till will appear as '{identity['host']}'")
    if not args.send:
        print("       (not sending; add --send to post one real snapshot)")
        return report()

    class Fixed:
        def get(self, key, default=None):
            return {"support.enabled": True, "support.url": url, "support.token": token,
                    "support.ca_file": sup.get("ca_file", "")}.get(key, default)

    channel = remote_support.RemoteSupport(
        Fixed(), lambda: {"check_support": True, "at": time.strftime("%Y-%m-%d %H:%M:%S")}, lambda n, a: {})
    try:
        reply = channel.test()
        say(OK, f"the snapshot was accepted: {json.dumps(reply, ensure_ascii=False)}")
        print(f"\n       '{identity['host']}' should now be on the dashboard.")
    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            say(BAD, "the service rejected the key (HTTP 401).\n"
                     "         the key in config.json is wrong; --use-builtin clears it")
        else:
            say(BAD, f"the service refused the snapshot: HTTP {exc.code} {exc.reason}")
    except Exception as exc:
        say(BAD, f"could not post the snapshot: {exc}")
    return report()


def report() -> int:
    print("\n" + "=" * 70)
    if problems:
        print(f"{len(problems)} PROBLEM(S) FOUND\n")
        for text in problems:
            print("  - " + text)
    else:
        print("ALL CHECKS PASSED - this till can reach the support service.")
    print("=" * 70)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
