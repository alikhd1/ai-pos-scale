"""
support_config.py
-----------------
The support service this build reports to, baked in.

These tills are deployed to shops by somebody who is not going to type a URL
and a 32-character token into a settings dialog on a machine they cannot
reach.  So the address and the key ship inside the build, and a fresh install
starts reporting on its own.

Both are only DEFAULTS: anything in config.json wins, so a particular till can
be pointed somewhere else, or switched off entirely, without a new build.
Setting them to "" turns the whole channel back off by default.

A word on the token
===================
A secret inside a shipped executable is not really a secret.  Anyone with the
exe can read it out - there is no way around that, and no amount of obfuscation
changes it.  What it protects is correspondingly modest, and that is why the
service has two separate keys: this one only lets a till POST its own status.
It cannot read another till's data, cannot open the dashboard, and cannot queue
an action for anybody.  The admin token, which can do those things, is never
shipped and lives only on the server.

If this device token does leak, the worst case is somebody posting junk
snapshots.  Rotate it by editing /etc/pos-support/tokens.env on the server,
restarting pos-support, and shipping a build with the new value here.
"""

#: Where tills report.  https:// unless it is the loopback.
DEFAULT_URL = "https://rpa.adadsoft.ir/pos"

#: The device token from /etc/pos-support/tokens.env on the support server.
DEFAULT_TOKEN = "8vo3VFpXSwLpIm1yqdZPKn2pUqrpYFwk"


def have_defaults() -> bool:
    """True when this build knows where to report without being told."""
    return bool(DEFAULT_URL.strip() and DEFAULT_TOKEN.strip())
