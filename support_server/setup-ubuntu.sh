#!/usr/bin/env bash
# Install the POS support service on an Ubuntu box that already runs nginx with
# a TLS certificate.
#
# It is deliberately additive: the service listens on the loopback only, and
# nginx is given one extra location block on an existing site, so no DNS record
# and no new certificate are needed.  Everything it writes is namespaced under
# /srv/projects/pos-support and /etc/pos-support.
#
#   sudo ./setup-ubuntu.sh --site /etc/nginx/sites-available/deploypal.conf --path /pos
#
# Re-running it is safe: tokens are generated once and kept.
set -euo pipefail

SITE=""
MOUNT="/pos"
PORT="8900"
ROOT="/srv/projects/pos-support"
ETC="/etc/pos-support"
SERVICE="pos-support"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --site)  SITE="$2"; shift 2 ;;
    --path)  MOUNT="$2"; shift 2 ;;
    --port)  PORT="$2"; shift 2 ;;
    --root)  ROOT="$2"; shift 2 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

[[ $EUID -eq 0 ]] || { echo "run with sudo" >&2; exit 1; }
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==> installing to $ROOT"
install -d -m 755 "$ROOT" "$ROOT/data" "$ETC"
install -m 755 "$HERE/pos_support.py" "$ROOT/pos_support.py"

if ! id -u pos-support >/dev/null 2>&1; then
  echo "==> creating the pos-support service user"
  useradd --system --home "$ROOT" --shell /usr/sbin/nologin pos-support
fi
chown -R pos-support:pos-support "$ROOT/data"

# ---- tokens, generated once and then left alone
if [[ ! -f "$ETC/tokens.env" ]]; then
  echo "==> generating tokens"
  umask 077
  cat > "$ETC/tokens.env" <<EOF
POS_SUPPORT_DEVICE_TOKEN=$(head -c 24 /dev/urandom | base64 | tr -d '=+/' | cut -c1-32)
POS_SUPPORT_ADMIN_TOKEN=$(head -c 24 /dev/urandom | base64 | tr -d '=+/' | cut -c1-32)
EOF
fi
chmod 600 "$ETC/tokens.env"
chown root:root "$ETC/tokens.env"

echo "==> writing the systemd unit"
cat > "/etc/systemd/system/$SERVICE.service" <<EOF
[Unit]
Description=AI POS Scale support service
After=network.target

[Service]
Type=simple
User=pos-support
Group=pos-support
EnvironmentFile=$ETC/tokens.env
Environment=POS_SUPPORT_HOST=127.0.0.1
Environment=POS_SUPPORT_PORT=$PORT
Environment=POS_SUPPORT_DATA=$ROOT/data
ExecStart=/usr/bin/python3 $ROOT/pos_support.py
Restart=always
RestartSec=3
# it only ever needs its own data directory
NoNewPrivileges=yes
PrivateTmp=yes
ProtectSystem=strict
ProtectHome=yes
ReadWritePaths=$ROOT/data

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now "$SERVICE"
sleep 1
systemctl is-active --quiet "$SERVICE" || { journalctl -u "$SERVICE" -n 30 --no-pager; exit 1; }

# ---- nginx: one extra location on an existing, already-certificated site
if [[ -n "$SITE" ]]; then
  if [[ ! -f "$SITE" ]]; then echo "no such nginx site: $SITE" >&2; exit 1; fi
  if grep -q "location ${MOUNT}/" "$SITE"; then
    echo "==> nginx already has ${MOUNT}/, leaving it alone"
  else
    echo "==> adding ${MOUNT}/ to $SITE"
    cp "$SITE" "$SITE.bak.$(date +%s)"
    # insert into the TLS server block (the last one that listens on 443)
    python3 - "$SITE" "$MOUNT" "$PORT" <<'PY'
import re, sys
path, mount, port = sys.argv[1], sys.argv[2], sys.argv[3]
text = open(path).read()
block = f"""
    location {mount}/ {{
        proxy_pass http://127.0.0.1:{port}/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        client_max_body_size 16m;
        proxy_read_timeout 120s;
    }}
"""
# the last "listen 443" server block is the TLS one; close it after our block
matches = list(re.finditer(r"listen[^;]*443[^;]*;", text))
if not matches:
    sys.exit("no TLS server block found in " + path)
start = text.rfind("server", 0, matches[-1].start())
depth, i = 0, text.index("{", start)
end = None
for j in range(i, len(text)):
    if text[j] == "{":
        depth += 1
    elif text[j] == "}":
        depth -= 1
        if depth == 0:
            end = j
            break
if end is None:
    sys.exit("could not find the end of the server block")
open(path, "w").write(text[:end] + block + text[end:])
print("inserted")
PY
  fi
  nginx -t
  systemctl reload nginx
fi

echo
echo "==== done ===================================================="
systemctl --no-pager --lines=0 status "$SERVICE" | head -3
echo
echo "tokens (keep these safe):"
cat "$ETC/tokens.env"
echo
if [[ -n "$SITE" ]]; then
  echo "dashboard:  https://<your-host>${MOUNT}/    (user: anything, password: the ADMIN token)"
  echo "till URL :  https://<your-host>${MOUNT}"
fi
echo "health   :  curl -s http://127.0.0.1:$PORT/healthz"
