#!/usr/bin/env bash
# OpenBrain — one-command install.
#
#   ./install.sh
#
# Creates the venv, installs deps, registers the daemon as a launchd service
# (starts at login, restarts on crash), waits for it to come up, then offers to
# seed your persona. Safe to re-run.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"
PYTHON="$PROJECT_DIR/.venv/bin/python"
PORT="${MEMORY_PORT:-3111}"

echo "→ OpenBrain install (project: $PROJECT_DIR)"

# 1. venv + deps
if [[ ! -x "$PYTHON" ]]; then
  echo "→ creating virtualenv (.venv)"
  python3 -m venv .venv
fi
echo "→ installing dependencies"
"$PROJECT_DIR/.venv/bin/pip" install -q -r requirements.txt

# 2. register + start the launchd service
echo "→ registering background service"
bash "$PROJECT_DIR/scripts/install-launchd.sh"

# 3. wait for health
echo -n "→ waiting for the daemon"
for _ in $(seq 1 30); do
  if curl -s --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo " — up"; break
  fi
  echo -n "."; sleep 0.5
done

if ! curl -s --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
  echo ""
  echo "⚠️  Daemon did not respond on port $PORT. Check logs:"
  echo "    tail -f ~/.myagent/logs/myagent.err.log"
  exit 1
fi

# 4. offer to seed persona (only interactively, and only if brain looks empty)
count="$(curl -s "http://127.0.0.1:$PORT/health" \
  | "$PYTHON" -c "import json,sys;print(json.load(sys.stdin).get('memory_count',0))" 2>/dev/null || echo 0)"

echo ""
echo "✅  OpenBrain is running:  http://127.0.0.1:$PORT/"

# 5. install the `openbrain` CLI shim onto PATH
BIN_DIR="$HOME/.local/bin"
mkdir -p "$BIN_DIR"
chmod +x "$PROJECT_DIR/bin/openbrain"
ln -sf "$PROJECT_DIR/bin/openbrain" "$BIN_DIR/openbrain"
echo "→ installed CLI: $BIN_DIR/openbrain"
case ":$PATH:" in
  *":$BIN_DIR:"*) : ;;
  *) echo "  ⚠ add to PATH (then restart your shell):"
     echo "      echo 'export PATH=\"$BIN_DIR:\$PATH\"' >> ~/.zshrc" ;;
esac

# 6. auto-wire every detected AI tool (zero-config)
echo ""
"$PYTHON" -m myagent connect

# 7. offer to seed persona (only interactively, and only if brain is empty)
if [[ -t 0 && "$count" == "0" ]]; then
  read -r -p "→ Set up your always-on persona now? [Y/n] " ans
  if [[ -z "$ans" || "$ans" =~ ^[Yy]$ ]]; then
    "$PYTHON" -m myagent init
  fi
else
  echo "Seed your persona anytime:  $PYTHON -m myagent init"
fi
