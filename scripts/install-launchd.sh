#!/usr/bin/env bash
# Install myagent as a launchd agent so the daemon starts at login and
# restarts on crash. Re-run after moving the project; safe to run repeatedly.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$PROJECT_DIR/.venv/bin/python"
LABEL="com.myagent.memory"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
LOG_DIR="$HOME/.myagent/logs"

if [[ ! -x "$PYTHON" ]]; then
  echo "error: $PYTHON not found — create the venv first:" >&2
  echo "  python3 -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
  exit 1
fi

mkdir -p "$LOG_DIR" "$HOME/Library/LaunchAgents"

cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>$LABEL</string>
    <key>ProgramArguments</key>
    <array>
        <string>$PYTHON</string>
        <string>-m</string>
        <string>myagent</string>
    </array>
    <key>WorkingDirectory</key>
    <string>$PROJECT_DIR</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <dict>
        <key>SuccessfulExit</key>
        <false/>
    </dict>
    <key>StandardOutPath</key>
    <string>$LOG_DIR/myagent.out.log</string>
    <key>StandardErrorPath</key>
    <string>$LOG_DIR/myagent.err.log</string>
</dict>
</plist>
EOF

# Reload cleanly if already installed.
launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"

echo "Installed and started $LABEL"
echo "  status:  launchctl print gui/$(id -u)/$LABEL | head -20"
echo "  logs:    tail -f $LOG_DIR/myagent.err.log"
echo "  remove:  scripts/uninstall-launchd.sh"
