#!/usr/bin/env bash
# Stop the myagent daemon and remove its launchd agent.
set -euo pipefail

LABEL="com.myagent.memory"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"

launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
rm -f "$PLIST"
echo "Removed $LABEL (data in ~/.myagent is untouched)"
