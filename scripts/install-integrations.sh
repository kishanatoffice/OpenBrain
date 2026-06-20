#!/usr/bin/env bash
# Drop OpenBrain integration files into a target project for every supported
# AI coding tool. Run from (or pass) the project root you want to wire up.
#
#   scripts/install-integrations.sh [TARGET_PROJECT_DIR]
#
# Project-level files are copied into place. Global files (Cursor global,
# Antigravity ~/.gemini) are printed as manual steps, since clobbering a
# shared global config automatically would be rude.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$REPO_DIR/integrations"
TARGET="${1:-$(pwd)}"

if [ ! -d "$SRC" ]; then
  echo "error: $SRC not found" >&2
  exit 1
fi

place() {  # place <src-relative> <dest-relative>
  local src="$SRC/$1" dest="$TARGET/$2"
  mkdir -p "$(dirname "$dest")"
  cp "$src" "$dest"
  echo "  ✓ $2"
}

echo "Installing OpenBrain integrations into: $TARGET"
echo ""

echo "Cursor:"
place cursor/mcp.json          .cursor/mcp.json
place cursor/openbrain.mdc     .cursor/rules/openbrain.mdc

echo "VS Code (Copilot):"
place vscode/mcp.json                 .vscode/mcp.json
place vscode/openbrain-hooks.json     .github/hooks/openbrain.json
place vscode/copilot-instructions.md  .github/copilot-instructions.md

echo "JetBrains (Junie):"
place jetbrains/mcp.json       .junie/mcp/mcp.json
place jetbrains/guidelines.md  .junie/guidelines.md

echo "Antigravity (project rule):"
place antigravity/openbrain.md .antigravity/rules/openbrain.md

echo ""
echo "Manual (global / shared) steps:"
echo "  • Antigravity MCP: merge $SRC/antigravity/mcp_config.json into"
echo "      ~/.gemini/config/mcp_config.json"
echo "  • Cursor (global, optional): copy $SRC/cursor/mcp.json to ~/.cursor/mcp.json"
echo ""
echo "Claude Code is wired in the OpenBrain repo itself (.claude/settings.json)."
echo "Make sure the daemon is running on http://127.0.0.1:3111 before use."
