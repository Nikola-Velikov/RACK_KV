#!/bin/bash
#
# WorkflowTemplate Setup Script v3.0
# Copies agent definitions, memory files, and project config into your project.
#
# Usage:
#   ./scripts/setup.sh /path/to/your/project
#
# What it does:
#   1. Copies .claude/agents/ and .claude/settings.json to your project root
#   2. Copies memory/ files to Claude's project memory directory
#   3. Copies CLAUDE.md and private/PROJECT-INFO.md templates
#   4. Reminds you to register MCP server (if using HAOps)
#

set -e

# --- Arguments ---
PROJECT_ROOT="${1:?Usage: ./scripts/setup.sh /path/to/your/project}"

# Resolve to absolute path
PROJECT_ROOT="$(cd "$PROJECT_ROOT" 2>/dev/null && pwd)" || {
  echo "ERROR: Directory does not exist: $1"
  exit 1
}

echo "Setting up agent workspace for: $PROJECT_ROOT"
echo ""

# --- Locate template root (where this script lives) ---
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TEMPLATE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# --- 1. Copy agent definitions ---
echo "1. Copying .claude/agents/ to project..."
mkdir -p "$PROJECT_ROOT/.claude/agents"
cp "$TEMPLATE_ROOT/claude/agents/"*.md "$PROJECT_ROOT/.claude/agents/"

# Copy settings.json (base permissions)
cp "$TEMPLATE_ROOT/claude/settings.json" "$PROJECT_ROOT/.claude/settings.json"

# Copy settings.local template (developer customizes)
if [ ! -f "$PROJECT_ROOT/.claude/settings.local.json" ]; then
  cp "$TEMPLATE_ROOT/claude/settings.local.json.template" "$PROJECT_ROOT/.claude/settings.local.json"
  echo "   Created .claude/settings.local.json (customize permissions)"
else
  echo "   .claude/settings.local.json already exists, skipping"
fi

echo "   Done."
echo ""

# --- 2. Copy memory files ---
# Claude Code stores project memory at:
#   ~/.claude/projects/-<path-with-dashes>/memory/
# Convert project path: /Users/john/Projects/myapp → -Users-john-Projects-myapp
ESCAPED_PATH="$(echo "$PROJECT_ROOT" | tr '/' '-')"
MEMORY_DIR="$HOME/.claude/projects/$ESCAPED_PATH/memory"

echo "2. Copying memory files to: $MEMORY_DIR"
mkdir -p "$MEMORY_DIR"

for f in "$TEMPLATE_ROOT/memory/"*.md; do
  BASENAME="$(basename "$f")"
  if [ ! -f "$MEMORY_DIR/$BASENAME" ]; then
    cp "$f" "$MEMORY_DIR/$BASENAME"
    echo "   Copied: $BASENAME"
  else
    echo "   Exists: $BASENAME (skipping)"
  fi
done

echo "   Done."
echo ""

# --- 3. Copy project files ---
echo "3. Copying project configuration files..."

# CLAUDE.md (main project instructions)
if [ -f "$TEMPLATE_ROOT/templates/CLAUDE-TEMPLATE.md" ]; then
  if [ ! -f "$PROJECT_ROOT/CLAUDE.md" ]; then
    cp "$TEMPLATE_ROOT/templates/CLAUDE-TEMPLATE.md" "$PROJECT_ROOT/CLAUDE.md"
    echo "   Created: CLAUDE.md (fill in your project details)"
  else
    echo "   Exists: CLAUDE.md (skipping)"
  fi
fi

# private/PROJECT-INFO.md (credentials, git-ignored)
if [ -f "$TEMPLATE_ROOT/templates/PROJECT-INFO-TEMPLATE.md" ]; then
  mkdir -p "$PROJECT_ROOT/private"
  if [ ! -f "$PROJECT_ROOT/private/PROJECT-INFO.md" ]; then
    cp "$TEMPLATE_ROOT/templates/PROJECT-INFO-TEMPLATE.md" "$PROJECT_ROOT/private/PROJECT-INFO.md"
    echo "   Created: private/PROJECT-INFO.md (fill in credentials)"
  else
    echo "   Exists: private/PROJECT-INFO.md (skipping)"
  fi
fi

echo "   Done."
echo ""

# --- 4. Summary ---
echo "═══════════════════════════════════════════"
echo "  Setup Complete"
echo "═══════════════════════════════════════════"
echo ""
echo "Next steps:"
echo ""
echo "  1. Replace [PLACEHOLDERS] in all files:"
echo "     grep -r '{{' $PROJECT_ROOT/.claude/ $MEMORY_DIR/ $PROJECT_ROOT/CLAUDE.md"
echo ""
echo "  2. Add to .gitignore (if not already):"
echo "     .claude/settings.local.json"
echo "     private/"
echo ""
echo "  3. (Optional) Register HAOps Science MCP server:"
echo "     claude mcp add-json --scope user haops '{\"type\":\"stdio\",\"command\":\"node\",\"args\":[\"/path/to/haops-science-mcp-server/dist/index.js\"],\"env\":{\"HAOPS_API_URL\":\"https://science.haops.eu\",\"HAOPS_API_KEY\":\"your-key\"}}'"
echo ""
echo "  4. Start your first session:"
echo "     cd $PROJECT_ROOT"
echo "     claude"
echo "     > Tell Claude: 'You are the architect' (or dev/qa/devops agent)"
echo ""
