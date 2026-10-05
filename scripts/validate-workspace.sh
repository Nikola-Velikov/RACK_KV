#!/usr/bin/env bash
# Scan workspace files for unresolved [PLACEHOLDER] patterns
set -euo pipefail
FOUND=0
for f in CLAUDE.md private/PROJECT-INFO.md memory/*.md; do
  [ -f "$f" ] || continue
  matches=$(grep -oE '\{\{[A-Z_]+\}\}' "$f" 2>/dev/null || true)
  if [ -n "$matches" ]; then
    echo "WARN: $f has unresolved placeholders:"
    echo "$matches" | sort -u | sed 's/^/  /'
    FOUND=1
  fi
done
if [ $FOUND -eq 0 ]; then
  echo "PASS: No unresolved placeholders found."
else
  echo "FAIL: Fix the above placeholders before starting."
  exit 1
fi
