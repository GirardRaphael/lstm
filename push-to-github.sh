#!/usr/bin/env bash
# Push the current branch to GitHub (GirardRaphael/lstm).
# Usage: GITHUB_TOKEN=ghp_... ./push-to-github.sh
# Or:    ./push-to-github.sh ghp_...
set -euo pipefail

TOKEN="${GITHUB_TOKEN:-${1:-}}"
if [ -z "$TOKEN" ]; then
  echo "Usage: GITHUB_TOKEN=<token> $0"
  echo "   or: $0 <token>"
  echo ""
  echo "Get a token at https://github.com/settings/tokens (repo scope)"
  exit 1
fi

cd "$(dirname "$0")"
BRANCH=$(git branch --show-current)
echo "Pushing $BRANCH to github.com/GirardRaphael/lstm..."
git push "https://x-access-token:${TOKEN}@github.com/GirardRaphael/lstm.git" "$BRANCH"
echo "Done. https://github.com/GirardRaphael/lstm/tree/$BRANCH"
