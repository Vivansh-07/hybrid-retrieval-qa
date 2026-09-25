#!/usr/bin/env bash
# Initialise this folder as a git repo and push it to GitHub.
# Usage: ./init_and_push.sh <github-username> [repo-name]
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <github-username> [repo-name]"
  exit 1
fi

USERNAME="$1"
REPO="${2:-hybrid-retrieval-qa}"
cd "$(dirname "$0")"

[ -d .git ] || git init
git add .
git commit -m "Initial commit: hybrid retrieval pipeline, evaluation harness, paper skeleton" || true
git branch -M main

if git remote get-url origin >/dev/null 2>&1; then
  git remote set-url origin "https://github.com/$USERNAME/$REPO.git"
else
  git remote add origin "https://github.com/$USERNAME/$REPO.git"
fi

echo "Pushing to https://github.com/$USERNAME/$REPO.git"
git push -u origin main
