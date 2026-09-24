#!/bin/bash
# Copy code from the main checkout into each strategy profile directory.
# Profiles keep their own data/, reports/ and .env; strategy settings live
# in each profile's launchd plist as POLYBOT_* environment overrides.
set -e
MAIN="$HOME/Projects/polymarket-bot"
for name in $("$MAIN/venv/bin/python" "$MAIN/scripts/profiles.py" --names); do
  dst="$HOME/Projects/polybot-profiles/$name"
  mkdir -p "$dst/data" "$dst/reports"
  rsync -a --delete \
    --exclude venv --exclude data --exclude reports --exclude .git \
    --exclude '__pycache__' --exclude '.pytest_cache' \
    "$MAIN/" "$dst/"
  [ -e "$dst/venv" ] || ln -s "$MAIN/venv" "$dst/venv"
  [ -f "$dst/.env" ] || cp "$MAIN/.env" "$dst/.env"
  echo "synced $name"
done
