#!/bin/bash
# Weekly (Wednesday 05:00 ET, via launchd com.polymarket.weekly): rebuild the
# college team-code map from this week's Polymarket questions, copy code to
# every strategy profile, and restart the bots so they load the new map.
set -e
cd "$(dirname "$0")/.."
echo "== $(date '+%Y-%m-%d %H:%M:%S')"
./venv/bin/python scripts/build_cfb_teams.py
bash scripts/sync_profiles.sh
for j in bot bot.aggressive bot.ride bot.balanced; do
  launchctl kickstart -k "gui/$(id -u)/com.polymarket.$j" && echo "restarted $j"
done
