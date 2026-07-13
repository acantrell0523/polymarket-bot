#!/bin/bash
# Install the bot + supervisor as macOS launchd user services:
#   - start automatically at login/reboot (RunAtLoad)
#   - restart automatically on crash (KeepAlive, verified by kill test)
# NOTE: launchd agents only run while you're logged in, and NOTHING runs
# while the Mac sleeps. For true 24/7, deploy docker-compose to an
# always-on box; meanwhile prevent sleep on AC power with:
#   sudo pmset -c sleep 0
set -e
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PY="$REPO/venv/bin/python"
mkdir -p ~/Library/LaunchAgents "$REPO/reports"
for name in bot supervisor; do
  MODULE=$([ "$name" = "bot" ] && echo "bot.trading_loop" || echo "bot.supervisor")
  PLIST=~/Library/LaunchAgents/com.polymarket.$name.plist
  cat > "$PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.polymarket.$name</string>
  <key>ProgramArguments</key>
  <array><string>$PY</string><string>-m</string><string>$MODULE</string></array>
  <key>WorkingDirectory</key><string>$REPO</string>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>ThrottleInterval</key><integer>15</integer>
  <key>StandardOutPath</key><string>$REPO/reports/launchd_$name.log</string>
  <key>StandardErrorPath</key><string>$REPO/reports/launchd_$name.log</string>
</dict>
</plist>
PLIST
  launchctl unload "$PLIST" 2>/dev/null || true
  launchctl load "$PLIST"
done
echo "installed. status:"; launchctl list | grep polymarket
