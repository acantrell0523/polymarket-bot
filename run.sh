#!/bin/bash
# Launch trading loop + supervisor from THIS repo (wherever it lives).
# The old hardcoded ~/Documents/polymarket-bot pointed at a different
# checkout entirely.
cd "$(dirname "$0")"
source venv/bin/activate
mkdir -p reports
nohup python -m bot.trading_loop > reports/live_output.log 2>&1 &
echo "trading loop started (pid $!)"
nohup python -m bot.supervisor > reports/supervisor.log 2>&1 &
echo "supervisor started (pid $!)"
