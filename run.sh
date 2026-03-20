#!/bin/bash
cd ~/Documents/polymarket-bot
source venv/bin/activate
nohup python -m bot.trading_loop > reports/live_output.log 2>&1 &
nohup python -m bot.supervisor > reports/supervisor.log 2>&1 &
echo "Bot and supervisor started"
