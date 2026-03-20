# Polymarket Trading Bot

A modular Python bot for detecting pricing inefficiencies on [Polymarket US](https://polymarket.us) prediction markets, with backtesting, parameter sweeps, live in-game sports betting, and Slack alerts.

> **Disclaimer:** This is for educational purposes only. Trading prediction markets involves risk of loss. This is not financial advice. Use at your own risk.

## Features

- **8 trading signals** from free public data (momentum, volume, order book imbalance, mean reversion, volatility, smart money, whale flow, sentiment)
- **Live sports betting** with aggressive in-game parameters (3s scan, tighter thresholds)
- **Paper trading mode** for risk-free testing
- **Live trading** via the official [polymarket-us](https://pypi.org/project/polymarket-us/) SDK
- **Backtesting engine** with slippage/fee simulation and benchmark comparison
- **Parameter sweeps** to find optimal settings across edge thresholds, sizing methods, and stop-loss levels
- **Slack alerts** for trade opens, closes, daily summaries, and errors
- **Risk controls** — per-position stop-loss, portfolio exposure cap, daily loss limit

## Architecture

```
polymarket-bot/
├── bot/
│   ├── signals/
│   │   ├── signals.py          # 5 core signals
│   │   ├── unusual_whales.py   # 3 on-chain enrichment signals
│   │   └── estimator.py        # Combines signals, detects edge
│   ├── strategies/
│   │   ├── sizing.py           # Kelly Criterion & fixed fractional sizing
│   │   └── risk.py             # Stop-loss, take-profit, daily loss limit
│   ├── market_data.py          # Polymarket US API + CLOB client
│   ├── execution.py            # Order execution via polymarket-us SDK
│   ├── portfolio.py            # Position tracking, P&L accounting
│   ├── alerts.py               # Slack webhook alerts
│   └── trading_loop.py         # Main orchestrator loop
├── backtest/
│   ├── engine.py               # Replay engine with slippage & fee simulation
│   ├── sweep.py                # Parameter sensitivity analysis
│   ├── reporting.py            # Charts, JSON/CSV exports
│   └── runner.py               # CLI entry point
├── data/
│   └── loader.py               # Historical data loader + synthetic generator
├── utils/
│   ├── config.py               # YAML + .env config loader
│   ├── logger.py               # Structured JSON logging
│   └── models.py               # Shared data models
├── tests/
│   └── test_core.py            # Unit tests (44 tests)
├── configs/
│   └── config.yaml             # Master configuration
├── .env.example                # Environment variable template
└── requirements.txt            # Python dependencies
```

## Setup

### 1. Prerequisites

- **Python 3.10+** (the `polymarket-us` SDK requires it)
- A [Polymarket US](https://polymarket.us) account with identity verification

### 2. Install dependencies

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install polymarket-us
```

### 3. Get API keys

1. Go to [polymarket.us/developer](https://polymarket.us/developer)
2. Sign in and generate a new API key
3. Copy both the **Key ID** and **Secret Key** (the secret is only shown once)

### 4. Configure environment

```bash
cp .env.example .env
```

Edit `.env` with your credentials:
```
POLYMARKET_KEY_ID=your_key_id_here
POLYMARKET_SECRET_KEY=your_secret_key_here
```

### 5. Run tests

```bash
python -m pytest tests/ -v
```

## Usage

### Paper Trading (no API keys needed)

Set `paper_trading: true` in `configs/config.yaml`, then:

```bash
python -m bot.trading_loop
```

The bot will scan markets, detect mispricings, and simulate trades without placing real orders.

### Live Trading

1. Ensure your `.env` has valid API keys and your account is funded
2. Set `paper_trading: false` in `configs/config.yaml`
3. Start conservatively — set `max_position_size_usd: 20.0`

```bash
python -m bot.trading_loop
```

The bot uses the official `polymarket-us` SDK to place limit orders on the Polymarket US exchange.

**WARNING:** Live trading uses real funds. Start with small position sizes and monitor closely.

### Backtesting

```bash
# Run a single backtest with current config
python -m backtest.runner

# Run a parameter sweep to find optimal settings
python -m backtest.runner --sweep
```

Output goes to `reports/`.

## Slack Alerts (Optional)

Get real-time notifications for trades, daily summaries, and errors.

### Setup

1. Create a [Slack Incoming Webhook](https://api.slack.com/messaging/webhooks) for your workspace
2. Add the webhook URL to `.env`:
   ```
   SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL
   ```
3. Enable alerts in `configs/config.yaml`:
   ```yaml
   alerts:
     enabled: true
     on_trade_open: true
     on_trade_close: true
     on_daily_summary: true
     on_error: true
   ```

### Alert types

- **Trade Opened** — market, side, price, size, edge
- **Trade Closed** — market, entry/exit price, P&L, close reason
- **Daily Summary** — trades, win rate, P&L, bankroll
- **Bot Started** — mode, bankroll, markets in scope
- **Error** — scan cycle failures

## How Edge Detection Works

Each market passes through 8 independent signals combined via confidence-weighted averaging:

| Signal | Weight | Source |
|--------|--------|--------|
| Order Book Imbalance | 25% | US API `/markets/{slug}/book` |
| Price Momentum | 20% | CLOB `/prices-history` |
| Volume | 15% | US API `/v1/markets` |
| Mean Reversion | 10% | CLOB `/prices-history` |
| Volatility | 5% | CLOB `/prices-history` |
| Smart Money | 10% | CLOB `/trades` |
| Whale Flow | 10% | CLOB `/trades` |
| Market Sentiment | 5% | US API `/v1/markets` |

**Edge** = estimated probability − market price. If this exceeds the configured threshold (default 2%), a trade signal is generated.

### Live Game Mode

For sports markets currently in progress (`gameStartTime` in the past):
- Scan interval drops to **3 seconds**
- Order book imbalance weight increases to **35%**, momentum to **30%**
- Edge threshold lowers to **1.5%**
- Take-profit tightens to **0.5%**
- Games older than 4 hours are automatically skipped

## Configuration

All parameters are in `configs/config.yaml`. Key settings:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `paper_trading` | `true` | Simulate trades without real orders |
| `min_edge_threshold` | `0.02` | Minimum edge to trigger a trade (2%) |
| `max_position_size_usd` | `20.0` | Max USD per trade |
| `position_sizing_method` | `fixed_fractional` | `kelly` or `fixed_fractional` |
| `stop_loss_threshold` | `0.20` | Close at 20% loss |
| `take_profit_threshold` | `0.01` | Close when edge narrows to 1% |
| `daily_loss_limit_usd` | `200.0` | Halt trading after this daily loss |
| `max_open_positions` | `40` | Max concurrent positions |

## License

MIT — see [LICENSE](LICENSE).
