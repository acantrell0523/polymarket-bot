# Polymarket Trading Bot

An autonomous Python bot for detecting and trading pricing inefficiencies on
[Polymarket US](https://polymarket.us) prediction markets — with cost-aware
edge detection, fractional Kelly sizing, layered risk controls, full decision
auditability, backtesting, and Slack observability.

**Strictly focused on current sports.** A central league registry
(`bot/leagues.py`) covers MLB, WNBA, MLS, NBA, NHL, NFL, NCAA basketball, and
EPL; off-season leagues return zero games from ESPN and reactivate
automatically when their seasons start — the bot always follows whatever is
being played today. A daily recorder (supervisor job at 05:30 ET, or
`python scripts/ingest_historical.py`) builds the backtest dataset forward
from today: game prices plus de-vigged sportsbook consensus per game.

> **Disclaimer:** This is for educational purposes only. Trading prediction
> markets involves risk of loss. This is not financial advice. Use at your own
> risk.

## How It Makes Decisions

1. **Estimate probability.** External sources (sportsbook consensus via
   the-odds-api / ESPN / FanDuel / Pinnacle, PredictIt cross-market prices, a
   log-normal crypto model) plus market microstructure signals (order-book and
   liquidity imbalance, on-chain whale/smart-money flow) are pooled with a
   **Bayesian log-odds update anchored on the market price as the prior**.
   With no external evidence the estimate *is* the market price — the bot
   cannot manufacture edge from nothing (the "external validation gate").

2. **Compute NET edge.** Gross edge (estimate − price) is re-priced at the
   **executable** price (best ask for buys, best bid for sells) minus the
   taker fee. A 5% gross edge across a 4-cent spread with a 2% fee is a ~1%
   real edge — the bot gates on the real number (`trading.min_net_edge`).

3. **Filter.** Seven-plus-two checks: ≥2 sportsbooks agree, per-league minimum
   edge, tradeable price band (15–85¢), ≥$1k book depth, no correlated
   position on the same game, daily trade limit, no entries in the last 5
   minutes of a game, spread narrow enough to exit (`max_spread`), and the
   net-edge gate.

4. **Size with fractional Kelly.** `f* = (p·b − q)/b`, computed on the
   fee-grossed executable cost, scaled by `kelly_fraction` (default 0.25 =
   quarter-Kelly) — bet size proportional to edge *and* bankroll, shrunk for
   estimation error. Hard caps: per-trade max, portfolio exposure max, never
   >50% of cash.

5. **Manage the position.** Estimates are refreshed every scan as new
   information arrives. Exits: stop-loss (25%), aggressive take (up 30%),
   trailing stop, edge-convergence take-profit, and "let it ride" (hold to
   resolution when ≥70% in your favor).

6. **Log everything.** Every opportunity that reaches validation — executed
   or rejected — lands in the `decision_log` SQLite table with the estimated
   probability, market/executable price, gross and net edge, spread, fee,
   size, and the exact reason. Closed positions get full exit telemetry.

## Safety & Autonomy

- **Daily loss limit** — trading pauses durably (survives restarts) until the
  next UTC day after losing `daily_loss_limit_usd`.
- **Kill switch** — supervisor halts the bot if account value drops below 50%
  of start; remove `data/kill_switch` to resume.
- **Heartbeat monitoring** — the loop writes `data/heartbeat.json` every
  cycle; the supervisor Slack-alerts if it goes stale.
- **Degraded mode** — repeated market-data failures back off exponentially
  (up to 5 min) with a single alert per outage, instead of hammering a dead API.
- **Paper mode** — full pipeline with simulated fills; no credentials needed.

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install polymarket-us          # live trading only; paper mode works without it

cp .env.example .env               # then fill in your keys
python -m pytest tests/ -q         # 152 tests
```

Required keys in `.env`:

| Variable | Needed for |
|----------|-----------|
| `THE_ODDS_API_KEY` | Sports trading (≥2 books required; free tier at the-odds-api.com) |
| `POLYMARKET_KEY_ID` / `POLYMARKET_SECRET_KEY` | Live trading (polymarket.us/developer) |
| `SLACK_WEBHOOK_URL` | Alerts (optional) |

## Usage

```bash
# Paper trading (default; no credentials needed)
python -m bot.trading_loop

# Live trading + supervisor
bash run.sh

# Docker (bot + supervisor, shared data volume, heartbeat healthcheck)
docker compose up -d --build

# Backtest (SQLite historical data → JSON → synthetic fallback)
python -m backtest.runner
python -m backtest.runner --sweep    # parameter sensitivity sweep

# Post-hoc analysis
python scripts/analyze_exits.py --days 30
python scripts/inspect_historical.py
```

### Configuration

Everything lives in `configs/config.yaml`. Any scalar value can be overridden
per-environment without editing files:

```bash
POLYBOT_TRADING__PAPER_TRADING=false \
POLYBOT_TRADING__KELLY_FRACTION=0.25 \
POLYBOT_TRADING__DAILY_LOSS_LIMIT_USD=50 \
python -m bot.trading_loop
```

Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `trading.paper_trading` | `true` | Simulate trades without real orders |
| `trading.min_edge_threshold` | `0.05` | Minimum gross edge (per-league minimums may be higher) |
| `trading.min_net_edge` | `0.02` | Minimum edge after fees + spread |
| `trading.taker_fee_rate` | `0.02` | Exchange taker fee used everywhere |
| `trading.position_sizing_method` | `kelly` | `kelly`, `tiered_kelly`, or `fixed_fractional` |
| `trading.kelly_fraction` | `0.25` | Fraction of full Kelly |
| `trading.max_position_size_usd` | `50` | Hard cap per trade |
| `trading.max_portfolio_exposure_usd` | `500` | Total exposure cap |
| `trading.stop_loss_threshold` | `0.25` | Close at 25% loss |
| `trading.daily_loss_limit_usd` | `75` | Durable pause after this daily loss |
| `trading.max_spread` | `0.10` | Widest book the bot will enter |
| `signals.combination_method` | `logodds` | Bayesian log-odds pooling (`linear` = legacy) |

See `CLAUDE.md` for the full architecture reference, database schema, and the
current state of every subsystem.

## License

MIT — see [LICENSE](LICENSE).
