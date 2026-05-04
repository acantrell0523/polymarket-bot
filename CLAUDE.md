# CLAUDE.md — Polymarket Trading Bot Codebase Guide

This document is a comprehensive reference for AI assistants (and human developers) working in this codebase. It covers architecture, key subsystems, configuration, and the current state of the code.

---

## Table of Contents
1. [Overall Structure](#1-overall-structure)
2. [Sniper Strategy Logic](#2-sniper-strategy-logic)
3. [Signal System & Edge Detection](#3-signal-system--edge-detection)
4. [Kelly / Position Sizing](#4-kelly--position-sizing)
5. [Risk Management](#5-risk-management)
6. [Execution & Portfolio](#6-execution--portfolio)
7. [External Data Sources](#7-external-data-sources)
8. [Supervisor & Observability](#8-supervisor--observability)
9. [Backtesting](#9-backtesting)
10. [Configuration](#10-configuration)
11. [Database Schema](#11-database-schema)
12. [Current State: What Works, What's Broken, Rough Edges](#12-current-state-what-works-whats-broken-rough-edges)

---

## 1. Overall Structure

```
polymarket-bot/
├── bot/                        # Live trading core
│   ├── trading_loop.py         # Main orchestrator (TradingBot class + run loop)
│   ├── market_data.py          # Polymarket US API + CLOB API HTTP client
│   ├── execution.py            # Order placement via polymarket-us SDK
│   ├── portfolio.py            # Exchange-backed position tracker
│   ├── alerts.py               # Slack webhook alerts
│   ├── edge_log.py             # SQLite edge analytics + CLV tracking
│   ├── trade_db.py             # SQLite trade history + signal log
│   ├── supervisor.py           # Read-only scheduled agent (reports, kill switch)
│   ├── game_schedule.py        # ESPN schedule fetcher; drives scan sleep logic
│   ├── test_signals.py         # (likely a manual scratch file, not a test suite)
│   ├── signals/
│   │   ├── signals.py          # 5 signal functions (OB imbalance, line movement,
│   │   │                       #   odds value, liquidity imbalance, cross-market,
│   │   │                       #   sports context, crypto model)
│   │   ├── estimator.py        # ProbabilityEstimator: routes by market type,
│   │   │                       #   external validation gate, edge detection
│   │   ├── odds_api.py         # the-odds-api.com client + ESPN fallback
│   │   ├── book_scrapers.py    # FanDuel + Pinnacle free-API clients
│   │   ├── cross_market.py     # PredictIt API client (politics/other markets)
│   │   ├── crypto_api.py       # CoinGecko + log-normal crypto model
│   │   ├── sports_data.py      # ESPN game context (home advantage, fatigue)
│   │   └── live_odds.py        # Live in-game NCAA edge tracker
│   └── strategies/
│       ├── sizing.py           # Kelly / tiered-Kelly / fixed-fractional sizing
│       ├── risk.py             # Stop-loss, trailing stop, take-profit, let-it-ride
│       └── trade_filter.py     # Pre-trade validation checklist (validate_trade)
├── backtest/
│   ├── engine.py               # Replay engine with slippage/fee simulation
│   ├── sweep.py                # Parameter sensitivity sweep
│   ├── reporting.py            # Matplotlib charts + JSON/CSV exports
│   └── runner.py               # CLI entry point (python -m backtest.runner)
├── data/
│   ├── loader.py               # Historical JSON loader + GBM synthetic generator
│   ├── trades.db               # SQLite trade log (created at runtime)
│   └── settled_slugs.txt       # Persisted set of auto-settled slugs
├── utils/
│   ├── config.py               # YAML + .env loader; typed dataclass config
│   ├── logger.py               # Structured JSON logger
│   └── models.py               # Shared dataclasses (OrderBook, TradeSignal, etc.)
├── tests/
│   └── test_core.py            # pytest suite (currently broken — see §12)
├── onchain.py                  # Free on-chain enrichment client (not wired in live)
├── configs/config.yaml         # Master configuration file
├── config.yaml                 # Duplicate/root config (searched as fallback)
├── .env.example                # Environment variable template
├── requirements.txt            # Python dependencies
└── run.sh                      # Launch script (starts trading_loop + supervisor)
```

### How the modules fit together

```
run.sh
  ├── python -m bot.trading_loop    ← main loop
  │     TradingBot.run()
  │       ├── GameSchedule.should_be_scanning()   ← ESPN schedule gate
  │       ├── MarketDataClient.get_active_markets() → List[dict] from US API
  │       ├── MarketDataClient.build_snapshot()    → MarketSnapshot
  │       ├── ProbabilityEstimator.detect_edge()   → TradeSignal | None
  │       ├── trade_filter.validate_trade()        → rejection reason | None
  │       ├── PositionSizer.size_position()        → USD size
  │       ├── ExecutionEngine.execute_trade()      → Trade | None  (SDK or paper)
  │       ├── Portfolio.open_position()            → Position
  │       └── RiskManager / Portfolio.close_position() ← position lifecycle
  │
  └── python -m bot.supervisor     ← scheduled agent
        Supervisor.run()
          ├── daily_review()     @ 6 AM ET
          ├── morning_briefing() @ 8 AM ET
          └── check_kill_switch() every 15 min
```

---

## 2. Sniper Strategy Logic

**Philosophy**: "Sniper, not machine gun." Every scan cycle collects ALL edge opportunities, ranks them, and takes only the best 1–2 — subject to a strict validation checklist. This is implemented in `TradingBot.process_markets()` (`bot/trading_loop.py`).

### Dual-Speed Scan Loop

```
Every 3 seconds (live games only):
  - Split cached market list into live vs. pre-game
  - If any live games: fast-scan only those markets
  - Check open positions (tighter take-profit = 3%)

Every 60 seconds (full scan):
  - Fetch all active markets from US API
  - Build snapshots for all markets
  - Run full process_markets() pipeline
  - Check positions
```

The loop gates on `GameSchedule.should_be_scanning()` — if no game is live and none starts within 2 hours, the bot sleeps 120 seconds between checks (live mode only; paper mode always scans).

### Opportunity Detection Flow

1. **Fetch markets** — `MarketDataClient.get_active_markets()` returns markets filtered to:
   - Sports: upcoming within 24 hours, OR live (started ≤4 hours ago)
   - Non-sports: `endDate` within 14 days
   - Both: must be ≥1 hour to expiry

2. **Build snapshot** — `MarketDataClient.build_snapshot()` fetches the order book for each market and assembles a `MarketSnapshot`.

3. **Detect edge** — `ProbabilityEstimator.detect_edge()` (see §3). Returns a `TradeSignal` or `None`.

4. **Collect & rank** — All opportunities are passed to `rank_opportunities()`, which sorts by `abs(edge) + 0.005 if sell`. Shorts score slightly higher (short-bias bonus).

5. **Validate each** — `validate_trade()` in `bot/strategies/trade_filter.py` applies a 7-point checklist:
   1. Minimum 2 books from the-odds-api
   2. Per-league minimum edge (NHL 4%, NCAA 5%, NBA 7%)
   3. Price in range 15%–85% (no extreme favorites/longshots)
   4. Order book liquidity ≥ $1,000
   5. No existing position on the same game (correlated position check via `extract_game_id()`)
   6. Daily trade limit not reached
   7. Last-5-minutes block (buzzer-beater risk): if `game_time_remaining < 300s`, reject

6. **Size & execute** — Passes `PositionSizer.size_position()` then `ExecutionEngine.execute_trade()`.

7. **Cooldown** — After closing a position, the slug goes on a 10-minute cooldown (unless P&L > $2, in which case immediate re-entry is allowed).

### Live Game Mode

When `snapshot.is_live == True`, the bot uses a separate `self.live_estimator` with reweighted signals:
```
odds_value:              0.40
order_book_imbalance:    0.25
line_movement:           0.15
liquidity_imbalance:     0.20
```
Take-profit threshold tightens to 3% during live scans.

---

## 3. Signal System & Edge Detection

All signal logic lives in `bot/signals/`. The estimator auto-detects market type and routes to the correct signal set.

### Market Type Detection (`estimator.py`)

| Type      | Detection                                                      |
|-----------|----------------------------------------------------------------|
| `sports`  | Slug starts with `aec-`, `asc-`, `tsc-`, or `atc-`           |
| `crypto`  | Question/slug contains BTC, ETH, SOL, DOGE, XRP, etc.        |
| `politics`| Question contains election/president/congress/senate/etc.     |
| `other`   | Everything else                                               |

### Signal Weights by Market Type

| Signal              | Sports | Crypto | Politics | Other |
|---------------------|--------|--------|----------|-------|
| `odds_value`        | 40%    | —      | —        | —     |
| `sports_context`    | 15%    | —      | —        | —     |
| `line_movement`     | 20%    | —      | 15%      | 20%   |
| `order_book_imbal.` | 15%    | 20%    | 25%      | 25%   |
| `liquidity_imbal.`  | 10%    | 10%    | 15%      | 15%   |
| `crypto_model`      | —      | 45%    | —        | —     |
| `cross_market`      | —      | 25%    | 45%      | 40%   |

### Signal Descriptions

- **`odds_value`**: The primary signal. Fetches consensus probability from the-odds-api.com (or ESPN/FanDuel/Pinnacle fallback). Edge = `consensus_prob - polymarket_price`. Requires ≥2 books; ≥3 books for edges >7%. Sharp books (Pinnacle) are blended in at 60% when they disagree by ≥3%.
- **`order_book_imbalance`**: `bid_depth / (bid_depth + ask_depth)`. Mapped to [0.3, 0.7]. Confidence scales with total depth up to $1,000.
- **`liquidity_imbalance`**: Focuses on top-3 order book levels. Concentrated bids = informed buying signal.
- **`line_movement`**: Proxy using distance from 0.5. Markets far from even money have had significant line movement; signal endorses the current direction.
- **`sports_context`**: ESPN game data — home advantage (NBA: +3%, NCAAB: +5%), back-to-back detection, neutral site adjustment, conference tournament flag.
- **`cross_market`**: PredictIt API fuzzy-match. Returns confidence=0 if no match (blocks trade for politics/other markets).
- **`crypto_model`**: CoinGecko price + 30-day volatility → log-normal `P(S > K)` model for "Will X hit $Y by date Z?" markets.

### External Validation Gate

**If the primary external signal (odds_value / crypto_model / cross_market) has `confidence == 0`, the trade is blocked entirely.** No external data = no trade. This is the single most important guardrail.

Additional caps:
- If external edge < 2%: all non-primary signal confidences are halved
- Combined edge cannot exceed `external_edge + 1%` (edge cap)

### Probability Combination Formula

```
estimated_prob = sum(w_i * c_i * v_i) / sum(w_i * c_i)
```
where `w_i` = weight, `c_i` = confidence, `v_i` = signal value. Clamped to [0.01, 0.99].

### Edge Calculation

```
edge = estimated_prob - snapshot.price
side = "buy" if edge > 0 else "sell"
```
Trade fires if `min_edge ≤ |edge| ≤ max_edge`.

---

## 4. Kelly / Position Sizing

Implemented in `bot/strategies/sizing.py`. Three sizing methods:

### Tiered Kelly (`tiered_kelly`) — current default

Hard-coded tiers based on absolute edge (ignoring bankroll):

| Edge      | Size   |
|-----------|--------|
| 5–7%      | $20    |
| 7–10%     | $30    |
| 10–15%    | $40    |
| 15%+      | $50    |

### Full Kelly (`kelly`)

```python
b = (1 - price) / price          # net odds (buy side)
f* = kelly_fraction * (p*b - q) / b
size = f* * bankroll
```
`kelly_fraction` defaults to 0.5 (half-Kelly). For sells, `b = price / (1 - price)`.

### Fixed Fractional (`fixed_fractional`)

```python
size = fixed_fraction * bankroll   # default 15%
```

### Guardrails (applied after all three methods)

1. `size = min(size, max_position_size_usd)` — hard cap ($50 in config)
2. `size = min(size, available_exposure)` — portfolio exposure cap ($500 total)
3. `size = min(size, bankroll * 0.5)` — never more than 50% of cash
4. If `size < min_position_size_usd` ($5), return 0 (skip trade)

---

## 5. Risk Management

Implemented in `bot/strategies/risk.py`.

### Position Exit Hierarchy

Checked in this order on every cycle:

1. **Resolved** — price at 0.00 or 1.00 (`current_price ≤ 0.001` or `≥ 0.999`)
2. **Stop-loss** — down ≥25% from entry (`loss_pct ≥ stop_loss_threshold`)
3. **Minimum hold gate** — no exit before 10 minutes **except** stop-loss or resolved
4. **Let it ride** — if price ≥70% in your favor (`buy + price ≥ 0.70`, or `sell + price ≤ 0.30`), suppress all exits except stop-loss/resolved. Sends a Slack alert instead.
5. **Aggressive exit** — up ≥30% from entry
6. **Trailing stop** — was up ≥15%, dropped back ≥10% from peak
7. **Take-profit** — edge converged to ≤5% **and** unrealized P&L ≥ $5 minimum

### Portfolio-Level Controls

- `max_open_positions`: 5 concurrent positions
- `daily_loss_limit_usd`: halt if daily P&L ≤ −$75
- `max_daily_trades`: 5 trades per day

### Kill Switch

`bot/supervisor.py` writes `data/kill_switch` file if account value drops below 50% of starting value. The trading loop checks for this file every cycle and stops trading while it exists.

**Pause mechanism**: `data/pause_until` file containing an ISO timestamp. Trading loop skips cycles while paused.

---

## 6. Execution & Portfolio

### `ExecutionEngine` (`bot/execution.py`)

**Paper mode** (`paper_trading: true`): Simulates fills at signal price, applies 2% synthetic fee.

**Live mode**: Uses the `polymarket-us` Python SDK (`pip install polymarket-us`). Submits IOC (Immediate-Or-Cancel) limit orders. A fill of 0 executions = no position created (important: prevents ghost positions).

Close orders: submits a sell at $0.01 to sweep the book. After 3 consecutive failed closes, marks as `auto-settling` (market resolved on exchange, position will settle automatically).

Live order intent strings:
- Buy YES: `ORDER_INTENT_BUY_LONG`
- Sell YES: `ORDER_INTENT_SELL_LONG` (for closing longs or shorting)

### `Portfolio` (`bot/portfolio.py`)

The exchange is the **single source of truth** for balances and positions in live mode. The portfolio class is a thin wrapper that:
- Fetches `buyingPower` from `client.account.balances()` for the cash balance
- Fetches `client.portfolio.positions()` for open positions
- Caches exchange data for 2 seconds to avoid hammering the API within a single cycle
- Tracks bot-opened positions internally by slug (set `self._bot_positions`)
- Writes closed trades to SQLite via `bot.trade_db`
- Sends Slack alerts on open/close events

In paper mode, it maintains an in-process list `_paper_positions` and `_paper_bankroll`.

**P&L**: Exchange P&L is preferred (fetched fresh after close). If unavailable, falls back to local calculation. A warning is logged if they differ by >$0.50.

### `MarketDataClient` (`bot/market_data.py`)

Primary API: `https://gateway.polymarket.us`
Fallback order book: CLOB (`https://clob.polymarket.com/book`)

Key endpoints used:
- `GET /v1/markets` — active market list (with `active=true&closed=false`)
- `GET /v1/markets/{slug}/book` — order book (bids/offers)
- `GET /v1/markets/{slug}/bbo` — best bid/offer for price updates
- `GET https://clob.polymarket.com/prices-history` — price history (fallback, rarely used now)

Rate limiting: 5 req/s max. Uses `urllib3.Retry` with exponential backoff on 429/5xx.

---

## 7. External Data Sources

### the-odds-api.com (`bot/signals/odds_api.py`)

- Paid API (500 req/month free tier). Enabled when `THE_ODDS_API_KEY` is set.
- Falls back to ESPN if API key absent or quota exhausted.
- Supported sports: NBA, NHL, NCAA basketball, EPL, NFL, MLB, MLS, La Liga, Bundesliga, Serie A, Ligue 1.
- Sharp books (Pinnacle) weighted 60% when they diverge ≥3% from soft books.

### FanDuel + Pinnacle (free scrapers) (`bot/signals/book_scrapers.py`)

- `FanDuelClient`: Hits FanDuel's public content-managed-page API. No auth needed.
- `PinnacleClient`: Hits Pinnacle's public guest API (`guest.api.arcadia.pinnacle.com`). No auth needed.
- `MultiBookAggregator`: Combines both. Sharp books weighted 1.5× in consensus.
- Supports NBA, NCAA basketball, NHL.

### ESPN (free) (`bot/signals/odds_api.py`, `bot/signals/sports_data.py`, `bot/game_schedule.py`)

- `_fetch_espn_odds()`: Converts ESPN spread/moneyline to implied probability.
- `ESPNCache`: Game data (scores, records, venue, neutral site) for sports context signal.
- `GameSchedule`: Today's schedule across NBA/NCAA/NHL. Drives sleep-when-no-games logic.

### PredictIt (`bot/signals/cross_market.py`)

- Free public API: `https://www.predictit.org/api/marketdata/all`
- Used for politics/other markets. Fuzzy string match against question text.

### CoinGecko (`bot/signals/crypto_api.py`)

- Free public API: current price + 30-day history.
- Parses questions like "Will Bitcoin hit $100,000 by June 2026?" into a log-normal probability estimate.

### ESPN Live Odds / NCAA In-Game (`bot/signals/live_odds.py`)

- Fetches live NCAA scoreboard every 30 seconds during March Madness.
- Compares ESPN live implied probability (from spread) to Polymarket price.
- Detects "lag" edges: sportsbook line moved on a momentum swing but Polymarket hasn't repriced yet.

---

## 8. Supervisor & Observability

### `Supervisor` (`bot/supervisor.py`)

Read-only agent. **Never modifies config.yaml.** Runs as a separate process via `run.sh`.

Scheduled jobs (APScheduler, US/Eastern):
- **6:00 AM ET**: `daily_review()` — 24h trade stats by market type, edge validation report, kill switch check, Slack post
- **8:00 AM ET**: `morning_briefing()` — top 10 edge opportunities from morning scan, posted to Slack
- **Every 15 min**: `check_kill_switch()` — halts bot if account value < 50% of starting value

Kill switch: creates `data/kill_switch` file. Remove the file to resume trading.

### Slack Alerts (`bot/alerts.py`)

Color-coded attachments:
- 🟢 Green: trade closed (profit)
- 🔴 Red: trade closed (loss) or error
- 🔵 Blue: trade opened (pre-game)
- 🟠 Orange: trade opened (live game)
- ⚫ Gray: daily summary, startup, supervisor

Requires `SLACK_WEBHOOK_URL` in `.env` and `alerts.enabled: true` in config.

### Edge Log (`bot/edge_log.py`)

SQLite tables in `data/trades.db`:
- `edge_log`: Full signal snapshot for every trade opened (consensus price, books used, signal values, edge pattern, CLV data, outcome)
- `line_movement`: Time-series of consensus vs. Polymarket price for each slug

**CLV tracking** (Closing Line Value): When a game tips off, the bot records the closing line consensus probability. CLV = `entry_price - closing_line_consensus`. Positive = bought cheaper than the closing line (value).

**Edge patterns** classified: `late_line_movement`, `mean_reversion`, `liquidity_gap`, `weak_consensus`, `sharp_divergence`.

### `TradingLogger` (`utils/logger.py`)

All log events are structured JSON: `{"timestamp": ..., "level": ..., "event": ..., "data": {...}}`. Writes to both stdout and `reports/trading.log`.

---

## 9. Backtesting

Entry point: `python -m backtest.runner` or `python -m backtest.runner --sweep`

### `BacktestEngine` (`backtest/engine.py`)

Replays a list of `List[List[MarketSnapshot]]` (markets × time) through the full signal → sizing → risk pipeline. Applies slippage and fees at execution time.

**Slippage models**:
- `fixed`: configurable bps (default 50 bps)
- `depth_based`: `multiplier / (total_depth / 1000)`, capped at 5%

**Fees**: Taker 200 bps (2%), Maker 0 bps.

Benchmark comparison: default 62% win rate / 366 trades.

### Parameter Sweep (`backtest/sweep.py`)

Grid over:
- Edge thresholds: 3%, 5%, 7%, 10%, 15%
- Sizing methods: Kelly (4 fractions), Fixed Fractional (4 fractions)
- Stop-loss levels: 5%, 10%, 15%, 20%, 30%

Total combinations: 5 × 5 × (4 + 4) = 200. Results sorted by Sharpe ratio.

### Data (`data/loader.py`)

No real historical data is included in the repo. The runner falls back to `generate_synthetic_markets()` which uses geometric Brownian motion (GBM) to generate price series.

Real historical data would go in `./data/historical/` as JSON files with the schema expected by `load_historical_data()`.

---

## 10. Configuration

### Required Environment Variables (`.env`)

| Variable                    | Required? | Description                                    |
|-----------------------------|-----------|------------------------------------------------|
| `POLYMARKET_KEY_ID`         | Live only | Polymarket US API key ID                       |
| `POLYMARKET_SECRET_KEY`     | Live only | Polymarket US API secret key                   |
| `THE_ODDS_API_KEY`          | Yes (strongly) | the-odds-api.com key. Without it, sports markets won't trade (ESPN fallback only gets 1 book). |
| `SLACK_WEBHOOK_URL`         | Optional  | Slack incoming webhook for alerts              |
| `POLYMARKET_PRIVATE_KEY`    | Unused    | Legacy field, not read by any active code      |

Get Polymarket US API credentials at `polymarket.us/developer`.

### `configs/config.yaml` — Key Parameters

```yaml
trading:
  paper_trading: true              # true = simulate, false = real money
  min_edge_threshold: 0.05         # 5% min edge to trade
  max_edge_threshold: 0.40         # 40% max (sanity cap)
  position_sizing_method: tiered_kelly
  max_position_size_usd: 50.0      # Hard cap per trade
  min_position_size_usd: 5.0       # Skip if too small
  max_portfolio_exposure_usd: 500.0
  stop_loss_threshold: 0.25        # Close at 25% loss
  take_profit_threshold: 0.05      # Close when edge ≤ 5%
  minimum_take_profit_usd: 5.0     # Don't take profit below $5
  daily_loss_limit_usd: 75.0       # Halt after $75 daily loss
  max_open_positions: 5
  max_daily_trades: 5
  aggressive_exit_pct: 0.30        # Close if up 30%+
  trailing_stop_activation_pct: 0.15
  trailing_stop_pct: 0.10

filters:
  min_daily_volume_usd: 500.0
  min_liquidity_usd: 200.0
  min_hours_to_expiry: 1
  max_hours_to_expiry: 48          # Note: this filter is set but effectively
                                   # overridden by per-category logic in market_data.py
```

### Config Loading Priority

`utils/config.py` loads in this order:
1. `configs/config.yaml` (primary)
2. `config.yaml` (root fallback)
3. `.env` file (overlays API keys and webhook URL)

`SignalConfig` weights in `config.yaml` are loaded but the estimator uses its own `WEIGHTS` dict (hardcoded in `estimator.py`) — the config weights are not currently used by the live estimator.

---

## 11. Database Schema

**`data/trades.db`** — SQLite, WAL mode.

```sql
trades           -- Closed positions (P&L source for stats)
  slug, market_id, side, entry_price, close_price, quantity, size_usd,
  realized_pnl, close_reason, market_type, entry_time, close_time

daily_summaries  -- One row per calendar day
  date, total_trades, wins, losses, total_pnl, win_rate,
  avg_win, avg_loss, profit_factor, bankroll,
  moneyline_trades, moneyline_win_rate, spread_trades, spread_win_rate,
  totals_trades, totals_win_rate

signal_log       -- Latest signal values per slug (PRIMARY KEY slug, signal_name)
  slug, market_type, timestamp, signal_name, value, confidence,
  direction, metadata_json, edge, polymarket_price

parameter_changes -- Audit log of manual config changes
  timestamp, parameter, old_value, new_value, reason

edge_log         -- Full signal snapshot for every trade opened
  slug, timestamp, market_type, league, polymarket_price, consensus_price,
  books_used, num_books, edge_at_entry, signal_snapshot (JSON), edge_pattern,
  final_outcome, actual_pnl, time_held_seconds, price_at_close, close_reason,
  is_live_game, entry_time, close_time, closing_line_value, closing_line_price,
  resolution_flag

line_movement    -- Time-series of consensus vs. Polymarket price
  slug, timestamp, consensus_prob, polymarket_price, num_books,
  sharp_consensus, overall_consensus
```

---

## 12. Current State: What Works, What's Broken, Rough Edges

### ✅ Working

- **Live trading loop** — `TradingBot.run()` is the production path. Dual-speed scanning (3s live / 60s full), sniper trade selection, full validation checklist, position lifecycle.
- **Sports signal pipeline** — `odds_value` signal + ESPN/FanDuel/Pinnacle fallback is the practical engine for all sports trades. Works without `THE_ODDS_API_KEY` (uses 1 free book), but you need ≥2 books to pass `validate_trade()`.
- **Paper trading mode** — fully functional, no API keys needed.
- **Tiered Kelly sizing** — simple, predictable, works well.
- **Risk management** — stop-loss, let-it-ride, trailing stop, take-profit all active.
- **Supervisor** — daily review, morning briefing, kill switch all operational.
- **Slack alerts** — all alert types send correctly.
- **SQLite persistence** — trades, edge logs, signal log all written correctly.
- **Backtest engine** — runs on synthetic data (no historical data included).

### ❌ Broken / Not Working

**Tests (`tests/test_core.py`) are completely broken** — they were written against an older version of the codebase and import modules/functions that no longer exist:
- Imports `bot.signals.unusual_whales` — this module doesn't exist (was removed/renamed)
- Imports `price_momentum_signal`, `volume_signal`, `mean_reversion_signal`, `volatility_signal` from `bot.signals.signals` — these functions don't exist in the current `signals.py`
- Calls `estimator.compute_signals(snapshot)` without the now-required `market_type` argument
- References `config.signals.price_momentum_weight`, `volume_signal_weight`, etc. — these fields don't exist in the current `SignalConfig`
- Tests `assert abs(total - 1.0) < 0.01` on signal weights that sum to ~0.85 in current config

Running `pytest` will fail with import errors before any tests execute.

### ⚠️ Rough Edges / TODOs

1. **`onchain.py` is dead code** — The `OnChainEnrichmentClient` at the root level was designed to provide whale/smart-money/sentiment signals from free Polymarket APIs. It's not imported or called anywhere in the live signal pipeline or the active estimator. It would replace the `unusual_whales` module (which the tests still reference) but hasn't been wired in.

2. **`config.yaml` at root** — The loader searches for both `configs/config.yaml` and a root `config.yaml` as a fallback. If both exist with different values, behavior depends on which is found first. Only `configs/config.yaml` should be edited.

3. **Signal config weights are unused** — `configs/config.yaml` has a `signals:` section with `odds_value_weight: 0.4`, etc. These values are loaded into `SignalConfig` but the active `ProbabilityEstimator` uses its own hardcoded `WEIGHTS` dict in `estimator.py`. Changes to config signal weights have no effect.

4. **Backtest `Portfolio` API mismatch** — `backtest/engine.py` calls `Portfolio(initial_bankroll_usd)` with a positional float argument, and references `portfolio.positions`, `portfolio.initial_bankroll`, `portfolio.trades` — attributes that don't exist in the current exchange-backed `Portfolio` class. The backtest would fail on a real run with the current code.

5. **No Playwright / browser automation** — The task prompt mentions "Playwright automation" and "login flow." This does not exist anywhere in the codebase. All live trading goes through the `polymarket-us` Python SDK with API key + secret authentication. There is no browser, no Selenium, no Playwright.

6. **`THE_ODDS_API_KEY` is effectively required** — Without it, `OddsCache` falls back to ESPN only (1 book). The `validate_trade()` checklist requires ≥2 books, so all sports trades will be rejected. The bot will run but won't place any trades in live mode without this key.

7. **`data/historical/` is empty** — The backtest always falls back to synthetic GBM data. To backtest against real data, you'd need to populate `./data/historical/` with JSON files in the loader's expected format.

8. **`max_hours_to_expiry: 48` filter** — Set in config but the actual filtering in `market_data.py` uses different windows per category (sports: 24h, non-sports: 14 days), effectively ignoring this config value.

9. **`bot/test_signals.py`** — File exists but wasn't examined; likely a manual scratch/debug script, not a proper test.

10. **Paper mode portfolio is a class-level list** — `Portfolio._paper_positions` is defined as a class-level attribute (`_paper_positions: List[Position] = []`), not an instance-level one. If you create multiple Portfolio instances in the same process (e.g., in tests), they share the same position list. This is a subtle but real bug.

---

## Running the Bot

```bash
# Install
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install polymarket-us   # not in requirements.txt — must install separately

# Configure
cp .env.example .env
# Edit .env with your credentials

# Paper trading (no credentials needed)
python -m bot.trading_loop

# Live trading + supervisor (background)
bash run.sh

# Backtest
python -m backtest.runner

# Parameter sweep
python -m backtest.runner --sweep
```

Logs: `reports/trading.log` (bot), `reports/supervisor.log` (supervisor), `reports/live_output.log` (nohup stdout).
