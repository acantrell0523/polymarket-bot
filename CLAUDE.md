# CLAUDE.md — Polymarket Trading Bot Codebase Guide

This document is a comprehensive reference for AI assistants (and human developers) working in this codebase. It covers architecture, key subsystems, configuration, and the current state of the code.

---

## Latest Changes — 2026-07-11 PM (external audit: blockers 1-5 fixed)

An external audit (GPT session) flagged 4 critical execution blockers + a
signal-pooling flaw. All 5 verified against the code and live APIs, then
fixed. paper_trading remains TRUE — live trading stays blocked until the
remaining audit items (position-state persistence, ESPN clock parsing,
backtest rebuild, CI) land.

| # | Blocker | Fix |
|---|---------|-----|
| 1 | **Market discovery saw nothing** — single limit=500 request; measured live: 6,000+ active markets, WNBA games at offset ~2,500, MLB ~3,500 | `get_active_markets()` paginates limit+offset until a short page (20k cap); first-page failure → [] (degraded mode), later-page failure keeps partial results |
| 2 | **Fees modeled flat 2%** — US schedule (eff. 2026-07-01, verified at docs.polymarket.us/fees) is `fee = Θ·C·p·(1−p)`, taker Θ=0.06, maker −0.0125 | New `bot/strategies/fees.py` single source (banker's-rounded bookings); net-edge, Kelly, paper fills, live fills, backtest all use `trading.taker_fee_coefficient`. Flat model overcharged favorites ~3× and undercharged mid-priced markets |
| 3 | **Unsafe fill reconciliation** — submitted the MID as IOC limit; `int()` truncated fractional fills; unknown execution fields summed to 0 then FELL BACK TO FULL QUANTITY (phantom fills); no automated-order flag | Submits at `signal.exec_price` (refuses if missing); `automaticOrder: true`; `reconcile_executions()` parses float quantities across field-name variants, VWAPs fill price, zero-parseable ⇒ NO trade; fees booked on actual fills |
| 4 | **False close success** — API errors read as "position absent"; 3 unfilled IOCs became "auto-settling"; shorts closed with BUY_SHORT @ $0.01 (can never fill) | `get_exchange_positions()` returns None on failure vs {} when empty; close success requires post-close verification; auto-settle requires the market-status endpoint to confirm resolution; shorts close BUY_LONG @ $0.99 |
| 5 | **Aux signals could reverse the books** — 0.5-anchored absolute values dragged the blend across the price (BUY at 0.31 when market 0.25, consensus 0.20); the ext+1% cap clamps magnitude, not sign | `_direction_reversed()` guard in `detect_edge()`: the primary external signal defines the only permitted trade direction; aux signals may temper it, never flip it. Integration test reproduces the audit's exact scenario |

| 6 | **Position state reset every scan/restart** — live positions rebuilt with entry_time=now, estimated_prob=0.5, peak/telemetry zeroed: the 10-min hold gate never elapsed (only stop-loss could ever fire live), take-profit/trailing/let-it-ride ran on garbage | New `live_position_state` table (slug PK); persisted at open, overlaid on reconstruction (exchange stays authoritative for qty/cost), deleted at close; paper positions restore on restart (`restore_state=True`); `tests/conftest.py` autouse fixture isolates all tests onto throwaway DBs |

**Multi-book consensus (same session):** `book_scrapers.py` extended to summer
sports — Pinnacle league ids 246/578/2663 (MLB/WNBA/MLS, discovered live),
FanDuel mlb/wnba pages (no MLS page exists; DK direct API is bot-blocked, DK
comes via ESPN pickcenter). New league-scoped nickname fragments fix flat-dict
collisions ("Las Vegas Aces" matched NHL vgk:"vegas"; "Chicago Cubs" matched
NBA chi). The recorder blends FanDuel+Pinnacle into the DK line for games the
books still list (pre-game = point-in-time, zero lookahead) weighted by book
count, with a NO-DOWNGRADE guard so post-game reruns (1-book) never overwrite
richer pre-game consensus. **Verified live: upcoming games record
num_books=3-5; the ">7% edge needs 3 books" rule is now unlockable in
backtests.**

**Live in-game edge engine (same session, late PM):** the flagship feature.
New `bot/signals/live_win_prob.py` — `LiveWinProbCache` sweeps registered
leagues' scoreboards for in-progress games (TTL 20s) and reads ESPN's
per-play `winprobability` model from the game summary (verified live against
TOR@SD in the 8th: model 16.6% vs Polymarket 17.7c). New `live_win_prob`
signal (freshness-gated: conf 0.90 under 30s old, decays to 0 at 150s —
stale model output gets NO vote) becomes the PRIMARY external signal for
live sports snapshots, replacing the stale pregame line; the external gate
and direction guard operate on it. Weight 0.55 in-game, strict no-op
pregame. Wired into both estimators via `live_cache`.

**Audit #7 FIXED (same session):** `displayClock`/`period` were read off
`status["type"]` where they don't exist (verified against a live payload) —
always defaulting to full-game time remaining, defeating the last-5-minutes
gate. Now read from `event["status"]`, and a live clocked game with an
unreadable clock FAILS CLOSED (0s remaining = entries blocked).

**Book dedup (same session):** FanDuel sometimes double-lists a game's
moneyline; the aggregator now keeps one entry per book, so num_books counts
DISTINCT books (a 3-book game had reported 5).

**ABBR_MAP** moved to `bot/leagues.py` (canonical; scripts import from there;
`normalize_abbr()` helper).

Tests: 252 passing. Remaining audit items: aux signals as deltas around the
prior (full fix behind the direction guard), point-in-time backtest rebuild,
CI + Ruff cleanup (83 findings).

---

## Changes — 2026-07-11 AM (live verification on real APIs)

First session on a machine with real network access. The recorder, consensus
backfill, and real-data backtest are now all VERIFIED WORKING end-to-end.

| Area | Change |
|------|--------|
| **Recorder verified live** | `python scripts/ingest_historical.py` (no args) now matches **71/71 games** across MLB/WNBA for the last 3 days, with **219/221 snapshots (99%) carrying consensus**, 0 errors. Off-season leagues correctly return zero games. |
| **Slug discovery (real conventions)** | Gamma game markets use BARE slugs — `{league}-{away}-{home}-{date}` (e.g. `mlb-mil-pit-2026-07-10`), NOT the `aec-` prefixed family (that's polymarket.us). Spread/total markets are separate `-spread-*`/`-total-*` slugs. `candidate_slugs()` tries bare first, prefixed as fallback. `detect_market_type()`/`get_league_from_slug()` now recognize bare league prefixes via `bot.leagues.league_from_slug()`. |
| **Closed-market lookup fix** | Gamma `/markets?slug=` EXCLUDES closed markets by default — resolved games (2+ days old) returned 0 rows and silently missed. `find_game_market()` retries each candidate with `closed=true`. |
| **Verified abbreviation maps** | MLB: ESPN `chw`→PM `cws`, `ath`→`oak`. WNBA: `gs`→`gsv`, `con`→`conn`, `lv`→`las`, `ny`→`nyl`. All confirmed against real Polymarket slugs. |
| **token0_side (exact YES-team)** | Game markets aren't Yes/No — `outcomes` is `["Away Team", "Home Team"]` and token 0 prices `outcomes[0]`. New `historical_markets.token0_side` column (migration-guarded) records home/away by matching outcomes[0] to ESPN names at ingest; the consensus backfill uses it before falling back to inference. All 70 recorded markets: `token0_side=away`. |
| **combination_method default reverted to `linear`** | Real-data replay exposed a calibration regression: log-odds pooling shrinks combined edge by overall confidence, and with the odds_value confidence formula (`min(books/5)·min(edge·5)`) a 5-7% single-book edge collapses below every threshold — live trading would have been silently disabled too. All edge thresholds/benchmarks were tuned on linear pooling, so linear is the default again; `logodds` remains available but EXPERIMENTAL (needs threshold recalibration). |
| **Backtest runs on real data** | `python -m backtest.runner` out of the box: 70 real markets → 4 trades through the full pipeline (real prices + real DraftKings consensus + Kelly + risk exits). Two config fixes made this work: `backtest.initial_bankroll_usd` 200→1000 (quarter-Kelly at 5% edge sizes ~2% of bankroll; below ~$600 everything lands under the $5 min position), and new backtest-scoped `backtest.min_price_history_length: 1` (engine previously hardcoded 10, which skipped every real daily-candle snapshot). |
| **Known data limitation** | ESPN pickcenter carries ONE provider (DraftKings) per game → `num_books` clamps to 2 → the ">7% edge needs 3 books" safety rule blocks all large edges in replays. Highest-value next step: when `THE_ODDS_API_KEY` is set, have the recorder also snapshot point-in-time multi-book consensus for today's games (true num_books, zero lookahead). |
| **Tests** | 182 passing. |

---

## Changes — 2026-07-05 (production-hardening audit)

A full audit + upgrade pass landed. Key changes to know before reading the rest
of this document (sections below have been updated in place):

| Area | Change |
|------|--------|
| **Net-edge accounting** | New `bot/strategies/edge.py`. Edge is now computed at the EXECUTABLE price (best ask for buys / best bid for sells) net of the taker fee. `TradeSignal` carries `net_edge/exec_price/spread/fee_rate`; the trade filter gates on `trading.min_net_edge` (default 2%) and rejects books wider than `trading.max_spread`. |
| **Bayesian pooling** | Log-odds pooling anchored on the market price added as `signals.combination_method: logodds`. *(2026-07-11: default reverted to `linear` — see Latest Changes; logodds is experimental pending threshold recalibration.)* |
| **Fee-aware Kelly** | Default sizing is fractional Kelly (`position_sizing_method: kelly`, `kelly_fraction: 0.25`) computed on the executable price grossed up by the taker fee. Tiered flat-dollar sizing remains available. |
| **On-chain signal wired** | `OnChainEnrichmentClient` is now instantiated in the trading loop (when `onchain.enabled`) and feeds a new supporting `onchain_flow` signal (weight 0.10, confidence capped at 0.5) in all market types. |
| **Backtest 0-trades gap CLOSED** | New `backtest/historical_odds.py` — a time-aware, lookahead-free `HistoricalOddsCache` satisfies the external validation gate in replays. Synthetic backtests generate synthetic consensus and now exercise the full pipeline (gate → ranking → sizing → risk). SQLite-backed backtests read `historical_snapshots.espn_consensus_prob`, populated by the recorder's consensus phase as data accumulates. |
| **Decision audit trail** | New `decision_log` table: EVERY opportunity that reaches validation is recorded (executed / rejected / skipped_sizing / execution_failed) with estimated_prob, prices, gross+net edge, spread, fee, size, and reason. |
| **Health & degraded mode** | New `bot/health.py`. Trading loop writes `data/heartbeat.json` every cycle; supervisor alerts when it goes stale (5-min check). Repeated market-data failures trigger exponential backoff (max 5 min) with one Slack alert per outage. |
| **Durable daily-loss halt** | Breaching `daily_loss_limit_usd` now writes `data/pause_until` (next UTC midnight) so the halt survives restarts, plus a Slack alert. |
| **Position re-estimation** | On every full scan, open positions' `estimated_prob` is refreshed from current signals (`estimate_for_snapshot`), so take-profit tracks current beliefs, not entry-time ones. |
| **Env-var config overrides** | Any scalar config value: `POLYBOT_<SECTION>__<FIELD>` (e.g. `POLYBOT_TRADING__PAPER_TRADING=false`). Applied after YAML. |
| **Containerized** | `Dockerfile` + `docker-compose.yml` (bot + supervisor services, shared data volume, heartbeat-based healthcheck). |
| **Cleanup** | Dead `bot/test_signals.py` deleted; dead subgraph URL removed from config; `apscheduler`/`python-dateutil` added to requirements (supervisor previously crashed on missing dep); expiry windows now config-driven (`filters.sports_window_hours` / `nonsports_window_days`); paper/live fee simulation reads `trading.taker_fee_rate`. |
| **Current-sports focus** | New `bot/leagues.py` central registry (MLB, WNBA, MLS + NBA/NHL/NFL/CBB/EPL). `GameSchedule` previously watched only NBA/NCAA/NHL — all off-season in July — so the live bot would have slept all summer; it now sweeps every registered league (off-season leagues return zero games, auto-reactivate when seasons start). Ingest is a forward-looking daily recorder: no-arg run sweeps the last 3 days → today across all leagues; NBA outrights are opt-in (`--include-nba-outrights`); `--prune-leagues nba` deletes finished-season data. Supervisor runs the recorder daily at 05:30 ET. Per-league clock math (incl. clockless MLB, count-up soccer clocks) drives the last-5-minutes block. |
| **Consensus backfill** | `scripts/ingest_historical.py` Phase 3 (+ standalone `--consensus-only` mode): ESPN pickcenter odds → de-vigged consensus → `historical_snapshots.espn_consensus_prob`, with YES-team inference and a closing-line lookahead guard (`--consensus-window-days`, default 3). See §10. |
| **Tests** | 181 tests passing (was 107): net-edge math, log-odds pooling, fee-aware Kelly, spread/net-edge filters, onchain signal, env overrides, health monitor, decision log, historical odds cache, backtest-fires-trades integration, consensus backfill (de-vig math, YES-team inference, window guard, end-to-end into HistoricalOddsCache). |

---

## Changes — 2026-05-04

Ten commits landed that day; retained for history.

| Commit | Summary |
|--------|---------|
| `f91e418` | **Exit proximity telemetry** — `compute_exit_proximity()` added to `bot/trading_loop.py`; `exit_proximity_json` column added to `exit_log`; `scripts/analyze_exits.py` query 7 consumes it |
| `7d92b20` | **Regression suite** — 14 new tests locking in live-loop `let_it_ride` safety and exit proximity correctness |
| `38f103d` | **SQLite-aware backtest runner** — `backtest/runner.py` now tries SQLite first; `data/loader.py` gained `load_historical_data_from_db()`; `backtest/portfolio.py` gained `BacktestPortfolio` |
| `40342ba` | **Historical data ingestion** — `data/historical_db.py`, `scripts/ingest_historical.py`, `scripts/inspect_historical.py`; two-phase NBA ingest (ESPN game sweep + outright futures) |
| `fe5e610` | **Exit telemetry** — `exit_log` table created in `data/trades.db`; `Position` gained `max_favorable_pnl_usd`, `max_adverse_pnl_usd`, `let_it_ride_count`; `Portfolio.close_position()` now accepts `exit_proximity` kwarg |
| `f625210` | **Housekeeping** — `onchain.py` moved from repo root → `bot/signals/onchain.py`; root `config.yaml` deleted; `Portfolio._paper_positions` fixed to instance-level (not class-level) |
| `ae03c5c` | **Config-driven weights** — `ProbabilityEstimator.detect_edge()` now overlays `config.signals.weights` on top of hardcoded defaults; config changes take effect at runtime |
| `84bb7d7` | **BacktestPortfolio** — separated from live `Portfolio`; fixed `let_it_ride` bug in backtest engine; fixed bankroll accounting |
| `3bed2e3` | **Gitignore** — `data/trades.db`, `data/kill_switch`, etc. excluded from version control |
| `785514d` | **Test suite rewrite** — old broken tests replaced; suite passes 100/100 |

**Net result for future sessions**: tests pass, backtest runs (but fires 0 trades — see §12), exit telemetry is live, historical NBA data is ingested, signal weights are configurable.

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
10. [Historical Data Ingestion](#10-historical-data-ingestion)
11. [Configuration](#11-configuration)
12. [Database Schema](#12-database-schema)
13. [Current State: What Works, What's Broken, Rough Edges](#13-current-state-what-works-whats-broken-rough-edges)

---

## 1. Overall Structure

```
polymarket-bot/
├── bot/                        # Live trading core
│   ├── trading_loop.py         # Main orchestrator (TradingBot + compute_exit_proximity)
│   ├── market_data.py          # Polymarket US API + CLOB API HTTP client
│   ├── execution.py            # Order placement via polymarket-us SDK
│   ├── portfolio.py            # Exchange-backed position tracker + exit telemetry write
│   ├── alerts.py               # Slack webhook alerts
│   ├── edge_log.py             # SQLite edge analytics + CLV tracking
│   ├── trade_db.py             # SQLite trade history, signal log, exit_log
│   ├── supervisor.py           # Read-only scheduled agent (reports, kill switch)
│   ├── game_schedule.py        # ESPN schedule fetcher; drives scan sleep logic
│   ├── leagues.py              # Central league registry (current sports; ESPN paths,
│   │                           #   odds-api keys, per-league game clock math)
│   ├── health.py               # Heartbeat + API failure tracking / degraded mode
│   ├── signals/
│   │   ├── signals.py          # 8 signal functions (OB imbalance, line movement,
│   │   │                       #   odds value, liquidity imbalance, cross-market,
│   │   │                       #   sports context, crypto model, onchain flow)
│   │   ├── estimator.py        # ProbabilityEstimator: routes by market type,
│   │   │                       #   external validation gate, config-driven weights
│   │   ├── onchain.py          # OnChainEnrichmentClient (moved from repo root;
│   │   │                       #   not yet wired into live pipeline — see §13)
│   │   ├── odds_api.py         # the-odds-api.com client + ESPN fallback
│   │   ├── book_scrapers.py    # FanDuel + Pinnacle free-API clients
│   │   ├── cross_market.py     # PredictIt API client (politics/other markets)
│   │   ├── crypto_api.py       # CoinGecko + log-normal crypto model
│   │   ├── sports_data.py      # ESPN game context (home advantage, fatigue)
│   │   └── live_odds.py        # Live in-game NCAA edge tracker
│   └── strategies/
│       ├── edge.py             # Cost-aware net-edge accounting (fees + spread)
│       ├── sizing.py           # Fee-aware Kelly / tiered-Kelly / fixed-fractional sizing
│       ├── risk.py             # Stop-loss, trailing stop, take-profit, let-it-ride
│       └── trade_filter.py     # Pre-trade validation checklist (validate_trade)
├── backtest/
│   ├── engine.py               # Replay engine with slippage/fee simulation
│   ├── historical_odds.py      # Time-aware HistoricalOddsCache (no lookahead)
│   ├── portfolio.py            # BacktestPortfolio — pure in-memory, no SQLite side effects
│   ├── sweep.py                # Parameter sensitivity sweep
│   ├── reporting.py            # Matplotlib charts + JSON/CSV exports
│   └── runner.py               # CLI entry: SQLite → JSON → synthetic GBM priority order
├── data/
│   ├── historical_db.py        # historical_markets + historical_snapshots schema & helpers
│   ├── loader.py               # JSON loader + GBM synthetic generator +
│   │                           #   load_historical_data_from_db() (SQLite path)
│   ├── trades.db               # SQLite (created at runtime; not in version control)
│   └── settled_slugs.txt       # Persisted set of auto-settled slugs
├── scripts/
│   ├── ingest_historical.py    # Two-phase NBA data ingest (ESPN + outright futures)
│   ├── inspect_historical.py   # Prints DB summary: markets, snapshots, outcomes
│   └── analyze_exits.py        # 7 canned queries against exit_log for post-hoc analysis
├── utils/
│   ├── config.py               # YAML + .env loader; typed dataclass config
│   ├── logger.py               # Structured JSON logger
│   └── models.py               # Shared dataclasses (OrderBook, TradeSignal, Position…)
├── tests/
│   └── test_core.py            # pytest suite — 182 tests, all passing
├── configs/config.yaml         # CANONICAL master configuration file (only config file)
├── .env.example                # Environment variable template
├── requirements.txt            # Python dependencies
└── run.sh                      # Launch script (starts trading_loop + supervisor)
```

> **Note**: the root `config.yaml` that existed this morning has been **deleted**. `configs/config.yaml` is now the only config file. The loader fallback path still references it but will never find it.

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
  │       └── check_positions()
  │             ├── RiskManager.check_position()  ← per-position risk checks
  │             ├── compute_exit_proximity()       ← signed distance to all 5 thresholds
  │             └── Portfolio.close_position()     ← writes trades + exit_log to SQLite
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

| Type      | Detection                                                        |
|-----------|------------------------------------------------------------------|
| `sports`  | Slug starts with `aec-`, `asc-`, `tsc-`, or `atc-`            |
| `sports`  | Slug contains `-nba-` anywhere (catches outright/futures slugs) |
| `crypto`  | Question/slug contains BTC, ETH, SOL, DOGE, XRP, etc.          |
| `politics`| Question contains election/president/congress/senate/etc.        |
| `other`   | Everything else                                                  |

> **New today**: the `-nba-` slug rule was added so NBA outright/futures markets (e.g. `"will-the-celtics-win-the-2026-nba-finals"`) are classified as `sports` and routed through the odds_value signal rather than cross_market. This is required now that game-level NBA markets don't exist for the 2026 playoff window — only outrights do.

### Signal Weights by Market Type

Weights are now **config-driven**: `detect_edge()` starts from the hardcoded `WEIGHTS` dict, then overlays any values present in `config.signals.weights` (from `configs/config.yaml`). If a key is absent from the config the hardcoded default remains, so deployments without the new config key continue working unchanged.

| Signal              | Sports | Crypto | Politics | Other |
|---------------------|--------|--------|----------|-------|
| `odds_value`        | 40%    | —      | —        | —     |
| `sports_context`    | 15%    | —      | —        | —     |
| `line_movement`     | 20%    | —      | 15%      | 20%   |
| `order_book_imbal.` | 15%    | 20%    | 25%      | 25%   |
| `liquidity_imbal.`  | 10%    | 10%    | 15%      | 15%   |
| `crypto_model`      | —      | 45%    | —        | —     |
| `cross_market`      | —      | 25%    | 45%      | 40%   |

To change weights, edit the `signals.weights` block in `configs/config.yaml` (see §11). Changes take effect on the next bot startup — no code edit needed.

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

Selected by `signals.combination_method` (default `linear` — all edge
thresholds and benchmarks were calibrated against the linear pool; see the
2026-07-11 Latest Changes for why logodds is not the default):

**Log-odds Bayesian pooling (experimental)** — market price is the prior; each
signal shifts the posterior in log-odds space proportional to its
confidence-scaled weight:

```
posterior = logit(price) + sum_i k_i * (logit(v_i) - logit(price))
k_i = (w_i * c_i) / sum_j(w_j)                    # sum(k_i) <= 1
estimated_prob = sigmoid(posterior)               # clamped [0.01, 0.99]
```

Key property: with no signal evidence the estimate IS the market price (zero
edge) — the math itself is humble, not just the gates.

**Linear pooling (default, `combination_method: linear`)**:

```
estimated_prob = sum(w_i * c_i * v_i) / sum(w_i * c_i)
```

### Edge Calculation

```
edge = estimated_prob - snapshot.price            # gross edge (vs mid/last)
side = "buy" if edge > 0 else "sell"
```
Trade fires if `min_edge ≤ |edge| ≤ max_edge`, then must ALSO clear the
cost-aware net-edge gate (`bot/strategies/edge.py`):

```
buy : net_edge = estimated_prob - best_ask * (1 + taker_fee)
sell: net_edge = best_bid * (1 - taker_fee) - estimated_prob
```
requiring `net_edge ≥ trading.min_net_edge` (default 2%) and
`spread ≤ trading.max_spread` (default 0.10).

---

## 4. Kelly / Position Sizing

Implemented in `bot/strategies/sizing.py`. Three sizing methods. **The default
is now fractional Kelly (`kelly`, `kelly_fraction: 0.25`)** — fee- and
spread-aware, proportional to edge AND bankroll. It uses the executable price
(`signal.exec_price`, set by `compute_edge_breakdown`) grossed up by the taker
fee as the cost basis, so costs shrink the Kelly fraction exactly as they
shrink the real edge. Zero net edge ⇒ zero size.

### Tiered Kelly (`tiered_kelly`) — legacy option

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
4. **Let it ride** — if price ≥70% in your favor (`buy + price ≥ 0.70`, or `sell + price ≤ 0.30`), suppress all exits except stop-loss/resolved. Sends a Slack alert instead. Increments `position.let_it_ride_count` each cycle it holds.
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
- Writes closed trades to SQLite via `bot.trade_db.insert_trade()`
- Writes rich exit telemetry to `exit_log` via `bot.trade_db.insert_exit_log()` (see §Exit Telemetry)
- Sends Slack alerts on open/close events

In paper mode, it maintains an in-process `_paper_positions: List[Position]` and `_paper_bankroll: float` — both **instance-level** attributes (the class-level mutable default bug was fixed today).

**P&L**: Exchange P&L is preferred (fetched fresh after close). If unavailable, falls back to local calculation. A warning is logged if they differ by >$0.50.

### Exit Telemetry (`compute_exit_proximity`)

Every position close now produces two data products:

**1. `exit_log` table row** (see §12 for full schema) — written by `Portfolio.close_position()`. Contains:
- P&L extremes across the position lifetime: `max_favorable_pnl_usd`, `max_adverse_pnl_usd`
- Let-it-ride summary: `let_it_ride_triggered`, `num_let_it_ride_triggers`
- Stop-loss boundary distance at close: `exit_threshold_distance`
- The full proximity blob: `exit_proximity_json`

**2. `exit_proximity_json`** — computed by `compute_exit_proximity()` in `bot/trading_loop.py`. Returns a 5-key dict with signed distances to every exit threshold at the moment of close:

```python
{
    "stop_loss_distance_pct":       loss_pct - stop_loss_threshold,
    "take_profit_edge_distance":    take_profit_threshold - edge_remaining,
    "aggressive_exit_distance_pct": gain_pct - aggressive_exit_pct,
    "trailing_stop_distance_pct":   drop_from_peak - trailing_stop_pct,  # None when unarmed
    "let_it_ride_distance_pct":     favorable_price - LET_IT_RIDE_THRESHOLD,
}
```

**Sign convention**: negative = threshold not yet reached; positive = threshold already passed. Stored as a JSON string in `exit_log.exit_proximity_json`. Queryable with SQLite's `json_extract()`.

**Position tracking fields** (updated each cycle by `TradingBot.check_positions()`):
- `position.max_favorable_pnl_usd` — highest positive unrealized P&L seen ($)
- `position.max_adverse_pnl_usd` — most negative unrealized P&L seen ($, e.g. −6.20)
- `position.let_it_ride_count` — number of cycles let_it_ride was triggered

**BacktestPortfolio does not call `insert_exit_log()`** — the backtest layer is SQLite-free by design. `insert_exit_log()` is only called from the live `Portfolio`.

### `MarketDataClient` (`bot/market_data.py`)

Primary API: `https://gateway.polymarket.us`
Fallback order book: CLOB (`https://clob.polymarket.com/book`)

Key endpoints used:
- `GET /v1/markets` — active market list (with `active=true&closed=false`)
- `GET /v1/markets/{slug}/book` — order book (bids/offers)
- `GET /v1/markets/{slug}/bbo` — best bid/offer for price updates
- `GET https://clob.polymarket.com/prices-history` — price history (used by historical ingest)

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
- Also used by `scripts/ingest_historical.py` as Phase 1 data source for game-level market discovery.

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

### Polymarket CLOB + Gamma APIs (historical ingest only)

- `CLOB /prices-history` — daily price candles per token; used by `scripts/ingest_historical.py` to populate `historical_snapshots`.
- `Gamma /markets?slug=…` — market metadata lookup for game-level markets.
- `Gamma /events?slug=…` — all markets under a named event slug (NBA outrights).

---

## 8. Supervisor & Observability

### `Supervisor` (`bot/supervisor.py`)

Read-only agent. **Never modifies config.yaml.** Runs as a separate process via `run.sh`.

Scheduled jobs (APScheduler, US/Eastern):
- **5:30 AM ET**: `daily_data_recorder()` — records yesterday+today's games (prices + consensus) across all registered leagues into the historical tables
- **6:00 AM ET**: `daily_review()` — 24h trade stats by market type, edge validation report, kill switch check, Slack post
- **8:00 AM ET**: `morning_briefing()` — top 10 edge opportunities from morning scan, posted to Slack
- **Every 15 min**: `check_kill_switch()` — halts bot if account value < 50% of starting value
- **Every 5 min**: `check_heartbeat()` — Slack alert when the trading loop's heartbeat goes stale

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

### Exit Analysis Script (`scripts/analyze_exits.py`)

Seven canned SQL queries against `exit_log`:
1. Close reason distribution (count, win rate, avg P&L per reason)
2. Peak unrealized P&L by close reason
3. Let-it-ride win rate (triggered vs not triggered)
4. Stop-loss tightness (distance from SL boundary, was winning first?)
5. Average hold time by close reason
6. P&L by entry_estimated_prob bucket
7. **Stop-loss tightness diagnostic** — reads `exit_proximity_json` to show how close `take_profit` was when `stop_loss` fired (uses `json_extract()`)

```bash
python scripts/analyze_exits.py --days 30
```

### `TradingLogger` (`utils/logger.py`)

All log events are structured JSON: `{"timestamp": ..., "level": ..., "event": ..., "data": {...}}`. Writes to both stdout and `reports/trading.log`.

---

## 9. Backtesting

Entry point: `python -m backtest.runner` or `python -m backtest.runner --sweep`

### Data Priority Order (`backtest/runner.py`)

The runner tries three data sources in order:

1. **SQLite** (`load_historical_data_from_db()`) — reads `historical_markets` + `historical_snapshots` from `data/trades.db`. Returns `None` if either table is empty, triggering fallback.
2. **JSON files** (`load_historical_data()`) — reads `./data/historical/*.json` files. Returns `None` if directory doesn't exist or is empty.
3. **Synthetic GBM** (`generate_synthetic_markets()`) — always succeeds; 20 markets × 500 snapshots.

When SQLite data is loaded, the runner prints an explanatory note:
```
NOTE: NBA outright markets route to 'sports' type; trades require
      OddsCache to be populated. Zero trades = gate working as designed.
```

### `BacktestPortfolio` (`backtest/portfolio.py`)

A **separate class** from the live `Portfolio`. Pure in-memory — zero exchange calls, zero SQLite writes, zero side effects from the live trading path.

Key differences from live `Portfolio`:
- `bankroll` decreases on `open_position()` (cost + fees deducted immediately)
- `bankroll` recovers on `close_position()` (cost + realized P&L returned)
- No `insert_exit_log()` call on close
- No Slack alerts
- `initial_bankroll`, `positions`, `trades`, `equity_curve`, `timestamps` all available as attributes

The `BacktestEngine` (`backtest/engine.py`) instantiates `BacktestPortfolio`, not `Portfolio`.

### `BacktestEngine` (`backtest/engine.py`)

Replays a list of `List[List[MarketSnapshot]]` (markets × time) through the full signal → sizing → risk pipeline. Applies slippage and fees at execution time.

**Slippage models**:
- `fixed`: configurable bps (default 50 bps)
- `depth_based`: `multiplier / (total_depth / 1000)`, capped at 5%

**Fees**: Taker 200 bps (2%), Maker 0 bps.

Benchmark comparison: default 62% win rate / 366 trades.

**`let_it_ride` is handled correctly** in the backtest engine: the engine checks `if close_reason == "let_it_ride":` explicitly (not a truthiness check) and skips to the next snapshot. This is verified by a regression test in `tests/test_core.py`.

### Why the Backtest Currently Fires 0 Trades

The historical data ingested so far consists entirely of **NBA outright/futures markets** (Finals winner, MVP, Conference champion, etc.). These slugs contain `-nba-` and are classified as `sports` type by `detect_market_type()`. The `sports` signal path requires `odds_value_signal` to return `confidence > 0`, which requires the `OddsCache` to be populated with live API data. The backtest runs without live API calls, so `OddsCache` is empty, the external validation gate blocks every trade, and 0 trades are placed.

**This is correct behavior, not a bug.** The architectural gate is working as designed. The fix is to populate historical consensus probabilities into the `OddsCache` (or a new `HistoricalOddsCache`) before the replay — this is the **next major piece** of work for meaningful backtests.

### Parameter Sweep (`backtest/sweep.py`)

Grid over:
- Edge thresholds: 3%, 5%, 7%, 10%, 15%
- Sizing methods: Kelly (4 fractions), Fixed Fractional (4 fractions)
- Stop-loss levels: 5%, 10%, 15%, 20%, 30%

Total combinations: 5 × 5 × (4 + 4) = 200. Results sorted by Sharpe ratio.

---

## 10. Historical Data Ingestion

### Overview

Two SQLite tables (`historical_markets`, `historical_snapshots`) in `data/trades.db` store market data for backtesting. Data is recorded by `scripts/ingest_historical.py` from free public APIs (ESPN + Polymarket CLOB/Gamma).

**The recorder is strictly focused on CURRENT sports.** With no date arguments it sweeps the last 3 days up to today across every league in `bot/leagues.py` (July: MLB, WNBA, MLS have games; NBA/NHL/NFL return zero until their seasons start, then auto-reactivate). Run it daily — cron, or the supervisor's built-in 05:30 ET `daily_data_recorder` job — and the dataset builds forward from today. Season backfills are still possible by passing explicit `--start/--end`, and `--prune-leagues nba` clears out finished-season data.

### Three-Phase Ingestion (`scripts/ingest_historical.py`)

**Phase 1 — Game-by-game (ESPN scoreboard, every registered league)**:
1. For each date in the window and each league, fetch games from that league's ESPN scoreboard (off-season leagues return none)
2. For each game, construct candidate Polymarket slugs (`aec-{league}-{away}-{home}-{date}`; soccer uses the 3-outcome `atc-…-{team}` family)
3. Look up the market on Gamma API
4. Fetch CLOB `prices-history` for the YES token
5. Filter history to the requested date window
6. Insert into `historical_markets` (market metadata) and `historical_snapshots` (price series)

**Phase 2 — NBA outright/futures (OPT-IN via `--include-nba-outrights`; off by default, season over)**:
1. For each slug in `NBA_OUTRIGHT_EVENT_SLUGS` (Finals winner, MVP, Conference champion, etc.)
2. Fetch all markets under that Gamma event
3. Fetch CLOB `prices-history` for each market's YES token
4. Insert into both tables

**Phase 3 — Consensus backfill (ESPN pickcenter)** — populates `espn_consensus_prob`, which `HistoricalOddsCache.from_db()` serves to the backtest's external validation gate:
1. For every `moneyline_game` market with an `espn_game_id` (including ones ingested on previous runs), fetch the ESPN game **summary** endpoint's `pickcenter` block (free; retained for past games)
2. Per provider: de-vig home/away moneylines into a win probability (normalize so they sum to 1); fall back to the point spread via the same calibrated logistic the live bot uses (ESPN's `spread` is the home spread, negative = favored — sign is flipped)
3. Average across providers → consensus; provider count → `num_books`
4. Infer which team the market's YES token refers to, strongest evidence first: (a) settled winner + decisive final price (≥0.75/≤0.25) — the data itself says which side YES was; (b) first team nickname mentioned in the question; (c) slug convention fallback (`aec-nba-{away}-{home}` → away)
5. `UPDATE` `espn_consensus_prob`/`num_books` on snapshots within `--consensus-window-days` (default 3) before game start through 1 day after. **The pickcenter line is the closing line** — stamping it onto much older snapshots would leak tip-off-time information into the backtest, so earlier snapshots keep 0 (lookahead guard). Outrights have no consensus source and stay gated.

**Re-running is fully idempotent**: `INSERT OR IGNORE` for price phases, plain `UPDATE` for the consensus backfill. Timestamps are bucketed to UTC day boundaries (floor to 86400 s) before insert, so the `UNIQUE(slug, timestamp, source)` constraint catches duplicates even when the CLOB API returns slightly different intra-day timestamps across runs. Verified by regression tests (`TestHistoricalDB.test_ingest_idempotency`, `TestUpdateSnapshotConsensus.test_rerun_is_idempotent`).

```bash
# Daily recorder (default): last 3 days → today, all registered leagues
python scripts/ingest_historical.py

# Specific leagues / window
python scripts/ingest_historical.py --leagues mlb,wnba,mls --days-back 7

# Drop finished-season NBA data
python scripts/ingest_historical.py --prune-leagues nba --leagues mlb,wnba,mls

# Backfill consensus for an already-populated database (no re-ingest):
python scripts/ingest_historical.py --consensus-only

# Options: --consensus-window-days N (default 3), --skip-consensus,
#          --include-nba-outrights, --start/--end for explicit windows
```

### Current State of the Historical Database

As of 2026-05-04, the database contains **NBA outright markets only**. Game-level markets (`aec-nba-*` slugs) for the 2026 NBA playoffs are not available because Polymarket no longer creates game-level markets for this window — only season-long outrights (Finals winner, MVP, Conference champion). The ingestion script Phase 1 will find 0 game matches for 2026 playoff dates.

The database was populated by running the script with a broad date window covering the 2024–25 and 2025–26 seasons.

### Database Schema — Historical Tables

See §12 for the full column list. Key points:
- `historical_markets.market_type` is `"moneyline_game"` for game-level or `"nba_outright"` for futures
- `historical_snapshots.timestamp` is a **UTC day boundary** (Unix epoch, floored to 86400 s)
- `historical_snapshots.polymarket_price` is clamped to [0.01, 0.99] by the loader
- `historical_snapshots.trade_size_usd` is 0 for all CLOB-sourced data (not available from `prices-history`)

### Inspection Tool (`scripts/inspect_historical.py`)

```bash
python scripts/inspect_historical.py
```

Prints:
- Market count by type (moneyline_game vs nba_outright)
- Total snapshot count
- Game markets by date
- Market count per ESPN game
- Snapshot density per market
- Settled outcome distribution
- Sample snapshots for the two highest-density markets

---

## 11. Configuration

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

signals:
  # Scalar weights (legacy — kept for SignalConfig compat; not read by estimator directly)
  odds_value_weight: 0.4
  sports_context_weight: 0.15
  line_movement_weight: 0.2
  order_book_imbalance_weight: 0.15
  liquidity_imbalance_weight: 0.1

  # Per-market-type weight dicts — these ARE read by ProbabilityEstimator.detect_edge().
  # Edit here to tune signal behavior without touching code.
  # Values here overlay the hardcoded WEIGHTS defaults in estimator.py.
  weights:
    sports:
      odds_value: 0.40
      sports_context: 0.15
      line_movement: 0.20
      order_book_imbalance: 0.15
      liquidity_imbalance: 0.10
    crypto:
      crypto_model: 0.45
      cross_market: 0.25
      order_book_imbalance: 0.20
      liquidity_imbalance: 0.10
    politics:
      cross_market: 0.45
      order_book_imbalance: 0.25
      line_movement: 0.15
      liquidity_imbalance: 0.15
    other:
      cross_market: 0.40
      order_book_imbalance: 0.25
      line_movement: 0.20
      liquidity_imbalance: 0.15

filters:
  min_daily_volume_usd: 500.0
  min_liquidity_usd: 200.0
  min_hours_to_expiry: 1
  max_hours_to_expiry: 48          # Note: effectively overridden by per-category logic
                                   # in market_data.py (sports: 24h, non-sports: 14 days)
```

### Config Loading Priority

`utils/config.py` loads in this order:
1. `configs/config.yaml` (primary — **this is the only config file now**)
2. `config.yaml` (root fallback — **file was deleted today; fallback never fires**)
3. `.env` file (overlays API keys and webhook URL)

---

## 12. Database Schema

**`data/trades.db`** — SQLite, WAL mode. Created at runtime by `bot/trade_db.py` on import. Not version-controlled.

```sql
-- Closed positions (P&L source of record for stats)
trades
  slug, market_id, side, entry_price, close_price, quantity, size_usd,
  realized_pnl, close_reason, market_type, entry_time, close_time

-- One row per calendar day
daily_summaries
  date, total_trades, wins, losses, total_pnl, win_rate,
  avg_win, avg_loss, profit_factor, bankroll,
  moneyline_trades, moneyline_win_rate, spread_trades, spread_win_rate,
  totals_trades, totals_win_rate

-- Latest signal values per slug (PRIMARY KEY slug, signal_name)
signal_log
  slug, market_type, timestamp, signal_name, value, confidence,
  direction, metadata_json, edge, polymarket_price

-- Audit log of manual config changes
parameter_changes
  timestamp, parameter, old_value, new_value, reason

-- Full signal snapshot for every trade opened
edge_log
  slug, timestamp, market_type, league, polymarket_price, consensus_price,
  books_used, num_books, edge_at_entry, signal_snapshot (JSON), edge_pattern,
  final_outcome, actual_pnl, time_held_seconds, price_at_close, close_reason,
  is_live_game, entry_time, close_time, closing_line_value, closing_line_price,
  resolution_flag

-- Time-series of consensus vs. Polymarket price
line_movement
  slug, timestamp, consensus_prob, polymarket_price, num_books,
  sharp_consensus, overall_consensus

-- Rich exit telemetry — one row per closed position (NEW today)
exit_log
  id, slug, market_id, side,
  entry_time, close_time, hold_seconds,
  entry_price, close_price, size_usd, quantity, realized_pnl,
  close_reason,
  entry_estimated_prob,
  max_favorable_pnl_usd,   -- highest positive unrealized P&L seen ($)
  max_adverse_pnl_usd,     -- most negative unrealized P&L seen ($)
  peak_unrealized_pct,     -- max_favorable_pnl_usd / size_usd
  let_it_ride_triggered,   -- 1 if let_it_ride fired at least once
  num_let_it_ride_triggers,
  exit_threshold_distance, -- distance from stop-loss boundary at close
  metadata_json,           -- paper_mode, is_live_game, etc.
  exit_proximity_json,     -- signed distance to every exit threshold (5 keys)
  created_at

-- Historical market metadata (NEW today)
historical_markets
  id, slug, market_id, condition_id, league, sport, question,
  home_team, away_team, home_abbr, away_abbr,
  game_start_time, espn_game_id, home_score, away_score,
  settled_outcome,    -- "YES", "NO", "home", "away", or "open"
  market_type,        -- "moneyline_game" or "nba_outright"
  token_id_0, token_id_1,
  ingest_time
  UNIQUE(slug)

-- Decision audit trail — one row per opportunity that reached validation
-- (decision: executed | rejected | skipped_sizing | execution_failed)
decision_log
  id, timestamp, slug, market_type, side,
  polymarket_price, exec_price, estimated_prob,
  gross_edge, net_edge, spread, fee_rate,
  position_size_usd, decision, reason, metadata_json

-- Historical price time-series (NEW today)
historical_snapshots
  id, slug, timestamp,         -- timestamp is UTC day boundary (unix epoch)
  polymarket_price,            -- CLOB daily close price, clamped [0.01, 0.99]
  espn_consensus_prob,         -- de-vigged sportsbook consensus for the YES
                               --   outcome (populated by ingest Phase 3;
                               --   0.0 = no consensus available)
  num_books,                   -- pickcenter provider count (0 = no consensus)
  source,                      -- "clob_prices_history"
  trade_size_usd               -- 0.0 for CLOB-sourced data
  UNIQUE(slug, timestamp, source)
```

**Migration note**: The `exit_proximity_json` column is added via `ALTER TABLE ... ADD COLUMN` with a guard catch, so running `init_db()` on a pre-existing database is always safe.

---

## 13. Current State: What Works, What's Broken, Rough Edges

### ✅ Working

- **Test suite** — 182 tests passing in `tests/test_core.py`. Run `pytest tests/test_core.py`. Covers: order book, signals, market type detection, probability estimator, validate_trade, position sizing, risk manager, portfolio, data generation, backtest engine, config, exit telemetry, live-loop let_it_ride safety, historical DB schema + idempotency, DB loader, exit proximity computation, and the analyze_exits script.

  ```
  ============================= 182 passed in 0.88s ==============================
  ```

- **Live trading loop** — `TradingBot.run()` is the production path. Dual-speed scanning (3s live / 60s full), sniper trade selection, full validation checklist, position lifecycle.

- **Exit telemetry** — `exit_log` table is populated on every position close. `exit_proximity_json` records signed distance to all 5 exit thresholds. `scripts/analyze_exits.py` can query all of it, including the new stop-loss tightness diagnostic.

- **Sports signal pipeline** — `odds_value` signal + ESPN/FanDuel/Pinnacle fallback is the practical engine for all sports trades. Works without `THE_ODDS_API_KEY` (uses 1 free book), but you need ≥2 books to pass `validate_trade()`.

- **Paper trading mode** — fully functional, no API keys needed.

- **Tiered Kelly sizing** — simple, predictable, works well.

- **Risk management** — stop-loss, let-it-ride, trailing stop, take-profit all active and correct. `let_it_ride` handling is verified in both the live loop and the backtest engine by regression tests.

- **Supervisor** — daily review, morning briefing, kill switch all operational.

- **Slack alerts** — all alert types send correctly.

- **SQLite persistence** — trades, edge logs, signal log, exit_log all written correctly.

- **Backtest engine** — runs without crashing. `BacktestPortfolio` correctly tracks bankroll. `let_it_ride` bug (was: bare truthiness check closing winning positions) is fixed.

- **Historical data ingestion** — `scripts/ingest_historical.py` runs end-to-end; idempotency is verified; `historical_markets` and `historical_snapshots` tables are populated with NBA outright data.

- **Signal weights are config-driven** — editing `signals.weights` in `configs/config.yaml` takes effect on next bot start. Config overlay is tested.

- **NBA outright slug classification** — slugs containing `-nba-` now correctly route to `sports` type, not `other`. Prevents these markets from being blocked by the cross_market external validation gate.

- **`onchain.py` in correct location** — moved to `bot/signals/onchain.py`; the root-level dead import issue is resolved.

### ❌ Current Gaps / Not Yet Working

1. **Single-provider consensus limits big-edge replay** — ESPN pickcenter carries only DraftKings for MLB/WNBA games, so `num_books` clamps to 2 and the ">7% edge needs 3 books" rule blocks all large edges in backtests (working as designed, but it caps what replays can evaluate). Next step: when `THE_ODDS_API_KEY` is set, extend the recorder to also snapshot point-in-time multi-book consensus from the live `OddsCache` for today's games — true `num_books`, zero lookahead, and a real closing-line-value dataset. (The recorder itself is now VERIFIED live: 71/71 games matched, 99% consensus coverage, 0 errors — see Latest Changes.)

2. **Backtest data accumulates forward, so it starts thin** — The strategy is strictly current-sports: the recorder builds the dataset from today onward (MLB/WNBA/MLS right now). That means meaningful sports backtests need a few weeks of recorded data before they carry statistical weight. The old 2026 NBA outright data can be dropped with `--prune-leagues nba`. ESPN may carry few pickcenter providers per game (`num_books` is clamped to ≥2 by `HistoricalOddsCache.from_db`, and the "3 books for >7% edges" rule still applies).

3. **`THE_ODDS_API_KEY` is effectively required for live trading** — Without it, `OddsCache` falls back to ESPN only (1 book). The `validate_trade()` checklist requires ≥2 books, so all sports trades will be rejected. The bot will run but won't place any trades in live mode without this key.

4. **`polymarket-us` SDK availability** — Not on all package indexes; the Dockerfile tolerates a failed install (paper mode works without it). Live deployments must confirm the SDK installed.

### ✅ Gaps Closed on 2026-07-05

- **Backtest fires 0 trades** → pipeline gap closed via `backtest/historical_odds.py` (time-aware `HistoricalOddsCache`, no lookahead; synthetic consensus for GBM data). Remaining data-ingestion work tracked as gap #1 above.
- **`onchain.py` not wired** → `OnChainEnrichmentClient` instantiated in `TradingBot.__init__` (gated by `onchain.enabled`), feeds the new `onchain_flow` signal in both estimators.
- **The Graph hosted subgraph dead** → dead `subgraph_url` removed from `configs/config.yaml`; on-chain data comes from free CLOB/Gamma endpoints.
- **`max_hours_to_expiry` ignored** → scan windows are now config-driven: `filters.sports_window_hours` (default 24) and `filters.nonsports_window_days` (default 14).
- **`bot/test_signals.py` scratch file** → deleted.
- **Missing deps** → `apscheduler` and `python-dateutil` added to `requirements.txt` (supervisor previously crashed on a clean install).

### ✅ Bugs Fixed Today (no longer issues)

- **Class-level paper portfolio bug** — `Portfolio._paper_positions` was a class-level mutable default (`[]`) so all instances shared the same list. Fixed: `_paper_positions` is now set in `__init__()` as an instance attribute. Tested by `TestPortfolio.test_independent_paper_positions_and_bankrolls`.
- **Backtest let_it_ride bug** — `backtest/engine.py` was using a bare `if close_reason:` truthiness check, which fires on the non-close string `"let_it_ride"` and incorrectly closed winning positions. Fixed with an explicit `if close_reason == "let_it_ride":` guard. Verified by `TestLiveLoopLetItRideSafety`.
- **BacktestPortfolio API mismatch** — The old backtest was calling `Portfolio(initial_bankroll_usd)` with a positional float, then accessing `.positions`, `.initial_bankroll`, `.trades` — attributes that don't exist in the exchange-backed `Portfolio`. Fixed by creating a separate `BacktestPortfolio` class in `backtest/portfolio.py`.
- **Signal config weights no-op** — Previously, editing `configs/config.yaml` signal weights had no effect because `ProbabilityEstimator` used a hardcoded `WEIGHTS` dict and ignored the config. Fixed: `detect_edge()` now overlays `config.signals.weights` on top of the defaults.
- **Broken test suite** — Old `tests/test_core.py` imported removed modules and called APIs with stale signatures; all tests failed at import time. Replaced with a complete 100-test suite that passes cleanly.
- **Root `config.yaml` ambiguity** — Two config files existed (root and `configs/`). Root `config.yaml` was deleted; `configs/config.yaml` is now canonical and unambiguous.

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

# Backtest (SQLite → JSON → synthetic fallback)
python -m backtest.runner

# Parameter sweep
python -m backtest.runner --sweep

# Ingest historical NBA data
python scripts/ingest_historical.py --start 2025-01-01 --end 2026-05-04

# Inspect what was ingested
python scripts/inspect_historical.py

# Analyze exit telemetry (after running bot and closing some positions)
python scripts/analyze_exits.py --days 30

# Containerized deployment (bot + supervisor, shared data volume)
docker compose up -d --build
docker compose logs -f bot

# Override any config value per-environment without editing YAML
POLYBOT_TRADING__PAPER_TRADING=false POLYBOT_TRADING__KELLY_FRACTION=0.25 python -m bot.trading_loop
```

Logs: `reports/trading.log` (bot), `reports/supervisor.log` (supervisor), `reports/live_output.log` (nohup stdout).
