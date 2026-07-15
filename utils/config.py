"""YAML + .env config loader with validation."""

import os
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

import yaml
from dotenv import load_dotenv


@dataclass
class APIConfig:
    clob_url: str = "https://clob.polymarket.com"
    gamma_url: str = "https://gamma-api.polymarket.com"
    subgraph_url: str = ""
    scan_interval_seconds: int = 10
    max_requests_per_second: int = 5
    max_retries: int = 3
    retry_backoff_base: float = 2.0


@dataclass
class OnChainConfig:
    enabled: bool = True
    max_requests_per_second: int = 3
    cache_ttl_seconds: int = 120
    enrich_all_markets: bool = False
    enrich_on_edge_only: bool = True
    min_trades_for_analysis: int = 20


@dataclass
class TradingConfig:
    min_edge_threshold: float = 0.05
    max_edge_threshold: float = 0.40
    position_sizing_method: str = "kelly"
    fixed_fraction: float = 0.02
    kelly_fraction: float = 0.5
    max_position_size_usd: float = 50.0
    min_position_size_usd: float = 1.0
    max_portfolio_exposure_usd: float = 500.0
    stop_loss_threshold: float = 0.25
    take_profit_threshold: float = 0.05
    minimum_take_profit_usd: float = 5.0
    aggressive_exit_pct: float = 0.30
    trailing_stop_activation_pct: float = 0.15
    trailing_stop_pct: float = 0.10
    daily_loss_limit_usd: float = 200.0
    max_open_positions: int = 5
    max_daily_trades: int = 15
    paper_trading: bool = True
    # --- Cost-aware edge accounting ---
    # Polymarket US fee schedule (effective 2026-07-01, docs.polymarket.us/fees):
    #   fee = coefficient * contracts * price * (1 - price)
    # Takers pay 0.06; see bot/strategies/fees.py. Edge math, Kelly sizing,
    # paper fills, live fills, and the backtest all read this ONE value.
    taker_fee_coefficient: float = 0.06
    # DEPRECATED: legacy flat fee-on-notional model. No longer read by edge,
    # sizing, execution, or backtest. Kept so old env overrides don't crash.
    taker_fee_rate: float = 0.02
    # Minimum edge AFTER subtracting fees and the cost of crossing the spread.
    # Gross edge thresholds alone overstate profitability: a 5% gross edge with
    # a 2% fee and a 2% half-spread is only a 1% real edge.
    min_net_edge: float = 0.02
    # Tradeable price band (avoid extreme longshots/favorites where resolution
    # risk dominates) and order-book quality gates. Previously hardcoded in
    # trade_filter.py; now configurable per deployment.
    min_price: float = 0.15
    max_price: float = 0.85
    min_book_liquidity_usd: float = 1000.0
    # Widest acceptable bid-ask spread (absolute, in price units). A book wider
    # than this cannot be exited cleanly, so we never enter it.
    max_spread: float = 0.10
    # --- Live in-game validation (validate_live_trade) ---
    # Mid-game trades can't use the pregame checklist: books pull their lines
    # at tip-off, so the >=2-books rule blocked every live trade. Instead the
    # ESPN win-prob model is the external validation, gated by:
    # Freshness: model data older than this gets no trade (stale model during
    # a live game is misinformation, not reduced information).
    max_live_model_age_seconds: float = 45.0
    # Divergence sanity cap: a model-vs-market gap LARGER than this is a red
    # flag, not an opportunity — verified on Sky@Wings Q4 1:20 (model 24.9%
    # vs market 13%: the market was right, Chicago lost). Genuine repricing
    # lag lives in single digits.
    max_live_divergence: float = 0.15
    # Game-phase window: skip the chaotic opening minutes (model warming up)
    # and rely on the existing last-5-minutes block for the endgame.
    live_min_elapsed_seconds: float = 300.0
    # Clockless sports (baseball): no clock to gate on, so block entries from
    # this period (inning) onward — late innings are the endgame equivalent.
    live_clockless_max_period: int = 7
    # --- Live discipline (Jul 13 postmortem: edge detection was right, the
    # LEAK was re-entry churn — 6 stop-outs from re-buying choppy games,
    # including buying the 52c top seconds after a profitable exit) ---
    # Hard caps per game (game_id includes the date, so no daily reset needed):
    max_entries_per_game: int = 3
    max_stops_per_game: int = 2          # 2 stop-losses on a game = locked out
    # Live stops need room: 25% of a 25c entry is 6c — inside normal live
    # noise. Wider stop, smaller size = same $ risk with fewer whipsaws.
    live_stop_loss_threshold: float = 0.35
    live_max_position_size_usd: float = 25.0
    # Two-scan confirmation: a live edge must persist across scans (first
    # sighting at least min, at most max seconds old) before entering —
    # filters single-play model spikes.
    live_confirm_min_seconds: float = 3.0
    live_confirm_max_seconds: float = 30.0


# Default per-market-type weights — must stay in sync with WEIGHTS in estimator.py
# onchain_flow has weight everywhere but returns confidence=0 unless an
# OnChainEnrichmentClient is wired in, so it is a strict no-op when disabled.
_DEFAULT_SIGNAL_WEIGHTS: Dict[str, Dict[str, float]] = {
    "sports": {
        "odds_value": 0.40,
        "sports_context": 0.15,
        "line_movement": 0.20,
        "order_book_imbalance": 0.15,
        "liquidity_imbalance": 0.10,
        "onchain_flow": 0.10,
        "live_win_prob": 0.55,   # in-game only (confidence 0 pregame)
        "derivative_line": 0.40,  # spread/total markets only
    },
    "crypto": {
        "crypto_model": 0.45,
        "cross_market": 0.25,
        "order_book_imbalance": 0.20,
        "liquidity_imbalance": 0.10,
        "onchain_flow": 0.10,
    },
    "politics": {
        "cross_market": 0.45,
        "order_book_imbalance": 0.25,
        "line_movement": 0.15,
        "liquidity_imbalance": 0.15,
        "onchain_flow": 0.10,
    },
    "other": {
        "cross_market": 0.40,
        "order_book_imbalance": 0.25,
        "line_movement": 0.20,
        "liquidity_imbalance": 0.15,
        "onchain_flow": 0.10,
    },
}


@dataclass
class SignalConfig:
    order_book_imbalance_weight: float = 0.15
    line_movement_weight: float = 0.20
    odds_value_weight: float = 0.40
    liquidity_imbalance_weight: float = 0.10
    sports_context_weight: float = 0.15
    # How signals are pooled into one probability:
    #   "linear"  — confidence-weighted arithmetic mean (DEFAULT). The edge
    #               thresholds (league 4-7% minimums), benchmark win rates,
    #               and exit tuning were all calibrated against this pool,
    #               where a lone external signal passes its consensus through
    #               at full strength.
    #   "logodds" — Bayesian update in log-odds space anchored on the market
    #               price as the prior. Better multi-signal math, but it
    #               shrinks the combined edge by overall confidence — with
    #               the odds_value confidence formula (min(books/5)*min(edge*5))
    #               a 6% single-book edge collapses to <1% combined and
    #               nothing ever clears min_edge_threshold. EXPERIMENTAL:
    #               requires recalibrating every edge threshold before use.
    combination_method: str = "linear"
    weights: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: {
            market: dict(signal_weights)
            for market, signal_weights in _DEFAULT_SIGNAL_WEIGHTS.items()
        }
    )


@dataclass
class FilterConfig:
    min_daily_volume_usd: float = 500.0
    min_liquidity_usd: float = 200.0
    min_hours_to_expiry: float = 1.0
    max_hours_to_expiry: float = 48.0
    min_price_history_length: int = 10
    # Time-to-resolution windows actually enforced by MarketDataClient.
    # Sports markets: only scan games starting within this many hours.
    # Non-sports markets: only scan markets resolving within this many days.
    # (max_hours_to_expiry above is legacy and superseded by these two.)
    sports_window_hours: float = 24.0
    nonsports_window_days: float = 14.0
    include_categories: List[str] = field(default_factory=list)
    exclude_categories: List[str] = field(default_factory=list)


@dataclass
class BacktestConfig:
    data_dir: str = "./data/historical"
    start_date: str = "2024-01-01"
    end_date: str = "2024-12-31"
    initial_bankroll_usd: float = 1000.0
    slippage_model: str = "depth_based"
    fixed_slippage_bps: int = 50
    depth_slippage_multiplier: float = 0.001
    maker_fee_bps: int = 0
    taker_fee_bps: int = 200
    latency_ms: int = 500
    benchmark_win_rate: float = 0.62
    benchmark_trade_count: int = 366
    # Minimum prior price points before a replayed snapshot is tradeable.
    # Separate from filters.min_price_history_length (live scanning, 10):
    # recorded game markets carry only a few daily candles, so the live value
    # would silently skip every real snapshot.
    min_price_history_length: int = 1


@dataclass
class SweepConfig:
    edge_thresholds: List[float] = field(default_factory=lambda: [0.03, 0.05, 0.07, 0.10, 0.15])
    sizing_methods: List[str] = field(default_factory=lambda: ["kelly", "fixed_fractional"])
    kelly_fractions: List[float] = field(default_factory=lambda: [0.25, 0.5, 0.75, 1.0])
    fixed_fractions: List[float] = field(default_factory=lambda: [0.01, 0.02, 0.05, 0.10])
    stop_loss_levels: List[float] = field(default_factory=lambda: [0.05, 0.10, 0.15, 0.20, 0.30])
    max_workers: int = 4


@dataclass
class LoggingConfig:
    level: str = "INFO"
    file: str = "./reports/trading.log"
    console: bool = True
    log_scans: bool = True
    log_order_books: bool = False


@dataclass
class ReportingConfig:
    output_dir: str = "./reports"
    chart_format: str = "png"
    export_json: bool = True
    export_csv: bool = True


@dataclass
class AlertConfig:
    enabled: bool = True
    slack_webhook_url: str = ""
    on_trade_open: bool = True
    on_trade_close: bool = True
    on_daily_summary: bool = True
    daily_summary_hour: int = 22
    on_error: bool = True


@dataclass
class WalletConfig:
    chain_id: int = 137
    approval_amount: str = "1000000000"
    key_id: str = ""
    secret_key: str = ""


@dataclass
class BotConfig:
    api: APIConfig = field(default_factory=APIConfig)
    onchain: OnChainConfig = field(default_factory=OnChainConfig)
    wallet: WalletConfig = field(default_factory=WalletConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    signals: SignalConfig = field(default_factory=SignalConfig)
    filters: FilterConfig = field(default_factory=FilterConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    sweep: SweepConfig = field(default_factory=SweepConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    reporting: ReportingConfig = field(default_factory=ReportingConfig)
    alerts: AlertConfig = field(default_factory=AlertConfig)
    private_key: str = ""
    odds_api_key: str = ""


def _apply_dict(obj, d: Dict[str, Any]):
    """Apply a dict of values to a dataclass instance."""
    for k, v in d.items():
        if hasattr(obj, k):
            current = getattr(obj, k)
            if isinstance(current, list) and isinstance(v, list):
                setattr(obj, k, v)
            else:
                setattr(obj, k, type(current)(v) if not isinstance(v, type(current)) and not isinstance(v, list) else v)


def load_config(config_path: str = "configs/config.yaml", env_path: str = ".env") -> BotConfig:
    """Load configuration from YAML file and environment variables."""
    load_dotenv(env_path)

    config = BotConfig()

    # Try multiple paths for the config file
    paths_to_try = [config_path, f"./{config_path}", os.path.join(os.getcwd(), config_path)]
    # Also try config.yaml in project root
    if config_path == "configs/config.yaml":
        paths_to_try.append("config.yaml")
        paths_to_try.append(os.path.join(os.getcwd(), "config.yaml"))

    raw = None
    for path in paths_to_try:
        if os.path.exists(path):
            with open(path, "r") as f:
                raw = yaml.safe_load(f)
            break

    if raw and isinstance(raw, dict):
        if "api" in raw:
            _apply_dict(config.api, raw["api"])
        if "onchain" in raw:
            _apply_dict(config.onchain, raw["onchain"])
        if "wallet" in raw:
            _apply_dict(config.wallet, raw["wallet"])
        if "trading" in raw:
            _apply_dict(config.trading, raw["trading"])
        if "signals" in raw:
            signals_raw = dict(raw["signals"])
            # Handle weights separately: deep-merge per-market-type and per-signal
            # so that missing keys keep their hardcoded defaults.
            weights_from_yaml = signals_raw.pop("weights", None)
            _apply_dict(config.signals, signals_raw)
            if weights_from_yaml and isinstance(weights_from_yaml, dict):
                for market_type, market_weights in weights_from_yaml.items():
                    if isinstance(market_weights, dict):
                        # Start from defaults for this market type, then overlay config values
                        base = dict(config.signals.weights.get(market_type, {}))
                        base.update(market_weights)
                        config.signals.weights[market_type] = base
        if "filters" in raw:
            _apply_dict(config.filters, raw["filters"])
        if "backtest" in raw:
            _apply_dict(config.backtest, raw["backtest"])
        if "sweep" in raw:
            _apply_dict(config.sweep, raw["sweep"])
        if "logging" in raw:
            _apply_dict(config.logging, raw["logging"])
        if "reporting" in raw:
            _apply_dict(config.reporting, raw["reporting"])
        if "alerts" in raw:
            _apply_dict(config.alerts, raw["alerts"])

    # Load private key from environment
    config.private_key = os.environ.get("POLYMARKET_PRIVATE_KEY", "")

    # Load Polymarket US API credentials from environment
    config.wallet.key_id = os.environ.get("POLYMARKET_KEY_ID", "")
    config.wallet.secret_key = os.environ.get("POLYMARKET_SECRET_KEY", "")

    # Load Slack webhook URL from environment
    env_webhook = os.environ.get("SLACK_WEBHOOK_URL", "")
    if env_webhook:
        config.alerts.slack_webhook_url = env_webhook

    # Load The Odds API key from environment
    config.odds_api_key = os.environ.get("THE_ODDS_API_KEY", "")

    # Generic env-var override layer — applied LAST so it wins over YAML.
    # Any scalar config field can be overridden without editing files:
    #   POLYBOT_TRADING__MIN_EDGE_THRESHOLD=0.07
    #   POLYBOT_TRADING__PAPER_TRADING=false
    #   POLYBOT_ALERTS__ENABLED=true
    # Format: POLYBOT_<SECTION>__<FIELD> (double underscore between section
    # and field). Nested dicts (signals.weights) are YAML-only.
    _apply_env_overrides(config)

    return config


def _coerce_env_value(raw: str, current):
    """Coerce an env-var string to the type of the existing config value."""
    if isinstance(current, bool):
        return raw.strip().lower() in ("1", "true", "yes", "on")
    if isinstance(current, int) and not isinstance(current, bool):
        return int(float(raw))
    if isinstance(current, float):
        return float(raw)
    return raw


def _apply_env_overrides(config: "BotConfig", environ: Optional[Dict[str, str]] = None):
    """Apply POLYBOT_<SECTION>__<FIELD> environment overrides to config.

    Only overrides fields that already exist on the section dataclass, and
    coerces the string to the field's current type. Unknown sections/fields
    are ignored silently so stray env vars can't crash startup.
    """
    env = environ if environ is not None else os.environ
    sections = {
        "api": config.api,
        "onchain": config.onchain,
        "wallet": config.wallet,
        "trading": config.trading,
        "signals": config.signals,
        "filters": config.filters,
        "backtest": config.backtest,
        "sweep": config.sweep,
        "logging": config.logging,
        "reporting": config.reporting,
        "alerts": config.alerts,
    }
    prefix = "POLYBOT_"
    for key, raw in env.items():
        if not key.startswith(prefix) or "__" not in key:
            continue
        section_name, _, field_name = key[len(prefix):].partition("__")
        section = sections.get(section_name.lower())
        if section is None:
            continue
        attr = field_name.lower()
        if not hasattr(section, attr):
            continue
        current = getattr(section, attr)
        if isinstance(current, (dict, list)):
            continue  # complex structures stay YAML-only
        try:
            setattr(section, attr, _coerce_env_value(raw, current))
        except (ValueError, TypeError):
            continue  # malformed value — keep the YAML/default value
