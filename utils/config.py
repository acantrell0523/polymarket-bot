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


@dataclass
class SignalConfig:
    order_book_imbalance_weight: float = 0.15
    line_movement_weight: float = 0.20
    odds_value_weight: float = 0.40
    liquidity_imbalance_weight: float = 0.10
    sports_context_weight: float = 0.15


@dataclass
class FilterConfig:
    min_daily_volume_usd: float = 500.0
    min_liquidity_usd: float = 200.0
    min_hours_to_expiry: float = 1.0
    max_hours_to_expiry: float = 48.0
    min_price_history_length: int = 10
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
            _apply_dict(config.signals, raw["signals"])
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

    return config
