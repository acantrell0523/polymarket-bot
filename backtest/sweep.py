"""Parameter sensitivity analysis."""

import copy
import itertools
from typing import List, Dict, Any, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

from utils.config import BotConfig, SweepConfig
from utils.models import BacktestResult
from backtest.engine import BacktestEngine
from utils.models import MarketSnapshot


def _run_single_backtest(args: Tuple) -> Dict[str, Any]:
    """Run a single backtest with specific parameters. Used by multiprocessing."""
    config_dict, market_data_serialized, params = args

    # Reconstruct config
    config = _dict_to_config(config_dict)

    # Apply sweep parameters
    config.trading.min_edge_threshold = params["edge_threshold"]
    config.trading.position_sizing_method = params["sizing_method"]
    config.trading.kelly_fraction = params.get("kelly_fraction", 0.5)
    config.trading.fixed_fraction = params.get("fixed_fraction", 0.02)
    config.trading.stop_loss_threshold = params["stop_loss"]

    engine = BacktestEngine(config)
    result = engine.run(market_data_serialized)

    return {
        "params": params,
        "total_return": result.total_return,
        "total_return_pct": result.total_return_pct,
        "win_rate": result.win_rate,
        "total_trades": result.total_trades,
        "sharpe_ratio": result.sharpe_ratio,
        "max_drawdown": result.max_drawdown,
        "max_drawdown_pct": result.max_drawdown_pct,
        "profit_factor": result.profit_factor,
        "avg_trade_pnl": result.avg_trade_pnl,
    }


def _config_to_dict(config: BotConfig) -> Dict:
    """Serialize config to dict for multiprocessing."""
    import dataclasses
    result = {}
    for f in dataclasses.fields(config):
        val = getattr(config, f.name)
        if dataclasses.is_dataclass(val):
            result[f.name] = dataclasses.asdict(val)
        else:
            result[f.name] = val
    return result


def _dict_to_config(d: Dict) -> BotConfig:
    """Deserialize config from dict."""
    from utils.config import (
        BotConfig, APIConfig, OnChainConfig, TradingConfig,
        SignalConfig, FilterConfig, BacktestConfig, SweepConfig as SC,
        LoggingConfig, ReportingConfig,
    )
    config = BotConfig()
    if "api" in d:
        for k, v in d["api"].items():
            if hasattr(config.api, k):
                setattr(config.api, k, v)
    if "onchain" in d:
        for k, v in d["onchain"].items():
            if hasattr(config.onchain, k):
                setattr(config.onchain, k, v)
    if "trading" in d:
        for k, v in d["trading"].items():
            if hasattr(config.trading, k):
                setattr(config.trading, k, v)
    if "signals" in d:
        for k, v in d["signals"].items():
            if hasattr(config.signals, k):
                setattr(config.signals, k, v)
    if "filters" in d:
        for k, v in d["filters"].items():
            if hasattr(config.filters, k):
                setattr(config.filters, k, v)
    if "backtest" in d:
        for k, v in d["backtest"].items():
            if hasattr(config.backtest, k):
                setattr(config.backtest, k, v)
    if "sweep" in d:
        for k, v in d["sweep"].items():
            if hasattr(config.sweep, k):
                setattr(config.sweep, k, v)
    if "logging" in d:
        for k, v in d["logging"].items():
            if hasattr(config.logging, k):
                setattr(config.logging, k, v)
    if "reporting" in d:
        for k, v in d["reporting"].items():
            if hasattr(config.reporting, k):
                setattr(config.reporting, k, v)
    return config


def generate_parameter_combinations(sweep_config: SweepConfig) -> List[Dict[str, Any]]:
    """Generate all parameter combinations for the sweep."""
    combinations = []

    for edge in sweep_config.edge_thresholds:
        for stop_loss in sweep_config.stop_loss_levels:
            # Kelly combinations
            for kelly_f in sweep_config.kelly_fractions:
                combinations.append({
                    "edge_threshold": edge,
                    "sizing_method": "kelly",
                    "kelly_fraction": kelly_f,
                    "fixed_fraction": 0.02,
                    "stop_loss": stop_loss,
                })

            # Fixed fractional combinations
            for fixed_f in sweep_config.fixed_fractions:
                combinations.append({
                    "edge_threshold": edge,
                    "sizing_method": "fixed_fractional",
                    "kelly_fraction": 0.5,
                    "fixed_fraction": fixed_f,
                    "stop_loss": stop_loss,
                })

    return combinations


def run_sweep(
    config: BotConfig,
    market_data: List[List[MarketSnapshot]],
    max_workers: int = 4,
) -> List[Dict[str, Any]]:
    """
    Run parameter sweep across all combinations.

    Uses sequential execution (multiprocessing with market data
    is complex due to serialization).
    """
    combinations = generate_parameter_combinations(config.sweep)
    results = []

    total = len(combinations)
    print(f"\nRunning parameter sweep: {total} combinations")
    print(f"{'='*60}")

    for i, params in enumerate(combinations):
        # Apply parameters
        sweep_config = copy.deepcopy(config)
        sweep_config.trading.min_edge_threshold = params["edge_threshold"]
        sweep_config.trading.position_sizing_method = params["sizing_method"]
        sweep_config.trading.kelly_fraction = params["kelly_fraction"]
        sweep_config.trading.fixed_fraction = params["fixed_fraction"]
        sweep_config.trading.stop_loss_threshold = params["stop_loss"]

        engine = BacktestEngine(sweep_config)
        result = engine.run(market_data)

        entry = {
            "params": params,
            "total_return": result.total_return,
            "total_return_pct": result.total_return_pct,
            "win_rate": result.win_rate,
            "total_trades": result.total_trades,
            "sharpe_ratio": result.sharpe_ratio,
            "max_drawdown": result.max_drawdown,
            "max_drawdown_pct": result.max_drawdown_pct,
            "profit_factor": result.profit_factor,
            "avg_trade_pnl": result.avg_trade_pnl,
        }
        results.append(entry)

        if (i + 1) % 10 == 0 or (i + 1) == total:
            print(f"  Progress: {i+1}/{total} combinations tested")

    # Sort by Sharpe ratio
    results.sort(key=lambda x: x["sharpe_ratio"], reverse=True)

    print(f"\n{'='*60}")
    print(f"  SWEEP COMPLETE - {total} combinations tested")
    print(f"{'='*60}")

    if results:
        best = results[0]
        print(f"\n  Best by Sharpe Ratio ({best['sharpe_ratio']:.2f}):")
        print(f"    Edge threshold: {best['params']['edge_threshold']*100:.0f}%")
        print(f"    Sizing: {best['params']['sizing_method']}")
        if best['params']['sizing_method'] == 'kelly':
            print(f"    Kelly fraction: {best['params']['kelly_fraction']}")
        else:
            print(f"    Fixed fraction: {best['params']['fixed_fraction']}")
        print(f"    Stop-loss: {best['params']['stop_loss']*100:.0f}%")
        print(f"    Return: ${best['total_return']:.2f} ({best['total_return_pct']:.1f}%)")
        print(f"    Win rate: {best['win_rate']*100:.1f}%")
        print(f"    Trades: {best['total_trades']}")
        print(f"    Max drawdown: {best['max_drawdown_pct']:.1f}%")

    return results
