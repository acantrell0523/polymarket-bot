"""CLI entry point for backtests."""

import sys
import os

from utils.config import load_config
from data.loader import generate_synthetic_markets, load_historical_data, load_historical_data_from_db
from backtest.engine import BacktestEngine
from backtest.reporting import generate_report, generate_sweep_report
from backtest.sweep import run_sweep


def main():
    """Run backtest or parameter sweep."""
    # Parse arguments
    config_path = "configs/config.yaml"
    do_sweep = False

    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg == "--config" and i + 1 < len(args):
            config_path = args[i + 1]
        elif arg == "--sweep":
            do_sweep = True

    config = load_config(config_path)

    # Load or generate data — priority order:
    #   1. SQLite (historical_markets / historical_snapshots in data/trades.db)
    #   2. JSON files in config.backtest.data_dir
    #   3. Synthetic GBM data (fallback when both real sources are empty)
    print("\nLoading market data...")
    market_data = load_historical_data_from_db()

    if market_data is not None:
        total_snaps = sum(len(m) for m in market_data)
        print(f"  Loaded {len(market_data)} markets ({total_snaps} snapshots) from SQLite")
        print(f"  NOTE: NBA outright markets route to 'sports' type; trades require")
        print(f"        OddsCache to be populated. Zero trades = gate working as designed.")
    else:
        market_data = load_historical_data(config.backtest.data_dir)
        if market_data is not None:
            total_snaps = sum(len(m) for m in market_data)
            print(f"  Loaded {len(market_data)} markets ({total_snaps} snapshots) from JSON")

    if market_data is None:
        print("  No historical data found, generating synthetic data...")
        market_data = generate_synthetic_markets(
            num_markets=20,
            history_length=200,
            num_snapshots=500,
            seed=42,
        )
        print(f"  Generated {len(market_data)} synthetic markets with 500 snapshots each")

    if do_sweep:
        # Parameter sweep mode
        sweep_results = run_sweep(
            config, market_data, max_workers=config.sweep.max_workers
        )
        report_dir = generate_sweep_report(
            sweep_results,
            output_dir=config.reporting.output_dir,
            chart_format=config.reporting.chart_format,
        )
    else:
        # Single backtest mode
        print("\nRunning backtest...")
        engine = BacktestEngine(config)
        result = engine.run(market_data)

        report_dir = generate_report(
            result,
            output_dir=config.reporting.output_dir,
            benchmark_win_rate=config.backtest.benchmark_win_rate,
            benchmark_trade_count=config.backtest.benchmark_trade_count,
            chart_format=config.reporting.chart_format,
        )


if __name__ == "__main__":
    main()
