"""Charts, JSON/CSV exports, benchmark comparison."""

import os
import json
import csv
from datetime import datetime
from typing import List, Dict, Any, Optional

import numpy as np

from utils.models import BacktestResult


def generate_report(
    result: BacktestResult,
    output_dir: str = "./reports",
    benchmark_win_rate: float = 0.62,
    benchmark_trade_count: int = 366,
    chart_format: str = "png",
) -> str:
    """
    Generate a full backtest report with charts and data exports.

    Returns the path to the report directory.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = os.path.join(output_dir, f"backtest_{timestamp}")
    os.makedirs(report_dir, exist_ok=True)

    # Print summary to console
    _print_summary(result, benchmark_win_rate, benchmark_trade_count)

    # Generate charts
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        _plot_equity_curve(result, report_dir, chart_format)
        _plot_drawdown(result, report_dir, chart_format)
        _plot_trade_distribution(result, report_dir, chart_format)
        _plot_benchmark_comparison(result, report_dir, chart_format,
                                   benchmark_win_rate, benchmark_trade_count)
    except ImportError:
        print("  [WARNING] matplotlib not available, skipping charts")

    # Export JSON
    _export_json(result, report_dir, benchmark_win_rate, benchmark_trade_count)

    # Export CSV
    _export_csv(result, report_dir)

    print(f"\n  Report saved to: {report_dir}/")
    return report_dir


def generate_sweep_report(
    sweep_results: List[Dict[str, Any]],
    output_dir: str = "./reports",
    chart_format: str = "png",
) -> str:
    """Generate sweep-specific reports."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = os.path.join(output_dir, f"sweep_{timestamp}")
    os.makedirs(report_dir, exist_ok=True)

    # Export sweep results CSV
    _export_sweep_csv(sweep_results, report_dir)

    # Generate heatmap
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        _plot_sensitivity_heatmap(sweep_results, report_dir, chart_format)
    except ImportError:
        print("  [WARNING] matplotlib not available, skipping charts")

    # Export sweep summary JSON
    summary = {
        "total_combinations": len(sweep_results),
        "best_by_sharpe": sweep_results[0] if sweep_results else None,
        "best_by_return": max(sweep_results, key=lambda x: x["total_return"]) if sweep_results else None,
        "best_by_win_rate": max(sweep_results, key=lambda x: x["win_rate"]) if sweep_results else None,
    }
    with open(os.path.join(report_dir, "sweep_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n  Sweep report saved to: {report_dir}/")
    return report_dir


def _print_summary(result: BacktestResult, benchmark_wr: float, benchmark_tc: int):
    """Print backtest summary to console."""
    print(f"\n{'='*60}")
    print(f"  BACKTEST COMPLETE")
    print(f"{'='*60}")
    print(f"  Trades:         {result.total_trades}")
    print(f"  Win Rate:       {result.win_rate*100:.1f}%")
    print(f"  Total Return:   ${result.total_return:.2f} ({result.total_return_pct:.1f}%)")
    print(f"  Max Drawdown:   ${result.max_drawdown:.2f} ({result.max_drawdown_pct:.1f}%)")
    print(f"  Sharpe Ratio:   {result.sharpe_ratio:.2f}")
    print(f"  Profit Factor:  {result.profit_factor:.2f}")
    print(f"  Avg Duration:   {result.avg_duration_hours:.1f} hours")
    print(f"{'='*60}")

    # Benchmark comparison
    wr_pass = "PASS" if result.win_rate >= benchmark_wr else "FAIL"
    tc_pass = "PASS" if result.total_trades >= benchmark_tc else "FAIL"
    print(f"  Benchmark ({benchmark_wr*100:.0f}% WR / {benchmark_tc} trades):")
    print(f"    Win Rate:     [{wr_pass}] {result.win_rate*100:.1f}% vs {benchmark_wr*100:.1f}%")
    print(f"    Trade Count:  [{tc_pass}] {result.total_trades} vs {benchmark_tc}")
    print(f"{'='*60}")


def _plot_equity_curve(result: BacktestResult, report_dir: str, fmt: str):
    """Plot equity curve."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(result.equity_curve, label="Portfolio Value", color="blue", linewidth=1.5)
    ax.axhline(y=result.initial_bankroll, color="gray", linestyle="--", label="Initial Bankroll")
    ax.set_title("Equity Curve")
    ax.set_xlabel("Time (snapshots)")
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, f"equity_curve.{fmt}"), dpi=150)
    plt.close()


def _plot_drawdown(result: BacktestResult, report_dir: str, fmt: str):
    """Plot drawdown chart."""
    import matplotlib.pyplot as plt

    equity = np.array(result.equity_curve)
    peaks = np.maximum.accumulate(equity)
    drawdowns = (peaks - equity) / peaks * 100

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(range(len(drawdowns)), drawdowns, color="red", alpha=0.3)
    ax.plot(drawdowns, color="red", linewidth=1)
    ax.set_title("Drawdown")
    ax.set_xlabel("Time (snapshots)")
    ax.set_ylabel("Drawdown (%)")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, f"drawdown.{fmt}"), dpi=150)
    plt.close()


def _plot_trade_distribution(result: BacktestResult, report_dir: str, fmt: str):
    """Plot trade P&L distribution."""
    import matplotlib.pyplot as plt

    closed = [p for p in result.positions if p.status == "closed"]
    if not closed:
        return

    pnls = [p.realized_pnl for p in closed]
    edges = [abs(p.estimated_prob - p.entry_price) for p in closed]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # P&L histogram
    axes[0].hist(pnls, bins=30, color="steelblue", edgecolor="white", alpha=0.8)
    axes[0].axvline(x=0, color="red", linestyle="--")
    axes[0].set_title("Trade P&L Distribution")
    axes[0].set_xlabel("P&L ($)")
    axes[0].set_ylabel("Count")

    # Edge distribution
    axes[1].hist(edges, bins=20, color="green", edgecolor="white", alpha=0.8)
    axes[1].set_title("Edge Size Distribution")
    axes[1].set_xlabel("Edge (absolute)")
    axes[1].set_ylabel("Count")

    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, f"trade_distribution.{fmt}"), dpi=150)
    plt.close()


def _plot_benchmark_comparison(result: BacktestResult, report_dir: str, fmt: str,
                                benchmark_wr: float, benchmark_tc: int):
    """Plot benchmark comparison bar chart."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Win rate comparison
    axes[0].bar(["Bot", "Benchmark"], [result.win_rate * 100, benchmark_wr * 100],
                color=["steelblue", "gray"])
    axes[0].set_title("Win Rate Comparison")
    axes[0].set_ylabel("Win Rate (%)")
    axes[0].set_ylim(0, 100)

    # Trade count comparison
    axes[1].bar(["Bot", "Benchmark"], [result.total_trades, benchmark_tc],
                color=["steelblue", "gray"])
    axes[1].set_title("Trade Count Comparison")
    axes[1].set_ylabel("Trades")

    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, f"benchmark_comparison.{fmt}"), dpi=150)
    plt.close()


def _plot_sensitivity_heatmap(sweep_results: List[Dict], report_dir: str, fmt: str):
    """Plot sensitivity heatmap for edge threshold vs stop-loss."""
    import matplotlib.pyplot as plt

    if not sweep_results:
        return

    # Extract unique edge thresholds and stop-losses for kelly 0.5 only
    kelly_results = [r for r in sweep_results
                     if r["params"]["sizing_method"] == "kelly"
                     and r["params"]["kelly_fraction"] == 0.5]

    if not kelly_results:
        kelly_results = sweep_results

    edges = sorted(set(r["params"]["edge_threshold"] for r in kelly_results))
    stops = sorted(set(r["params"]["stop_loss"] for r in kelly_results))

    if len(edges) < 2 or len(stops) < 2:
        return

    # Build heatmap matrix
    matrix = np.zeros((len(stops), len(edges)))
    for r in kelly_results:
        e_idx = edges.index(r["params"]["edge_threshold"])
        s_idx = stops.index(r["params"]["stop_loss"])
        matrix[s_idx, e_idx] = r["sharpe_ratio"]

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(edges)))
    ax.set_xticklabels([f"{e*100:.0f}%" for e in edges])
    ax.set_yticks(range(len(stops)))
    ax.set_yticklabels([f"{s*100:.0f}%" for s in stops])
    ax.set_xlabel("Edge Threshold")
    ax.set_ylabel("Stop-Loss Level")
    ax.set_title("Sharpe Ratio Sensitivity (Kelly 0.5x)")
    plt.colorbar(im)

    # Add text annotations
    for i in range(len(stops)):
        for j in range(len(edges)):
            ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, f"sensitivity_heatmap.{fmt}"), dpi=150)
    plt.close()


def _export_json(result: BacktestResult, report_dir: str,
                  benchmark_wr: float, benchmark_tc: int):
    """Export results as JSON."""
    data = {
        "summary": {
            "initial_bankroll": result.initial_bankroll,
            "final_bankroll": result.final_bankroll,
            "total_return": result.total_return,
            "total_return_pct": result.total_return_pct,
            "win_rate": result.win_rate,
            "total_trades": result.total_trades,
            "winning_trades": result.winning_trades,
            "losing_trades": result.losing_trades,
            "max_drawdown": result.max_drawdown,
            "max_drawdown_pct": result.max_drawdown_pct,
            "sharpe_ratio": result.sharpe_ratio,
            "profit_factor": result.profit_factor,
            "avg_trade_pnl": result.avg_trade_pnl,
            "avg_win": result.avg_win,
            "avg_loss": result.avg_loss,
            "avg_duration_hours": result.avg_duration_hours,
        },
        "config": result.config,
        "benchmark": {
            "win_rate": benchmark_wr,
            "trade_count": benchmark_tc,
            "win_rate_pass": result.win_rate >= benchmark_wr,
            "trade_count_pass": result.total_trades >= benchmark_tc,
        },
    }

    with open(os.path.join(report_dir, "summary.json"), "w") as f:
        json.dump(data, f, indent=2, default=str)


def _export_csv(result: BacktestResult, report_dir: str):
    """Export trades as CSV."""
    closed = [p for p in result.positions if p.status == "closed"]
    if not closed:
        return

    filepath = os.path.join(report_dir, "trades.csv")
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "market_id", "side", "entry_price", "close_price",
            "size_usd", "pnl", "close_reason", "entry_time",
            "close_time", "duration_hours",
        ])
        for p in closed:
            duration = 0
            if p.close_time and p.entry_time:
                duration = (p.close_time - p.entry_time).total_seconds() / 3600
            writer.writerow([
                p.market_id, p.side, round(p.entry_price, 4),
                round(p.close_price, 4), round(p.size_usd, 2),
                round(p.realized_pnl, 2), p.close_reason,
                p.entry_time.isoformat() if p.entry_time else "",
                p.close_time.isoformat() if p.close_time else "",
                round(duration, 1),
            ])


def _export_sweep_csv(sweep_results: List[Dict], report_dir: str):
    """Export sweep results as CSV."""
    if not sweep_results:
        return

    filepath = os.path.join(report_dir, "sweep_results.csv")
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "edge_threshold", "sizing_method", "kelly_fraction",
            "fixed_fraction", "stop_loss", "total_return",
            "total_return_pct", "win_rate", "total_trades",
            "sharpe_ratio", "max_drawdown_pct", "profit_factor",
        ])
        for r in sweep_results:
            p = r["params"]
            writer.writerow([
                p["edge_threshold"], p["sizing_method"],
                p["kelly_fraction"], p["fixed_fraction"],
                p["stop_loss"], round(r["total_return"], 2),
                round(r["total_return_pct"], 1), round(r["win_rate"], 3),
                r["total_trades"], round(r["sharpe_ratio"], 2),
                round(r["max_drawdown_pct"], 1), round(r["profit_factor"], 2),
            ])
