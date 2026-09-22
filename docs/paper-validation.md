# Paper trading validation

The older paper sessions are exploratory. Entries used the snapshot midpoint, closes used a last/current quote, the closed-trade table excluded fees, and restarts reset simulated cash while restoring positions. Those records cannot show a reliable net return. `scripts/audit_paper.py` estimates the fee impact on closed legacy trades, but it cannot reconstruct executable prices, available depth, or quote timing.

The v2 simulator uses the visible best ask for buys and best bid for sells, subject to the IOC limit and whole-contract depth. It reserves purchase cost or short collateral `(1 - bid)` plus the entry fee. It marks equity at the executable exit price and records both entry and exit fees in net realized P&L. Paper cash, open positions, and each closed trade persist in SQLite, including across restarts. A legacy DB without this ledger fails closed instead of resetting cash. The paper supervisor reads a fresh v2 paper heartbeat and never compares a real account balance with the paper starting bankroll.

This is a conservative top-of-book model. It assumes the quoted size remains available through execution and does not model queue priority, latency, market impact, or partial exits. If the best exit level has fewer contracts than held, it keeps the position open and records the reason; this can delay a stop loss. Market resolution still needs an explicit confirmed outcome source. A near-zero or near-one quote is not settlement, so unresolved positions remain open and marked. These limits make forward paper results a screening tool, not proof that live orders will earn the same return.

For a clean forward comparison:

1. Preserve each existing `data/trades.db` as historical evidence. Start the v2 simulator with a new, empty DB in an isolated checkout/profile. Do not copy legacy `trades.db` or a legacy `live_position_state` table into the new profile.
2. Keep `trading.paper_trading: true`. Use one baseline profile first. Re-run the full test suite before starting it. Observe the heartbeat's `accounting_version: 2`, cash, equity, and trade rows' `entry_fees` and `exit_fees`.
3. Collect at least 100 **resolved, independent games** over multiple weeks and assess net P&L after fees, max drawdown, fills rejected for thin books, time to close, and results by market type and league. Multiple correlated markets on one game count as one independent observation. Compare forecasts against actual outcomes and quoted market probabilities on data gathered strictly before each trade. Reserve a later period for validation before changing thresholds.
4. Do not tune Kelly, edge thresholds, or exit levels to these first two days of paper results. The baseline's 29 closed trades are too few and had biased fills. Retire a profile only on a fresh, fee-aware, executable-price comparison.

Read-only legacy audit (paths shown as examples):

```bash
python scripts/audit_paper.py --profile baseline=/path/to/baseline/trades.db \
  --profile balanced=/path/to/balanced/trades.db
```

The script labels legacy fee estimates separately from v2 net accounting. It does not connect to Polymarket or submit orders.
