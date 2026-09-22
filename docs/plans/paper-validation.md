# Paper validation repair

Scope: honest forward paper results, with the existing strategy unchanged.

- [x] Simulate entries at executable prices, integer contracts and visible depth; size shorts by collateral.
- [x] Book both fees, reserve collateral, persist paper cash and positions atomically; reject legacy state instead of resetting losses.
- [x] Use executable exit quotes and sufficient depth; isolate paper supervisor from live balances.
- [x] Add regression tests and a read-only fee sensitivity audit; document a fresh forward evaluation.
- [x] Review the patch and prepare a draft PR. Do not launch, fund, or enable live trading.

Historical results cannot be repaired into a backtest: exit books, queue position and fill timing were not recorded. Fee sensitivity is diagnostic only. Any durable profitability claim requires new forward data.
