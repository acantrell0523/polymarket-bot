#!/usr/bin/env python3
"""Read-only fee sensitivity for old paper fills and net accounting for new fills.

Usage: python scripts/audit_paper.py --profile baseline=/path/to/trades.db
Never opens a trading client and never changes the source database.
"""
import argparse
import sqlite3
from decimal import Decimal, ROUND_HALF_EVEN


def fee(quantity, price, coefficient=0.06):
    raw = quantity * coefficient * price * (1.0 - price)
    return float(Decimal(str(raw)).quantize(Decimal('0.01'), rounding=ROUND_HALF_EVEN))


def audit(path):
    connection = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
    connection.row_factory = sqlite3.Row
    try:
        columns = {row[1] for row in connection.execute('PRAGMA table_info(trades)')}
        version = 'accounting_version' if 'accounting_version' in columns else '1 AS accounting_version'
        fees = ('entry_fees, exit_fees' if 'entry_fees' in columns and 'exit_fees' in columns
                else '0 AS entry_fees, 0 AS exit_fees')
        rows = connection.execute(
            f'SELECT side, entry_price, close_price, quantity, realized_pnl, '
            f'{version}, {fees} FROM trades'
        )
        cohorts = {1: {'count': 0, 'reported': 0.0, 'fees': 0.0, 'adjusted': 0.0},
                   2: {'count': 0, 'reported': 0.0, 'fees': 0.0, 'adjusted': 0.0}}
        for row in rows:
            bucket = cohorts.get(row['accounting_version'])
            if bucket is None:
                continue
            costs = (fee(row['quantity'], row['entry_price']) +
                     fee(row['quantity'], row['close_price']) if row['accounting_version'] == 1
                     else row['entry_fees'] + row['exit_fees'])
            bucket['count'] += 1
            bucket['reported'] += row['realized_pnl']
            bucket['fees'] += costs
            bucket['adjusted'] += (row['realized_pnl'] - costs if row['accounting_version'] == 1
                                   else row['realized_pnl'])
        return cohorts
    finally:
        connection.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', action='append', required=True, metavar='NAME=DB_PATH')
    args = parser.parse_args()
    print('Profile | Accounting | Closed | Reported P&L | Est. round-trip fees | Fee-adjusted P&L')
    print('--- | --- | ---: | ---: | ---: | ---:')
    for spec in args.profile:
        if '=' not in spec:
            parser.error('--profile must be NAME=DB_PATH')
        name, path = spec.split('=', 1)
        for version, b in audit(path).items():
            if not b['count']:
                continue
            label = 'legacy fee estimate' if version == 1 else 'net v2'
            print(f"{name} | {label} | {b['count']} | ${b['reported']:.2f} | "
                  f"${b['fees']:.2f} | ${b['adjusted']:.2f}")
    print('\nLegacy figures subtract estimated fees only. Historic midpoint fills, exit liquidity, '
          'and restarts cannot be reconstructed, so the adjusted P&L is still optimistic.')


if __name__ == '__main__':
    main()
