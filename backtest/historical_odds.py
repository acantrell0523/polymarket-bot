"""Time-aware consensus odds for backtests — closes the "0 trades" gap.

The live pipeline refuses to trade a sports market unless odds_value_signal
gets external consensus from OddsCache (the external validation gate). In a
backtest there are no live API calls, so the gate blocked every replayed
trade. This module provides a drop-in OddsCache replacement whose answers
come from historical data instead of HTTP:

  * HistoricalOddsCache.from_db()   — serves espn_consensus_prob values stored
    in the historical_snapshots table (populated by scripts/ingest_historical.py
    when a consensus source is available for the ingested window).

  * HistoricalOddsCache.synthetic_from_market_data() — for synthetic GBM
    backtests: generates a noisy consensus around each snapshot's own price
    plus a persistent per-market bias. This has NO lookahead (it only sees the
    price at the same timestamp) and exists to exercise the full pipeline —
    gate, ranking, sizing, risk — not to prove profitability.

The cache is time-aware: BacktestEngine calls set_time(snapshot.timestamp)
before each detect_edge() so lookups only ever return consensus values at or
before the replay clock. That prevents lookahead bias when real historical
consensus is loaded.

The `is_historical` flag tells odds_value_signal / the estimator to skip all
SQLite writes (line_movement, signal_log) during replays.
"""

import os
import random
from bisect import bisect_right
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

_DEFAULT_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "trades.db"
)


class HistoricalOddsCache:
    """OddsCache-compatible consensus source backed by historical data.

    Implements the subset of the OddsCache interface the signal layer uses:
    enabled, get_probability_for_slug(), get_consensus_odds(), _team_matches().
    """

    #: signals check this to skip live-DB side effects during replays
    is_historical = True

    def __init__(self, series: Dict[str, List[Tuple[float, float, int]]]):
        """
        Args:
            series: slug -> time-ordered list of (unix_ts, consensus_prob, num_books)
        """
        self.enabled = True
        self._series: Dict[str, List[Tuple[float, float, int]]] = {}
        for slug, points in series.items():
            self._series[slug] = sorted(points, key=lambda p: p[0])
        self._now: Optional[float] = None  # replay clock (unix ts); None = no limit

    # ------------------------------------------------------------------
    # Replay clock
    # ------------------------------------------------------------------

    def set_time(self, when: datetime):
        """Advance the replay clock. Lookups return data at or before this time."""
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        self._now = when.timestamp()

    def _lookup(self, slug: str) -> Optional[Tuple[float, int]]:
        points = self._series.get(slug)
        if not points:
            return None
        if self._now is None:
            _, prob, books = points[-1]
            return prob, books
        # Latest point with ts <= replay clock (no lookahead).
        timestamps = [p[0] for p in points]
        idx = bisect_right(timestamps, self._now) - 1
        if idx < 0:
            return None
        _, prob, books = points[idx]
        return prob, books

    # ------------------------------------------------------------------
    # OddsCache-compatible interface
    # ------------------------------------------------------------------

    def get_probability_for_slug(self, slug: str) -> Optional[Tuple[float, int]]:
        """(consensus_prob, num_books) at the replay clock, or None."""
        return self._lookup(slug)

    def get_consensus_odds(self, slug: str) -> Optional[Dict]:
        """Minimal consensus dict — no sharp-book data in historical mode."""
        result = self._lookup(slug)
        if result is None:
            return None
        prob, books = result
        return {
            "consensus_prob": prob,
            "num_books": books,
            "books_used": "historical",
            "sharp_probs": {},
        }

    @staticmethod
    def _team_matches(abbr: str, full_name: str) -> bool:
        """No team metadata in historical mode — sharp-blend never activates."""
        return False

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_db(cls, db_path: Optional[str] = None) -> "HistoricalOddsCache":
        """Build from historical_snapshots rows with a real consensus value.

        Rows with espn_consensus_prob <= 0 carry no information (the CLOB
        ingest writes 0.0 when no consensus source was available) and are
        skipped. An empty cache is valid — every lookup returns None and the
        external validation gate keeps blocking, exactly as designed.
        """
        path = db_path or _DEFAULT_DB_PATH
        series: Dict[str, List[Tuple[float, float, int]]] = {}
        if os.path.exists(path):
            try:
                import sqlite3
                conn = sqlite3.connect(path)
                rows = conn.execute(
                    "SELECT slug, timestamp, espn_consensus_prob, num_books "
                    "FROM historical_snapshots "
                    "WHERE espn_consensus_prob > 0 "
                    "ORDER BY slug, timestamp"
                ).fetchall()
                conn.close()
                for slug, ts, prob, books in rows:
                    series.setdefault(slug, []).append(
                        (float(ts), float(prob), max(int(books or 0), 2))
                    )
            except Exception:
                series = {}
        return cls(series)

    @classmethod
    def synthetic_from_market_data(
        cls,
        market_data: List[List],
        noise_sigma: float = 0.03,
        bias_sigma: float = 0.04,
        num_books: int = 4,
        seed: int = 42,
    ) -> "HistoricalOddsCache":
        """Fabricate consensus for synthetic backtests (pipeline smoke test).

        consensus(t) = clamp(price(t) + market_bias + noise(t))

        The per-market bias makes some markets look persistently mispriced so
        the pipeline actually fires; the per-snapshot noise makes edges come
        and go. Only the price at the SAME timestamp is used — no lookahead.
        """
        rng = random.Random(seed)
        series: Dict[str, List[Tuple[float, float, int]]] = {}
        for market_snapshots in market_data:
            if not market_snapshots:
                continue
            bias = rng.gauss(0.0, bias_sigma)
            points = []
            for snap in market_snapshots:
                consensus = snap.price + bias + rng.gauss(0.0, noise_sigma)
                consensus = min(max(consensus, 0.02), 0.98)
                ts = snap.timestamp
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                points.append((ts.timestamp(), consensus, num_books))
            slug = market_snapshots[0].slug or market_snapshots[0].market_id
            series[slug] = points
        return cls(series)

    def __len__(self) -> int:
        return sum(len(p) for p in self._series.values())
