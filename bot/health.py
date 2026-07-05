"""Health monitoring: heartbeat file, API failure tracking, degraded mode.

Two consumers:

  * TradingBot writes a heartbeat (data/heartbeat.json) every scan cycle and
    records success/failure per external data source. After N consecutive
    failures a source enters "degraded" mode and the monitor hands back an
    exponential backoff so the loop slows down instead of hammering a dead API.

  * Supervisor reads the heartbeat on its 15-minute schedule; a stale
    heartbeat means the trading loop is hung or dead and triggers a Slack
    alert even though the loop itself can no longer report anything.
"""

import json
import os
import time
from typing import Dict, Optional

HEARTBEAT_FILENAME = "heartbeat.json"

# A heartbeat older than this is considered stale (trading loop dead/hung).
# The full-scan interval is 60s and sleep-mode checks are 120s, so 10 minutes
# of silence means something is actually wrong, not just a slow cycle.
STALE_HEARTBEAT_SECONDS = 600


class HealthMonitor:
    """Tracks liveness and per-source API health for the trading loop."""

    def __init__(
        self,
        data_dir: str,
        logger=None,
        alerter=None,
        failure_threshold: int = 5,
        base_backoff_seconds: float = 10.0,
        max_backoff_seconds: float = 300.0,
    ):
        self.data_dir = data_dir
        self.logger = logger
        self.alerter = alerter
        self.failure_threshold = failure_threshold
        self.base_backoff_seconds = base_backoff_seconds
        self.max_backoff_seconds = max_backoff_seconds

        # source name -> consecutive failure count
        self._failures: Dict[str, int] = {}
        # sources we've already alerted about (one alert per outage, not per cycle)
        self._alerted: set = set()

    # ------------------------------------------------------------------
    # Heartbeat
    # ------------------------------------------------------------------

    def beat(self, status: Optional[Dict] = None):
        """Write the heartbeat file atomically (write temp + rename).

        Atomic rename prevents the supervisor from ever reading a half-written
        JSON file.
        """
        payload = {"timestamp": time.time(), **(status or {})}
        path = os.path.join(self.data_dir, HEARTBEAT_FILENAME)
        tmp_path = path + ".tmp"
        try:
            os.makedirs(self.data_dir, exist_ok=True)
            with open(tmp_path, "w") as f:
                json.dump(payload, f)
            os.replace(tmp_path, path)
        except OSError:
            # A failed heartbeat write must never take down the trading loop.
            if self.logger:
                self.logger.warning("heartbeat_write_failed", {"path": path})

    @staticmethod
    def read_heartbeat(data_dir: str) -> Optional[Dict]:
        """Read the heartbeat file; None if missing or unparseable."""
        path = os.path.join(data_dir, HEARTBEAT_FILENAME)
        try:
            with open(path, "r") as f:
                return json.load(f)
        except (OSError, ValueError):
            return None

    @staticmethod
    def heartbeat_age_seconds(data_dir: str) -> Optional[float]:
        """Seconds since the last heartbeat; None if no heartbeat exists."""
        hb = HealthMonitor.read_heartbeat(data_dir)
        if not hb or "timestamp" not in hb:
            return None
        return max(0.0, time.time() - float(hb["timestamp"]))

    @staticmethod
    def is_heartbeat_stale(data_dir: str, max_age: float = STALE_HEARTBEAT_SECONDS) -> bool:
        """True when the heartbeat is missing or older than max_age.

        A missing heartbeat also counts as stale — callers who want a startup
        grace period should check read_heartbeat() first.
        """
        age = HealthMonitor.heartbeat_age_seconds(data_dir)
        return age is None or age > max_age

    # ------------------------------------------------------------------
    # Per-source API health / degraded mode
    # ------------------------------------------------------------------

    def record_success(self, source: str):
        """Clear failure state for a source (and its outage alert latch)."""
        if self._failures.get(source, 0) >= self.failure_threshold:
            if self.logger:
                self.logger.info("api_source_recovered", {"source": source})
            if self.alerter and source in self._alerted:
                try:
                    self.alerter._post("#36a64f", f":white_check_mark: `{source}` recovered")
                except Exception:
                    pass
        self._failures[source] = 0
        self._alerted.discard(source)

    def record_failure(self, source: str) -> float:
        """Record a failure. Returns backoff seconds (0.0 while healthy).

        Backoff doubles per failure beyond the threshold, capped at
        max_backoff_seconds, so a dead upstream degrades the loop to a slow
        poll instead of a tight retry storm.
        """
        count = self._failures.get(source, 0) + 1
        self._failures[source] = count

        if count < self.failure_threshold:
            return 0.0

        # Degraded: alert once per outage, then back off exponentially.
        if source not in self._alerted:
            self._alerted.add(source)
            if self.logger:
                self.logger.error("api_source_degraded", {
                    "source": source,
                    "consecutive_failures": count,
                })
            if self.alerter:
                try:
                    self.alerter._post("#ff0000",
                        f":warning: `{source}` degraded — "
                        f"{count} consecutive failures, backing off")
                except Exception:
                    pass

        exponent = count - self.failure_threshold
        backoff = self.base_backoff_seconds * (2 ** min(exponent, 10))
        return min(backoff, self.max_backoff_seconds)

    def is_degraded(self, source: str) -> bool:
        return self._failures.get(source, 0) >= self.failure_threshold

    def consecutive_failures(self, source: str) -> int:
        return self._failures.get(source, 0)
