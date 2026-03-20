"""Structured JSON logging for the trading bot."""

import json
import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict, Optional


class TradingLogger:
    """Structured logger that outputs JSON-formatted log entries."""

    def __init__(self, name: str = "trading_bot", level: str = "INFO",
                 log_file: Optional[str] = None, console: bool = True):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        self.logger.handlers = []

        formatter = logging.Formatter("%(message)s")

        if console:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            fh = logging.FileHandler(log_file)
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

    def _log(self, level: str, event: str, data: Optional[Dict[str, Any]] = None):
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": level,
            "event": event,
        }
        if data:
            entry["data"] = data
        msg = json.dumps(entry, default=str)
        getattr(self.logger, level.lower())(msg)

    def info(self, event: str, data: Optional[Dict[str, Any]] = None):
        self._log("INFO", event, data)

    def warning(self, event: str, data: Optional[Dict[str, Any]] = None):
        self._log("WARNING", event, data)

    def error(self, event: str, data: Optional[Dict[str, Any]] = None):
        self._log("ERROR", event, data)

    def debug(self, event: str, data: Optional[Dict[str, Any]] = None):
        self._log("DEBUG", event, data)
