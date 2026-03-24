"""Kelly Criterion & fixed fractional position sizing."""

from utils.models import TradeSignal
from utils.config import TradingConfig


class PositionSizer:
    """Determines position size for each trade."""

    def __init__(self, config: TradingConfig):
        self.config = config

    def size_position(self, signal: TradeSignal, bankroll: float, current_exposure: float) -> float:
        """
        Calculate position size in USD.

        Args:
            signal: The trade signal with edge and probability estimates
            bankroll: Current total bankroll
            current_exposure: Current total portfolio exposure in USD

        Returns:
            Position size in USD (0 if trade should be skipped)
        """
        available = self.config.max_portfolio_exposure_usd - current_exposure
        if available <= 0:
            return 0.0

        if self.config.position_sizing_method == "tiered_kelly":
            size = self._tiered_kelly_size(signal)
        elif self.config.position_sizing_method == "kelly":
            size = self._kelly_size(signal, bankroll)
        else:
            size = self._fixed_fractional_size(bankroll)

        # Apply constraints
        size = min(size, self.config.max_position_size_usd)
        size = min(size, available)
        size = min(size, bankroll * 0.5)  # Never risk more than 50% of bankroll

        if size < self.config.min_position_size_usd:
            return 0.0

        return round(size, 2)

    def _kelly_size(self, signal: TradeSignal, bankroll: float) -> float:
        """
        Kelly Criterion position sizing.

        f* = kelly_fraction * (p * b - q) / b

        where:
            p = estimated win probability
            q = 1 - p
            b = net odds = (1 - price) / price for binary markets
        """
        p = signal.estimated_prob
        q = 1 - p
        price = signal.market_price

        if price <= 0 or price >= 1:
            return 0.0

        # Net odds (payout ratio)
        if signal.side == "buy":
            b = (1 - price) / price
        else:
            b = price / (1 - price)

        if b <= 0:
            return 0.0

        kelly_f = (p * b - q) / b
        kelly_f = max(0, kelly_f)

        # Apply Kelly fraction (e.g., half-Kelly)
        kelly_f *= self.config.kelly_fraction

        return kelly_f * bankroll

    def _tiered_kelly_size(self, signal: TradeSignal) -> float:
        """Tiered Kelly: go bigger on high conviction.

        5-7% edge   → $20
        7-10% edge  → $30
        10-15% edge → $40
        15%+ edge   → $50 (max)
        """
        abs_edge = abs(signal.edge) * 100  # convert to percentage
        if abs_edge >= 15:
            return 50.0
        elif abs_edge >= 10:
            return 40.0
        elif abs_edge >= 7:
            return 30.0
        else:
            return 20.0

    def _fixed_fractional_size(self, bankroll: float) -> float:
        """Fixed fractional: flat percentage of bankroll."""
        return self.config.fixed_fraction * bankroll
