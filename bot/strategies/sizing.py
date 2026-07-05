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
        Fractional Kelly position sizing, fee- and spread-aware.

        Kelly for a binary bet:  f* = (p*b - q) / b
        where p = win probability, q = 1-p, and b = net odds per $1 staked.

        The cost basis is the EXECUTABLE price (best ask for buys, best bid for
        sells — set on the signal by compute_edge_breakdown) grossed up by the
        taker fee. Using the mid price and ignoring fees systematically
        oversizes: Kelly is very sensitive to edge, and fees+spread eat 2-4
        points of it.

        For a BUY of YES at cost c = ask*(1+fee):
            win  -> receive $1, profit (1-c) per share  =>  b = (1-c)/c
        For a SELL (short YES) with proceeds c = bid*(1-fee), the bet risks
        (1-c) per share to win c, and wins with probability q = 1-p:
            b = c/(1-c), win probability = 1-p

        kelly_fraction (default 0.5 = half-Kelly, 0.25 recommended for live)
        scales down f* because our p estimate is noisy — full Kelly on an
        overestimated edge is how bankrolls die.
        """
        p = signal.estimated_prob
        fee = signal.fee_rate if signal.fee_rate > 0 else getattr(self.config, "taker_fee_rate", 0.0)
        exec_price = signal.exec_price if signal.exec_price > 0 else signal.market_price

        if exec_price <= 0 or exec_price >= 1:
            return 0.0

        if signal.side == "buy":
            cost = min(exec_price * (1 + fee), 0.999)  # $ per share
            win_prob = p
        else:
            # Short YES: proceeds per share after fee; risk is (1-cost) to win cost.
            cost = max(exec_price * (1 - fee), 0.001)
            win_prob = 1 - p

        if signal.side == "buy":
            b = (1 - cost) / cost
        else:
            b = cost / (1 - cost)

        if b <= 0:
            return 0.0

        kelly_f = (win_prob * b - (1 - win_prob)) / b
        kelly_f = max(0, kelly_f)

        # Fractional Kelly: shrink toward zero to pay for estimation error.
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
