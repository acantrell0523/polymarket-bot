"""Shared data models for the trading bot and backtesting framework."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any


@dataclass
class OrderBookLevel:
    price: float
    size: float


@dataclass
class OrderBook:
    bids: List[OrderBookLevel] = field(default_factory=list)
    asks: List[OrderBookLevel] = field(default_factory=list)

    @property
    def best_bid(self) -> Optional[float]:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> Optional[float]:
        return self.asks[0].price if self.asks else None

    @property
    def mid_price(self) -> Optional[float]:
        if self.best_bid is not None and self.best_ask is not None:
            return (self.best_bid + self.best_ask) / 2
        return None

    @property
    def bid_depth(self) -> float:
        return sum(level.size for level in self.bids)

    @property
    def ask_depth(self) -> float:
        return sum(level.size for level in self.asks)


@dataclass
class MarketSnapshot:
    market_id: str
    token_id: str
    question: str
    price: float
    volume_24h: float
    liquidity: float
    order_book: OrderBook
    price_history: List[float]
    timestamp: datetime
    category: str = ""
    slug: str = ""
    hours_to_expiry: float = 72.0
    is_live: bool = False  # True if game is currently in progress
    outcomes: List[str] = field(default_factory=lambda: ["Yes", "No"])


@dataclass
class Signal:
    name: str
    value: float  # Signal's probability estimate component
    confidence: float  # 0.0 to 1.0
    direction: str = "neutral"  # "bullish", "bearish", "neutral"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TradeSignal:
    market_id: str
    token_id: str
    side: str  # "buy" or "sell"
    estimated_prob: float
    market_price: float
    edge: float
    position_size_usd: float
    signals: List[Signal] = field(default_factory=list)
    timestamp: Optional[datetime] = None
    slug: str = ""


@dataclass
class Position:
    market_id: str
    token_id: str
    side: str
    entry_price: float
    size_usd: float
    quantity: float
    estimated_prob: float
    entry_time: datetime
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    status: str = "open"  # "open", "closed"
    close_reason: str = ""
    close_price: float = 0.0
    close_time: Optional[datetime] = None
    realized_pnl: float = 0.0
    slug: str = ""


@dataclass
class Trade:
    market_id: str
    token_id: str
    side: str
    price: float
    quantity: float
    size_usd: float
    timestamp: datetime
    trade_type: str = "entry"  # "entry" or "exit"
    fees: float = 0.0
    slippage: float = 0.0
    is_paper: bool = True


@dataclass
class BacktestResult:
    trades: List[Trade]
    positions: List[Position]
    equity_curve: List[float]
    timestamps: List[datetime]
    initial_bankroll: float
    final_bankroll: float
    total_return: float
    total_return_pct: float
    win_rate: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    max_drawdown: float
    max_drawdown_pct: float
    sharpe_ratio: float
    profit_factor: float
    avg_trade_pnl: float
    avg_win: float
    avg_loss: float
    avg_duration_hours: float
    config: Dict[str, Any] = field(default_factory=dict)
