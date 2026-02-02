"""
Backtest model for storing backtest results
"""

from sqlalchemy import Column, Integer, String, DateTime, JSON, ForeignKey, Float
from sqlalchemy.sql import func
from .database import Base


class Backtest(Base):
    """Backtest model"""

    __tablename__ = "backtests"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)

    # Model and datasets
    model_id = Column(Integer, ForeignKey("trained_models.id"), nullable=False)
    prediction_dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=True)
    execution_dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=True)

    # Strategy
    strategy_id = Column(Integer, ForeignKey("strategies.id"), nullable=True)
    strategy_params = Column(JSON, nullable=True)  # Specific param values used

    # Configuration
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    initial_capital = Column(Float, default=10000.0)
    position_sizing_type = Column(String(50), default="fixed")  # fixed/percent
    position_sizing_value = Column(Float, default=1000.0)
    commission = Column(Float, default=0.1)
    slippage = Column(Float, default=0.05)
    fitness_metric = Column(String(50), nullable=True)

    # Results
    status = Column(String(50), default="pending")  # pending/running/completed/failed
    results = Column(JSON, nullable=True)
    trades = Column(JSON, nullable=True)
    equity_curve = Column(JSON, nullable=True)
    drawdown_curve = Column(JSON, nullable=True)

    # Performance metrics
    total_return = Column(Float, nullable=True)
    sharpe_ratio = Column(Float, nullable=True)
    max_drawdown = Column(Float, nullable=True)
    win_rate = Column(Float, nullable=True)
    profit_factor = Column(Float, nullable=True)
    total_trades = Column(Integer, nullable=True)
    winning_trades = Column(Integer, nullable=True)
    losing_trades = Column(Integer, nullable=True)
    avg_trade_duration = Column(Float, nullable=True)
    final_equity = Column(Float, nullable=True)
    best_trade = Column(Float, nullable=True)
    worst_trade = Column(Float, nullable=True)

    error_message = Column(String(1000), nullable=True)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    def __repr__(self):
        return f"<Backtest(id={self.id}, name='{self.name}', return={self.total_return})>"

    def to_dict(self):
        """Convert to dictionary for API response"""
        return {
            "id": self.id,
            "name": self.name,
            "modelId": self.model_id,
            "predictionDatasetId": self.prediction_dataset_id,
            "executionDatasetId": self.execution_dataset_id,
            "strategyId": self.strategy_id,
            "strategyParams": self.strategy_params,
            "startDate": self.start_date.isoformat() if self.start_date else None,
            "endDate": self.end_date.isoformat() if self.end_date else None,
            "initialCapital": self.initial_capital,
            "positionSizingType": self.position_sizing_type,
            "positionSizingValue": self.position_sizing_value,
            "commission": self.commission,
            "slippage": self.slippage,
            "fitnessMetric": self.fitness_metric,
            "status": self.status,
            "results": self.results,
            "trades": self.trades,
            "equityCurve": self.equity_curve,
            "drawdownCurve": self.drawdown_curve,
            "totalReturn": self.total_return,
            "sharpeRatio": self.sharpe_ratio,
            "maxDrawdown": self.max_drawdown,
            "winRate": self.win_rate,
            "profitFactor": self.profit_factor,
            "totalTrades": self.total_trades,
            "winningTrades": self.winning_trades,
            "losingTrades": self.losing_trades,
            "avgTradeDuration": self.avg_trade_duration,
            "finalEquity": self.final_equity,
            "bestTrade": self.best_trade,
            "worstTrade": self.worst_trade,
            "errorMessage": self.error_message,
            "createdAt": self.created_at.isoformat() if self.created_at else None,
            "startedAt": self.started_at.isoformat() if self.started_at else None,
            "completedAt": self.completed_at.isoformat() if self.completed_at else None,
        }
