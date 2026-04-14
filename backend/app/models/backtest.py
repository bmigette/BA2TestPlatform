"""
Backtest model for storing backtest results
"""

from sqlalchemy import Column, Integer, String, Text, DateTime, JSON, ForeignKey, Float, Boolean
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
    description = Column(Text, nullable=True)  # User/agent notes about the backtest

    # Results
    status = Column(String(50), default="pending")  # pending/running/completed/failed
    results = Column(JSON, nullable=True)
    trades = Column(JSON, nullable=True)
    equity_curve = Column(JSON, nullable=True)
    drawdown_curve = Column(JSON, nullable=True)

    # Performance metrics (from backtesting.py)
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

    # Additional metrics from backtesting.py
    exposure_time = Column(Float, nullable=True)  # % of time in position
    buy_hold_return = Column(Float, nullable=True)  # Benchmark B&H return
    annualized_return = Column(Float, nullable=True)  # Annualized return %
    volatility = Column(Float, nullable=True)  # Annualized volatility %
    sortino_ratio = Column(Float, nullable=True)  # Downside risk-adjusted return
    calmar_ratio = Column(Float, nullable=True)  # Return / Max Drawdown
    sqn = Column(Float, nullable=True)  # System Quality Number
    expectancy = Column(Float, nullable=True)  # Average expected return per trade
    avg_drawdown = Column(Float, nullable=True)  # Average drawdown %
    max_drawdown_duration = Column(Float, nullable=True)  # Max DD duration in days
    avg_trade = Column(Float, nullable=True)  # Average trade return % (geometric)
    equity_peak = Column(Float, nullable=True)  # Peak equity reached

    error_message = Column(String(1000), nullable=True)

    # Save status
    is_saved = Column(Boolean, default=False)  # Whether backtest has been explicitly saved with a name

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    def __repr__(self):
        return f"<Backtest(id={self.id}, name='{self.name}', return={self.total_return})>"

    def _transform_trades_for_frontend(self):
        """Transform trade data to match frontend expected format."""
        if not self.trades:
            return []

        transformed = []
        for i, trade in enumerate(self.trades):
            # Map backend field names to frontend expected names
            # Backend: entry_time, exit_time, direction (buy/sell), pnl_pct, bars_held
            # Frontend: entryDate, exitDate, direction (long/short), pnlPercent, duration
            direction = trade.get('direction', 'buy')
            if direction == 'buy':
                direction = 'long'
            elif direction == 'sell':
                direction = 'short'

            transformed.append({
                'id': i + 1,
                'entryDate': trade.get('entry_time', ''),
                'exitDate': trade.get('exit_time', ''),
                'entryPrice': trade.get('entry_price', 0),
                'exitPrice': trade.get('exit_price', 0),
                'size': trade.get('size', 0),
                'direction': direction,
                'pnl': trade.get('pnl', 0),
                'pnlPercent': trade.get('pnl_pct', 0),
                'duration': trade.get('bars_held', 0),
                'exitReason': trade.get('exit_reason', 'unknown'),
            })
        return transformed

    def to_summary_dict(self):
        """Convert to lightweight dictionary for list endpoints (no curves/trades)."""
        return {
            "id": self.id,
            "name": self.name,
            "modelId": self.model_id,
            "predictionDatasetId": self.prediction_dataset_id,
            "executionDatasetId": self.execution_dataset_id,
            "strategyId": self.strategy_id,
            "startDate": self.start_date.isoformat() if self.start_date else None,
            "endDate": self.end_date.isoformat() if self.end_date else None,
            "initialCapital": self.initial_capital,
            "fitnessMetric": self.fitness_metric,
            "status": self.status,
            "totalReturn": self.total_return,
            "sharpeRatio": self.sharpe_ratio,
            "maxDrawdown": self.max_drawdown,
            "winRate": self.win_rate,
            "profitFactor": self.profit_factor,
            "totalTrades": self.total_trades,
            "avgTradeDuration": self.avg_trade_duration,
            "finalEquity": self.final_equity,
            "bestTrade": self.best_trade,
            "worstTrade": self.worst_trade,
            "errorMessage": self.error_message,
            "description": self.description,
            "isSaved": self.is_saved or False,
            "createdAt": self.created_at.isoformat() if self.created_at else None,
            "completedAt": self.completed_at.isoformat() if self.completed_at else None,
        }

    def to_dict(self):
        """Convert to full dictionary for detail endpoints (includes curves/trades)."""
        # Transform trades to frontend format
        transformed_trades = self._transform_trades_for_frontend()

        # Build results object that frontend expects
        results = {
            "equityCurve": self.equity_curve or [],
            "drawdownCurve": self.drawdown_curve or [],
            "trades": transformed_trades,
            "priceData": [],  # Price data would need to be fetched separately
        }

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
            "results": results,  # Nested results object for frontend
            "trades": transformed_trades,  # Also at top level for backwards compat
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
            # Additional metrics from backtesting.py
            "exposureTime": self.exposure_time,
            "buyHoldReturn": self.buy_hold_return,
            "annualizedReturn": self.annualized_return,
            "volatility": self.volatility,
            "sortinoRatio": self.sortino_ratio,
            "calmarRatio": self.calmar_ratio,
            "sqn": self.sqn,
            "expectancy": self.expectancy,
            "avgDrawdown": self.avg_drawdown,
            "maxDrawdownDuration": self.max_drawdown_duration,
            "avgTrade": self.avg_trade,
            "equityPeak": self.equity_peak,
            "errorMessage": self.error_message,
            "description": self.description,
            "isSaved": self.is_saved or False,
            "createdAt": self.created_at.isoformat() if self.created_at else None,
            "startedAt": self.started_at.isoformat() if self.started_at else None,
            "completedAt": self.completed_at.isoformat() if self.completed_at else None,
        }
