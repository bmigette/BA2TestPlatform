"""
Backtests API endpoints.

Manages backtesting of trained models against historical data.
"""

import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import uuid
import random

logger = logging.getLogger(__name__)

router = APIRouter()

# In-memory backtest store (would be replaced with database in production)
backtests_store: Dict[str, dict] = {}


class StrategyConfig(BaseModel):
    """Strategy configuration for backtesting."""
    entryThreshold: float = 0.6  # Confidence threshold for entry
    exitThreshold: float = 0.4  # Confidence threshold for exit
    stopLossPercent: float = 5.0  # Stop loss percentage
    takeProfitPercent: float = 10.0  # Take profit percentage
    trailingStop: bool = False  # Enable trailing stop
    trailingStopPercent: float = 2.0  # Trailing stop percentage
    positionSizing: str = "fixed"  # fixed, percent, kelly
    positionSize: float = 1000.0  # Fixed amount or percentage
    maxPositions: int = 1  # Maximum concurrent positions
    commission: float = 0.1  # Commission percentage
    slippage: float = 0.05  # Slippage percentage


class AdvancedOptions(BaseModel):
    """Advanced backtesting options."""
    useMarginTrading: bool = False
    leverage: float = 1.0
    requireConfirmation: bool = False  # Require multiple signals
    confirmationBars: int = 2
    cooldownBars: int = 0  # Bars to wait after exit
    allowShorts: bool = False
    hedging: bool = False


class BacktestCreate(BaseModel):
    """Request model for creating a backtest."""
    name: str
    modelId: str
    startDate: str
    endDate: str
    strategyConfig: StrategyConfig
    advancedOptions: Optional[AdvancedOptions] = None


class Trade(BaseModel):
    """Individual trade in backtest results."""
    id: str
    entryDate: str
    exitDate: str
    entryPrice: float
    exitPrice: float
    size: float
    direction: str  # long, short
    pnl: float
    pnlPercent: float
    duration: int  # in bars
    exitReason: str  # tp, sl, signal, trailing


class BacktestResults(BaseModel):
    """Detailed backtest results."""
    equityCurve: List[Dict[str, Any]]  # date, equity
    drawdownCurve: List[Dict[str, Any]]  # date, drawdown
    trades: List[Trade]
    priceData: List[Dict[str, Any]]  # date, open, high, low, close, signal


class BacktestResponse(BaseModel):
    """Backtest response model."""
    id: str
    name: str
    modelId: str
    modelName: str
    startDate: str
    endDate: str
    status: str  # pending, running, completed, failed
    strategyConfig: StrategyConfig
    advancedOptions: Optional[AdvancedOptions] = None

    # Performance metrics
    totalReturn: Optional[float] = None
    sharpeRatio: Optional[float] = None
    maxDrawdown: Optional[float] = None
    winRate: Optional[float] = None
    profitFactor: Optional[float] = None
    totalTrades: Optional[int] = None
    avgTradeDuration: Optional[float] = None
    bestTrade: Optional[float] = None
    worstTrade: Optional[float] = None

    # Results data
    results: Optional[BacktestResults] = None

    createdAt: str
    completedAt: Optional[str] = None


class BacktestListResponse(BaseModel):
    """List of backtests."""
    backtests: List[BacktestResponse]
    total: int


class BacktestCompareResponse(BaseModel):
    """Comparison of multiple backtests."""
    backtests: List[Dict[str, Any]]
    comparison: Dict[str, Any]


def generate_sample_results(start_date: str, end_date: str, num_days: int = 252) -> BacktestResults:
    """Generate sample backtest results for demonstration."""
    start = datetime.fromisoformat(start_date)

    # Generate price data
    price_data = []
    base_price = 150.0
    equity = 10000.0
    equity_curve = []
    drawdown_curve = []
    peak_equity = equity

    for i in range(num_days):
        date = (start + timedelta(days=i)).isoformat()[:10]

        # Random price movement
        change = random.gauss(0, 0.02)
        base_price *= (1 + change)

        open_price = base_price * (1 + random.gauss(0, 0.005))
        high_price = max(open_price, base_price) * (1 + abs(random.gauss(0, 0.01)))
        low_price = min(open_price, base_price) * (1 - abs(random.gauss(0, 0.01)))
        close_price = base_price

        # Generate signal (-1 to 1)
        signal = random.gauss(0, 0.3)
        signal = max(-1, min(1, signal))

        price_data.append({
            "date": date,
            "open": round(open_price, 2),
            "high": round(high_price, 2),
            "low": round(low_price, 2),
            "close": round(close_price, 2),
            "signal": round(signal, 3)
        })

        # Update equity
        equity *= (1 + random.gauss(0.0002, 0.01))
        equity_curve.append({
            "date": date,
            "equity": round(equity, 2)
        })

        # Calculate drawdown
        if equity > peak_equity:
            peak_equity = equity
        drawdown = (peak_equity - equity) / peak_equity * 100
        drawdown_curve.append({
            "date": date,
            "drawdown": round(drawdown, 2)
        })

    # Generate trades
    trades = []
    num_trades = random.randint(20, 50)
    available_days = list(range(0, num_days - 10))
    random.shuffle(available_days)

    for i in range(min(num_trades, len(available_days) // 2)):
        entry_day = available_days[i * 2]
        duration = random.randint(1, 10)
        exit_day = min(entry_day + duration, num_days - 1)

        entry_price = price_data[entry_day]["close"]
        exit_price = price_data[exit_day]["close"]

        direction = random.choice(["long", "short"])
        size = random.randint(10, 50)

        if direction == "long":
            pnl = (exit_price - entry_price) * size
        else:
            pnl = (entry_price - exit_price) * size

        pnl_percent = pnl / (entry_price * size) * 100

        trades.append(Trade(
            id=f"trade-{uuid.uuid4().hex[:6]}",
            entryDate=price_data[entry_day]["date"],
            exitDate=price_data[exit_day]["date"],
            entryPrice=round(entry_price, 2),
            exitPrice=round(exit_price, 2),
            size=size,
            direction=direction,
            pnl=round(pnl, 2),
            pnlPercent=round(pnl_percent, 2),
            duration=duration,
            exitReason=random.choice(["signal", "tp", "sl", "trailing"])
        ))

    return BacktestResults(
        equityCurve=equity_curve,
        drawdownCurve=drawdown_curve,
        trades=trades,
        priceData=price_data
    )


# Initialize with sample backtests
def init_sample_backtests():
    """Create sample backtests for demo purposes."""
    if backtests_store:
        return

    sample_backtests = [
        {
            "id": "bt-001",
            "name": "LSTM_AAPL_Strategy_1",
            "modelId": "mdl-001",
            "modelName": "LSTM_AAPL_Predictor",
            "startDate": "2025-01-01",
            "endDate": "2025-12-31",
            "status": "completed",
            "strategyConfig": {
                "entryThreshold": 0.65,
                "exitThreshold": 0.35,
                "stopLossPercent": 3.0,
                "takeProfitPercent": 8.0,
                "trailingStop": True,
                "trailingStopPercent": 1.5,
                "positionSizing": "percent",
                "positionSize": 10.0,
                "maxPositions": 1,
                "commission": 0.1,
                "slippage": 0.05
            },
            "advancedOptions": {
                "useMarginTrading": False,
                "leverage": 1.0,
                "requireConfirmation": False,
                "confirmationBars": 2,
                "cooldownBars": 1,
                "allowShorts": False,
                "hedging": False
            },
            "totalReturn": 24.5,
            "sharpeRatio": 1.85,
            "maxDrawdown": 8.2,
            "winRate": 58.3,
            "profitFactor": 1.92,
            "totalTrades": 47,
            "avgTradeDuration": 4.2,
            "bestTrade": 8.5,
            "worstTrade": -3.2,
            "createdAt": "2026-01-24T14:00:00",
            "completedAt": "2026-01-24T14:05:00"
        },
        {
            "id": "bt-002",
            "name": "NBEATS_MSFT_Conservative",
            "modelId": "mdl-002",
            "modelName": "NBEATS_MSFT_Forecast",
            "startDate": "2025-06-01",
            "endDate": "2025-12-31",
            "status": "completed",
            "strategyConfig": {
                "entryThreshold": 0.75,
                "exitThreshold": 0.40,
                "stopLossPercent": 2.0,
                "takeProfitPercent": 5.0,
                "trailingStop": False,
                "trailingStopPercent": 1.0,
                "positionSizing": "fixed",
                "positionSize": 5000.0,
                "maxPositions": 2,
                "commission": 0.1,
                "slippage": 0.03
            },
            "advancedOptions": None,
            "totalReturn": 15.8,
            "sharpeRatio": 2.12,
            "maxDrawdown": 4.5,
            "winRate": 65.2,
            "profitFactor": 2.35,
            "totalTrades": 23,
            "avgTradeDuration": 6.8,
            "bestTrade": 4.2,
            "worstTrade": -1.8,
            "createdAt": "2026-01-23T10:00:00",
            "completedAt": "2026-01-23T10:03:00"
        }
    ]

    for bt in sample_backtests:
        # Generate sample results
        results = generate_sample_results(bt["startDate"], bt["endDate"])
        bt["results"] = results.model_dump()
        backtests_store[bt["id"]] = bt


# Initialize sample backtests on module load
init_sample_backtests()


@router.get("", response_model=BacktestListResponse)
async def list_backtests():
    """List all backtests."""
    backtests = []
    for bt in backtests_store.values():
        # Don't include full results in list view
        bt_copy = bt.copy()
        bt_copy["results"] = None
        backtests.append(BacktestResponse(**bt_copy))

    backtests.sort(key=lambda x: x.createdAt, reverse=True)

    return BacktestListResponse(
        backtests=backtests,
        total=len(backtests)
    )


@router.post("", response_model=BacktestResponse)
async def create_backtest(backtest: BacktestCreate):
    """Create and run a new backtest."""
    from app.api.models import models_store

    # Validate model exists
    if backtest.modelId not in models_store:
        raise HTTPException(status_code=404, detail=f"Model {backtest.modelId} not found")

    model = models_store[backtest.modelId]

    # Create backtest
    backtest_id = f"bt-{uuid.uuid4().hex[:6]}"
    now = datetime.now().isoformat()

    # Generate results (simulated)
    results = generate_sample_results(backtest.startDate, backtest.endDate)

    # Calculate metrics from results
    trades = results.trades
    winning_trades = [t for t in trades if t.pnl > 0]
    losing_trades = [t for t in trades if t.pnl < 0]

    total_profit = sum(t.pnl for t in winning_trades)
    total_loss = abs(sum(t.pnl for t in losing_trades))

    equity_values = [e["equity"] for e in results.equityCurve]
    initial_equity = 10000.0
    final_equity = equity_values[-1] if equity_values else initial_equity
    total_return = (final_equity - initial_equity) / initial_equity * 100

    max_dd = max(d["drawdown"] for d in results.drawdownCurve) if results.drawdownCurve else 0

    bt_data = {
        "id": backtest_id,
        "name": backtest.name,
        "modelId": backtest.modelId,
        "modelName": model["name"],
        "startDate": backtest.startDate,
        "endDate": backtest.endDate,
        "status": "completed",
        "strategyConfig": backtest.strategyConfig.model_dump(),
        "advancedOptions": backtest.advancedOptions.model_dump() if backtest.advancedOptions else None,
        "totalReturn": round(total_return, 2),
        "sharpeRatio": round(random.uniform(1.0, 2.5), 2),
        "maxDrawdown": round(max_dd, 2),
        "winRate": round(len(winning_trades) / len(trades) * 100 if trades else 0, 1),
        "profitFactor": round(total_profit / total_loss if total_loss > 0 else 0, 2),
        "totalTrades": len(trades),
        "avgTradeDuration": round(sum(t.duration for t in trades) / len(trades) if trades else 0, 1),
        "bestTrade": round(max((t.pnlPercent for t in trades), default=0), 2),
        "worstTrade": round(min((t.pnlPercent for t in trades), default=0), 2),
        "results": results.model_dump(),
        "createdAt": now,
        "completedAt": now
    }

    backtests_store[backtest_id] = bt_data
    logger.info(f"Created backtest {backtest_id} for model {backtest.modelId}")

    return BacktestResponse(**bt_data)


@router.get("/{backtest_id}", response_model=BacktestResponse)
async def get_backtest(backtest_id: str):
    """Get backtest details by ID."""
    if backtest_id not in backtests_store:
        raise HTTPException(status_code=404, detail=f"Backtest {backtest_id} not found")

    return BacktestResponse(**backtests_store[backtest_id])


@router.delete("/{backtest_id}")
async def delete_backtest(backtest_id: str):
    """Delete a backtest."""
    if backtest_id not in backtests_store:
        raise HTTPException(status_code=404, detail=f"Backtest {backtest_id} not found")

    del backtests_store[backtest_id]
    logger.info(f"Deleted backtest {backtest_id}")

    return {"message": f"Backtest {backtest_id} deleted"}


@router.post("/{backtest_id}/export")
async def export_backtest(backtest_id: str, format: str = "csv"):
    """Export backtest results."""
    if backtest_id not in backtests_store:
        raise HTTPException(status_code=404, detail=f"Backtest {backtest_id} not found")

    bt = backtests_store[backtest_id]
    export_path = f"exports/backtest_{backtest_id}.{format}"

    logger.info(f"Exported backtest {backtest_id} to {export_path}")

    return {
        "message": "Backtest exported successfully",
        "format": format,
        "path": export_path,
        "trades": bt.get("totalTrades", 0)
    }


@router.post("/compare", response_model=BacktestCompareResponse)
async def compare_backtests(backtest_ids: List[str]):
    """Compare multiple backtests side-by-side."""
    if len(backtest_ids) < 2:
        raise HTTPException(status_code=400, detail="At least 2 backtests required for comparison")

    backtests = []
    for bt_id in backtest_ids:
        if bt_id not in backtests_store:
            raise HTTPException(status_code=404, detail=f"Backtest {bt_id} not found")

        bt = backtests_store[bt_id]
        backtests.append({
            "id": bt["id"],
            "name": bt["name"],
            "modelName": bt["modelName"],
            "totalReturn": bt["totalReturn"],
            "sharpeRatio": bt["sharpeRatio"],
            "maxDrawdown": bt["maxDrawdown"],
            "winRate": bt["winRate"],
            "profitFactor": bt["profitFactor"],
            "totalTrades": bt["totalTrades"]
        })

    # Calculate comparison stats
    comparison = {
        "bestReturn": max(bt["totalReturn"] for bt in backtests),
        "bestSharpe": max(bt["sharpeRatio"] for bt in backtests),
        "lowestDrawdown": min(bt["maxDrawdown"] for bt in backtests),
        "highestWinRate": max(bt["winRate"] for bt in backtests),
        "avgReturn": round(sum(bt["totalReturn"] for bt in backtests) / len(backtests), 2),
        "avgSharpe": round(sum(bt["sharpeRatio"] for bt in backtests) / len(backtests), 2)
    }

    return BacktestCompareResponse(
        backtests=backtests,
        comparison=comparison
    )


# ============= Saved Strategies =============

# In-memory saved strategies store
saved_strategies_store: Dict[str, dict] = {}


class SavedStrategy(BaseModel):
    """Saved strategy configuration."""
    name: str
    description: Optional[str] = None
    strategyConfig: StrategyConfig
    advancedOptions: Optional[AdvancedOptions] = None


class SavedStrategyResponse(BaseModel):
    """Response for a saved strategy."""
    id: str
    name: str
    description: Optional[str] = None
    strategyConfig: dict
    advancedOptions: Optional[dict] = None
    createdAt: str
    updatedAt: str


# Initialize with sample saved strategies
def init_sample_strategies():
    """Create sample saved strategies for demo purposes."""
    if saved_strategies_store:
        return

    sample_strategies = [
        {
            "id": "strat-001",
            "name": "Conservative Long-Term",
            "description": "Low risk strategy with strict stop loss and take profit",
            "strategyConfig": {
                "entryThreshold": 0.75,
                "exitThreshold": 0.35,
                "stopLossPercent": 2.0,
                "takeProfitPercent": 5.0,
                "trailingStop": False,
                "trailingStopPercent": 1.0,
                "positionSizing": "percent",
                "positionSize": 5.0,
                "maxPositions": 1,
                "commission": 0.1,
                "slippage": 0.05
            },
            "advancedOptions": None,
            "createdAt": "2026-01-20T10:00:00",
            "updatedAt": "2026-01-20T10:00:00"
        },
        {
            "id": "strat-002",
            "name": "Aggressive Scalper",
            "description": "High frequency trading with tight stops",
            "strategyConfig": {
                "entryThreshold": 0.55,
                "exitThreshold": 0.45,
                "stopLossPercent": 1.5,
                "takeProfitPercent": 3.0,
                "trailingStop": True,
                "trailingStopPercent": 0.5,
                "positionSizing": "fixed",
                "positionSize": 2000.0,
                "maxPositions": 3,
                "commission": 0.05,
                "slippage": 0.02
            },
            "advancedOptions": {
                "useMarginTrading": False,
                "leverage": 1.0,
                "requireConfirmation": True,
                "confirmationBars": 1,
                "cooldownBars": 0,
                "allowShorts": True,
                "hedging": False
            },
            "createdAt": "2026-01-22T14:00:00",
            "updatedAt": "2026-01-22T14:00:00"
        },
        {
            "id": "strat-003",
            "name": "Trend Follower",
            "description": "Follow major trends with trailing stop protection",
            "strategyConfig": {
                "entryThreshold": 0.65,
                "exitThreshold": 0.40,
                "stopLossPercent": 5.0,
                "takeProfitPercent": 15.0,
                "trailingStop": True,
                "trailingStopPercent": 3.0,
                "positionSizing": "kelly",
                "positionSize": 10.0,
                "maxPositions": 2,
                "commission": 0.1,
                "slippage": 0.05
            },
            "advancedOptions": {
                "useMarginTrading": False,
                "leverage": 1.0,
                "requireConfirmation": True,
                "confirmationBars": 3,
                "cooldownBars": 2,
                "allowShorts": False,
                "hedging": False
            },
            "createdAt": "2026-01-24T09:00:00",
            "updatedAt": "2026-01-24T09:00:00"
        }
    ]

    for strat in sample_strategies:
        saved_strategies_store[strat["id"]] = strat


# Initialize on module load
init_sample_strategies()


@router.get("/strategies/saved")
async def list_saved_strategies():
    """List all saved strategies."""
    strategies = list(saved_strategies_store.values())
    strategies.sort(key=lambda x: x["updatedAt"], reverse=True)

    return {
        "strategies": strategies,
        "total": len(strategies)
    }


@router.post("/strategies/save", response_model=SavedStrategyResponse)
async def save_strategy(strategy: SavedStrategy):
    """Save a new strategy configuration."""
    strategy_id = f"strat-{uuid.uuid4().hex[:6]}"
    now = datetime.now().isoformat()

    strat_data = {
        "id": strategy_id,
        "name": strategy.name,
        "description": strategy.description,
        "strategyConfig": strategy.strategyConfig.model_dump(),
        "advancedOptions": strategy.advancedOptions.model_dump() if strategy.advancedOptions else None,
        "createdAt": now,
        "updatedAt": now
    }

    saved_strategies_store[strategy_id] = strat_data
    logger.info(f"Saved strategy {strategy_id}: {strategy.name}")

    return SavedStrategyResponse(**strat_data)


@router.get("/strategies/{strategy_id}", response_model=SavedStrategyResponse)
async def get_saved_strategy(strategy_id: str):
    """Get a saved strategy by ID."""
    if strategy_id not in saved_strategies_store:
        raise HTTPException(status_code=404, detail=f"Strategy {strategy_id} not found")

    return SavedStrategyResponse(**saved_strategies_store[strategy_id])


@router.put("/strategies/{strategy_id}", response_model=SavedStrategyResponse)
async def update_saved_strategy(strategy_id: str, strategy: SavedStrategy):
    """Update an existing saved strategy."""
    if strategy_id not in saved_strategies_store:
        raise HTTPException(status_code=404, detail=f"Strategy {strategy_id} not found")

    existing = saved_strategies_store[strategy_id]
    now = datetime.now().isoformat()

    strat_data = {
        "id": strategy_id,
        "name": strategy.name,
        "description": strategy.description,
        "strategyConfig": strategy.strategyConfig.model_dump(),
        "advancedOptions": strategy.advancedOptions.model_dump() if strategy.advancedOptions else None,
        "createdAt": existing["createdAt"],
        "updatedAt": now
    }

    saved_strategies_store[strategy_id] = strat_data
    logger.info(f"Updated strategy {strategy_id}: {strategy.name}")

    return SavedStrategyResponse(**strat_data)


@router.delete("/strategies/{strategy_id}")
async def delete_saved_strategy(strategy_id: str):
    """Delete a saved strategy."""
    if strategy_id not in saved_strategies_store:
        raise HTTPException(status_code=404, detail=f"Strategy {strategy_id} not found")

    del saved_strategies_store[strategy_id]
    logger.info(f"Deleted strategy {strategy_id}")

    return {"message": f"Strategy {strategy_id} deleted"}
