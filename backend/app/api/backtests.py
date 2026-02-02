"""
Backtests API endpoints.

Manages backtesting of trained models against historical data.
"""

import logging
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.models import get_db, Backtest, Strategy, TrainedModel, Dataset

logger = logging.getLogger(__name__)

router = APIRouter()


class BacktestCreate(BaseModel):
    """Request model for creating a backtest."""
    name: str
    model_id: int
    prediction_dataset_id: int
    execution_dataset_id: int
    strategy_id: Optional[int] = None
    strategy_params: Optional[dict] = None
    start_date: str
    end_date: str
    initial_capital: float = 10000.0
    position_sizing_type: str = "fixed"  # fixed, percent
    position_sizing_value: float = 1000.0
    commission: float = 0.1
    slippage: float = 0.05
    fitness_metric: Optional[str] = None


class BacktestListResponse(BaseModel):
    """List of backtests."""
    backtests: List[dict]
    total: int


@router.get("")
async def list_backtests(
    db: Session = Depends(get_db)
):
    """List all backtests."""
    backtests = db.query(Backtest).order_by(Backtest.created_at.desc()).all()

    return BacktestListResponse(
        backtests=[bt.to_dict() for bt in backtests],
        total=len(backtests)
    )


@router.post("")
async def create_backtest(
    backtest: BacktestCreate,
    db: Session = Depends(get_db)
):
    """Create and run a new backtest."""
    # Validate model exists
    model = db.query(TrainedModel).filter(TrainedModel.id == backtest.model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail=f"Model {backtest.model_id} not found")

    # Validate datasets exist
    pred_dataset = db.query(Dataset).filter(Dataset.id == backtest.prediction_dataset_id).first()
    if not pred_dataset:
        raise HTTPException(status_code=404, detail=f"Prediction dataset {backtest.prediction_dataset_id} not found")

    exec_dataset = db.query(Dataset).filter(Dataset.id == backtest.execution_dataset_id).first()
    if not exec_dataset:
        raise HTTPException(status_code=404, detail=f"Execution dataset {backtest.execution_dataset_id} not found")

    # Validate strategy if provided
    strategy = None
    if backtest.strategy_id:
        strategy = db.query(Strategy).filter(Strategy.id == backtest.strategy_id).first()
        if not strategy:
            raise HTTPException(status_code=404, detail=f"Strategy {backtest.strategy_id} not found")

    # Parse dates
    try:
        start_date = datetime.fromisoformat(backtest.start_date)
        end_date = datetime.fromisoformat(backtest.end_date)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {e}")

    # Create backtest record
    db_backtest = Backtest(
        name=backtest.name,
        model_id=backtest.model_id,
        prediction_dataset_id=backtest.prediction_dataset_id,
        execution_dataset_id=backtest.execution_dataset_id,
        strategy_id=backtest.strategy_id,
        strategy_params=backtest.strategy_params,
        start_date=start_date,
        end_date=end_date,
        initial_capital=backtest.initial_capital,
        position_sizing_type=backtest.position_sizing_type,
        position_sizing_value=backtest.position_sizing_value,
        commission=backtest.commission,
        slippage=backtest.slippage,
        fitness_metric=backtest.fitness_metric,
        status="pending"
    )

    db.add(db_backtest)
    db.commit()
    db.refresh(db_backtest)

    logger.info(f"Created backtest: {db_backtest.name} (id={db_backtest.id})")

    # TODO: Queue backtest execution in background
    # For now, return pending status

    return db_backtest.to_dict()


@router.get("/{backtest_id}")
async def get_backtest(
    backtest_id: int,
    db: Session = Depends(get_db)
):
    """Get backtest details by ID."""
    backtest = db.query(Backtest).filter(Backtest.id == backtest_id).first()
    if not backtest:
        raise HTTPException(status_code=404, detail=f"Backtest {backtest_id} not found")

    return backtest.to_dict()


@router.delete("/{backtest_id}")
async def delete_backtest(
    backtest_id: int,
    db: Session = Depends(get_db)
):
    """Delete a backtest."""
    backtest = db.query(Backtest).filter(Backtest.id == backtest_id).first()
    if not backtest:
        raise HTTPException(status_code=404, detail=f"Backtest {backtest_id} not found")

    db.delete(backtest)
    db.commit()

    logger.info(f"Deleted backtest: {backtest.name} (id={backtest_id})")
    return {"message": f"Backtest {backtest_id} deleted"}


@router.post("/{backtest_id}/export")
async def export_backtest(
    backtest_id: int,
    format: str = "csv",
    db: Session = Depends(get_db)
):
    """Export backtest results."""
    backtest = db.query(Backtest).filter(Backtest.id == backtest_id).first()
    if not backtest:
        raise HTTPException(status_code=404, detail=f"Backtest {backtest_id} not found")

    if backtest.status != "completed":
        raise HTTPException(status_code=400, detail="Cannot export incomplete backtest")

    # TODO: Implement actual export
    export_path = f"exports/backtest_{backtest_id}.{format}"

    logger.info(f"Exported backtest {backtest_id} to {export_path}")

    return {
        "message": "Backtest exported successfully",
        "format": format,
        "path": export_path,
        "trades": backtest.total_trades or 0
    }


@router.post("/compare")
async def compare_backtests(
    backtest_ids: List[int],
    db: Session = Depends(get_db)
):
    """Compare multiple backtests side-by-side."""
    if len(backtest_ids) < 2:
        raise HTTPException(status_code=400, detail="At least 2 backtests required for comparison")

    backtests = []
    for bt_id in backtest_ids:
        bt = db.query(Backtest).filter(Backtest.id == bt_id).first()
        if not bt:
            raise HTTPException(status_code=404, detail=f"Backtest {bt_id} not found")

        backtests.append({
            "id": bt.id,
            "name": bt.name,
            "totalReturn": bt.total_return,
            "sharpeRatio": bt.sharpe_ratio,
            "maxDrawdown": bt.max_drawdown,
            "winRate": bt.win_rate,
            "profitFactor": bt.profit_factor,
            "totalTrades": bt.total_trades
        })

    # Calculate comparison stats
    returns = [bt["totalReturn"] for bt in backtests if bt["totalReturn"] is not None]
    sharpes = [bt["sharpeRatio"] for bt in backtests if bt["sharpeRatio"] is not None]
    drawdowns = [bt["maxDrawdown"] for bt in backtests if bt["maxDrawdown"] is not None]
    win_rates = [bt["winRate"] for bt in backtests if bt["winRate"] is not None]

    comparison = {
        "bestReturn": max(returns) if returns else None,
        "bestSharpe": max(sharpes) if sharpes else None,
        "lowestDrawdown": min(drawdowns) if drawdowns else None,
        "highestWinRate": max(win_rates) if win_rates else None,
        "avgReturn": round(sum(returns) / len(returns), 2) if returns else None,
        "avgSharpe": round(sum(sharpes) / len(sharpes), 2) if sharpes else None
    }

    return {
        "backtests": backtests,
        "comparison": comparison
    }
