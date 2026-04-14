"""
Backtests API endpoints.

Manages backtesting of trained models against historical data.
"""

import logging
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session, defer

from app.models import get_db, Backtest, Strategy, TrainedModel, Dataset

logger = logging.getLogger(__name__)

router = APIRouter()


class BacktestCreate(BaseModel):
    """Request model for creating a backtest."""
    name: str
    model_id: str  # String model ID like "mdl-abc123"
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
    """List all backtests (summary only, no curves/trades)."""
    from sqlalchemy import text

    # Use raw SQL to avoid loading huge blob columns (equity_curve, drawdown_curve, trades
    # can be 2-5MB each; with 200+ backtests the ORM query loads 1GB+ even with defer)
    result = db.execute(text("""
        SELECT id, name, model_id, prediction_dataset_id, execution_dataset_id,
               strategy_id, start_date, end_date, initial_capital, fitness_metric,
               status, total_return, sharpe_ratio, max_drawdown, win_rate,
               profit_factor, total_trades, avg_trade_duration, final_equity,
               best_trade, worst_trade, error_message, is_saved, created_at, completed_at
        FROM backtests
        ORDER BY created_at DESC
    """))

    backtests = []
    for row in result:
        backtests.append({
            "id": row[0], "name": row[1], "modelId": row[2],
            "predictionDatasetId": row[3], "executionDatasetId": row[4],
            "strategyId": row[5],
            "startDate": row[6].isoformat() if row[6] else None,
            "endDate": row[7].isoformat() if row[7] else None,
            "initialCapital": row[8], "fitnessMetric": row[9],
            "status": row[10], "totalReturn": row[11],
            "sharpeRatio": row[12], "maxDrawdown": row[13],
            "winRate": row[14], "profitFactor": row[15],
            "totalTrades": row[16], "avgTradeDuration": row[17],
            "finalEquity": row[18], "bestTrade": row[19],
            "worstTrade": row[20], "errorMessage": row[21],
            "isSaved": row[22] or False,
            "createdAt": row[23].isoformat() if row[23] else None,
            "completedAt": row[24].isoformat() if row[24] else None,
        })

    return {"backtests": backtests, "total": len(backtests)}


@router.post("")
async def create_backtest(
    backtest: BacktestCreate,
    db: Session = Depends(get_db)
):
    """Create and run a new backtest."""
    # Validate model exists (lookup by model_id string, not integer id)
    model = db.query(TrainedModel).filter(TrainedModel.model_id == backtest.model_id).first()
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

    # Create backtest record (use integer id for database FK)
    db_backtest = Backtest(
        name=backtest.name,
        model_id=model.id,  # Use the integer database id
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

    # Queue backtest execution in background
    from app.services.task_queue import get_task_queue
    task_queue = get_task_queue()
    task_id = task_queue.queue_task(
        task_type='backtest',
        name=f'Backtest: {db_backtest.name}',
        payload={'backtest_id': db_backtest.id},
        description=f'Running backtest on model {backtest.model_id}'
    )

    logger.info(f"Queued backtest task: {task_id}")

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
    import json
    import csv
    from pathlib import Path
    import io

    backtest = db.query(Backtest).filter(Backtest.id == backtest_id).first()
    if not backtest:
        raise HTTPException(status_code=404, detail=f"Backtest {backtest_id} not found")

    if backtest.status != "completed":
        raise HTTPException(status_code=400, detail="Cannot export incomplete backtest")

    # Ensure exports directory exists
    exports_dir = Path("exports")
    exports_dir.mkdir(exist_ok=True)

    # Build export data
    export_data = {
        "backtest": {
            "id": backtest.id,
            "name": backtest.name,
            "model_id": backtest.model_id,
            "start_date": backtest.start_date.isoformat() if backtest.start_date else None,
            "end_date": backtest.end_date.isoformat() if backtest.end_date else None,
            "initial_capital": backtest.initial_capital,
            "final_equity": backtest.final_equity,
            "total_return": backtest.total_return,
            "sharpe_ratio": backtest.sharpe_ratio,
            "max_drawdown": backtest.max_drawdown,
            "win_rate": backtest.win_rate,
            "profit_factor": backtest.profit_factor,
            "total_trades": backtest.total_trades,
            "winning_trades": backtest.winning_trades,
            "losing_trades": backtest.losing_trades,
        },
        "trades": backtest.trades or [],
        "equity_curve": backtest.equity_curve or [],
    }

    if format == "json":
        export_path = exports_dir / f"backtest_{backtest_id}.json"
        with open(export_path, 'w') as f:
            json.dump(export_data, f, indent=2)
    elif format == "csv":
        # Export trades as CSV
        trades_path = exports_dir / f"backtest_{backtest_id}_trades.csv"
        equity_path = exports_dir / f"backtest_{backtest_id}_equity.csv"

        # Write trades
        trades = backtest.trades or []
        if trades:
            with open(trades_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=trades[0].keys())
                writer.writeheader()
                writer.writerows(trades)

        # Write equity curve
        equity_curve = backtest.equity_curve or []
        if equity_curve:
            with open(equity_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=equity_curve[0].keys())
                writer.writeheader()
                writer.writerows(equity_curve)

        export_path = trades_path  # Return trades path as main export
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {format}. Use 'csv' or 'json'")

    logger.info(f"Exported backtest {backtest_id} to {export_path}")

    return {
        "message": "Backtest exported successfully",
        "format": format,
        "path": str(export_path),
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


class BacktestSave(BaseModel):
    """Request model for saving a backtest."""
    name: str


@router.post("/{backtest_id}/save")
async def save_backtest(
    backtest_id: int,
    save_data: BacktestSave,
    db: Session = Depends(get_db)
):
    """Save a backtest with a custom name (marks it as saved)."""
    backtest = db.query(Backtest).filter(Backtest.id == backtest_id).first()
    if not backtest:
        raise HTTPException(status_code=404, detail=f"Backtest {backtest_id} not found")

    backtest.name = save_data.name
    backtest.is_saved = True
    db.commit()
    db.refresh(backtest)

    logger.info(f"Saved backtest: {backtest.name} (id={backtest_id})")
    return backtest.to_dict()


@router.delete("/unsaved")
async def clear_unsaved_backtests(
    db: Session = Depends(get_db)
):
    """Delete all unsaved backtests."""
    unsaved = db.query(Backtest).filter(Backtest.is_saved == False).all()
    count = len(unsaved)

    for bt in unsaved:
        db.delete(bt)

    db.commit()

    logger.info(f"Cleared {count} unsaved backtests")
    return {"message": f"Deleted {count} unsaved backtests", "count": count}
