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
    """Request model for creating a backtest.

    Two engines share this endpoint, discriminated by ``engine``:

      * ``engine="ml"`` (default) — the legacy model-driven path. Requires ``model_id`` +
        ``prediction_dataset_id`` + ``execution_dataset_id`` (validated in the route so the
        existing ML behaviour is byte-for-byte unchanged).
      * ``engine="daily_expert"`` — the daily multi-asset expert engine. Requires ``expert``
        ({"class", "settings"}) + ``universe`` ({"mode", "symbols", "screener_settings"}).
        The ML model/dataset fields are unused (and not required) on this path.
    """
    name: str
    engine: str = "ml"  # "ml" (default, legacy) | "daily_expert"
    # ML-engine fields (required only when engine == "ml"; validated in the route).
    model_id: Optional[str] = None  # String model ID like "mdl-abc123"
    prediction_dataset_id: Optional[int] = None
    execution_dataset_id: Optional[int] = None
    strategy_id: Optional[int] = None
    strategy_params: Optional[dict] = None
    # daily_expert-engine fields (required only when engine == "daily_expert").
    expert: Optional[dict] = None  # {"class": "FMPRating", "settings": {...}}
    universe: Optional[dict] = None  # {"mode": "static"|"screener", "symbols": [...], "screener_settings": {...}}
    # Shared trading parameters.
    start_date: str
    end_date: str
    initial_capital: float = 10000.0
    position_sizing_type: str = "fixed"  # fixed, percent
    position_sizing_value: float = 1000.0
    commission: float = 0.1
    slippage: float = 0.05
    fitness_metric: Optional[str] = None
    # daily_expert engine knobs (used only on that path; sensible explicit values required).
    fill_model: Optional[str] = None       # "next_bar_open" | "same_bar_close"
    seed: Optional[int] = None
    warmup_days: Optional[int] = None


class DailyExpertSpec(BaseModel):
    """One expert in a daily backtest: a ba2_experts class name + optional setting overrides."""
    class_name: str  # serialised as "class" below via alias
    settings: Optional[dict] = None

    class Config:
        fields = {"class_name": "class"}


class DailyBacktestCreate(BaseModel):
    """Request model for creating a daily multi-asset (expert) backtest.

    No-defaults rule: every trading parameter is explicit. ``experts`` is a list of either
    bare class-name strings or ``{"class": ..., "settings": {...}}`` objects. Datasets/model
    are NOT used by the daily engine (the universe is ``enabled_instruments``)."""
    name: str
    enabled_instruments: List[str]
    experts: List[dict]  # [{"class": "FMPEarningsDrift", "settings": {...}}] or ["FMPEarningsDrift"]
    start_date: str
    end_date: str
    initial_capital: float
    commission: float        # flat $ per fill (BacktestAccount commission_per_trade)
    slippage: float          # slippage in basis points (BacktestAccount slippage_bps)
    fill_model: str          # "next_bar_open" | "same_bar_close"
    seed: int
    fitness_metric: Optional[str] = None
    warmup_days: Optional[int] = None


class BacktestListResponse(BaseModel):
    """List of backtests."""
    backtests: List[dict]
    total: int


@router.get("")
async def list_backtests(
    expert: Optional[str] = None,
    optimization_id: Optional[int] = None,
    saved: Optional[bool] = None,
    db: Session = Depends(get_db)
):
    """List all backtests (summary only, no curves/trades).

    Optional filters (applied only when provided):
      * ``expert``         — only runs of that expert (``Backtest.expert_name``).
      * ``optimization_id``— only runs belonging to that optimization job.
      * ``saved``          — only saved (``True``) / only unsaved (``False``) runs.
    """
    from sqlalchemy import text

    # Use raw SQL to avoid loading huge blob columns (equity_curve, drawdown_curve, trades
    # can be 2-5MB each; with 200+ backtests the ORM query loads 1GB+ even with defer)
    # Check if description column exists (migration may not have run yet)
    col_check = db.execute(text("PRAGMA table_info(backtests)"))
    columns = [r[1] for r in col_check]
    has_description = 'description' in columns

    desc_col = ", description" if has_description else ""

    # Build the optional WHERE clause from the provided filters (parameterised — never
    # string-interpolate user input).
    where_clauses = []
    params: dict = {}
    if expert is not None:
        where_clauses.append("b.expert_name = :expert")
        params["expert"] = expert
    if optimization_id is not None:
        where_clauses.append("b.optimization_id = :optimization_id")
        params["optimization_id"] = optimization_id
    if saved is not None:
        where_clauses.append("b.is_saved = :saved")
        params["saved"] = 1 if saved else 0
    where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

    result = db.execute(text(f"""
        SELECT b.id, b.name, b.model_id, b.prediction_dataset_id, b.execution_dataset_id,
               b.strategy_id, b.start_date, b.end_date, b.initial_capital, b.fitness_metric,
               b.status, b.total_return, b.sharpe_ratio, b.max_drawdown, b.win_rate,
               b.profit_factor, b.total_trades, b.winning_trades, b.losing_trades,
               b.avg_trade_duration, b.final_equity,
               b.best_trade, b.worst_trade, b.error_message, b.is_saved, b.created_at, b.completed_at,
               m.name as model_name, b.expert_name, b.optimization_id, b.engine_type
               {desc_col}
        FROM backtests b
        LEFT JOIN trained_models m ON b.model_id = m.id
        {where_sql}
        ORDER BY b.created_at DESC
    """), params)

    backtests = []
    for row in result:
        bt = {
            "id": row[0], "name": row[1], "modelId": row[2],
            "predictionDatasetId": row[3], "executionDatasetId": row[4],
            "strategyId": row[5],
            "startDate": str(row[6]) if row[6] else None,
            "endDate": str(row[7]) if row[7] else None,
            "initialCapital": row[8], "fitnessMetric": row[9],
            "status": row[10], "totalReturn": row[11],
            "sharpeRatio": row[12], "maxDrawdown": row[13],
            "winRate": row[14], "profitFactor": row[15],
            "totalTrades": row[16], "winningTrades": row[17], "losingTrades": row[18],
            "avgTradeDuration": row[19],
            "finalEquity": row[20], "bestTrade": row[21],
            "worstTrade": row[22], "errorMessage": row[23],
            "isSaved": row[24] or False,
            "createdAt": str(row[25]) if row[25] else None,
            "completedAt": str(row[26]) if row[26] else None,
            "modelName": row[27],
            "expertName": row[28],
            "optimizationId": row[29],
            "engineType": row[30] or "ml",
            "description": row[31] if has_description else None,
        }
        backtests.append(bt)

    return {"backtests": backtests, "total": len(backtests)}


@router.post("")
async def create_backtest(
    backtest: BacktestCreate,
    db: Session = Depends(get_db)
):
    """Create and run a new backtest.

    Dispatches on ``engine``: ``daily_expert`` builds the daily-engine payload (expert spec +
    static-universe instruments) and queues a ``daily_backtest`` task; everything else (the
    default ``ml``) keeps the legacy model-driven path byte-for-byte.
    """
    if backtest.engine == "daily_expert":
        return _create_daily_expert_backtest(backtest, db)

    # ----- legacy ML engine path (unchanged behaviour) -----
    if not backtest.model_id:
        raise HTTPException(status_code=400, detail="model_id is required for engine='ml'")
    if backtest.prediction_dataset_id is None:
        raise HTTPException(status_code=400, detail="prediction_dataset_id is required for engine='ml'")
    if backtest.execution_dataset_id is None:
        raise HTTPException(status_code=400, detail="execution_dataset_id is required for engine='ml'")

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

    # Copy strategy params into backtest record for persistence
    # (so backtest results are self-contained even if strategy is later deleted)
    strategy_params = backtest.strategy_params or {}
    if strategy and not strategy_params:
        strategy_params = {
            'initialTpPercent': strategy.initial_tp_percent,
            'initialSlPercent': strategy.initial_sl_percent,
            'buyEntryConditions': strategy.buy_entry_conditions,
            'sellEntryConditions': strategy.sell_entry_conditions,
            'exitConditions': strategy.exit_conditions,
            'strategyName': strategy.name,
        }

    # Create backtest record (use integer id for database FK)
    db_backtest = Backtest(
        name=backtest.name,
        model_id=model.id,  # Use the integer database id
        prediction_dataset_id=backtest.prediction_dataset_id,
        execution_dataset_id=backtest.execution_dataset_id,
        strategy_id=backtest.strategy_id,
        strategy_params=strategy_params,
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


def _create_daily_expert_backtest(backtest: "BacktestCreate", db: Session) -> dict:
    """Create + queue a daily multi-asset (expert) backtest from the unified create request.

    Builds the daily-engine payload from ``backtest.expert`` + ``backtest.universe`` and the
    shared trading parameters, persists a ``Backtest`` results row (``engine_type='daily_expert'``,
    ``model_id=None``, ``expert_name`` set for per-expert filtering), and enqueues the
    ``daily_backtest`` task whose handler (``handle_daily_backtest``) loads the row by id and runs
    the engine.

    Fail-early validation (``backend/CLAUDE.md``): the expert class must be supported, and a
    ``static`` universe must be non-empty.
    """
    from app.services.backtest.daily_backtest_handler import _SUPPORTED_EXPERTS

    expert = backtest.expert or {}
    expert_class = expert.get("class")
    if not expert_class:
        raise HTTPException(status_code=400, detail="expert.class is required for engine='daily_expert'")
    if expert_class not in _SUPPORTED_EXPERTS:
        raise HTTPException(
            status_code=400,
            detail=f"unsupported expert '{expert_class}'; supported: {sorted(_SUPPORTED_EXPERTS)}",
        )
    expert_settings = expert.get("settings") or {}

    universe = backtest.universe or {}
    mode = universe.get("mode")
    if mode not in ("static", "screener"):
        raise HTTPException(
            status_code=400,
            detail="universe.mode must be 'static' or 'screener' for engine='daily_expert'",
        )

    # Screener mode: the daily handler resolves instruments from the OFFLINE screener-history
    # cache (built via ``ba2-test fetch-screener``) at run time — READ-ONLY, fail-early on a
    # cache miss. The create path validates that the screener block carries the criteria +
    # cache location (no-defaults rule) and passes it through; it does NOT resolve here (the
    # resolution + fail-fast happens in the task so a cache miss surfaces on the run row).
    screener_universe = None
    symbols: list = []
    if mode == "screener":
        screener_settings = universe.get("screener_settings")
        if not screener_settings:
            raise HTTPException(
                status_code=400,
                detail="universe.screener_settings is required for universe.mode='screener'",
            )
        cache_db = universe.get("cache_db")
        if not cache_db:
            raise HTTPException(
                status_code=400,
                detail="universe.cache_db is required for universe.mode='screener' "
                       "(the offline screener-history cache built via ba2-test fetch-screener)",
            )
        group = universe.get("group")
        if not group:
            raise HTTPException(
                status_code=400,
                detail="universe.group is required for universe.mode='screener' "
                       "(the --group label used when building the cache)",
            )
        screener_universe = {
            "mode": "screener",
            "screener_settings": screener_settings,
            "cache_db": cache_db,
            "group": group,
        }
    else:
        symbols = universe.get("symbols") or []
        if not symbols:
            raise HTTPException(status_code=400, detail="universe.symbols must be non-empty for static mode")

    # Fail-early on the daily-engine trading knobs (no-defaults rule).
    if not backtest.fill_model:
        raise HTTPException(status_code=400, detail="fill_model is required for engine='daily_expert'")
    if backtest.seed is None:
        raise HTTPException(status_code=400, detail="seed is required for engine='daily_expert'")

    try:
        start_date = datetime.fromisoformat(backtest.start_date)
        end_date = datetime.fromisoformat(backtest.end_date)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {e}")

    db_backtest = Backtest(
        name=backtest.name,
        model_id=None,  # daily expert runs are not model-driven
        expert_name=expert_class,  # per-expert filtering / best-N retention
        start_date=start_date,
        end_date=end_date,
        initial_capital=backtest.initial_capital,
        commission=backtest.commission,
        slippage=backtest.slippage,
        fitness_metric=backtest.fitness_metric,
        status="pending",
        engine_type="daily_expert",
    )

    db.add(db_backtest)
    db.commit()
    db.refresh(db_backtest)

    logger.info(f"Created daily expert backtest: {db_backtest.name} (id={db_backtest.id})")

    experts_payload = [{"class": expert_class, "settings": expert_settings}]

    # Universe plumbing: static runs carry the explicit symbol list; screener runs carry the
    # ``universe`` block (mode/screener_settings/cache_db/group) which the handler resolves
    # from the offline cache (read-only, fail-fast on a miss).
    payload = {
        'backtest_id': db_backtest.id,
        'name': backtest.name,
        'experts': experts_payload,
        'start_date': backtest.start_date,
        'end_date': backtest.end_date,
        'initial_capital': backtest.initial_capital,
        'commission': backtest.commission,
        'slippage': backtest.slippage,
        'fill_model': backtest.fill_model,
        'seed': backtest.seed,
        'warmup_days': backtest.warmup_days,
    }
    if screener_universe is not None:
        payload['universe'] = screener_universe
        universe_desc = f"screener cache (group {screener_universe['group']})"
    else:
        payload['enabled_instruments'] = list(symbols)
        universe_desc = f"{len(symbols)} instruments"

    from app.services.task_queue import get_task_queue
    task_queue = get_task_queue()
    task_id = task_queue.queue_task(
        task_type='daily_backtest',
        name=f'Daily Backtest: {db_backtest.name}',
        payload=payload,
        description=f'Daily expert backtest ({expert_class}) over {universe_desc}',
    )

    logger.info(f"Queued daily backtest task: {task_id}")

    return {"taskId": task_id, "backtestId": db_backtest.id, **db_backtest.to_dict()}


@router.patch("/{backtest_id}")
async def update_backtest(
    backtest_id: int,
    update: dict,
    db: Session = Depends(get_db)
):
    """Update backtest fields (description, name)."""
    backtest = db.query(Backtest).filter(Backtest.id == backtest_id).first()
    if not backtest:
        raise HTTPException(status_code=404, detail=f"Backtest {backtest_id} not found")

    if 'description' in update:
        backtest.description = update['description']
    if 'name' in update:
        backtest.name = update['name']
    db.commit()
    return {"status": "updated", "id": backtest_id}


@router.post("/daily")
async def create_daily_backtest(
    request: DailyBacktestCreate,
    db: Session = Depends(get_db)
):
    """Create + queue a daily multi-asset (expert) backtest.

    Creates a ``Backtest`` results row (``status="pending"``, ``model_id=None`` — the daily
    engine is not model-driven; ``engine_type="daily_expert"`` to distinguish it from legacy
    ML runs) and queues a ``daily_backtest`` task whose payload carries the run config + the
    new row id.
    The ``daily_backtest`` handler runs the engine and persists the results onto the row.
    """
    # Validate fail-early (no defaults).
    if not request.enabled_instruments:
        raise HTTPException(status_code=400, detail="enabled_instruments must be non-empty")
    if not request.experts:
        raise HTTPException(status_code=400, detail="experts must be non-empty")

    try:
        start_date = datetime.fromisoformat(request.start_date)
        end_date = datetime.fromisoformat(request.end_date)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {e}")

    db_backtest = Backtest(
        name=request.name,
        model_id=None,  # daily expert runs are not model-driven (Task-7 migration makes this nullable)
        start_date=start_date,
        end_date=end_date,
        initial_capital=request.initial_capital,
        commission=request.commission,
        slippage=request.slippage,
        fitness_metric=request.fitness_metric,
        status="pending",
        engine_type="daily_expert",  # discriminates from legacy ML runs (migration 018)
    )

    db.add(db_backtest)
    db.commit()
    db.refresh(db_backtest)

    logger.info(f"Created daily backtest: {db_backtest.name} (id={db_backtest.id})")

    from app.services.task_queue import get_task_queue
    task_queue = get_task_queue()
    task_id = task_queue.queue_task(
        task_type='daily_backtest',
        name=f'Daily Backtest: {db_backtest.name}',
        payload={
            'backtest_id': db_backtest.id,
            'name': request.name,
            'enabled_instruments': request.enabled_instruments,
            'experts': request.experts,
            'start_date': request.start_date,
            'end_date': request.end_date,
            'initial_capital': request.initial_capital,
            'commission': request.commission,
            'slippage': request.slippage,
            'fill_model': request.fill_model,
            'seed': request.seed,
            'warmup_days': request.warmup_days,
        },
        description=f'Daily expert backtest over {len(request.enabled_instruments)} instruments',
    )

    logger.info(f"Queued daily backtest task: {task_id}")

    return {"taskId": task_id, "backtestId": db_backtest.id, **db_backtest.to_dict()}


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
