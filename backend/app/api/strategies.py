"""
Strategies API endpoints.

Manages trading strategies with entry/exit conditions.
"""

import logging
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.models import get_db, Strategy, TrainedModel, StrategyOptimization
from app.services.task_queue import get_task_queue

logger = logging.getLogger(__name__)

router = APIRouter()


# Pydantic models
class ConditionBase(BaseModel):
    id: str
    field: Optional[str] = None
    field_type: Optional[str] = None  # model_probability, model_class, position, time
    comparison: Optional[str] = None  # >, >=, <, <=, ==, !=, between
    value: Optional[float | int | List] = None
    optimize: bool = False
    value_min: Optional[float] = None
    value_max: Optional[float] = None
    value_step: Optional[float] = None
    optimize_enabled: bool = False
    confirmation_required: Optional[int] = None
    confirmation_bars: Optional[int] = None
    confirmation_bars_min: Optional[int] = None
    confirmation_bars_max: Optional[int] = None
    confirmation_bars_step: Optional[int] = None
    operator: Optional[str] = None  # AND, OR
    conditions: Optional[List["ConditionBase"]] = None


class ExitCondition(BaseModel):
    id: str
    name: Optional[str] = None
    conditions: ConditionBase
    action: str  # close, adjust_tp, adjust_sl, or option action (e.g. buy_call)
    toggle_optimize: bool = False                 # -> exit:<id>:enabled gene (optimizer drops the whole rule)
    reference_value: Optional[str] = None         # order_open_price | current_price | expert_target_price (adjust actions)
    action_value: Optional[float] = None
    action_value_optimize: bool = False
    action_value_min: Optional[float] = None
    action_value_max: Optional[float] = None
    action_value_step: Optional[float] = None
    # --- option-action fields (None for equity actions) ---
    option_strategy: Optional[str] = None
    option_strike_method: Optional[str] = None      # delta | percent_otm | consensus_target
    option_strike_param: Optional[float] = None
    option_dte_min: Optional[int] = None
    option_dte_max: Optional[int] = None
    option_sizing: Optional[float] = None           # % of equity
    option_strike_param_optimize: bool = False
    option_strike_param_min: Optional[float] = None
    option_strike_param_max: Optional[float] = None
    option_strike_param_step: Optional[float] = None
    option_dte_optimize: bool = False
    option_dte_min_range: Optional[int] = None
    option_dte_max_range: Optional[int] = None
    option_dte_step: Optional[int] = None


class StrategyCreate(BaseModel):
    name: str
    description: Optional[str] = None
    # Old single entry conditions (deprecated, for backwards compat)
    entry_conditions: Optional[dict] = None
    # New separate buy/sell entry conditions
    buy_entry_conditions: Optional[dict] = None
    sell_entry_conditions: Optional[dict] = None
    exit_conditions: Optional[List[dict]] = None
    initial_tp_percent: float = 5.0
    initial_tp_optimize: bool = False
    initial_tp_min: Optional[float] = None
    initial_tp_max: Optional[float] = None
    initial_tp_step: Optional[float] = None
    initial_sl_percent: float = 2.0
    initial_sl_optimize: bool = False
    initial_sl_min: Optional[float] = None
    initial_sl_max: Optional[float] = None
    initial_sl_step: Optional[float] = None


class StrategyUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    entry_conditions: Optional[dict] = None
    buy_entry_conditions: Optional[dict] = None
    sell_entry_conditions: Optional[dict] = None
    exit_conditions: Optional[List[dict]] = None
    initial_tp_percent: Optional[float] = None
    initial_tp_optimize: Optional[bool] = None
    initial_tp_min: Optional[float] = None
    initial_tp_max: Optional[float] = None
    initial_tp_step: Optional[float] = None
    initial_sl_percent: Optional[float] = None
    initial_sl_optimize: Optional[bool] = None
    initial_sl_min: Optional[float] = None
    initial_sl_max: Optional[float] = None
    initial_sl_step: Optional[float] = None


def extract_required_fields(
    first_arg=None,
    second_arg=None,
    *,
    buy_entry_conditions: dict = None,
    sell_entry_conditions: dict = None,
    exit_conditions: list = None,
    entry_conditions: dict = None
) -> List[str]:
    """Extract all model prediction fields used in conditions.

    Supports both old signature: extract_required_fields(entry_conditions, exit_conditions)
    and new signature with keyword args for buy/sell split.
    """
    # Handle backwards compatibility with old positional signature
    # Old: extract_required_fields(entry_conditions_dict, exit_conditions_list)
    if first_arg is not None:
        if isinstance(first_arg, dict):
            entry_conditions = first_arg
        if isinstance(second_arg, list):
            exit_conditions = second_arg

    fields = set()

    def traverse_conditions(cond):
        if cond is None:
            return
        if isinstance(cond, dict):
            if cond.get("field_type") in ("model_probability", "model_class"):
                if cond.get("field"):
                    fields.add(cond["field"])
            if cond.get("conditions"):
                for c in cond["conditions"]:
                    traverse_conditions(c)

    # Traverse all condition sources
    traverse_conditions(buy_entry_conditions)
    traverse_conditions(sell_entry_conditions)
    traverse_conditions(entry_conditions)
    for exit_cond in (exit_conditions or []):
        traverse_conditions(exit_cond.get("conditions"))

    return sorted(list(fields))


@router.get("")
async def list_strategies(
    search: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """List all strategies."""
    query = db.query(Strategy)

    if search:
        query = query.filter(Strategy.name.ilike(f"%{search}%"))

    strategies = query.order_by(Strategy.created_at.desc()).all()

    return {
        "strategies": [s.to_dict() for s in strategies],
        "total": len(strategies)
    }


@router.post("")
async def create_strategy(
    strategy: StrategyCreate,
    db: Session = Depends(get_db)
):
    """Create a new strategy."""
    required_fields = extract_required_fields(
        buy_entry_conditions=strategy.buy_entry_conditions,
        sell_entry_conditions=strategy.sell_entry_conditions,
        exit_conditions=strategy.exit_conditions,
        entry_conditions=strategy.entry_conditions
    )

    db_strategy = Strategy(
        name=strategy.name,
        description=strategy.description,
        required_fields=required_fields,
        entry_conditions=strategy.entry_conditions,
        buy_entry_conditions=strategy.buy_entry_conditions,
        sell_entry_conditions=strategy.sell_entry_conditions,
        exit_conditions=strategy.exit_conditions or [],
        initial_tp_percent=strategy.initial_tp_percent,
        initial_tp_optimize=strategy.initial_tp_optimize,
        initial_tp_min=strategy.initial_tp_min,
        initial_tp_max=strategy.initial_tp_max,
        initial_tp_step=strategy.initial_tp_step,
        initial_sl_percent=strategy.initial_sl_percent,
        initial_sl_optimize=strategy.initial_sl_optimize,
        initial_sl_min=strategy.initial_sl_min,
        initial_sl_max=strategy.initial_sl_max,
        initial_sl_step=strategy.initial_sl_step,
    )

    db.add(db_strategy)
    db.commit()
    db.refresh(db_strategy)

    logger.info(f"Created strategy: {db_strategy.name} (id={db_strategy.id})")
    return db_strategy.to_dict()


@router.get("/compatible/{model_id}")
async def get_compatible_strategies(
    model_id: int,
    db: Session = Depends(get_db)
):
    """Get strategies compatible with a model's prediction fields."""
    model = db.query(TrainedModel).filter(TrainedModel.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

    # Get model's prediction target fields
    model_fields = set()
    if model.prediction_targets:
        for target in model.prediction_targets:
            if isinstance(target, dict):
                # Add all possible field names the model might output
                target_type = target.get("type", "")
                if target_type:
                    model_fields.add(target_type)
                    model_fields.add(f"{target_type}_probability")
                    model_fields.add(f"{target_type}_class")

    # Get all strategies and filter by required fields
    strategies = db.query(Strategy).all()
    compatible = []

    for strategy in strategies:
        required = set(strategy.required_fields or [])
        if required.issubset(model_fields) or len(required) == 0:
            compatible.append(strategy.to_dict())

    return {
        "strategies": compatible,
        "total": len(compatible),
        "modelFields": sorted(list(model_fields))
    }


@router.get("/{strategy_id}")
async def get_strategy(
    strategy_id: int,
    db: Session = Depends(get_db)
):
    """Get strategy by ID."""
    strategy = db.query(Strategy).filter(Strategy.id == strategy_id).first()
    if not strategy:
        raise HTTPException(status_code=404, detail=f"Strategy {strategy_id} not found")
    return strategy.to_dict()


@router.put("/{strategy_id}")
async def update_strategy(
    strategy_id: int,
    update: StrategyUpdate,
    db: Session = Depends(get_db)
):
    """Update a strategy."""
    strategy = db.query(Strategy).filter(Strategy.id == strategy_id).first()
    if not strategy:
        raise HTTPException(status_code=404, detail=f"Strategy {strategy_id} not found")

    update_data = update.model_dump(exclude_unset=True)

    # Recalculate required fields if any conditions changed
    conditions_keys = ["entry_conditions", "buy_entry_conditions", "sell_entry_conditions", "exit_conditions"]
    if any(k in update_data for k in conditions_keys):
        update_data["required_fields"] = extract_required_fields(
            buy_entry_conditions=update_data.get("buy_entry_conditions", strategy.buy_entry_conditions),
            sell_entry_conditions=update_data.get("sell_entry_conditions", strategy.sell_entry_conditions),
            exit_conditions=update_data.get("exit_conditions", strategy.exit_conditions),
            entry_conditions=update_data.get("entry_conditions", strategy.entry_conditions)
        )

    for key, value in update_data.items():
        setattr(strategy, key, value)

    db.commit()
    db.refresh(strategy)

    logger.info(f"Updated strategy: {strategy.name} (id={strategy.id})")
    return strategy.to_dict()


@router.delete("/{strategy_id}")
async def delete_strategy(
    strategy_id: int,
    db: Session = Depends(get_db)
):
    """Delete a strategy."""
    strategy = db.query(Strategy).filter(Strategy.id == strategy_id).first()
    if not strategy:
        raise HTTPException(status_code=404, detail=f"Strategy {strategy_id} not found")

    db.delete(strategy)
    db.commit()

    logger.info(f"Deleted strategy: {strategy.name} (id={strategy_id})")
    return {"message": f"Strategy {strategy_id} deleted"}


class OptimizeRequest(BaseModel):
    """Request to launch a joint genetic optimization over a strategy.

    optimization_config MUST include the GA params (populationSize, generations,
    crossoverProb, mutationProb, earlyStoppingGenerations, elitismPercent, seed)
    plus a `backtest` block (engine/model/datasets/date-range/initial_capital/...).
    The handler validates these fail-early (no-defaults rule, backend/CLAUDE.md).
    """
    name: Optional[str] = None
    fitness_metric: str                      # sharpe/return/profit_factor/win_rate/max_drawdown/...
    optimization_type: str = "genetic"       # genetic | brute_force
    optimization_config: dict                # GA params + backtest{}
    expert_params: Optional[dict] = None     # {param:{optimize,min,max,step,type}}


@router.post("/{strategy_id}/optimize")
async def optimize_strategy(
    strategy_id: int,
    req: OptimizeRequest,
    db: Session = Depends(get_db)
):
    """Launch a joint genetic optimization (expert + ruleset params).

    Writes a StrategyOptimization row and enqueues a 'strategy_optimization' task.
    Any expert_params are folded into optimization_config so the handler is self-contained.
    """
    strategy = db.query(Strategy).filter(Strategy.id == strategy_id).first()
    if not strategy:
        raise HTTPException(status_code=404, detail=f"Strategy {strategy_id} not found")

    cfg = dict(req.optimization_config or {})
    if req.expert_params is not None:
        cfg["expert_params"] = req.expert_params

    row = StrategyOptimization(
        strategy_id=strategy_id,
        name=req.name,
        fitness_metric=req.fitness_metric,
        optimization_type=req.optimization_type,
        optimization_config=cfg,
        status="pending",
    )
    db.add(row)
    db.commit()
    db.refresh(row)

    task_name = req.name or f"Optimize strategy {strategy.name} ({req.fitness_metric})"
    task_id = get_task_queue().queue_task(
        task_type="strategy_optimization",
        name=task_name,
        payload={"optimization_id": row.id},
        description=f"Joint genetic optimization for strategy {strategy_id}",
    )
    logger.info(f"Enqueued strategy_optimization {row.id} (task {task_id})")
    return {"optimizationId": row.id, "taskId": task_id, **row.to_dict()}


def _top_individuals(row, n: int = 8) -> list:
    """Top-N distinct (by fitness) evaluated individuals from a (running) optimization's
    all_results, best first. Each entry carries its fitness + trade count (all_results stores a
    trade COUNT per trial, not the full trade list — full backtests are the persisted top-N)."""
    results = row.all_results or []
    seen, uniq = set(), []
    for e in sorted(results,
                    key=lambda x: (x.get("fitness") if x.get("fitness") is not None else -1e18),
                    reverse=True):
        f = e.get("fitness")
        if f is None or f in seen:
            continue
        seen.add(f)
        uniq.append(e)
        if len(uniq) >= n:
            break
    return [
        {"rank": i + 1, "fitness": e.get("fitness"), "nTrades": e.get("trades"),
         "params": e.get("params")}
        for i, e in enumerate(uniq)
    ]


# NOTE: register the two-segment /optimizations/* routes; they never collide with the
# one-segment GET /{strategy_id} (different path-segment count). /running before /{opt_id}.
@router.get("/optimizations/running")
def list_running_optimizations(db: Session = Depends(get_db)):
    """Running optimizations enriched with best fitness + top individuals (UI Running tab)."""
    rows = (db.query(StrategyOptimization)
            .filter(StrategyOptimization.status == "running")
            .order_by(StrategyOptimization.id.desc()).all())
    return {"optimizations": [
        {
            "id": r.id, "name": r.name, "status": r.status, "progress": r.progress,
            "fitnessMetric": r.fitness_metric, "bestFitness": r.best_fitness,
            "bestParams": r.best_params, "nEvaluated": len(r.all_results or []),
            "topIndividuals": _top_individuals(r, n=8),
        }
        for r in rows
    ]}


@router.get("/optimizations/{opt_id}")
def get_optimization(opt_id: int, db: Session = Depends(get_db)):
    """Full optimization detail (config, best params, top individuals) by id."""
    r = db.query(StrategyOptimization).filter(StrategyOptimization.id == opt_id).first()
    if not r:
        raise HTTPException(status_code=404, detail=f"Optimization {opt_id} not found")
    d = r.to_dict()
    d["topIndividuals"] = _top_individuals(r, n=15)
    return d
