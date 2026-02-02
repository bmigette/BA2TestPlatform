"""
Strategies API endpoints.

Manages trading strategies with entry/exit conditions.
"""

import logging
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.models import get_db, Strategy, TrainedModel

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
    action: str  # close, adjust_tp, adjust_sl
    action_value: Optional[float] = None
    action_value_optimize: bool = False
    action_value_min: Optional[float] = None
    action_value_max: Optional[float] = None
    action_value_step: Optional[float] = None


class StrategyCreate(BaseModel):
    name: str
    description: Optional[str] = None
    entry_conditions: dict
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


def extract_required_fields(entry_conditions: dict, exit_conditions: list) -> List[str]:
    """Extract all model prediction fields used in conditions."""
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
        strategy.entry_conditions,
        strategy.exit_conditions
    )

    db_strategy = Strategy(
        name=strategy.name,
        description=strategy.description,
        required_fields=required_fields,
        entry_conditions=strategy.entry_conditions,
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

    # Recalculate required fields if conditions changed
    if "entry_conditions" in update_data or "exit_conditions" in update_data:
        entry = update_data.get("entry_conditions", strategy.entry_conditions)
        exit = update_data.get("exit_conditions", strategy.exit_conditions)
        update_data["required_fields"] = extract_required_fields(entry, exit)

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
