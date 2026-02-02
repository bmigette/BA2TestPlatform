"""
Strategy Executor Service

Evaluates strategy conditions against data to generate trade signals.
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ExitActionType(Enum):
    CLOSE = "close"
    ADJUST_TP = "adjust_tp"
    ADJUST_SL = "adjust_sl"


@dataclass
class ExitAction:
    action: ExitActionType
    value: Optional[float] = None


@dataclass
class Position:
    entry_price: float
    entry_time: Any
    size: float
    direction: str  # "long" or "short"
    tp_percent: float
    sl_percent: float
    bars_held: int = 0
    days_held: int = 0

    @property
    def unrealized_pnl_pct(self) -> float:
        """Calculate unrealized P&L percentage (placeholder - needs current price)."""
        return 0.0


def evaluate_comparison(left: Any, operator: str, right: Any) -> bool:
    """Evaluate a comparison operation."""
    try:
        if operator == ">":
            return float(left) > float(right)
        elif operator == ">=":
            return float(left) >= float(right)
        elif operator == "<":
            return float(left) < float(right)
        elif operator == "<=":
            return float(left) <= float(right)
        elif operator == "==":
            return left == right
        elif operator == "!=":
            return left != right
        elif operator == "between":
            if isinstance(right, (list, tuple)) and len(right) == 2:
                return float(right[0]) <= float(left) <= float(right[1])
            return False
        else:
            logger.warning(f"Unknown operator: {operator}")
            return False
    except (TypeError, ValueError) as e:
        logger.warning(f"Comparison error: {e}")
        return False


def evaluate_condition(condition: dict, context: Dict[str, Any]) -> bool:
    """
    Evaluate a single condition against the context.

    Args:
        condition: Condition dict with field, comparison, value
        context: Dict with current values for all fields

    Returns:
        True if condition is met, False otherwise
    """
    # Handle nested AND/OR operators
    operator = condition.get("operator")
    if operator in ("AND", "OR"):
        sub_conditions = condition.get("conditions", [])
        if not sub_conditions:
            return True

        if operator == "AND":
            return all(evaluate_condition(c, context) for c in sub_conditions)
        else:  # OR
            return any(evaluate_condition(c, context) for c in sub_conditions)

    # Simple condition
    field = condition.get("field")
    comparison = condition.get("comparison")
    value = condition.get("value")

    if field is None or comparison is None:
        logger.warning(f"Invalid condition: missing field or comparison")
        return False

    # Get field value from context
    field_value = context.get(field)
    if field_value is None:
        logger.debug(f"Field {field} not found in context")
        return False

    return evaluate_comparison(field_value, comparison, value)


def evaluate_condition_tree(conditions: dict, context: Dict[str, Any]) -> bool:
    """Evaluate the full condition tree."""
    if not conditions:
        return False
    return evaluate_condition(conditions, context)


class StrategyExecutor:
    """Executes strategy conditions against data."""

    def __init__(self, strategy_config: dict):
        """
        Initialize executor with strategy configuration.

        Args:
            strategy_config: Dict with entry_conditions, exit_conditions, tp/sl settings
        """
        self.entry_conditions = strategy_config.get("entry_conditions", {})
        self.exit_conditions = strategy_config.get("exit_conditions", [])
        self.initial_tp_percent = strategy_config.get("initial_tp_percent", 5.0)
        self.initial_sl_percent = strategy_config.get("initial_sl_percent", 2.0)

    def check_entry(self, context: Dict[str, Any]) -> bool:
        """
        Check if entry conditions are met.

        Args:
            context: Dict with bar data and predictions

        Returns:
            True if should enter, False otherwise
        """
        return evaluate_condition_tree(self.entry_conditions, context)

    def check_exits(self, context: Dict[str, Any]) -> Optional[ExitAction]:
        """
        Check exit conditions and return action if any triggered.

        Args:
            context: Dict with bar data, predictions, and position state

        Returns:
            ExitAction if condition triggered, None otherwise
        """
        for exit_rule in self.exit_conditions:
            conditions = exit_rule.get("conditions", {})
            if evaluate_condition_tree(conditions, context):
                action_type = exit_rule.get("action", "close")
                action_value = exit_rule.get("action_value")

                try:
                    action_enum = ExitActionType(action_type)
                except ValueError:
                    action_enum = ExitActionType.CLOSE

                return ExitAction(action=action_enum, value=action_value)

        return None

    def build_context(
        self,
        bar_data: Dict[str, Any],
        predictions: Dict[str, float],
        position: Optional[Position] = None,
        current_price: float = 0.0
    ) -> Dict[str, Any]:
        """
        Build full context for condition evaluation.

        Args:
            bar_data: OHLCV and time data
            predictions: Model prediction probabilities
            position: Current position if any
            current_price: Current market price

        Returns:
            Combined context dict
        """
        context = {**bar_data, **predictions}

        if position:
            context["bars_in_trade"] = position.bars_held
            context["days_in_trade"] = position.days_held

            # Calculate P&L
            if position.direction == "long":
                pnl_pct = (current_price - position.entry_price) / position.entry_price * 100
            else:
                pnl_pct = (position.entry_price - current_price) / position.entry_price * 100

            context["position_pnl_pct"] = pnl_pct
            context["position_pnl_abs"] = pnl_pct * position.size / 100

        return context
