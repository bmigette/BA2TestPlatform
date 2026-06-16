"""Ruleset metadata API (Task A2).

Read-only single-source-of-truth endpoints so the exit-ruleset UI does NOT
hardcode or drift from the backend vocabulary:

  * ``GET /api/ruleset/vocabulary``    -> flags / numerics / operators / actions / reference_values
  * ``GET /api/ruleset/exit-presets``  -> the packaged default exit-rule presets

The vocabulary is derived entirely from ``ba2_common.core.types`` (no DB):
``ExpertEventType`` members split on the member-NAME prefix — ``F_*`` are boolean
flag conditions (no operator/value), ``N_*`` are numeric conditions (operator+value).
``ExpertActionType`` drives the action list; option actions are tagged via
``is_option_action`` and the two adjust actions are flagged ``needs_reference``.

A stub ``GET /api/experts/{expert_id}/open-positions-ruleset`` is included here as a
placeholder for Task A3 (import-from-live); it returns 503 until the live DB wiring lands.
"""
import logging

from fastapi import APIRouter, HTTPException

from ba2_common.core.types import (
    ExpertActionType,
    ExpertEventType,
    get_reference_value_options,
    is_option_action,
)
from app.services.ruleset_presets import EXIT_PRESETS

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")

OPERATORS = [">", ">=", "<", "<=", "==", "!=", "between"]
_NEEDS_REFERENCE = ("adjust_take_profit", "adjust_stop_loss")


def _label(value: str) -> str:
    """Human label for an enum value, e.g. ``profit_loss_percent`` -> ``Profit Loss Percent``."""
    return value.replace("_", " ").title()


@router.get("/ruleset/vocabulary")
def get_vocabulary():
    """Condition vocabulary, operators, actions, and reference-value options."""
    flags = [
        {"value": m.value, "label": _label(m.value)}
        for m in ExpertEventType
        if m.name.startswith("F_")
    ]
    numerics = [
        {"value": m.value, "label": _label(m.value)}
        for m in ExpertEventType
        if m.name.startswith("N_")
    ]
    actions = [
        {
            "value": m.value,
            "label": _label(m.value),
            "is_option": is_option_action(m.value),
            "needs_reference": m.value in _NEEDS_REFERENCE,
        }
        for m in ExpertActionType
    ]
    return {
        "flags": flags,
        "numerics": numerics,
        "operators": OPERATORS,
        "actions": actions,
        "reference_values": get_reference_value_options(),
    }


@router.get("/ruleset/exit-presets")
def get_exit_presets():
    """The packaged default exit-rule presets (each ``rule`` validates against ExitCondition)."""
    return {"presets": EXIT_PRESETS}


@router.get("/experts/{expert_id}/open-positions-ruleset")
def get_open_positions_ruleset(expert_id: int):
    """Stub for Task A3 (import-from-live open-positions ruleset).

    Returns 503 until the live-DB wiring is implemented.
    """
    raise HTTPException(status_code=503, detail="live DB not configured")
