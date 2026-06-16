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

``GET /api/experts/{expert_id}/open-positions-ruleset`` (Task A3, import-from-live) is
GRACEFUL/OPTIONAL: when the env var ``BA2_LIVE_DB`` points at the live BA2TradePlatform
sqlite, it reads that expert's ``open_positions`` ruleset READ-ONLY and converts each live
``EventAction`` into an ``ExitCondition``-shaped rule the UI can load (marked optimizable).
When ``BA2_LIVE_DB`` is unset or the DB is unreachable, it returns 503 so the UI falls back
to JSON-paste import. The read uses a dedicated ``mode=ro`` sqlite connection with raw SQL —
it never touches ba2_common's shared engine and can never write the live DB.
"""
import json
import logging
import os
import sqlite3

from fastapi import APIRouter, HTTPException

from ba2_common.core.types import (
    ExpertActionType,
    ExpertEventType,
    ReferenceValue,
    get_reference_value_options,
    is_option_action,
)
from app.services.ruleset_presets import EXIT_PRESETS

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")

OPERATORS = [">", ">=", "<", "<=", "==", "!=", "between"]
_NEEDS_REFERENCE = ("adjust_take_profit", "adjust_stop_loss")

# --- live-import conversion vocabulary (reverse of default_rulesets mappings) -------------
# ExpertEventType .value -> True if it is a numeric (N_*) condition (carries operator+value).
_NUMERIC_EVENT_VALUES = {m.value for m in ExpertEventType if m.name.startswith("N_")}
_FLAG_EVENT_VALUES = {m.value for m in ExpertEventType if m.name.startswith("F_")}
# live EventAction action_type .value -> API ``action`` value (close/sell/adjust_*/option_*).
# Identity for every ExpertActionType value (the API ``action`` vocabulary IS the enum value).
_ACTION_VALUES = {m.value for m in ExpertActionType}
# action_types whose action_value (the % offset) the UI/optimizer tunes.
_ADJUST_ACTIONS = {ExpertActionType.ADJUST_TAKE_PROFIT.value, ExpertActionType.ADJUST_STOP_LOSS.value}


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


def _opt_range(value):
    """Sensible default optimize range for a numeric leaf/action value: ±50%, step ≈ |v|/5.

    Returns ``(min, max, step)``. A zero/None value yields a small symmetric default so the
    optimizer still has a range to explore. Negative values (e.g. a -3% SL offset) keep the
    correct ordering (min <= max).
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = 0.0
    if v == 0.0:
        return (-1.0, 1.0, 0.2)
    lo, hi = sorted((v * 0.5, v * 1.5))
    step = abs(v) / 5.0
    return (lo, hi, step)


def _trigger_to_leaf(idx: int, trig: dict) -> dict | None:
    """Convert one live EventAction trigger to a ConditionBase-shaped leaf dict, or None.

    Flag triggers (F_*) become value-less leaves; numeric triggers (N_*) carry
    field/comparison/value and are marked optimizable with a default ±50% range. Unknown
    event types are skipped (return None) so a partial live rule never breaks the import.
    """
    et = trig.get("event_type")
    leaf_id = f"c{idx}"
    if et in _FLAG_EVENT_VALUES:
        return {"id": leaf_id, "field": et, "field_type": "flag"}
    if et in _NUMERIC_EVENT_VALUES:
        value = trig.get("value")
        vmin, vmax, vstep = _opt_range(value)
        return {
            "id": leaf_id,
            "field": et,
            "field_type": "numeric",
            "comparison": trig.get("operator") or ">",
            "value": value,
            "optimize": True,
            "optimize_enabled": True,
            "value_min": vmin,
            "value_max": vmax,
            "value_step": vstep,
        }
    return None


def _eventaction_to_exit_rule(ea_id, name: str, triggers: dict, actions: dict) -> dict | None:
    """Convert one live ``EventAction`` (open_positions) into an ExitCondition-shaped dict.

    Returns None when the action_type is unknown/unsupported (skip the rule rather than emit
    something that fails ExitCondition validation). Numeric conditions and adjust action_values
    are marked optimizable with sensible default ranges; the whole rule carries
    ``toggle_optimize`` so the optimizer can drop it.
    """
    # The live ``actions`` JSON is ``{"<key>": {"action_type": ..., reference_value?, value?}}``.
    # An open_positions rule has exactly one action; take the first usable one.
    action_cfg = None
    for cfg in (actions or {}).values():
        if isinstance(cfg, dict) and cfg.get("action_type") in _ACTION_VALUES:
            action_cfg = cfg
            break
    if action_cfg is None:
        return None
    action = action_cfg["action_type"]

    leaves = []
    for i, trig in enumerate((triggers or {}).values()):
        if not isinstance(trig, dict):
            continue
        leaf = _trigger_to_leaf(i, trig)
        if leaf is not None:
            leaves.append(leaf)

    rule: dict = {
        "id": f"live-{ea_id}",
        "name": name,
        "conditions": {"id": f"grp-{ea_id}", "operator": "AND", "conditions": leaves},
        "action": action,
        "toggle_optimize": True,
    }

    if action in _ADJUST_ACTIONS:
        rule["reference_value"] = action_cfg.get("reference_value") or ReferenceValue.ORDER_OPEN_PRICE.value
        av = action_cfg.get("value")
        rule["action_value"] = av
        amin, amax, astep = _opt_range(av)
        rule["action_value_optimize"] = True
        rule["action_value_min"] = amin
        rule["action_value_max"] = amax
        rule["action_value_step"] = astep

    return rule


def _read_live_open_positions_rules(db_path: str, expert_id: int) -> list[dict]:
    """READ-ONLY raw-SQL read of an expert's open_positions ruleset from the live sqlite.

    Opens the DB via a ``mode=ro`` URI (never writes, never touches ba2_common's engine),
    resolves ``expertinstance.open_positions_ruleset_id`` -> ordered ``eventaction`` rows, and
    converts each to an ExitCondition-shaped dict. Raises ``HTTPException(404)`` if the expert
    is absent. Connection errors propagate to the caller (mapped to 503 there).
    """
    uri = f"file:{db_path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT open_positions_ruleset_id FROM expertinstance WHERE id = ?",
            (expert_id,),
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail=f"expert {expert_id} not found in live DB")
        ruleset_id = row["open_positions_ruleset_id"]
        if ruleset_id is None:
            return []  # expert exists but has no open_positions ruleset configured
        ea_rows = conn.execute(
            "SELECT ea.id AS id, ea.name AS name, ea.triggers AS triggers, ea.actions AS actions "
            "FROM eventaction ea "
            "JOIN ruleset_eventaction_link link ON link.eventaction_id = ea.id "
            "WHERE link.ruleset_id = ? "
            "ORDER BY link.order_index",
            (ruleset_id,),
        ).fetchall()
    finally:
        conn.close()

    rules: list[dict] = []
    for ea in ea_rows:
        triggers = json.loads(ea["triggers"]) if ea["triggers"] else {}
        actions = json.loads(ea["actions"]) if ea["actions"] else {}
        rule = _eventaction_to_exit_rule(ea["id"], ea["name"], triggers, actions)
        if rule is not None:
            rules.append(rule)
    return rules


@router.get("/experts/{expert_id}/open-positions-ruleset")
def get_open_positions_ruleset(expert_id: int):
    """Import a LIVE expert's open_positions ruleset as ExitCondition-shaped rules (optional).

    Graceful/optional: when ``BA2_LIVE_DB`` is unset the UI falls back to JSON-paste import
    (503). When set, the live ruleset is read READ-ONLY and converted. DB/connection errors
    map to 503 (never 500) so the UI degrades gracefully; a missing expert is a 404.
    """
    db_path = os.environ.get("BA2_LIVE_DB")
    if not db_path:
        raise HTTPException(
            status_code=503,
            detail="live DB not configured; paste the ruleset JSON instead",
        )
    try:
        rules = _read_live_open_positions_rules(db_path, expert_id)
    except HTTPException:
        raise  # 404 (expert not found) passes through unchanged
    except Exception as exc:  # noqa: BLE001 — any DB/parse failure degrades to a graceful 503
        logger.warning("live open_positions import failed for expert %s: %s", expert_id, exc)
        raise HTTPException(
            status_code=503,
            detail="could not read live DB; paste the ruleset JSON instead",
        )
    return {"rules": rules}
