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

from ba2_common.core.rule_builders import FIELD_EVENT, FLAG_FIELD_EVENT
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

# --- enter_market import: reverse of triggers_from_condition_tree (event_type -> field) -----
# Inverse of the SHARED FIELD_EVENT / FLAG_FIELD_EVENT maps, keyed by ExpertEventType .value.
# Multiple synonym fields can map to the same event_type (e.g. expected_profit*); the reverse
# keeps the LAST field per event_type, which is a canonical UI field name — fine for import.
_EVENT_NUMERIC_FIELD = {et.value: field for field, et in FIELD_EVENT.items()}
_EVENT_FLAG_FIELD = {et.value: field for field, et in FLAG_FIELD_EVENT.items()}
# action_types that OPEN a position (which entry-tree the rule's leaves belong to).
_BUY_ACTION = ExpertActionType.BUY.value
_SELL_ACTION = ExpertActionType.SELL.value


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


def _expert_ruleset_eas(db_path: str, expert_id: int, ruleset_id_column: str) -> list[dict]:
    """READ-ONLY raw-SQL read of one expert ruleset's ordered EventActions from the live sqlite.

    Opens the DB via a ``mode=ro`` URI (never writes, never touches ba2_common's engine),
    resolves ``expertinstance.<ruleset_id_column>`` -> ordered ``eventaction`` rows, and returns
    each as ``{"id", "name", "triggers": <dict>, "actions": <dict>}`` (JSON already parsed).
    Raises ``HTTPException(404)`` if the expert is absent; returns ``[]`` when the expert has no
    such ruleset configured. Connection errors propagate to the caller (mapped to 503 there).

    ``ruleset_id_column`` is a fixed internal identifier (``open_positions_ruleset_id`` /
    ``enter_market_ruleset_id``), NOT user input, so interpolating it into the SQL is safe.
    """
    uri = f"file:{db_path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            f"SELECT {ruleset_id_column} AS ruleset_id FROM expertinstance WHERE id = ?",
            (expert_id,),
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail=f"expert {expert_id} not found in live DB")
        ruleset_id = row["ruleset_id"]
        if ruleset_id is None:
            return []  # expert exists but has no such ruleset configured
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

    eas: list[dict] = []
    for ea in ea_rows:
        eas.append(
            {
                "id": ea["id"],
                "name": ea["name"],
                "triggers": json.loads(ea["triggers"]) if ea["triggers"] else {},
                "actions": json.loads(ea["actions"]) if ea["actions"] else {},
            }
        )
    return eas


def _read_live_open_positions_rules(db_path: str, expert_id: int) -> list[dict]:
    """READ-ONLY read of an expert's open_positions ruleset as ExitCondition-shaped dicts.

    Resolves ``expertinstance.open_positions_ruleset_id`` -> ordered EventActions (via the shared
    ``_expert_ruleset_eas`` reader) and converts each to an ExitCondition-shaped dict.
    """
    rules: list[dict] = []
    for ea in _expert_ruleset_eas(db_path, expert_id, "open_positions_ruleset_id"):
        rule = _eventaction_to_exit_rule(ea["id"], ea["name"], ea["triggers"], ea["actions"])
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


# --- enter_market import (inverse of triggers_from_condition_tree) -------------------------

def _trigger_to_entry_leaf(idx: int, trig: dict) -> dict | None:
    """Convert one live enter_market trigger to a ConditionBase-shaped tree leaf, or None.

    Inverse of ``triggers_from_condition_tree``: flag triggers (event_type in the flag map)
    become value-less ``{id, field, field_type:"flag"}`` leaves; numeric triggers become
    ``{id, field, field_type:"numeric", comparison, value, optimize_enabled, value_min/max/step}``
    marked optimizable with a default ±50% range. Unknown event_types are skipped (return None)
    so a partial live rule never breaks the import.
    """
    et = trig.get("event_type")
    leaf_id = f"c{idx}"
    flag_field = _EVENT_FLAG_FIELD.get(et)
    if flag_field is not None:
        return {"id": leaf_id, "field": flag_field, "field_type": "flag"}
    num_field = _EVENT_NUMERIC_FIELD.get(et)
    if num_field is not None:
        value = trig.get("value")
        vmin, vmax, vstep = _opt_range(value)
        return {
            "id": leaf_id,
            "field": num_field,
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


def _eventaction_to_entry_group(ea_id, triggers: dict) -> tuple[dict, list[dict]] | None:
    """Convert one live enter_market EventAction's triggers into an AND-group of tree leaves.

    Returns ``(group, leaves)`` where ``group`` is ``{id, operator:"AND", conditions:[leaves]}``,
    or None if the EventAction yields no recognizable leaves (so an all-unknown rule is dropped
    rather than emitting an empty group).
    """
    leaves: list[dict] = []
    for i, trig in enumerate((triggers or {}).values()):
        if not isinstance(trig, dict):
            continue
        leaf = _trigger_to_entry_leaf(i, trig)
        if leaf is not None:
            leaves.append(leaf)
    if not leaves:
        return None
    group = {"id": f"grp-{ea_id}", "operator": "AND", "conditions": leaves}
    return group, leaves


def _entry_action_side(actions: dict) -> str | None:
    """Return "buy"/"sell" for an enter_market EventAction's open action, or None if neither."""
    for cfg in (actions or {}).values():
        if not isinstance(cfg, dict):
            continue
        at = cfg.get("action_type")
        if at == _BUY_ACTION:
            return "buy"
        if at == _SELL_ACTION:
            return "sell"
    return None


def _groups_to_tree(groups: list[dict]) -> dict | None:
    """Combine AND-groups into one entry condition tree: single group as-is, multiple OR-ed."""
    if not groups:
        return None
    if len(groups) == 1:
        return groups[0]
    return {"id": "grp-or", "operator": "OR", "conditions": groups}


def _read_live_enter_market_trees(db_path: str, expert_id: int) -> dict:
    """READ-ONLY read of an expert's enter_market ruleset -> buy/sell entry condition trees.

    Resolves ``expertinstance.enter_market_ruleset_id`` -> ordered EventActions (shared reader),
    converts each EventAction's triggers to an AND-group, routes it to buy/sell by the action's
    ``action_type``, and combines per-side groups (single -> as-is, multiple -> OR). Returns
    ``{"buy_entry_conditions": <tree|None>, "sell_entry_conditions": <tree|None>}``.
    """
    buy_groups: list[dict] = []
    sell_groups: list[dict] = []
    for ea in _expert_ruleset_eas(db_path, expert_id, "enter_market_ruleset_id"):
        side = _entry_action_side(ea["actions"])
        if side is None:
            continue
        converted = _eventaction_to_entry_group(ea["id"], ea["triggers"])
        if converted is None:
            continue
        group, _leaves = converted
        (buy_groups if side == "buy" else sell_groups).append(group)
    return {
        "buy_entry_conditions": _groups_to_tree(buy_groups),
        "sell_entry_conditions": _groups_to_tree(sell_groups),
    }


@router.get("/experts/{expert_id}/enter-market-ruleset")
def get_enter_market_ruleset(expert_id: int):
    """Import a LIVE expert's enter_market ruleset as buy/sell entry condition TREES (optional).

    The INVERSE of ``triggers_from_condition_tree``: each enter_market EventAction's triggers
    become a condition-tree AND-group (numeric leaves marked optimizable with default ranges,
    flag leaves value-less), routed to ``buy_entry_conditions`` / ``sell_entry_conditions`` by the
    action's ``action_type`` (buy/sell). Graceful/optional: 503 when ``BA2_LIVE_DB`` is unset or
    the DB is unreadable (UI falls back to JSON-paste), 404 for a missing expert; never 500.
    """
    db_path = os.environ.get("BA2_LIVE_DB")
    if not db_path:
        raise HTTPException(
            status_code=503,
            detail="live DB not configured; paste the ruleset JSON instead",
        )
    try:
        trees = _read_live_enter_market_trees(db_path, expert_id)
    except HTTPException:
        raise  # 404 (expert not found) passes through unchanged
    except Exception as exc:  # noqa: BLE001 — any DB/parse failure degrades to a graceful 503
        logger.warning("live enter_market import failed for expert %s: %s", expert_id, exc)
        raise HTTPException(
            status_code=503,
            detail="could not read live DB; paste the ruleset JSON instead",
        )
    return trees
