"""Seed the minimal enter_market ruleset(s) the daily engine drives.

The live platform configures rulesets through the UI; the backtest host seeds a small,
faithful default directly into the per-run backtest DB so ``TradeActionEvaluator`` has a
ruleset to evaluate (``ExpertInstance.enter_market_ruleset_id`` points at it).

The default enter ruleset is the standard "enter on a bullish recommendation when flat"
rule: a single ``EventAction`` whose triggers are the packaged ``BullishCondition``
(``recommended_action == BUY``) AND ``HasNoPositionCondition`` (no existing expert position
for the symbol — prevents duplicate entries), and whose single action is a ``buy``. This is
exactly the ruleset shape the live ``TradeActionEvaluator`` evaluates; the engine does NOT
invent a new evaluation path.

``enter_long_short_ruleset`` additionally adds the symmetric SELL-on-bearish rule for experts
that short (gated by the expert's ``enable_sell`` setting in the RM, so it is safe to include).

Trigger/action JSON shapes verified against ba2_common:
  * trigger: ``{"<key>": {"event_type": "<ExpertEventType value>"}}`` — empty/unknown
    operators are fine for flag conditions; ``BullishCondition``/``HasNoPositionCondition``
    take no value. ``ExpertEventType.F_BULLISH = "bullish"``, ``F_HAS_NO_POSITION =
    "has_no_position"``, ``F_BEARISH = "bearish"``.
  * action: ``{"<key>": {"action_type": "<ExpertActionType value>"}}`` — ``ExpertActionType.BUY
    = "buy"``, ``SELL = "sell"``. The action is parsed by ``_create_and_store_trade_actions``.
"""
from __future__ import annotations

from typing import List

from sqlmodel import Session

from ba2_common.core.db import add_instance, get_db
from ba2_common.core.models import EventAction, Ruleset, RulesetEventActionLink
from ba2_common.core.types import (
    AnalysisUseCase,
    ExpertActionType,
    ExpertEventRuleType,
    ExpertEventType,
    ReferenceValue,
)


def _make_event_action(name: str, triggers: dict, actions: dict,
                       subtype: "AnalysisUseCase" = AnalysisUseCase.ENTER_MARKET) -> int:
    """Create one ``EventAction`` (default enter_market subtype) and return its id."""
    ea = EventAction(
        name=name,
        type=ExpertEventRuleType.TRADING_RECOMMENDATION_RULE,
        subtype=subtype,
        triggers=triggers,
        actions=actions,
        extra_parameters={},
        continue_processing=False,
    )
    return add_instance(ea)


def _link(ruleset_id: int, event_action_ids: List[int]) -> None:
    """Attach the event actions to the ruleset (ordered) via the M2M link table.

    ``RulesetEventActionLink`` has a composite PK (ruleset_id + eventaction_id) and NO ``id``
    column, so ``add_instance`` (which reads ``.id`` after flush) cannot be used — insert via
    a session directly, mirroring the rules_export_import session-based link inserts.
    """
    with Session(get_db().bind) as session:
        for order_index, ea_id in enumerate(event_action_ids):
            session.add(
                RulesetEventActionLink(
                    ruleset_id=ruleset_id,
                    eventaction_id=ea_id,
                    order_index=order_index,
                )
            )
        session.commit()


# Strategy condition-tree field -> ExpertEventType for value (N_*) gates. These are the
# fields the optimizer's cond:<id>:value genes tune on a buy/sell entry tree; an unknown
# field is skipped (it never silently breaks the ruleset).
_FIELD_EVENT = {
    "confidence": ExpertEventType.N_CONFIDENCE,
    "expected_profit": ExpertEventType.N_EXPECTED_PROFIT_TARGET_PERCENT,
    "expected_profit_percent": ExpertEventType.N_EXPECTED_PROFIT_TARGET_PERCENT,
    "expected_profit_target_percent": ExpertEventType.N_EXPECTED_PROFIT_TARGET_PERCENT,
    # Cooldown gates (avoid re-buying the same symbol right after exiting it). Pair with ">"
    # so the entry only fires once N days have passed since the last (qualifying) close.
    "days_since_last_close": ExpertEventType.N_DAYS_SINCE_LAST_CLOSE,
    "days_since_last_profitable_close": ExpertEventType.N_DAYS_SINCE_LAST_PROFITABLE_CLOSE,
    "days_since_last_losing_close": ExpertEventType.N_DAYS_SINCE_LAST_LOSING_CLOSE,
    # Exit (open_positions) numeric conditions.
    "profit_loss_percent": ExpertEventType.N_PROFIT_LOSS_PERCENT,
    "profit_loss_amount": ExpertEventType.N_PROFIT_LOSS_AMOUNT,
    "days_opened": ExpertEventType.N_DAYS_OPENED,
    "percent_to_current_target": ExpertEventType.N_PERCENT_TO_CURRENT_TARGET,
    "new_target_percent": ExpertEventType.N_NEW_TARGET_PERCENT,
}

# Flag (boolean) condition fields -> ExpertEventType (no operator/value). Used by exit
# (open_positions) rules whose triggers include sentiment / term / risk / rating-change /
# position flags — exactly the live open_positions trigger vocabulary.
_FLAG_FIELD_EVENT = {
    "bullish": ExpertEventType.F_BULLISH,
    "bearish": ExpertEventType.F_BEARISH,
    "has_position": ExpertEventType.F_HAS_POSITION,
    "has_no_position": ExpertEventType.F_HAS_NO_POSITION,
    "has_buy_position": ExpertEventType.F_HAS_BUY_POSITION,
    "has_sell_position": ExpertEventType.F_HAS_SELL_POSITION,
    "short_term": ExpertEventType.F_SHORT_TERM,
    "medium_term": ExpertEventType.F_MEDIUM_TERM,
    "long_term": ExpertEventType.F_LONG_TERM,
    "highrisk": ExpertEventType.F_HIGHRISK,
    "mediumrisk": ExpertEventType.F_MEDIUMRISK,
    "lowrisk": ExpertEventType.F_LOWRISK,
    "new_target_higher": ExpertEventType.F_NEW_TARGET_HIGHER,
    "new_target_lower": ExpertEventType.F_NEW_TARGET_LOWER,
    "current_rating_positive": ExpertEventType.F_CURRENT_RATING_POSITIVE,
    "current_rating_negative": ExpertEventType.F_CURRENT_RATING_NEGATIVE,
}

# Exit action_type string -> (ExpertActionType, needs_reference_value). The adjust actions read
# reference_value (order_open_price/current_price/expert_target_price) + value (the % offset);
# close/sell take no params. Mirrors TradeActionEvaluator's action_config parsing.
_EXIT_ACTION = {
    "close": (ExpertActionType.CLOSE, False),
    "sell": (ExpertActionType.SELL, False),
    "adjust_take_profit": (ExpertActionType.ADJUST_TAKE_PROFIT, True),
    "adjust_stop_loss": (ExpertActionType.ADJUST_STOP_LOSS, True),
}


def _triggers_from_conditions(tree) -> dict:
    """Build an EventAction ``triggers`` dict (ANDed) from an exit-rule condition tree.

    Flag leaves (``_FLAG_FIELD_EVENT``) become value-less triggers; numeric leaves
    (``_FIELD_EVENT``) carry operator + value (the optimizer's cond:<id>:value gene). Unknown
    fields are skipped so a partial/edited tree never silently breaks the rule.
    """
    triggers: dict = {}
    for i, leaf in enumerate(_tree_leaves(tree)):
        field = str(leaf.get("field"))
        flag_et = _FLAG_FIELD_EVENT.get(field)
        if flag_et is not None:
            triggers[f"cond_{i}"] = {"event_type": flag_et.value}
            continue
        num_et = _FIELD_EVENT.get(field)
        if num_et is not None and leaf.get("value") is not None:
            triggers[f"cond_{i}"] = {
                "event_type": num_et.value,
                "operator": leaf.get("op") or leaf.get("operator") or ">",
                "value": leaf.get("value"),
            }
    return triggers


def _exit_action_json(rule: dict) -> dict | None:
    """Build an EventAction ``actions`` dict for one exit rule, or None if the action is unknown.

    ``rule['action_type']`` selects the action; adjust actions also carry ``reference_value``
    and ``action_value`` (the % offset the optimizer tunes via exit:<id>:action_value).
    """
    spec = _EXIT_ACTION.get(str(rule.get("action_type")))
    if spec is None:
        return None
    action_type, needs_ref = spec
    cfg: dict = {"action_type": action_type.value}
    if needs_ref:
        cfg["reference_value"] = rule.get("reference_value") or ReferenceValue.ORDER_OPEN_PRICE.value
        cfg["value"] = rule.get("action_value")
    return {"act": cfg}


def seed_open_positions_ruleset(exit_rules, name: str = "backtest-open-positions") -> int:
    """Seed an OPEN_POSITIONS ruleset from a Strategy exit-rule LIST; return its id.

    Each entry in ``exit_rules`` (the shape ``decode_params`` emits: ``{id, conditions,
    action_type, reference_value, action_value, enabled}``; enabled-off rules are already
    pruned) becomes ONE ordered ``EventAction`` whose triggers are the ANDed condition leaves
    and whose single action is Close/Sell/Adjust-TP/Adjust-SL. This is evaluated by the SAME
    packaged ``TradeActionEvaluator`` the live ``process_open_positions_recommendations`` uses
    (open_positions use case), so the backtest manages open positions identically to live.

    A rule with no usable action is skipped. Returns the ruleset id (with NO event actions if
    every rule was skipped — the caller can treat that as "no exit management").
    """
    ruleset = Ruleset(
        name=name,
        description="Backtest open_positions ruleset built from a Strategy exit-rule list.",
        type=ExpertEventRuleType.TRADING_RECOMMENDATION_RULE,
        subtype=AnalysisUseCase.OPEN_POSITIONS,
    )
    ruleset_id = add_instance(ruleset)

    ea_ids = []
    for idx, rule in enumerate(exit_rules or []):
        action = _exit_action_json(rule)
        if action is None:
            continue
        triggers = _triggers_from_conditions(rule.get("conditions"))
        ea_ids.append(
            _make_event_action(
                name=f"{name}-rule-{idx}",
                triggers=triggers,
                actions=action,
                subtype=AnalysisUseCase.OPEN_POSITIONS,
            )
        )
    if ea_ids:
        _link(ruleset_id, ea_ids)
    return ruleset_id


def _tree_leaves(node):
    """Yield leaf condition dicts (those with a ``field``) from an AND/OR condition tree."""
    if not isinstance(node, dict):
        return
    kids = node.get("conditions")
    if kids:
        for child in kids:
            yield from _tree_leaves(child)
    elif node.get("field"):
        yield node


def seed_ruleset_from_tree(buy_tree, name: str = "backtest-enter-tree") -> int:
    """Seed an enter_market ruleset from a Strategy buy-entry condition TREE; return its id.

    The base "BUY when bullish and flat" triggers are kept, AND each leaf value-condition in
    the tree is added as an extra trigger (event_type from _FIELD_EVENT, with the leaf's
    operator + value). Triggers in one EventAction are ANDed, so this realises a root-AND tree
    of entry gates (e.g. confidence > X AND expected_profit > Y) — exactly what the optimizer's
    cond:<id>:value / on-off-toggle genes tune. Unknown fields are skipped. (OR nesting and exit
    rules are a follow-up; falls back to the bullish+flat default when the tree adds nothing.)
    """
    triggers = {
        "bullish": {"event_type": ExpertEventType.F_BULLISH.value},
        "no_position": {"event_type": ExpertEventType.F_HAS_NO_POSITION.value},
    }
    for i, leaf in enumerate(_tree_leaves(buy_tree)):
        et = _FIELD_EVENT.get(str(leaf.get("field")))
        if et is None or leaf.get("value") is None:
            continue
        triggers[f"gate_{i}"] = {
            "event_type": et.value,
            "operator": leaf.get("op") or leaf.get("operator") or ">",
            "value": leaf.get("value"),
        }

    ruleset = Ruleset(
        name=name,
        description="Backtest enter ruleset built from a Strategy condition tree.",
        type=ExpertEventRuleType.TRADING_RECOMMENDATION_RULE,
        subtype=AnalysisUseCase.ENTER_MARKET,
    )
    ruleset_id = add_instance(ruleset)
    ea = _make_event_action(
        name=f"{name}-enter",
        triggers=triggers,
        actions={"buy": {"action_type": ExpertActionType.BUY.value}},
    )
    _link(ruleset_id, [ea])
    return ruleset_id


def seed_enter_long_ruleset(name: str = "backtest-enter-long") -> int:
    """Seed a "BUY when bullish and flat" enter_market ruleset; return its id.

    One rule: triggers = bullish AND has_no_position; action = buy.
    """
    ruleset = Ruleset(
        name=name,
        description="Backtest default: enter long when the expert recommends BUY and the "
        "expert has no open position for the symbol.",
        type=ExpertEventRuleType.TRADING_RECOMMENDATION_RULE,
        subtype=AnalysisUseCase.ENTER_MARKET,
    )
    ruleset_id = add_instance(ruleset)

    enter_long = _make_event_action(
        name=f"{name}-enter-long",
        triggers={
            "bullish": {"event_type": ExpertEventType.F_BULLISH.value},
            "no_position": {"event_type": ExpertEventType.F_HAS_NO_POSITION.value},
        },
        actions={"buy": {"action_type": ExpertActionType.BUY.value}},
    )
    _link(ruleset_id, [enter_long])
    return ruleset_id


def seed_enter_long_short_ruleset(name: str = "backtest-enter-long-short") -> int:
    """Seed a long+short enter_market ruleset; return its id.

    Two rules (evaluated in order, ``continue_processing=False`` so the first matching rule
    stops evaluation): bullish+flat -> buy; bearish+flat -> sell. The SELL leg is still
    gated by the expert's ``enable_sell`` permission inside the risk manager, so seeding it
    is safe for buy-only experts (their SELL orders are dropped by the RM permission filter).
    """
    ruleset = Ruleset(
        name=name,
        description="Backtest default: enter long on BUY / short on SELL when flat.",
        type=ExpertEventRuleType.TRADING_RECOMMENDATION_RULE,
        subtype=AnalysisUseCase.ENTER_MARKET,
    )
    ruleset_id = add_instance(ruleset)

    enter_long = _make_event_action(
        name=f"{name}-enter-long",
        triggers={
            "bullish": {"event_type": ExpertEventType.F_BULLISH.value},
            "no_position": {"event_type": ExpertEventType.F_HAS_NO_POSITION.value},
        },
        actions={"buy": {"action_type": ExpertActionType.BUY.value}},
    )
    enter_short = _make_event_action(
        name=f"{name}-enter-short",
        triggers={
            "bearish": {"event_type": ExpertEventType.F_BEARISH.value},
            "no_position": {"event_type": ExpertEventType.F_HAS_NO_POSITION.value},
        },
        actions={"sell": {"action_type": ExpertActionType.SELL.value}},
    )
    _link(ruleset_id, [enter_long, enter_short])
    return ruleset_id
