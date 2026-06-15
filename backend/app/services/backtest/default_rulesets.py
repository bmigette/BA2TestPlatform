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
)


def _make_event_action(name: str, triggers: dict, actions: dict) -> int:
    """Create one enter_market ``EventAction`` and return its id."""
    ea = EventAction(
        name=name,
        type=ExpertEventRuleType.TRADING_RECOMMENDATION_RULE,
        subtype=AnalysisUseCase.ENTER_MARKET,
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
}


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
