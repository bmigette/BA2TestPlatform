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
