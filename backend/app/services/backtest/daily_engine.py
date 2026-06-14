"""Custom daily multi-asset backtest engine.

Drives the REAL ba2trade decision/order path against the simulated ``BacktestAccount``,
with NO ``TradeManager`` import — the thin driver loop is re-implemented here from the
SAME packaged pieces the live ``TradeManager.process_expert_recommendations_after_analysis``
uses (BA2TradePlatform/.../core/TradeManager.py lines ~901-1190):

  per bar (a single simulated trading day ``as_of``):
    1. advance the virtual clock + BUST the per-account price cache (the gotcha);
    2. resolve the universe for the bar (static enabled_instruments, filtered to bars);
    3. for each (expert, settings):
         a. build the Phase-1 ``BacktestContext`` (providers / settings / account / as_of);
         b. for each symbol: ``rec = expert.analyze_as_of(as_of, ctx)`` — the SAME _gather+
            _process the live ``run_analysis`` runs — then, for a non-skip / non-HOLD
            actionable recommendation, persist an ``ExpertRecommendation`` row in the
            backtest DB and run it through the enter_market ruleset via
            ``TradeActionEvaluator.evaluate(...).execute(submit_to_broker=False)`` (creates a
            PENDING qty=0 ``TradingOrder``, exactly like live);
         c. once per expert: ``TradeRiskManagement(indicator_provider=<pandas indicators>)
            .review_and_prioritize_pending_orders(expert_instance_id)`` sizes the pending
            orders (classic RM + ``position_sizing.compute_risk_based_quantity`` /
            ``get_latest_atr``), then ``account.submit_order(order)`` for each sized order;
    4. ``account.refresh_orders()`` (the fill engine) + ``account.refresh_transactions()``
       (inherited WAITING->OPENED->CLOSED lifecycle) roll the bar's order/transaction state;
    5. ``account.snapshot_equity(as_of)`` records the per-bar equity curve point.

The decision logic is NOT perturbed: ``analyze_as_of`` is byte-identical to the Phase-1
golden path; the engine only wires the as_of clock + the order/RM driver around it.

Determinism: ``random``/``numpy`` are seeded from ``config["seed"]`` before the loop so a
run is reproducible (same cache + same params + same seed => identical equity curve).

Reuses (does NOT redefine):
  * ``ba2_common.core.backtest_context.BacktestContext`` + ``LiveProviderBundle`` (Phase 1).
  * ``ba2_common.core.TradeActionEvaluator.TradeActionEvaluator`` (enter/exit ruleset).
  * ``ba2_common.core.TradeRiskManagement.TradeRiskManagement`` (classic RM + sizing).
  * ``app.services.backtest.seam_wiring.make_indicator_provider`` (ATR injection seam).
  * the host ``BacktestAccount`` (submit_order / refresh_orders / refresh_transactions).
"""
from __future__ import annotations

import random
from datetime import date, datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ba2_common.core.backtest_context import BacktestContext, LiveProviderBundle
from ba2_common.core.db import add_instance
from ba2_common.core.models import ExpertRecommendation
from ba2_common.core.types import OrderRecommendation, RiskLevel, TimeHorizon
from ba2_common.logger import logger

from app.services.backtest.seam_wiring import make_indicator_provider


# ---------------------------------------------------------------------------
# Clock + universe hooks
# ---------------------------------------------------------------------------
def trading_days(start: datetime, end: datetime, price_source) -> List[Any]:
    """The backtest clock = the union of dataset bar keys in ``[start, end]``.

    Using the price source's own bar keys (not a synthetic calendar) keeps the clock
    aligned to available data: no phantom bars when nothing traded. Returns sorted bar
    keys — ``date`` for a daily source, ``datetime`` for an intraday source (so the
    loop steps once per intraday bar). Filtering is done on datetimes so a date key and
    a datetime key compare consistently against the ``[start, end]`` bounds.
    """
    lo = _to_dt(start)
    hi_intraday = getattr(price_source, "is_intraday", False)
    # For an intraday source compare to the exact end timestamp; for a daily source
    # keep the inclusive end-of-day bound (a date key compares within [lo_date, hi_date]).
    hi = _to_dt(end) if hi_intraday else _to_dt(end).replace(hour=23, minute=59, second=59)
    return [d for d in price_source.all_dates() if lo <= _to_dt(d) <= hi]


def resolve_universe(as_of: datetime, config: Dict[str, Any], price_source) -> List[str]:
    """v1 universe: the static ``enabled_instruments`` list, filtered to symbols that
    actually have a bar on ``as_of`` (a symbol with no bar today cannot be analysed/priced).

    Phase 3 replaces the body with the historical-screener reconstruction; the hook
    (signature + filter) is built now so the swap is body-only.
    """
    universe = config["enabled_instruments"]
    return [s for s in universe if price_source.bar_at(s, as_of) is not None]


def _to_dt(d: Any) -> datetime:
    """Normalise a date/datetime/str bar key to a tz-naive ``datetime`` for comparison.

    A ``date`` key becomes that day's midnight; a tz-aware datetime is converted to
    naive UTC. Lets daily (date) and intraday (datetime) clocks be range-filtered uniformly.
    """
    if isinstance(d, datetime):
        return d.astimezone(timezone.utc).replace(tzinfo=None) if d.tzinfo else d
    if isinstance(d, date):
        return datetime(d.year, d.month, d.day)
    if isinstance(d, str):
        return _to_dt(datetime.fromisoformat(d))
    raise TypeError(f"Cannot normalise {d!r} ({type(d)}) to a datetime")


def _as_date(d: Any) -> date:
    """Normalise a datetime/date to a calendar ``date`` (the bar-index key type)."""
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    if isinstance(d, str):
        return datetime.fromisoformat(d).date()
    raise TypeError(f"Cannot normalise {d!r} ({type(d)}) to a date")


# ---------------------------------------------------------------------------
# Recommendation -> ExpertRecommendation row
# ---------------------------------------------------------------------------
def _recommendation_to_expert_recommendation(
    rec: Any,
    *,
    expert_instance_id: int,
    symbol: str,
    as_of: datetime,
) -> Optional[int]:
    """Persist a Phase-1 ``Recommendation`` value object as an ``ExpertRecommendation`` row
    in the backtest DB and return its id (or ``None`` if not actionable).

    Mirrors live ``run_analysis`` step 6 (BA2TradePlatform core) which maps the value
    object to an ``ExpertRecommendation`` row. SKIP and HOLD are NOT persisted as actionable
    rows (the live enter loop filters ``recommended_action != HOLD`` and skips SKIP), so the
    engine returns ``None`` for them — they leave the ledger untouched this bar.

    The row's ``instance_id`` MUST equal the ExpertInstance id so the inherited
    ``_create_transaction_for_order`` derives the correct ``Transaction.expert_id`` (which
    the ruleset position conditions and the RM query by).
    """
    if getattr(rec, "skip", False):
        return None
    action = rec.signal
    if action == OrderRecommendation.HOLD or action == OrderRecommendation.ERROR:
        return None

    # expected_profit_percent / confidence are required (non-nullable) on the row; the
    # RM prioritises by expected_profit_percent. Live uses 0.0 when the expert leaves it
    # unset, but our clean experts populate it — fall back to 0.0 only if genuinely None.
    expected_profit = rec.expected_profit_percent
    if expected_profit is None:
        expected_profit = 0.0

    row = ExpertRecommendation(
        instance_id=expert_instance_id,
        market_analysis_id=None,
        symbol=symbol,
        recommended_action=action,
        expected_profit_percent=float(expected_profit),
        price_at_date=float(rec.current_price),
        details=rec.details or "",
        confidence=(None if rec.confidence is None else float(rec.confidence)),
        risk_level=RiskLevel.MEDIUM,
        time_horizon=TimeHorizon.MEDIUM_TERM,
        data=(dict(rec.raw_outputs) if rec.raw_outputs else None),
        created_at=as_of,
    )
    return add_instance(row)


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------
class DailyBacktestEngine:
    """Daily multi-asset simulator driving the real ba2trade order path.

    Args (keyword-only):
        account: the wired ``BacktestAccount`` (already registered on the resolver).
        experts: list of ``(expert_instance, expert_instance_id, expert_settings, ruleset_id)``
            tuples. ``expert_instance`` is a ba2_experts object (e.g. ``FMPEarningsDrift``)
            registered on the resolver under ``expert_instance_id``; ``expert_settings`` is the
            resolved settings dict fed to ``_process`` (the optimizer-override seam);
            ``ruleset_id`` is the enter_market ruleset to evaluate (seeded in the backtest DB);
            it is ignored (and may be ``None``) for a BYPASS expert that declares
            ``bypasses_classic_rm`` — such an expert rebalances to target weights via its own
            FactorPortfolioManager instead of the enter/exit ruleset + classic RM.
        price_source: the ``AsOfPriceSource`` (the virtual clock + bar store).
        config: the run config dict (validated fail-early by the handler). Required keys read
            here: ``start_date``, ``end_date``, ``enabled_instruments``, ``seed``. Optional:
            ``subtype``.
        progress_cb: ``callable(pct: float, msg: str)`` invoked once per bar (the handler
            wires pause/progress through it). Defaults to a no-op.
        indicator_provider: the injected indicators provider for ATR sizing. Defaults to
            ``make_indicator_provider()`` (the ohlcv/'fmp'-backed pandas indicator calc).
    """

    def __init__(
        self,
        *,
        account: Any,
        experts: List[Tuple[Any, int, Dict[str, Any], int]],
        price_source: Any,
        config: Dict[str, Any],
        progress_cb: Optional[Callable[[float, str], None]] = None,
        indicator_provider: Any = None,
    ) -> None:
        self.account = account
        self.experts = experts
        self.price = price_source
        self.config = config
        self.progress_cb = progress_cb or (lambda pct, msg: None)
        self.seed = config["seed"]
        self._indicator_provider = indicator_provider

    # -- the loop -----------------------------------------------------------
    def run(self) -> Dict[str, Any]:
        """Run the full simulation and return a results dict (Task 5 ``build_results`` shape).

        Task 4 returns a minimal results payload (equity_history + filled trades) so the
        loop is independently testable; Task 5's ``build_results`` consumes the SAME account
        (``get_balance_history``/``get_filled_trades``) to produce the final metrics blob.
        """
        # Determinism: seed BEFORE any decision so a run is byte-reproducible.
        random.seed(self.seed)
        np.random.seed(self.seed & 0xFFFFFFFF)

        # ATR injection seam: build once, reuse across bars/experts.
        indicator_provider = self._indicator_provider
        if indicator_provider is None:
            indicator_provider = make_indicator_provider()

        days = trading_days(self.config["start_date"], self.config["end_date"], self.price)
        total = max(len(days), 1)

        for i, as_of in enumerate(days):
            # Tz-AWARE UTC clock — the SAME contract the live path assumes: the experts'
            # _process does ``now = as_of or datetime.now(timezone.utc)`` and then subtracts
            # tz-aware report/transaction dates, so a NAIVE as_of would raise
            # "can't subtract offset-naive and offset-aware datetimes". Using aware UTC here
            # makes the backtest clock byte-identical to the live ``datetime.now(timezone.utc)``.
            # A daily key (date) becomes midnight UTC (historical behaviour); an intraday key
            # (datetime) keeps its time component so the bar timestamp is preserved.
            if isinstance(as_of, datetime):
                as_of_dt = as_of if as_of.tzinfo else as_of.replace(tzinfo=timezone.utc)
            else:
                as_of_dt = datetime(as_of.year, as_of.month, as_of.day, tzinfo=timezone.utc)

            # 1. advance the clock + bust the per-account price cache (the gotcha).
            self.price.set_clock(as_of_dt)
            self._bust_price_cache()

            # 2. universe for the bar.
            universe = resolve_universe(as_of_dt, self.config, self.price)

            # 3. each expert: analyze_as_of -> persist rec -> ruleset -> RM -> submit.
            #    BYPASS experts (piece 1b): an expert that declares ``bypasses_classic_rm``
            #    (e.g. FactorRanker) does NOT use the enter/exit ruleset OR the classic risk
            #    manager. It emits {symbol: weight} target weights once per bar and rebalances
            #    via its own FactorPortfolioManager — so we route its targets DIRECTLY to the
            #    portfolio manager (which itself prices + submits orders), SKIPPING
            #    TradeActionEvaluator/TradeConditions, TradeRiskManagement and position_sizing.
            for expert, expert_id, settings, ruleset_id in self.experts:
                if getattr(expert, "bypasses_classic_rm", False):
                    self._run_bypass_expert_bar(expert, expert_id, settings, as_of_dt)
                    continue
                created_any = self._run_expert_bar(
                    expert, expert_id, settings, ruleset_id, universe, as_of_dt
                )
                if created_any:
                    self._size_and_submit(expert_id, indicator_provider)

            # 4. fills on THIS bar's working orders; roll order state into transactions.
            self.account.refresh_orders()
            self.account.refresh_transactions()

            # 5. record per-bar equity / drawdown point.
            self.account.snapshot_equity(as_of_dt)

            self.progress_cb((i + 1) / total * 100.0, f"bar {as_of:%Y-%m-%d}")

        return self._build_minimal_results()

    # -- per-expert, per-bar ------------------------------------------------
    def _run_expert_bar(
        self,
        expert: Any,
        expert_id: int,
        settings: Dict[str, Any],
        ruleset_id: int,
        universe: List[str],
        as_of: datetime,
    ) -> bool:
        """Analyse every universe symbol for one expert and stage PENDING orders.

        Returns True iff at least one PENDING order was created (so the caller knows to run
        the risk manager). Per-symbol failures are logged and skipped (a bad symbol must not
        abort the whole bar) — matching the live loop's per-recommendation try/except.
        """
        from ba2_common.core.TradeActionEvaluator import TradeActionEvaluator

        providers = self._provider_bundle()
        created_any = False

        for symbol in universe:
            # The per-symbol expert decision: ``analyze_as_of`` -> ``_gather`` reads
            # ``self._gather_symbol`` (the live ``run_analysis`` sets it before _gather), so
            # the engine must pin the symbol on the shared expert object each iteration.
            # The STUB experts in the unit tests ignore it; the real ba2_experts require it.
            try:
                expert._gather_symbol = symbol
            except Exception:  # noqa: BLE001 — a stub without the attr is fine
                pass
            ctx = BacktestContext(
                providers=providers,
                settings=settings,
                as_of=as_of,
                account=self.account,
                subtype=self.config.get("subtype"),
            )
            try:
                rec = expert.analyze_as_of(as_of, ctx)
            except Exception as e:  # noqa: BLE001 — one symbol must not abort the bar
                self._log(f"analyze_as_of failed for {symbol} @ {as_of:%Y-%m-%d}: {e}")
                continue

            rec_id = _recommendation_to_expert_recommendation(
                rec, expert_instance_id=expert_id, symbol=symbol, as_of=as_of
            )
            if rec_id is None:
                continue  # SKIP / HOLD / ERROR — nothing to stage.

            # Re-read the persisted row so the evaluator/actions see a DB-attached object
            # carrying its id (BuyAction links the order to expert_recommendation.id).
            from ba2_common.core.db import get_instance as _get_instance

            recommendation = _get_instance(ExpertRecommendation, rec_id)
            if recommendation is None:
                continue

            try:
                evaluator = TradeActionEvaluator(
                    account=self.account,
                    instrument_name=symbol,
                    existing_transactions=None,
                )
                action_summaries = evaluator.evaluate(
                    instrument_name=symbol,
                    expert_recommendation=recommendation,
                    ruleset_id=ruleset_id,
                    existing_order=None,
                )
                if not action_summaries or any("error" in s for s in action_summaries):
                    continue  # conditions not met / evaluation error -> no order this symbol.

                # Create PENDING qty=0 orders (NOT submitted: RM sizes + submits next).
                results = evaluator.execute(submit_to_broker=False)
                if any(r.get("success") and (r.get("data") or {}).get("order_id") for r in results):
                    created_any = True
            except Exception as e:  # noqa: BLE001
                self._log(f"ruleset eval/execute failed for {symbol} @ {as_of:%Y-%m-%d}: {e}")
                continue

        return created_any

    def _run_bypass_expert_bar(
        self,
        expert: Any,
        expert_id: int,
        settings: Dict[str, Any],
        as_of: datetime,
    ) -> None:
        """Run ONE bar for a BYPASS expert (piece 1b): rebalance to target weights.

        A bypass expert (``getattr(expert, 'bypasses_classic_rm', False)`` is True, e.g.
        FactorRanker) resolves its OWN universe internally, so ``analyze_as_of`` is called
        ONCE for the bar (not per-symbol). The returned recommendation carries
        ``raw_outputs['targets']`` — the ``{symbol: weight}`` book — which is routed DIRECTLY
        through ``FactorPortfolioManager(expert_id).rebalance(targets)``. That manager prices
        each name off the account, diffs the targets against the expert's current holdings, and
        calls ``account.submit_order`` for each delta. The classic decision path is SKIPPED in
        full: NO TradeActionEvaluator/TradeConditions, NO ExpertRecommendation row, NO
        TradeRiskManagement / position_sizing.

        A skip / empty-targets recommendation is a no-op for the bar (nothing to rebalance).
        A per-bar failure is logged and swallowed (one bad bar must not abort the run), matching
        the classic path's per-bar try/except.
        """
        ctx = BacktestContext(
            providers=self._provider_bundle(),
            settings=settings,
            as_of=as_of,
            account=self.account,
            subtype=self.config.get("subtype"),
        )
        try:
            rec = expert.analyze_as_of(as_of, ctx)
        except Exception as e:  # noqa: BLE001 — one bar must not abort the run
            self._log(f"bypass analyze_as_of failed @ {as_of:%Y-%m-%d}: {e}")
            return

        if getattr(rec, "skip", False):
            return
        raw = getattr(rec, "raw_outputs", None) or {}
        targets = raw.get("targets")
        if not targets:
            return  # no target weights this bar -> nothing to rebalance.

        from ba2_experts.FactorRanker.portfolio import FactorPortfolioManager

        try:
            FactorPortfolioManager(expert_id).rebalance(targets)
        except Exception as e:  # noqa: BLE001 — a rebalance failure must not kill the run
            self._log(f"bypass rebalance failed for expert {expert_id} @ {as_of:%Y-%m-%d}: {e}")

    def _size_and_submit(self, expert_id: int, indicator_provider: Any) -> None:
        """Classic RM sizes the PENDING orders, then submit each sized order to the sim.

        This is the live ``process_expert_recommendations_after_analysis`` tail (lines
        ~1167-1190): ``TradeRiskManagement.review_and_prioritize_pending_orders`` sets each
        order's quantity (via ``compute_risk_based_quantity`` / ``get_latest_atr`` using the
        injected indicator provider), then ``account.submit_order(order)`` sends the sized
        ones. ATR injection is the exact Phase-0 seam: ``TradeRiskManagement(indicator_provider=...)``.
        """
        from ba2_common.core.TradeRiskManagement import TradeRiskManagement

        rm = TradeRiskManagement(indicator_provider=indicator_provider)
        try:
            updated_orders = rm.review_and_prioritize_pending_orders(expert_id)
        except Exception as e:  # noqa: BLE001 — RM failure for one expert must not kill the run
            self._log(f"risk manager failed for expert {expert_id}: {e}")
            return

        for order in updated_orders:
            if order.quantity and order.quantity > 0:
                try:
                    self.account.submit_order(order)
                except Exception as e:  # noqa: BLE001
                    self._log(f"submit_order failed for order {order.id}: {e}")

    # -- helpers ------------------------------------------------------------
    def _provider_bundle(self) -> Any:
        """The as_of-aware ProviderBundle fed to each expert's ``_gather``.

        Reuses Phase-1's ``LiveProviderBundle`` over the host's ba2_providers registry
        (resolved via the same ``TradeConditions`` provider resolver wired in Task 1). The
        providers are as_of-aware (the engine threads ``as_of`` into ``analyze_as_of``), so
        the bundle is constructed once and shared across bars.
        """
        bundle = getattr(self, "_bundle_cache", None)
        if bundle is None:
            from ba2_common.core.TradeConditions import _get_provider

            bundle = LiveProviderBundle(
                lambda category, name, **kw: _get_provider(category, name, **kw)
            )
            self._bundle_cache = bundle
        return bundle

    def _bust_price_cache(self) -> None:
        """Pop the per-account entry from the inherited wall-clock price cache.

        Belt-and-braces with ``BacktestAccount.get_instrument_current_price`` (which already
        bypasses the cache): any inherited caller that still routes through the cached path
        gets a fresh as-of price every bar instead of a stale virtual-day-N value.
        """
        cache = getattr(type(self.account), "_GLOBAL_PRICE_CACHE", None)
        if isinstance(cache, dict):
            cache.pop(self.account.id, None)

    def _build_minimal_results(self) -> Dict[str, Any]:
        """Task-4 results payload: the equity history + filled trades from the account.

        Task 5's ``build_results`` produces the full Backtest metric blob from the SAME
        account; this minimal dict keeps the engine independently testable and gives the
        handler a consistent return shape until Task 5 lands.
        """
        return {
            "equity_history": self.account.get_balance_history(),
            "trades": self.account.get_filled_trades(),
            "final_equity": self.account.equity(),
            "initial_capital": float(self.account._cfg["starting_cash"]),
        }

    @staticmethod
    def _log(msg: str) -> None:
        logger.warning(f"[daily_engine] {msg}")
