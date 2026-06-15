"""``daily_backtest`` task handler (Phase 2 Task 5).

Contract matches the existing handlers (``handle_backtest`` etc.):
``handler(task_id: str, payload: dict) -> result dict``; a returned ``{'status':'failed',...}``
marks the task failed (``TaskQueueService._process_task``). The handler:

  1. validates the payload fail-early (no-defaults rule, ``backend/CLAUDE.md``);
  2. loads the host ``Backtest`` row (the results row), sets it ``running``;
  3. wires the ba2 seams, opens a per-run backtest TRADING DB, seeds the AccountDefinition +
     ExpertInstance rows, builds the ``BacktestAccount`` + the (clean) expert instances;
  4. runs ``DailyBacktestEngine`` (the real ba2trade order path), polling ``is_task_paused``
     and pushing ``update_progress`` via the engine's progress callback;
  5. converts the finished account to the metric blob (``results.build_results``) and persists
     every metric column + the equity/drawdown/trades JSON onto the ``Backtest`` row,
     ``status='completed'``.

Two distinct DBs (per the replan): the ``Backtest`` RESULTS row lives in the host
``app.models.database.SessionLocal`` DB; the TRADING rows (TradingOrder/Transaction/...) live
in the separate per-run ``ba2_common.core.db`` sqlite (``backtest_trading_db``).
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

from app.models.backtest import Backtest
from app.models.database import SessionLocal
from app.services.task_queue import get_task_queue

logger = logging.getLogger(__name__)


# Payload keys the handler REQUIRES (validated fail-early, no defaults).
# ``enabled_instruments`` is NOT in this list: it is either supplied directly (static
# universe) or RESOLVED from the offline screener cache (screener universe) in
# ``_build_config``; the post-resolution non-empty check lives there so a screener run is
# not rejected for lacking a static symbol list.
REQUIRED_KEYS = [
    "backtest_id",
    "experts",
    "start_date",
    "end_date",
    "initial_capital",
    "commission",
    "slippage",
    "fill_model",
    "seed",
]

# The clean (no-LLM) experts this phase ships. The payload's ``experts`` list names a subset
# of these by class name; anything else is rejected fail-early (a typo must not silently no-op).
_SUPPORTED_EXPERTS = {
    "FMPEarningsDrift": "ba2_experts.FMPEarningsDrift",
    "FMPInsiderClusterBuy": "ba2_experts.FMPInsiderClusterBuy",
    # FMP analyst price-target consensus ("FMPConsensus"). Backtestable: analyze_as_of +
    # no-lookahead as_of reconstruction (grades-historical + v4/price-target history), no LLM.
    "FMPRating": "ba2_experts.FMPRating",
    # BYPASS expert (piece 1): FactorRanker declares ``bypasses_classic_rm`` — it does NOT use
    # the enter/exit ruleset or the classic RM, and rebalances to target weights via its own
    # FactorPortfolioManager. ``_build_experts`` detects the marker and skips ruleset seeding /
    # RM-gate enabling for it; the engine routes its targets straight to the portfolio manager.
    "FactorRanker": "ba2_experts.FactorRanker",
}


# Max indicator/lookback window (in trading BARS) each expert needs warmed up before the
# first trading bar. The classic-RM ATR (~14) is the floor; FactorRanker's 12-1 month
# momentum needs a full year. An expert may override via a ``BACKTEST_WARMUP_BARS`` class attr.
_EXPERT_WARMUP_BARS = {
    "FactorRanker": 252,          # momentum_12_1 lookback (12 months)
    "FMPRating": 10,
    "FMPEarningsDrift": 10,
    "FMPInsiderClusterBuy": 10,
}
_WARMUP_FLOOR_DAYS = 60           # never warm up less than this (ATR + safety)
_BARS_TO_CALDAYS = 1.45           # trading bars -> calendar days (≈252 bars/year -> ~365 days)


def derive_warmup_days(expert_specs: List[Any]) -> int:
    """Calendar-day warmup window derived from the run's experts' max bar lookback.

    Looks up each expert's lookback (the class's ``BACKTEST_WARMUP_BARS`` if present, else
    the ``_EXPERT_WARMUP_BARS`` table, else a small default), takes the max, converts BARS ->
    calendar days, and floors at ``_WARMUP_FLOOR_DAYS``. Used when the payload does not pin
    ``warmup_days`` so indicators (200-EMA, 252-bar momentum, ...) get enough history.
    """
    max_bars = 14  # ATR-14 floor
    for spec in expert_specs or []:
        name = spec.get("class") if isinstance(spec, dict) else spec
        bars = _EXPERT_WARMUP_BARS.get(name, 20)
        mod_path = _SUPPORTED_EXPERTS.get(name)
        if mod_path:
            try:
                import importlib
                cls = getattr(importlib.import_module(mod_path), name)
                bars = int(getattr(cls, "BACKTEST_WARMUP_BARS", bars))
            except Exception:  # noqa: BLE001 — fall back to the table value
                pass
        max_bars = max(max_bars, bars)
    return max(_WARMUP_FLOOR_DAYS, int(max_bars * _BARS_TO_CALDAYS) + 10)


class _Paused(Exception):
    """Raised from the progress callback when the task is paused (surfaces as a failure
    with a clear message — the queue's pause/resume re-queues the task)."""


def handle_daily_backtest(task_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Run a daily multi-asset expert backtest and persist the ``Backtest`` results row."""
    # --- validate fail-early (no defaults) ---------------------------------
    for key in REQUIRED_KEYS:
        if payload.get(key) is None:
            return {"status": "failed", "error": f"payload.{key} is required"}

    backtest_id = payload["backtest_id"]
    tq = get_task_queue()
    db = SessionLocal()
    try:
        bt = db.query(Backtest).filter(Backtest.id == backtest_id).first()
        if bt is None:
            return {"status": "failed", "error": f"Backtest {backtest_id} not found"}

        bt.status = "running"
        bt.started_at = datetime.now()
        db.commit()

        try:
            config = _build_config(payload)
        except (KeyError, ValueError) as e:
            _fail(db, bt, str(e))
            return {"status": "failed", "error": str(e)}

        def progress(pct: float, msg: str) -> None:
            if tq.is_task_paused(task_id):
                raise _Paused(msg)
            tq.update_progress(task_id, pct, msg)

        results = run_daily_backtest(config, progress_cb=progress)

        _persist_results(db, bt, results)
        bt.status = "completed"
        bt.completed_at = datetime.now()
        db.commit()
        logger.info(
            f"Daily backtest {backtest_id} completed: {results.get('total_trades', 0)} trades, "
            f"return={results.get('total_return')}%"
        )
        return {"status": "completed", "backtest_id": backtest_id, "results": results}

    except _Paused as e:
        # Leave the row 'running' status untouched? No — surface paused as a failure so the
        # row is not stuck; the queue handles re-queue on resume independently.
        _fail(db, bt, f"paused: {e}")
        return {"status": "failed", "error": "paused"}
    except Exception as e:  # noqa: BLE001 — any engine failure must fail the row, not crash the worker
        logger.error(f"Daily backtest {backtest_id} failed: {e}", exc_info=True)
        try:
            row = db.query(Backtest).filter(Backtest.id == backtest_id).first()
            if row is not None:
                _fail(db, row, str(e))
        except Exception:  # noqa: BLE001
            pass
        return {"status": "failed", "error": str(e)}
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def _build_config(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Parse + assemble the engine run config from the validated payload.

    ``account_settings`` is the BacktestAccount's resolved config dict (the no-defaults keys
    the account reads: starting_cash / commission_per_trade / slippage_bps / fill_model).
    Raises ValueError on a bad date / unsupported expert (fail-early).
    """
    start_date = _parse_dt(payload["start_date"], "start_date")
    end_date = _parse_dt(payload["end_date"], "end_date")
    if end_date < start_date:
        raise ValueError("payload.end_date must be on or after start_date")

    expert_specs = payload["experts"]
    if not isinstance(expert_specs, list) or not expert_specs:
        raise ValueError("payload.experts must be a non-empty list")
    for spec in expert_specs:
        name = spec.get("class") if isinstance(spec, dict) else spec
        if name not in _SUPPORTED_EXPERTS:
            raise ValueError(
                f"unsupported expert '{name}'; supported: {sorted(_SUPPORTED_EXPERTS)}"
            )

    enabled_instruments = _resolve_enabled_instruments(payload, start_date, end_date)

    initial_capital = float(payload["initial_capital"])
    account_settings = {
        "starting_cash": initial_capital,
        "commission_per_trade": float(payload["commission"]),
        "slippage_bps": float(payload["slippage"]),
        "fill_model": str(payload["fill_model"]),
    }

    # warmup_days: longest indicator/lookback window the experts need preloaded before
    # start_date. If the payload sets it explicitly, honour that; otherwise DERIVE it from
    # the experts' max indicator lookback (in bars -> calendar days) so e.g. a 200-EMA or
    # FactorRanker's 252-bar momentum actually has its lookback. This is a fetch-window
    # sizing knob, NOT a trading parameter, so deriving it never affects a decision.
    warmup_days = (int(payload["warmup_days"]) if payload.get("warmup_days") is not None
                   else derive_warmup_days(expert_specs))

    return {
        "backtest_id": payload["backtest_id"],
        "name": payload.get("name", f"daily-backtest-{payload['backtest_id']}"),
        "start_date": start_date,
        "end_date": end_date,
        "enabled_instruments": enabled_instruments,
        "experts": expert_specs,
        "initial_capital": initial_capital,
        "account_settings": account_settings,
        "warmup_days": warmup_days,
        "seed": int(payload["seed"]),
        "subtype": payload.get("subtype"),
        # Entry cadence (optimizer/CLI seam): {"days": {weekday: bool}, "times": [...]}.
        # None/absent -> analyse every bar (legacy). The engine's _entry_schedule honours it.
        "run_schedule_override": payload.get("run_schedule_override"),
        # Optimizer/API condition trees: when a buy-entry tree is present the engine builds the
        # enter ruleset FROM it (_build_experts -> seed_ruleset_from_tree) so the condition
        # thresholds + on/off toggles gate entries; else it falls back to the bullish+flat
        # default. The optimizer builds its config dict directly (bypassing _build_config), so
        # these forwards are what carry the trees through the API/CLI create path. sell_tree /
        # exit_rules are plumbed through for the engine's follow-up consumption.
        "buy_tree": payload.get("buy_tree"),
        "sell_tree": payload.get("sell_tree"),
        "exit_rules": payload.get("exit_rules"),
        # Initial TP/SL bracket percents applied per opened position so trades close (the
        # engine's _apply_initial_brackets reads these). Optional on the standalone path;
        # the optimizer always supplies them via the tp/sl genes.
        "initial_tp_percent": payload.get("initial_tp_percent"),
        "initial_sl_percent": payload.get("initial_sl_percent"),
        # Intraday fill clock (e.g. "1h"/"15m"); 1d default. Decoupled from entry cadence.
        "execution_interval": payload.get("execution_interval", "1d"),
    }


def _resolve_enabled_instruments(
    payload: Dict[str, Any], start_date: datetime, end_date: datetime
) -> List[str]:
    """Resolve the run's instrument list: static symbols OR an offline screener-cache union.

    Two universe shapes are supported (discriminated by ``payload['universe']['mode']``):

      * static (default, or ``universe.mode == 'static'``): the explicit ``enabled_instruments``
        list the payload carries — behaviour unchanged.
      * screener (``universe.mode == 'screener'``): resolve the symbols from the OFFLINE
        screener-history cache (built via ``ba2-test fetch-screener``) by unioning the cached
        survivors across the scan dates in ``[start_date, end_date]``. READ-ONLY — never
        live-screens; a cache miss raises ``ScreenerCacheMiss`` which propagates to fail the
        run early (build the cache first). The screener block carries ``screener_settings`` +
        ``cache_db`` + ``group`` (no defaults — fail-early per ``backend/CLAUDE.md``).
    """
    universe = payload.get("universe") or {}
    mode = universe.get("mode")

    if mode == "screener":
        from app.services.backtest.universe_resolver import resolve_screener_universe

        screener_settings = universe.get("screener_settings")
        if screener_settings is None:
            raise ValueError("universe.screener_settings is required for screener mode")
        cache_db = universe.get("cache_db")
        if not cache_db:
            raise ValueError("universe.cache_db is required for screener mode")
        group = universe.get("group")
        if not group:
            raise ValueError("universe.group is required for screener mode")

        # ScreenerCacheMiss (RuntimeError) is intentionally NOT caught here: it bubbles to the
        # handler's generic failure path so the run fails with the build-the-cache message.
        instruments = resolve_screener_universe(
            screener_settings=screener_settings,
            start=start_date,
            end=end_date,
            cache_db=cache_db,
            group=group,
        )
        if not instruments:
            raise ValueError(
                "screener universe resolved to zero symbols for the requested range"
            )
        return instruments

    # Static universe (default): the explicit instrument list (fail-early if absent/empty).
    instruments = payload.get("enabled_instruments")
    if not instruments:
        raise ValueError("payload.enabled_instruments is required for a static universe")
    return list(instruments)


# ---------------------------------------------------------------------------
# Engine run (the per-run trading DB scope)
# ---------------------------------------------------------------------------
def run_daily_backtest(
    config: Dict[str, Any],
    progress_cb: Optional[Callable[[float, str], None]] = None,
) -> Dict[str, Any]:
    """Run ONE daily multi-asset backtest synchronously, in-process, and return the
    results metric blob (the ``results.build_results`` shape).

    This is the SYNCHRONOUS core extracted from ``handle_daily_backtest`` so it can be
    called directly (e.g. by the joint genetic optimizer fitness function, which must
    NOT enqueue a sub-task under ``max_workers=1``). It opens the per-run trading DB,
    seeds the account + experts, runs ``DailyBacktestEngine``, and converts the finished
    account into the full metric blob.

    Determinism: the engine seeds ``random``/``numpy`` from ``config["seed"]`` at the start
    of ``run()`` so a run is byte-reproducible (same cache + same config + same seed =>
    identical equity curve / metrics).

    Args:
        config: the engine run config dict (the shape ``_build_config`` produces). Required
            keys: ``backtest_id``, ``account_settings``, ``enabled_instruments``,
            ``start_date``, ``end_date``, ``warmup_days``, ``experts``, ``seed``.
        progress_cb: optional ``callable(pct: float, msg: str)`` invoked once per bar
            (the handler wires pause/progress through it). Defaults to a no-op so a direct
            in-process call (the optimizer) needs no task queue.

    Returns:
        The results dict (``build_results`` output): total_trades / win_rate / total_return /
        sharpe_ratio / max_drawdown / profit_factor / ... + equity_curve / drawdown_curve /
        trades.
    """
    from ba2_common.core.db import activity_logging_disabled
    from ba2_providers import get_provider
    from ba2_providers.fmp_common import frozen_ttl_cache

    from app.services.backtest.backtest_account import BacktestAccount
    from app.services.backtest.backtest_db import (
        backtest_trading_db,
        seed_account_definition,
    )
    from app.services.backtest.daily_engine import DailyBacktestEngine
    from app.services.backtest.price_source import AsOfClampedOHLCVProvider, AsOfPriceSource
    from app.services.backtest.results import build_results
    from app.services.backtest.seam_wiring import make_indicator_provider, wire_backtest_seams

    progress = progress_cb or (lambda pct, msg: None)

    # Accept ISO string dates: the joint optimizer's _build_daily_trial_config forwards the
    # dates straight from the JSON optimization_config (strings), and AsOfPriceSource.preload
    # does start - timedelta. Coerce once here so every caller (CLI/API/optimizer) is safe.
    config = {
        **config,
        "start_date": _parse_dt(config["start_date"], "start_date"),
        "end_date": _parse_dt(config["end_date"], "end_date"),
    }

    resolver = wire_backtest_seams()
    account_id = 1

    # Backtest perf, scoped to the whole run (live path unaffected):
    #   * frozen_ttl_cache()      — FMP fundamentals are fetched once per symbol and reused
    #                               for every as_of bar (no 15-min TTL re-fetch mid-run).
    #   * activity_logging_disabled() — silence the per-bar ActivityLog write churn (from
    #                               TradeActionEvaluator / TradeRiskManagement), which would
    #                               otherwise serialize thousands of writes through the DB lock.
    with backtest_trading_db(config["backtest_id"]), frozen_ttl_cache(), activity_logging_disabled():
        seed_account_definition(account_id, config["account_settings"])

        # Time-machine price source backed by the FMP OHLCV provider (as_of-aware).
        # execution_interval governs the FILL clock granularity (default 1d). Intraday
        # values (e.g. "1h", "15m") give finer open/close fill detection; it is decoupled
        # from whatever interval the experts request via the provider seam in _gather.
        ohlcv = get_provider("ohlcv", "fmp")
        ps = AsOfPriceSource(ohlcv_provider=ohlcv, interval=config.get("execution_interval", "1d"))
        ps.preload(
            config["enabled_instruments"],
            config["start_date"],
            config["end_date"],
            warmup_days=config["warmup_days"],
        )

        account = BacktestAccount(account_id, ps, config["account_settings"])
        resolver.register_account(account_id, account)

        experts = _build_experts(config, resolver, account_id)

        # Clamp the indicator/ATR OHLCV fetches to the backtest clock: PandasIndicatorCalc
        # and get_latest_atr fetch with end_date=now(), which would leak future bars into the
        # ATR/indicators used for sizing + rule conditions. The clamp follows ps.set_clock().
        indicator_provider = make_indicator_provider(
            ohlcv_provider=AsOfClampedOHLCVProvider(ohlcv, ps)
        )

        engine = DailyBacktestEngine(
            account=account,
            experts=experts,
            price_source=ps,
            config=config,
            progress_cb=progress,
            indicator_provider=indicator_provider,
        )
        engine.run()

        # build_results consumes the SAME account (get_balance_history / get_filled_trades).
        return build_results(account, config)


def _build_experts(
    config: Dict[str, Any], resolver: Any, account_id: int
) -> List[Tuple[Any, int, Dict[str, Any], int]]:
    """Construct + register the (clean) expert instances and seed their backtest DB rows.

    Returns the ``(expert_instance, expert_instance_id, settings_dict, ruleset_id)`` tuples the
    ``DailyBacktestEngine`` iterates. Each expert:
      * gets a seeded ExpertInstance row (account_id + enter ruleset) so the inherited
        decision/RM code resolves it by id;
      * is constructed from its ba2_experts class (``__init__`` runs ``_load_expert_instance``);
      * has its automated-trading gates enabled (``allow_automated_trade_opening``/``enable_buy``)
        so the RM actually sizes + submits the pending orders;
      * is registered on the resolver so the inherited code (and the RM) finds it.

    ``settings_dict`` (fed to ``_process`` via the engine's BacktestContext) is the expert's
    declared decision settings: explicit overrides from the payload spec, else the declared
    defaults from ``get_settings_definitions`` (these are the expert's OWN defaults — not the
    host adding hidden config — so the no-defaults rule is honoured).
    """
    import importlib

    from app.services.backtest.backtest_db import seed_expert_instance
    from app.services.backtest.default_rulesets import seed_enter_long_ruleset, seed_ruleset_from_tree

    # When the (optimizer-decoded) buy-entry condition tree is present, build the enter ruleset
    # FROM it so the optimizer's cond:<id>:value thresholds + on/off toggles actually gate
    # entries; else fall back to the default "BUY when bullish & flat" ruleset.
    buy_tree = config.get("buy_tree")

    def _seed_enter(nm: str) -> int:
        return seed_ruleset_from_tree(buy_tree, name=nm) if buy_tree else seed_enter_long_ruleset(name=nm)

    out: List[Tuple[Any, int, Dict[str, Any], int]] = []
    for idx, spec in enumerate(config["experts"], start=1):
        if isinstance(spec, dict):
            class_name = spec["class"]
            overrides = spec.get("settings", {}) or {}
        else:
            class_name = spec
            overrides = {}

        module_path = _SUPPORTED_EXPERTS[class_name]
        module = importlib.import_module(module_path)
        expert_cls = getattr(module, class_name)

        # BYPASS expert (piece 1b): an expert that declares ``bypasses_classic_rm`` does NOT
        # use the enter/exit ruleset or the classic RM. For it we seed NO enter ruleset and
        # enable NO RM gates — it rebalances to target weights via its own
        # FactorPortfolioManager (the engine routes its analyze_as_of targets there directly).
        bypass = bool(getattr(expert_cls, "bypasses_classic_rm", False))

        if bypass:
            ruleset_id: Optional[int] = None
            expert_id = seed_expert_instance(
                account_id=account_id,
                expert_class_name=class_name,
                # The ExpertInstance FK is non-nullable; seed a ruleset row to satisfy it even
                # though the engine never evaluates it for a bypass expert.
                enter_market_ruleset_id=seed_enter_long_ruleset(
                    name=f"backtest-bypass-{class_name}-{idx}"
                ),
                instance_id=idx,
            )
        else:
            ruleset_id = _seed_enter(f"backtest-enter-{class_name}-{idx}")
            expert_id = seed_expert_instance(
                account_id=account_id,
                expert_class_name=class_name,
                enter_market_ruleset_id=ruleset_id,
                instance_id=idx,
            )

        # The expert's declared decision settings: its own defaults + payload overrides.
        decision_settings = _expert_decision_settings(expert_cls, overrides)

        expert = expert_cls(expert_id)
        if bypass:
            # No RM gates: a bypass expert never goes through TradeRiskManagement. It DOES need
            # its own universe (FactorRanker resolves it from the ``enabled_instruments``
            # setting), so seed that from the run's universe; persist the decision settings so
            # any self.settings read on the rebalance path matches the engine's _process dict.
            bypass_settings: Dict[str, Any] = {
                "enabled_instruments": (
                    {sym: {} for sym in config["enabled_instruments"]},
                    "json",
                ),
            }
            for k, v in decision_settings.items():
                bypass_settings[k] = (v, _setting_type(v))
            expert.save_settings(bypass_settings)
        else:
            # Enable the RM gates (interface defaults are restrictive) + persist the decision
            # settings so any self.settings read in the inherited path is consistent with the
            # dict the engine passes to _process.
            gate_settings: Dict[str, Any] = {
                "allow_automated_trade_opening": (True, "bool"),
                "enable_buy": (True, "bool"),
            }
            for k, v in decision_settings.items():
                gate_settings[k] = (v, _setting_type(v))
            expert.save_settings(gate_settings)

        resolver.register_expert(expert_id, expert)
        out.append((expert, expert_id, decision_settings, ruleset_id))

    return out


def _expert_decision_settings(expert_cls: Any, overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve the expert's decision settings dict: declared defaults overlaid by overrides.

    Pulls each ``_SETTING_KEYS`` entry from the class's ``get_settings_definitions`` default,
    then applies the payload overrides. The defaults belong to the EXPERT (its own contract),
    not the host — fail-early if a key has neither a default nor an override.
    """
    defs = expert_cls.get_settings_definitions()
    keys = getattr(expert_cls, "_SETTING_KEYS", tuple(defs.keys()))
    settings: Dict[str, Any] = {}
    for key in keys:
        if key in overrides:
            settings[key] = overrides[key]
        elif key in defs and "default" in defs[key]:
            settings[key] = defs[key]["default"]
        else:
            raise ValueError(
                f"{expert_cls.__name__} setting '{key}' has no default and no payload override"
            )
    # Pass through any EXPLICIT overrides that aren't among the expert's own decision keys —
    # e.g. the classic-RM sizing settings (risk_per_trade_pct / atr_multiplier / min_stop_loss_pct
    # / max_virtual_equity_per_instrument_percent) optimized via model:* but read by the RM off
    # the expert (get_setting_with_interface_default), not declared on the expert itself. An
    # override is an intentional caller/optimizer instruction, so it must reach the saved settings.
    for k, v in overrides.items():
        if k not in settings:
            settings[k] = v
    return settings


def _setting_type(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    return "str"


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def _persist_results(db: Any, bt: Backtest, results: Dict[str, Any]) -> None:
    """Map the results metric blob onto the ``Backtest`` row's columns + JSON blobs.

    Mirrors ``handle_backtest``'s assignment block so the SAME columns + ``to_dict`` camelCase
    contract + UI consume the daily-engine output unchanged.
    """
    # Basic trade metrics
    bt.total_trades = results["total_trades"]
    bt.winning_trades = results["winning_trades"]
    bt.losing_trades = results["losing_trades"]
    bt.win_rate = results["win_rate"]

    # Return metrics
    bt.total_return = results["total_return"]
    bt.annualized_return = results.get("annualized_return")
    bt.buy_hold_return = results.get("buy_hold_return")

    # Risk metrics
    bt.sharpe_ratio = results["sharpe_ratio"]
    bt.sortino_ratio = results.get("sortino_ratio")
    bt.calmar_ratio = results.get("calmar_ratio")
    bt.volatility = results.get("volatility")

    # Drawdown metrics
    bt.max_drawdown = results["max_drawdown"]
    bt.avg_drawdown = results.get("avg_drawdown")
    bt.max_drawdown_duration = results.get("max_drawdown_duration")

    # Trade quality metrics
    bt.profit_factor = results["profit_factor"]
    bt.expectancy = results.get("expectancy")
    bt.sqn = results.get("sqn")
    bt.avg_trade = results.get("avg_trade")
    bt.best_trade = results.get("best_trade")
    bt.worst_trade = results.get("worst_trade")

    # Duration metrics
    bt.avg_trade_duration = results["avg_trade_duration"]
    bt.exposure_time = results.get("exposure_time")

    # Equity metrics
    bt.final_equity = results["final_equity"]
    bt.equity_peak = results.get("equity_peak")

    # Curves + trades (JSON blobs the UI reads as equityCurve/drawdownCurve/trades).
    bt.equity_curve = results["equity_curve"]
    bt.drawdown_curve = results["drawdown_curve"]
    bt.trades = results["trades"]
    bt.results = {k: v for k, v in results.items()
                  if k not in ("equity_curve", "drawdown_curve", "trades")}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _parse_dt(value: Any, field: str) -> datetime:
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value))
    except (TypeError, ValueError) as e:
        raise ValueError(f"payload.{field} is not a valid ISO date: {value!r} ({e})")


def _fail(db: Any, bt: Backtest, message: str) -> None:
    bt.status = "failed"
    bt.error_message = message[:1000]
    bt.completed_at = datetime.now()
    db.commit()
