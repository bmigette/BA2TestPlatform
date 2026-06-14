"""Strategy optimization handler — joint genetic search over
expert + classic-RM + ruleset/condition params, scored by ONE backtest metric.

Registered as task type ``strategy_optimization`` (main.py). Mirrors the proven GA
wiring in ``job_handler.py`` (validate -> seed -> hoist -> fitness_function -> optimize
-> persist) but the fitness runs the DETERMINISTIC Phase-2 daily backtest
(``daily_backtest_handler.run_daily_backtest``, the synchronous in-process runner) and
reads ``results[<fitness_metric>]`` via ``strategy_fitness.compute_fitness``.

Determinism (the Phase-4 core gate):
  * the GA population/crossover/mutation is governed by seeding ``random`` AND ``np.random``
    from ``optimization_config.seed`` BEFORE ``optimize`` (so a seeded run reproduces an
    identical best individual);
  * each per-trial daily backtest is intrinsically deterministic (the engine seeds
    random/np.random from ``config['seed']`` at the start of ``run()``);
  * a param-independent pass is hoisted ONCE per run (``_build_hoisted_state``) and reused
    for every individual;
  * a content-hash trial memo (``trial_memo``) makes an elitism-reselected identical
    individual a FREE hit AND a self-check that the run is deterministic.

The GA must NEVER enqueue a sub-task: ``init_task_queue(max_workers=1)`` (main.py) would
deadlock. The fitness calls the synchronous runner in-process (confirmed in Replan).
"""
import logging
import random
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np

from app.models import (
    SessionLocal,
    Strategy as StrategyModel,
    StrategyOptimization,
    TaskQueue,
)
from app.services.genetic import GeneticOptimizer, DEAP_AVAILABLE
from app.services.task_queue import get_task_queue
from app.services.strategy_param_space import collect_param_space, decode_params
from app.services.strategy_fitness import compute_fitness, ZERO_TRADE_SENTINEL
from app.services.trial_memo import trial_key, TrialMemo

logger = logging.getLogger(__name__)

# Mirror job_handler.required_ga_keys (no-defaults rule, backend/CLAUDE.md) + add 'seed'
# (Phase-4 determinism). Every value is explicitly provided + validated fail-early.
REQUIRED_GA_KEYS = (
    "populationSize",
    "generations",
    "crossoverProb",
    "mutationProb",
    "earlyStoppingGenerations",
    "elitismPercent",
    "seed",
)


def _fail(opt_id: int, db: Any, msg: str) -> Dict[str, Any]:
    """Mark the StrategyOptimization row failed + return the failure dict."""
    logger.error(f"strategy_optimization {opt_id} failed: {msg}")
    row = db.query(StrategyOptimization).filter(StrategyOptimization.id == opt_id).first()
    if row:
        row.status = "failed"
        row.error_message = msg[:1000]
        row.completed_at = datetime.now()
        db.commit()
    return {"status": "failed", "error": msg}


def handle_strategy_optimization(task_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Run the joint genetic optimization for one StrategyOptimization row."""
    if not DEAP_AVAILABLE:
        return {"status": "failed", "error": "DEAP not available"}
    opt_id = payload.get("optimization_id")
    if not opt_id:
        return {"status": "failed", "error": "optimization_id is required"}

    db = SessionLocal()
    try:
        opt = db.query(StrategyOptimization).filter(
            StrategyOptimization.id == opt_id
        ).first()
        if not opt:
            return {"status": "failed", "error": f"StrategyOptimization {opt_id} not found"}
        opt.status = "running"
        opt.started_at = datetime.now()
        db.commit()

        strategy = db.query(StrategyModel).filter(
            StrategyModel.id == opt.strategy_id
        ).first()
        if not strategy:
            return _fail(opt_id, db, f"Strategy {opt.strategy_id} not found")

        # --- Fail-early config validation (no-defaults rule) ---
        ga = opt.optimization_config or {}
        for key in REQUIRED_GA_KEYS:
            if key not in ga:
                return _fail(opt_id, db, f"optimization_config.{key} is required")
        if not opt.fitness_metric:
            return _fail(opt_id, db, "fitness_metric is required")

        backtest_cfg = ga.get("backtest")
        if not backtest_cfg:
            return _fail(
                opt_id,
                db,
                "optimization_config.backtest is required "
                "(engine/datasets/date-range/initial_capital/...)",
            )
        expert_cfg = ga.get("expert_params")  # may be None (expert frozen)
        rm_cfg = ga.get("rm_params")  # may be None (RM not optimized)

        # BYPASS expert (piece 1c): if the backtest's expert declares ``bypasses_classic_rm``
        # (e.g. FactorRanker) the search space must EXCLUDE rm:*/tp/sl/cond:*/exit:* and search
        # ONLY the expert's own params (model:*). Detected from the backtest_cfg experts here so
        # the same flag drives both the param space and the per-trial config.
        bypass_expert = _is_bypass_expert(backtest_cfg)

        # --- Build the joint param space (Task 1) ---
        try:
            param_space = collect_param_space(
                strategy, expert_cfg=expert_cfg, rm_cfg=rm_cfg, bypass=bypass_expert
            )
        except ValueError as e:
            return _fail(opt_id, db, str(e))
        opt.parameter_ranges = param_space
        db.commit()

        # --- DETERMINISM: seed both RNGs (Task 4 / determinism_rule) ---
        seed = int(ga["seed"])
        random.seed(seed)
        np.random.seed(seed & 0xFFFFFFFF)

        # --- HOIST the param-independent pass out of the trial loop (lever 2) ---
        hoisted = _build_hoisted_state(backtest_cfg)

        memo = TrialMemo()
        all_results: list = []
        best = {"fitness": None, "params": None}

        tq = get_task_queue()

        def fitness_function(decoded_flat: Dict[str, Any]) -> float:
            if tq.is_task_paused(task_id):
                raise InterruptedError("paused/cancelled")
            decoded = decode_params(strategy, decoded_flat)
            key = trial_key(
                {
                    "engine": backtest_cfg.get("engine"),
                    "model_id": backtest_cfg.get("model_id"),
                    "pred_dataset_id": backtest_cfg.get("prediction_dataset_id"),
                    "exec_dataset_id": backtest_cfg.get("execution_dataset_id"),
                    "start": str(backtest_cfg.get("start_date")),
                    "end": str(backtest_cfg.get("end_date")),
                    "seed": backtest_cfg.get("seed"),
                    "params": decoded_flat,
                }
            )
            cached = memo.get(key)
            if cached is not None:
                return cached
            results = _run_trial_backtest(backtest_cfg, hoisted, decoded)
            fit = compute_fitness(opt.fitness_metric, results)
            memo.put(key, fit)
            all_results.append(
                {
                    "params": decoded_flat,
                    "fitness": fit,
                    "key": key,
                    "trades": results.get("total_trades") if results else 0,
                }
            )
            if best["fitness"] is None or fit > best["fitness"]:
                best["fitness"] = fit
                best["params"] = decoded_flat
            return fit

        # --- brute_force option for tiny spaces (optimization_type) ---
        if (opt.optimization_type or "genetic") == "brute_force":
            return _run_brute_force(
                opt, db, task_id, param_space, fitness_function, all_results
            )

        optimizer = GeneticOptimizer(
            param_ranges=param_space,
            population_size=int(ga["populationSize"]),
            n_generations=int(ga["generations"]),
            crossover_prob=float(ga["crossoverProb"]),
            mutation_prob=float(ga["mutationProb"]),
            early_stopping_generations=int(ga["earlyStoppingGenerations"]),
            elitism_percent=float(ga["elitismPercent"]),
        )

        gen_state = {"gen": 0}

        def on_generation_start(generation: int):
            gen_state["gen"] = generation

        def ga_callback(generation: int, best_fitness: float, best_params: Dict):
            pct = ((generation + 1) / int(ga["generations"])) * 100.0
            tq.update_progress(
                task_id,
                pct,
                f"Gen {generation + 1}/{ga['generations']} best={best_fitness:.4f}",
            )
            row = db.query(StrategyOptimization).filter(
                StrategyOptimization.id == opt_id
            ).first()
            row.progress = pct
            row.best_fitness = best_fitness
            row.best_params = best_params
            row.all_results = all_results
            db.commit()
            if tq.is_task_paused(task_id):
                raise InterruptedError("paused/cancelled")

        def checkpoint_cb(generation: int, population: list):
            _save_checkpoint(
                task_id, optimizer.get_checkpoint_data(generation, population)
            )

        start_gen, init_pop = 0, None
        ckpt = _load_checkpoint(task_id)
        if ckpt:
            start_gen, init_pop = optimizer.resume_from_checkpoint(ckpt)

        result = optimizer.optimize(
            fitness_function=fitness_function,
            callback=ga_callback,
            on_generation_start=on_generation_start,
            checkpoint_callback=checkpoint_cb,
            start_generation=start_gen,
            initial_population=init_pop,
        )

        # Trust guard: if EVERY trial failed (e.g. a bad backtest config), all_results is
        # empty and best_fitness is a meaningless default. The GA swallows per-trial
        # exceptions as warnings, so without this guard the optimization would report
        # "completed" having evaluated NOTHING. Fail loudly instead.
        if not all_results:
            return _fail(
                opt_id, db,
                "optimization produced 0 successful trials — every backtest failed. Check the "
                "logs for per-trial 'Fitness evaluation failed' warnings (e.g. a bad backtest "
                "config) before trusting any result.",
            )

        opt.status = "completed"
        opt.completed_at = datetime.now()
        opt.progress = 100.0
        opt.best_params = result["best_params"]
        opt.best_fitness = result["best_fitness"]
        opt.all_results = all_results
        db.commit()
        logger.info(
            f"strategy_optimization {opt_id} done: "
            f"best_fitness={result['best_fitness']:.4f} "
            f"memo hits/misses={memo.hits}/{memo.misses}"
        )
        return {
            "status": "completed",
            "optimization_id": opt_id,
            "best_fitness": result["best_fitness"],
            "best_params": result["best_params"],
        }

    except InterruptedError:
        return {"status": "paused"}
    except Exception as e:  # noqa: BLE001 — any crash must fail the row, not the worker
        logger.error(f"strategy_optimization {opt_id} crashed: {e}", exc_info=True)
        return _fail(opt_id, db, str(e))
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Bypass-expert detection (piece 1c)
# ---------------------------------------------------------------------------
def _is_bypass_expert(backtest_cfg: Dict[str, Any]) -> bool:
    """True iff ANY expert named in the daily backtest_cfg declares ``bypasses_classic_rm``.

    Resolves each expert class name through the daily handler's ``_SUPPORTED_EXPERTS`` map and
    reads the class-level marker (``getattr(cls, 'bypasses_classic_rm', False)``). A bypass
    expert (e.g. FactorRanker) rebalances to target weights via its own portfolio manager, so
    the optimizer must drop the rm:*/tp/sl/cond:*/exit:* namespaces and search only model:*.

    Only the ``daily`` engine has the expert-aware bypass concept; the ML engine path is never
    a bypass. An unresolvable / unknown class is treated as NON-bypass (the validating handler
    rejects unknown experts at run time — this stays defensive and never raises here).
    """
    if backtest_cfg.get("engine", "daily") != "daily":
        return False
    import importlib

    from app.services.backtest.daily_backtest_handler import _SUPPORTED_EXPERTS

    for spec in backtest_cfg.get("experts", []) or []:
        class_name = spec.get("class") if isinstance(spec, dict) else spec
        module_path = _SUPPORTED_EXPERTS.get(class_name)
        if not module_path:
            continue
        try:
            module = importlib.import_module(module_path)
            expert_cls = getattr(module, class_name)
        except Exception:  # noqa: BLE001 — never let detection raise; default to non-bypass
            continue
        if bool(getattr(expert_cls, "bypasses_classic_rm", False)):
            return True
    return False


# ---------------------------------------------------------------------------
# The Phase-2 seam (the GA fitness target)
# ---------------------------------------------------------------------------
def _build_hoisted_state(backtest_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Compute the param-INDEPENDENT pass ONCE per run (determinism lever 2).

    For the daily engine, the param-independent input is the fixed price/indicator
    cache the per-trial ``run_daily_backtest`` preloads over the (start,end) window.
    The cache content is identical across trials for a fixed (instruments, date range),
    so the hoisted state here just carries the resolved backtest_cfg through to the
    trial runner; the engine's intrinsic seeding makes each trial deterministic.

    NOTE (perf-todo, not a correctness issue): the current Phase-2 runner re-preloads
    the AsOfPriceSource per call. A future optimization can pre-build and reuse the
    AsOfPriceSource bundle here; until then this is an explicit known perf-todo.

    Returns an opaque dict consumed by ``_run_trial_backtest``.
    """
    return {"backtest_cfg": backtest_cfg}


def _run_trial_backtest(
    backtest_cfg: Dict[str, Any],
    hoisted: Dict[str, Any],
    decoded: Dict[str, Any],
) -> Dict[str, Any]:
    """Run ONE deterministic backtest with the decoded trial params; return results dict.

    The default (and the design's first-class path) is the Phase-2 SYNCHRONOUS daily
    runner (``daily_backtest_handler.run_daily_backtest``) for ba2-expert strategies with
    multi-asset classic RM. The decoded trial params are injected per the Replan seam:
      * ``decoded['rm']`` + ``decoded['expert_overrides']`` are MERGED into each expert's
        settings dict (the engine feeds settings to ``_process``; the RM reads its sizing
        params off the expert via ``get_setting_with_interface_default``), mapping the
        joint namespaces to the REAL ba2 RM setting names;
      * ``decoded['tp']`` / ``decoded['sl']`` set the initial TP/SL the ruleset applies;
      * ``decoded['buy_tree']`` / ``decoded['sell_tree']`` / ``decoded['exit_rules']`` are
        the substituted condition trees.

    The legacy ML-expert single-asset path (``backtest_handler.run_backtest``) is kept as a
    lazily-imported fallback for ``engine == 'ml'`` so it never pulls torch unless explicitly
    requested.
    """
    engine = backtest_cfg.get("engine", "daily")
    if engine == "daily":
        from app.services.backtest.daily_backtest_handler import run_daily_backtest

        config = _build_daily_trial_config(backtest_cfg, decoded)
        return run_daily_backtest(config)

    if engine == "ml":
        return _run_ml_trial_backtest(backtest_cfg, decoded)

    raise ValueError(
        f"Unknown backtest engine: {engine!r} (valid: 'daily', 'ml')"
    )


# Map the joint RM namespaces (decode_params 'rm' keys) onto the REAL ba2 RM setting names
# the daily engine's TradeRiskManagement reads off the expert (Replan):
#   risk_per_trade_pct        -> risk_per_trade_pct        (matches)
#   atr_stop_mult             -> atr_multiplier
#   min_stop_pct              -> min_stop_loss_pct
#   per_instrument_cap_pct    -> max_virtual_equity_per_instrument_percent
# max_concurrent_positions has NO enforcement hook in the current engine (it caps by
# equity %, not position count) — it is intentionally NOT forwarded as a setting.
_RM_SETTING_NAME = {
    "risk_per_trade_pct": "risk_per_trade_pct",
    "atr_stop_mult": "atr_multiplier",
    "min_stop_pct": "min_stop_loss_pct",
    "per_instrument_cap_pct": "max_virtual_equity_per_instrument_percent",
}


def _build_daily_trial_config(
    backtest_cfg: Dict[str, Any], decoded: Dict[str, Any]
) -> Dict[str, Any]:
    """Assemble the ``run_daily_backtest`` config for one trial from the run-level
    backtest_cfg + the decoded trial params.

    The expert settings the engine feeds to ``_process`` are merged with:
      * the decoded expert_overrides (model:* numeric decision settings), and
      * the decoded RM params (mapped to the real ba2 RM setting names) so the classic RM
        sizes against the trial's risk config.

    BYPASS expert (piece 1c): for an expert that declares ``bypasses_classic_rm`` the param
    space already excludes rm:*/tp/sl, so ``decoded`` carries none; but we ALSO refuse to inject
    any rm/tp/sl override defensively (the bypass rebalance path ignores them), forwarding ONLY
    the expert's own model:* overrides.
    """
    bypass = _is_bypass_expert(backtest_cfg)
    overrides = dict(decoded.get("expert_overrides") or {})
    if not bypass:
        rm = decoded.get("rm") or {}
        for joint_name, value in rm.items():
            real = _RM_SETTING_NAME.get(joint_name)
            if real is not None and value is not None:
                overrides[real] = value
        # Optional TP/SL forwarded as expert settings so the ruleset/RM can read them if it
        # consults the expert (the daily ruleset's initial TP/SL seam).
        if decoded.get("tp") is not None:
            overrides.setdefault("initial_tp_percent", decoded["tp"])
        if decoded.get("sl") is not None:
            overrides.setdefault("initial_sl_percent", decoded["sl"])

    # Merge the per-trial overrides into each expert spec's settings (do NOT mutate the
    # run-level backtest_cfg — build fresh spec dicts).
    experts_in = backtest_cfg["experts"]
    experts_out = []
    for spec in experts_in:
        if isinstance(spec, dict):
            merged_settings = dict(spec.get("settings") or {})
            merged_settings.update(overrides)
            experts_out.append({"class": spec["class"], "settings": merged_settings})
        else:
            experts_out.append({"class": spec, "settings": dict(overrides)})

    return {
        "backtest_id": backtest_cfg["backtest_id"],
        "name": backtest_cfg.get("name", f"opt-trial-{backtest_cfg['backtest_id']}"),
        "start_date": backtest_cfg["start_date"],
        "end_date": backtest_cfg["end_date"],
        "enabled_instruments": list(backtest_cfg["enabled_instruments"]),
        "experts": experts_out,
        "initial_capital": float(backtest_cfg["initial_capital"]),
        "account_settings": backtest_cfg["account_settings"],
        "warmup_days": int(backtest_cfg["warmup_days"]),
        "seed": int(backtest_cfg["seed"]),
        "subtype": backtest_cfg.get("subtype"),
    }


def _run_ml_trial_backtest(
    backtest_cfg: Dict[str, Any], decoded: Dict[str, Any]
) -> Dict[str, Any]:
    """Legacy ML-expert single-asset adapter (``backtest_handler.run_backtest``).

    Lazily imported so torch is only pulled when ``engine == 'ml'`` is explicitly
    requested with a real model/datasets present.
    """
    from app.services.backtest_handler import run_backtest, _empty_results
    import pandas as pd

    db = SessionLocal()
    try:
        from app.models import Dataset, TrainedModel

        model = db.query(TrainedModel).filter(
            TrainedModel.id == backtest_cfg["model_id"]
        ).first()
        pred = db.query(Dataset).filter(
            Dataset.id == backtest_cfg["prediction_dataset_id"]
        ).first()
        exe = db.query(Dataset).filter(
            Dataset.id == backtest_cfg["execution_dataset_id"]
        ).first()
        if not (model and pred and exe):
            return _empty_results(float(backtest_cfg.get("initial_capital", 10000.0)))
        pred_df = pd.read_csv(pred.file_path)
        exec_df = pd.read_csv(exe.file_path)
        for df in (pred_df, exec_df):
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"])
        strategy_params = {
            "initial_tp_percent": decoded["tp"],
            "initial_sl_percent": decoded["sl"],
        }
        return run_backtest(
            model=model,
            pred_df=pred_df,
            exec_df=exec_df,
            strategy_params=strategy_params,
            initial_capital=float(backtest_cfg.get("initial_capital", 10000.0)),
            position_sizing_type=backtest_cfg.get("position_sizing_type", "percent"),
            position_sizing_value=backtest_cfg.get("position_sizing_value", 10.0),
            commission=backtest_cfg.get("commission", 0.0),
            slippage=backtest_cfg.get("slippage", 0.0),
            buy_entry_conditions=decoded["buy_tree"],
            sell_entry_conditions=decoded["sell_tree"],
            exit_conditions=decoded["exit_rules"],
        )
    finally:
        db.close()


# ---------------------------------------------------------------------------
# brute force + checkpoint persistence
# ---------------------------------------------------------------------------
def _run_brute_force(
    opt: Any,
    db: Any,
    task_id: str,
    param_space: Dict[str, Any],
    fitness_function,
    all_results: list,
) -> Dict[str, Any]:
    """Exhaustive search over the stepped ranges (itertools.product) for tiny spaces."""
    import itertools

    axes: Dict[str, list] = {}
    for name, spec in param_space.items():
        vals, v = [], spec["min"]
        while v <= spec["max"] + 1e-9:
            vals.append(int(round(v)) if spec["type"] == "int" else round(v, 10))
            v += spec["step"]
        axes[name] = vals
    names = list(axes.keys())
    best = {"fitness": None, "params": None}
    for combo in itertools.product(*(axes[n] for n in names)):
        flat = dict(zip(names, combo))
        fit = fitness_function(flat)
        if best["fitness"] is None or fit > best["fitness"]:
            best = {"fitness": fit, "params": flat}
    opt.status = "completed"
    opt.completed_at = datetime.now()
    opt.progress = 100.0
    opt.best_params = best["params"]
    opt.best_fitness = best["fitness"]
    opt.all_results = all_results
    db.commit()
    return {
        "status": "completed",
        "optimization_id": opt.id,
        "best_fitness": best["fitness"],
        "best_params": best["params"],
    }


def _save_checkpoint(task_id: str, checkpoint_data: Dict[str, Any]) -> None:
    """Persist GA checkpoint to TaskQueue.checkpoint_data (keyed by task_id)."""
    db = SessionLocal()
    try:
        t = db.query(TaskQueue).filter(TaskQueue.task_id == task_id).first()
        if t:
            t.checkpoint_data = checkpoint_data
            db.commit()
    finally:
        db.close()


def _load_checkpoint(task_id: str) -> Optional[Dict[str, Any]]:
    """Load a GA checkpoint from TaskQueue.checkpoint_data (keyed by task_id)."""
    db = SessionLocal()
    try:
        t = db.query(TaskQueue).filter(TaskQueue.task_id == task_id).first()
        return t.checkpoint_data if (t and t.checkpoint_data) else None
    finally:
        db.close()
