"""``ba2-test`` — console CLI for the BA2 Test Platform (ML / backtest).

Installed as the ``ba2-test`` command (``pyproject.toml`` ``[project.scripts]``). It is a
subcommand dispatcher that runs against the ``backend/`` package (added to the path), so it
covers the platform's operations without the API:

  ba2-test serve [--host --port --reload]      launch the FastAPI API (uvicorn app.main:app)
  ba2-test backtest <run_daily_backtest args>  run a daily expert backtest (full passthrough)
  ba2-test fetch-cache --symbols .. [...]       populate the as-of OHLCV cache
  ba2-test fetch-screener --settings-json F ..  build the survivorship-free screener history
  ba2-test cache-usage                          show cache disk usage per type
  ba2-test cache-clear [--type T] [--before D]  clear cache (all, or one type, optional date)
  ba2-test runs list [--saved-only]             list tracked backtest runs (shared results table)
  ba2-test runs save <id> [--name N]            mark a run saved (survives clear-unsaved)
  ba2-test runs clear-unsaved                    delete all runs not marked saved
  ba2-test runs delete <id>                      delete one run

  (persist a CLI run with: ba2-test backtest ... --track  [or --save to keep it])

Run ``ba2-test <cmd> -h`` for per-command help. Works for an editable/source install (the
repo root is resolved from this module's location).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime


def _enter_backend() -> str:
    """Put ``backend/`` on the path and chdir into it (the app's import + cwd root)."""
    repo_root = os.path.dirname(os.path.abspath(__file__))
    backend = os.path.join(repo_root, "backend")
    if not os.path.isdir(backend):
        sys.exit(
            f"ba2-test: backend dir not found at {backend}. The console command requires "
            f"an editable/source install of the test-platform repo."
        )
    if backend not in sys.path:
        sys.path.insert(0, backend)
    os.chdir(backend)
    # Load .env (FMP_API_KEY etc.), mirroring run_daily_backtest.py.
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(backend, ".env"))
        load_dotenv(os.path.join(repo_root, ".env"))
    except Exception:  # noqa: BLE001 — dotenv optional
        pass
    # The test platform's legacy OHLCV providers read FMP_API_KEY from the ENV, but the key is
    # configured in the trade app-settings DB (ba2_common). Mirror it into the env (in-process
    # only — never written to disk) so fetch-cache/fetch-screener resolve it, matching how the
    # backtest path forwards the key. No-op if already set or unavailable.
    if not os.getenv("FMP_API_KEY"):
        try:
            from ba2_common.config import get_app_setting
            _k = get_app_setting("FMP_API_KEY")
            if _k:
                os.environ["FMP_API_KEY"] = _k
        except Exception:  # noqa: BLE001 — best-effort; absence just means env-only resolution
            pass
    return backend


def _find_npm() -> "str | None":
    """Locate npm: PATH first, then the standard Windows nodejs install."""
    import shutil
    for cand in ("npm", "npm.cmd"):
        p = shutil.which(cand)
        if p:
            return p
    for p in (r"C:\Program Files\nodejs\npm.cmd", r"C:\Program Files (x86)\nodejs\npm.cmd"):
        if os.path.isfile(p):
            return p
    return None


def _start_frontend(repo_root: str, port: int):
    """Launch the Vite dev server (npm run dev) as a subprocess. Returns the Popen or None."""
    import subprocess
    fe = os.path.join(repo_root, "frontend")
    if not os.path.isdir(os.path.join(fe, "node_modules")):
        print(f"ba2-test: frontend deps not installed; run `npm install` in {fe} first.")
        return None
    npm = _find_npm()
    if not npm:
        print("ba2-test: npm not found (install Node.js); cannot start the frontend.")
        return None
    env = dict(os.environ)
    # Node on PATH for the child (so vite's own node resolves).
    nodedir = os.path.dirname(npm)
    env["PATH"] = nodedir + os.pathsep + env.get("PATH", "")
    proc = subprocess.Popen([npm, "run", "dev", "--", "--port", str(port)], cwd=fe, env=env)
    print(f"frontend (vite)  -> http://localhost:{port}")
    return proc


def _cmd_serve(args) -> int:
    repo_root = os.path.dirname(os.path.abspath(__file__))
    mode = args.mode
    fe_proc = None
    if mode in ("both", "front"):
        fe_proc = _start_frontend(repo_root, args.frontend_port)

    if mode in ("both", "back"):
        try:
            import uvicorn
        except ImportError:
            if fe_proc:
                fe_proc.terminate()
            sys.exit("ba2-test: uvicorn not installed. Install backend/requirements.txt into this venv.")
        print(f"backend (api)    -> http://localhost:{args.port}  (docs: /docs)")
        try:
            uvicorn.run("app.main:app", host=args.host, port=args.port, reload=args.reload)
        finally:
            if fe_proc:
                fe_proc.terminate()
    elif mode == "front":
        if fe_proc is None:
            return 1
        try:
            fe_proc.wait()
        except KeyboardInterrupt:
            fe_proc.terminate()
    return 0


def _cmd_backtest(rest: list) -> int:
    # Full passthrough to the daily-backtest CLI (every flag it supports: --expert,
    # --universe, --start/--end, --interval, --seed, --initial-capital, --out, ...).
    from scripts.run_daily_backtest import main as bt_main
    return int(bt_main(rest) or 0)


def _cmd_fetch_cache(args) -> int:
    from app.services.ohlcv_cache_handler import handle_ohlcv_cache_fetch
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    timeframes = [t.strip() for t in args.timeframes.split(",") if t.strip()]
    overall = {"fetched": [], "failed": []}
    for sym in symbols:
        payload = {
            "provider": args.provider,
            "symbol": sym,
            "timeframes": timeframes,
            "start_date": args.start,
            "end_date": args.end,
            "executor_workers": args.workers,
        }
        res = handle_ohlcv_cache_fetch(f"cli-fetch-{sym}", payload)
        (overall["fetched"] if res.get("status") == "completed" else overall["failed"]).append({sym: res})
    print(json.dumps(overall, indent=2, default=str))
    return 0


def _cmd_fetch_screener(args) -> int:
    from app.services.screener_history_cache import ScreenerHistoryCache, screened_universe_for_bar
    with open(args.settings_json, "r", encoding="utf-8") as fh:
        settings = json.load(fh)
    start = datetime.fromisoformat(args.start)
    end = datetime.fromisoformat(args.end)
    cache = ScreenerHistoryCache(args.cache_db)
    # Walk scan dates at the requested cadence (calendar days) and build/replay each bar.
    from datetime import timedelta
    built = 0
    d = start
    while d <= end:
        rows = screened_universe_for_bar(settings, d, args.group, cache)
        print(f"{d.date()}: {len(rows)} survivors")
        built += 1
        d += timedelta(days=max(1, args.cadence_days))
    print(f"done: {built} scan dates into {args.cache_db}")
    return 0


def _cmd_cache_usage(_args) -> int:
    from app.services.cache_manager import get_usage
    print(json.dumps(get_usage(), indent=2, default=str))
    return 0


def _cmd_cache_clear(args) -> int:
    from app.services import cache_manager
    before = datetime.fromisoformat(args.before) if args.before else None
    if args.type:
        res = cache_manager.clear_type(args.type, before=before)
    else:
        res = cache_manager.clear_all(before=before)
    print(json.dumps(res, indent=2, default=str))
    return 0


# --- backtest run tracking (the shared `backtests` results table) ----------------------
def _runs_db():
    # Ensure the results schema exists so `runs`/`--track` work even before the API's
    # first start (init_db = Base.metadata.create_all, same call the platform makes).
    import app.models  # noqa: F401 — registers all ORM models on Base
    from app.models.database import SessionLocal, init_db
    init_db()
    return SessionLocal()


# Fitness/sort metric -> Backtest column (higher = better for all of these).
_METRIC_COL = {
    "sharpe": "sharpe_ratio",
    "calmar": "calmar_ratio",
    "return": "total_return",
    "total_return": "total_return",
    "profit_factor": "profit_factor",
    "sortino": "sortino_ratio",
}


def _cmd_report(args) -> int:
    """Write an HTML summary of tracked backtests: per-expert best performer + counts,
    overall leaderboard, and per-optimization-job stats."""
    import html as _html
    from app.models.backtest import Backtest
    db = _runs_db()
    try:
        rows = (db.query(Backtest).filter(Backtest.status == "completed")
                .order_by(Backtest.sharpe_ratio.desc()).all())
    finally:
        db.close()

    def esc(v):
        return _html.escape(str(v)) if v is not None else "-"

    def num(v, n=2):
        return f"{v:.{n}f}" if isinstance(v, (int, float)) else "-"

    # Per-expert grouping.
    by_expert: dict = {}
    for r in rows:
        by_expert.setdefault(r.expert_name or "(untagged)", []).append(r)

    parts = [
        "<!doctype html><meta charset='utf-8'><title>BA2 Backtest Report</title>",
        "<style>body{font:14px/1.5 system-ui,Segoe UI,Arial;margin:24px;color:#1e293b}"
        "h1{margin:0 0 4px}h2{margin:24px 0 8px;border-bottom:2px solid #e2e8f0;padding-bottom:4px}"
        "table{border-collapse:collapse;width:100%;margin:8px 0}"
        "th,td{border:1px solid #e2e8f0;padding:6px 10px;text-align:right}"
        "th:first-child,td:first-child,td.l{text-align:left}"
        "th{background:#f1f5f9}tr:nth-child(even){background:#f8fafc}"
        ".pos{color:#16a34a}.neg{color:#dc2626}.muted{color:#64748b}</style>",
        f"<h1>BA2 Backtest Optimization Report</h1>",
        f"<div class='muted'>Generated {datetime.now():%Y-%m-%d %H:%M} · "
        f"{len(rows)} completed run(s) · {len(by_expert)} expert(s)</div>",
    ]

    # Per-expert best performer + counts.
    parts.append("<h2>Per-expert summary (best by Sharpe)</h2>")
    parts.append("<table><tr><th>Expert</th><th>Runs</th><th>Best Sharpe</th>"
                 "<th>Best Return %</th><th>Best run</th><th>Trades</th></tr>")
    for expert, group in sorted(by_expert.items()):
        best = max(group, key=lambda r: (r.sharpe_ratio if r.sharpe_ratio is not None else -1e9))
        rc = "pos" if (best.total_return or 0) >= 0 else "neg"
        parts.append(
            f"<tr><td class='l'>{esc(expert)}</td><td>{len(group)}</td>"
            f"<td>{num(best.sharpe_ratio)}</td><td class='{rc}'>{num(best.total_return)}</td>"
            f"<td class='l'>#{best.id} {esc(best.name)}</td><td>{esc(best.total_trades)}</td></tr>")
    parts.append("</table>")

    # Overall leaderboard (top 20 by Sharpe).
    parts.append("<h2>Leaderboard (top 20 by Sharpe)</h2>")
    parts.append("<table><tr><th>#</th><th>Expert</th><th>Opt</th><th>Sharpe</th><th>Return %</th>"
                 "<th>MaxDD %</th><th>Win %</th><th>PF</th><th>Trades</th><th>Saved</th><th>Name</th></tr>")
    for r in rows[:20]:
        rc = "pos" if (r.total_return or 0) >= 0 else "neg"
        parts.append(
            f"<tr><td>{r.id}</td><td class='l'>{esc(r.expert_name)}</td>"
            f"<td>{esc(r.optimization_id)}</td><td>{num(r.sharpe_ratio)}</td>"
            f"<td class='{rc}'>{num(r.total_return)}</td><td>{num(r.max_drawdown)}</td>"
            f"<td>{num(r.win_rate,1)}</td><td>{num(r.profit_factor)}</td>"
            f"<td>{esc(r.total_trades)}</td><td>{'★' if r.is_saved else ''}</td>"
            f"<td class='l'>{esc(r.name)}</td></tr>")
    parts.append("</table>")

    # Per optimization-job stats.
    by_opt: dict = {}
    for r in rows:
        if r.optimization_id is not None:
            by_opt.setdefault(r.optimization_id, []).append(r)
    if by_opt:
        parts.append("<h2>Per optimization job</h2>")
        parts.append("<table><tr><th>Opt #</th><th>Expert</th><th>Trials</th>"
                     "<th>Best Sharpe</th><th>Avg Sharpe</th><th>Best Return %</th></tr>")
        for oid, group in sorted(by_opt.items()):
            shp = [r.sharpe_ratio for r in group if r.sharpe_ratio is not None]
            ret = [r.total_return for r in group if r.total_return is not None]
            exp = group[0].expert_name
            parts.append(
                f"<tr><td>{oid}</td><td class='l'>{esc(exp)}</td><td>{len(group)}</td>"
                f"<td>{num(max(shp)) if shp else '-'}</td>"
                f"<td>{num(sum(shp)/len(shp)) if shp else '-'}</td>"
                f"<td>{num(max(ret)) if ret else '-'}</td></tr>")
        parts.append("</table>")

    # Default INSIDE the repo (tracked ``reports/``) so the HTML is committed and syncs across
    # machines — not an out-of-tree absolute path. Resolve from this module's location (the
    # repo root), since _enter_backend() has chdir'd into backend/ by now.
    if args.out:
        out = args.out
    else:
        repo_root = os.path.dirname(os.path.abspath(__file__))
        out = os.path.join(repo_root, "reports", "ba2_backtest_report.html")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))
    print(f"wrote report -> {out} ({len(rows)} runs, {len(by_expert)} experts)")
    return 0


# Per-expert optimizable numeric decision settings (model:*) + the fixed (non-optimized)
# settings each expert still needs. RM params + TP/SL ranges are set on the Strategy below.
_EXPERT_OPT = {
    "FMPRating": {
        "expert_params": {
            "profit_ratio": {"optimize": True, "min": 0.5, "max": 1.5, "step": 0.1, "type": "float"},
            "min_analysts": {"optimize": True, "min": 5, "max": 25, "step": 5, "type": "int"},
            "price_target_window_days": {"optimize": True, "min": 30, "max": 180, "step": 30, "type": "int"},
        },
        "fixed_settings": {"target_price_type": "consensus"},
    },
    "FMPEarningsDrift": {
        "expert_params": {
            "surprise_min_pct": {"optimize": True, "min": 2.0, "max": 15.0, "step": 1.0, "type": "float"},
            "max_days_since_report": {"optimize": True, "min": 5, "max": 45, "step": 5, "type": "int"},
        },
        "fixed_settings": {},
    },
    "FMPInsiderClusterBuy": {
        "expert_params": {
            "lookback_days": {"optimize": True, "min": 30, "max": 120, "step": 15, "type": "int"},
            "min_insiders": {"optimize": True, "min": 2, "max": 6, "step": 1, "type": "int"},
        },
        "fixed_settings": {},
    },
}


# Classic-RM sizing/stop params the RM reads off the expert. The optimizer now searches RM
# through the model:* namespace (keyed by the REAL ba2 setting names), merged into each
# expert's expert_params — there is no separate rm:* namespace. risk_per_trade_pct spans
# 0.5%..5%. (max_concurrent_positions is omitted: the engine has no enforcement hook for it.)
_RM_OPT = {
    "risk_per_trade_pct": {"optimize": True, "min": 0.5, "max": 5.0, "step": 0.5, "type": "float"},
    "atr_multiplier": {"optimize": True, "min": 1.5, "max": 4.0, "step": 0.5, "type": "float"},
    "min_stop_loss_pct": {"optimize": True, "min": 3.0, "max": 10.0, "step": 1.0, "type": "float"},
    "max_virtual_equity_per_instrument_percent": {"optimize": True, "min": 5.0, "max": 30.0, "step": 5.0, "type": "float"},
}


def _build_strategy_row(name: str):
    """A Strategy whose TP/SL + the 5 classic-RM params (the RM's sizing/stop conditions &
    actions) are marked optimizable with ranges — the numeric RM space the optimizer searches."""
    from app.models.strategy import Strategy
    # Entry-gate tree: confidence + expected-profit thresholds, each value-optimizable AND
    # on/off-toggleable. The engine builds the enter ruleset from this (seed_ruleset_from_tree),
    # so these are the optimizer's "RM/entry conditions" — tuned thresholds + steps turned on/off.
    buy_entry_conditions = {
        "id": "root", "type": "AND", "conditions": [
            {"id": "gate_confidence", "field": "confidence", "op": ">", "value": 50,
             "optimize": True, "value_min": 40, "value_max": 80, "value_step": 5,
             "toggle_optimize": True},
            {"id": "gate_expected_profit", "field": "expected_profit", "op": ">", "value": 3,
             "optimize": True, "value_min": 0, "value_max": 15, "value_step": 1,
             "toggle_optimize": True},
            # Cooldown gates: only re-enter a symbol once N days have passed since the last
            # close (any / profitable / losing). Each is value-optimizable AND on/off-toggleable
            # so the optimizer can decide whether a cooldown helps and how long it should be.
            # 0 days never blocks; the optimizer can also turn the gate off entirely.
            {"id": "gate_days_since_close", "field": "days_since_last_close", "op": ">", "value": 0,
             "optimize": True, "value_min": 0, "value_max": 30, "value_step": 5,
             "toggle_optimize": True},
            {"id": "gate_days_since_profit", "field": "days_since_last_profitable_close", "op": ">",
             "value": 0, "optimize": True, "value_min": 0, "value_max": 30, "value_step": 5,
             "toggle_optimize": True},
            {"id": "gate_days_since_loss", "field": "days_since_last_losing_close", "op": ">",
             "value": 0, "optimize": True, "value_min": 0, "value_max": 60, "value_step": 10,
             "toggle_optimize": True},
        ],
    }
    # Exit (open_positions) ruleset: the dynamic-exit "movements", each a rule the backtest
    # evaluates via the real TradeActionEvaluator on the analysis cadence (identical to live).
    # Every rule is on/off-toggleable (toggle_optimize -> exit:<id>:enabled gene); numeric
    # condition thresholds (cond:<id>:value) and adjust-action %s (exit:<id>:action_value) are
    # value-optimized with steps. (The immediate initial TP/SL bracket stays via the tp/sl genes;
    # these rules ADJUST/CLOSE on top of it.)
    exit_conditions = [
        # Close the position when the expert turns bearish (sell signal).
        {"id": "exit_bearish", "action_type": "close", "toggle_optimize": True,
         "conditions": {"type": "AND", "conditions": [{"id": "xb", "field": "bearish"}]}},
        # Close when the expert's current rating goes negative (downgrade exit).
        {"id": "exit_downgrade", "action_type": "close", "toggle_optimize": True,
         "conditions": {"type": "AND", "conditions": [{"id": "xd", "field": "current_rating_negative"}]}},
        # Profit-lock: once +X% in profit, move the stop to entry +lock% (break-even / lock-in).
        {"id": "exit_belock", "action_type": "adjust_stop_loss", "reference_value": "order_open_price",
         "action_value": 0.0, "action_value_optimize": True,
         "action_value_min": -2.0, "action_value_max": 8.0, "action_value_step": 2.0,
         "toggle_optimize": True,
         "conditions": {"type": "AND", "conditions": [
             {"id": "xlk", "field": "profit_loss_percent", "op": ">", "value": 5,
              "optimize": True, "value_min": 3, "value_max": 20, "value_step": 2}]}},
        # Time exit: close after N days held (caps dead-money holds).
        {"id": "exit_time", "action_type": "close", "toggle_optimize": True,
         "conditions": {"type": "AND", "conditions": [
             {"id": "xt", "field": "days_opened", "op": ">", "value": 60,
              "optimize": True, "value_min": 20, "value_max": 120, "value_step": 20}]}},
    ]
    return Strategy(
        name=name,
        buy_entry_conditions=buy_entry_conditions,
        exit_conditions=exit_conditions,
        initial_tp_percent=10.0, initial_tp_optimize=True, initial_tp_min=5.0, initial_tp_max=40.0, initial_tp_step=3.0,
        initial_sl_percent=6.0, initial_sl_optimize=True, initial_sl_min=3.0, initial_sl_max=20.0, initial_sl_step=2.0,
    )


def _cmd_optimize(args) -> int:
    """Create a Strategy + StrategyOptimization and run a joint genetic optimization headless.

    Optimizes the expert's numeric decision settings + the 5 classic-RM params (sizing/stop
    'conditions & actions') + TP/SL, scored by --fitness, with parallel trials and suppressed
    per-trial logging. Persists the best trial as a tagged Backtest (optimization_id) and writes
    the HTML report.
    """
    from datetime import datetime as _dt
    import app.models  # noqa: F401 — register ORM models
    from app.models.database import SessionLocal, init_db
    from app.models.backtest import Backtest
    from app.models.strategy import Strategy
    from app.models.strategy_optimization import StrategyOptimization
    from app.services.backtest.daily_backtest_handler import derive_warmup_days
    from app.services.strategy_optimization_handler import handle_strategy_optimization

    expert = args.expert
    spec = _EXPERT_OPT.get(expert)
    if spec is None:
        sys.exit(f"ba2-test: optimize not configured for expert {expert!r}; have {sorted(_EXPERT_OPT)}")
    universe = [s.strip().upper() for s in args.universe.split(",") if s.strip()]
    if not universe:
        sys.exit("ba2-test: --universe must list at least one symbol")
    run_sched = None
    if args.run_schedule == "weekly":
        days = {d: (d == args.run_schedule_day) for d in
                ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")}
        run_sched = {"days": days}

    init_db()
    db = SessionLocal()
    try:
        strat = _build_strategy_row(args.name or f"opt-{expert}")
        db.add(strat); db.commit(); db.refresh(strat)

        backtest_block = {
            "engine": "daily",
            "enabled_instruments": universe,
            "experts": [{"class": expert, "settings": dict(spec["fixed_settings"])}],
            "start_date": args.start, "end_date": args.end,
            "initial_capital": float(args.initial_capital),
            "account_settings": {
                "starting_cash": float(args.initial_capital),
                "commission_per_trade": float(args.commission),
                "slippage_bps": float(args.slippage),
                "fill_model": args.fill_model,
            },
            "warmup_days": derive_warmup_days([expert]),
            "seed": int(args.seed),
            "subtype": "daily_expert",
            "run_schedule_override": run_sched,
            "execution_interval": args.interval,
            "backtest_id": int(_dt.now().timestamp()),
            "name": f"opt-{expert}-trial",
        }
        cfg = {
            "populationSize": int(args.population),
            "generations": int(args.generations),
            "crossoverProb": 0.6, "mutationProb": 0.3,
            "earlyStoppingGenerations": int(args.early_stop),
            "elitismPercent": 0.1, "seed": int(args.seed),
            "parallelIndividuals": int(args.parallel),
            # Expert decision params + the classic-RM sizing params (model:* namespace, real ba2
            # setting names). Without the RM block the RM stays fixed at its interface defaults.
            "expert_params": {**spec["expert_params"], **_RM_OPT},
            "backtest": backtest_block,
        }
        opt = StrategyOptimization(
            strategy_id=strat.id, name=args.name or f"opt-{expert}",
            fitness_metric=args.fitness, optimization_type="genetic",
            optimization_config=cfg, status="pending",
        )
        db.add(opt); db.commit(); db.refresh(opt)
        opt_id = opt.id
        print(f"optimize: strategy #{strat.id} + StrategyOptimization #{opt_id} "
              f"({expert} x {len(universe)} syms, pop={args.population} gen={args.generations} "
              f"parallel={args.parallel} fitness={args.fitness})")
    finally:
        db.close()

    res = handle_strategy_optimization("cli-optimize", {"optimization_id": opt_id})
    if res.get("status") != "completed":
        print(json.dumps(res, indent=2, default=str))
        sys.exit(f"ba2-test: optimization {opt_id} did not complete")

    # Re-run the best params as ONE tracked, tagged Backtest so it lands in runs/report.
    db = SessionLocal()
    try:
        opt = db.query(StrategyOptimization).filter(StrategyOptimization.id == opt_id).first()
        print(f"optimize: done. best_fitness={opt.best_fitness} best_params={json.dumps(opt.best_params, default=str)}")
    finally:
        db.close()
    nsaved = _persist_top_backtests(opt_id, expert, n=int(args.save_top))
    print(f"optimize: top {nsaved} persisted as tagged, saved Backtests (optimization_id={opt_id}); "
          f"run `ba2-test runs list --group {opt_id}` or `ba2-test report`.")
    return 0


def _persist_top_backtests(opt_id: int, expert: str, n: int = 5) -> int:
    """Re-run the optimization's TOP-N distinct param sets and persist each as a tagged,
    saved Backtest (best params + their metrics) so the top performers are kept for
    comparison and to warm-start future optimizations. Returns how many were persisted."""
    import json as _json
    from datetime import datetime as _dt
    import app.models  # noqa: F401
    from app.models.database import SessionLocal
    from app.models.backtest import Backtest
    from app.models.strategy import Strategy
    from app.models.strategy_optimization import StrategyOptimization
    from app.services.strategy_optimization_handler import _build_daily_trial_config  # noqa: SLF001
    from app.services.backtest.daily_backtest_handler import run_daily_backtest, _persist_results
    from app.services.strategy_param_space import decode_params

    db = SessionLocal()
    try:
        opt = db.query(StrategyOptimization).filter(StrategyOptimization.id == opt_id).first()
        strat = db.query(Strategy).filter_by(id=opt.strategy_id).first()
        cfg = opt.optimization_config or {}
        bt_block = dict(cfg["backtest"])

        # Top-N distinct param sets by fitness (fall back to best_params if all_results is thin).
        seen, ranked = set(), []
        for r in sorted(opt.all_results or [], key=lambda r: (r.get("fitness") if r.get("fitness") is not None else -1e9), reverse=True):
            key = _json.dumps(r.get("params"), sort_keys=True, default=str)
            if key in seen:
                continue
            seen.add(key)
            ranked.append(r["params"])
            if len(ranked) >= n:
                break
        if not ranked and opt.best_params:
            ranked = [opt.best_params]

        persisted = 0
        for rank, params in enumerate(ranked, start=1):
            trial_cfg = _build_daily_trial_config(bt_block, decode_params(strat, params))
            trial_cfg["name"] = f"TOP{rank}-{opt.name or expert}"
            # Persist this top-N run's trading DB (orders/transactions/recommendations) to disk
            # for post-mortem inspection — the GA trials run RAM-only for speed.
            trial_cfg["persist_trading_db"] = True
            results = run_daily_backtest(trial_cfg)
            bt = Backtest(
                name=trial_cfg["name"], model_id=None, engine_type="daily_expert",
                expert_name=expert, optimization_id=opt_id,
                strategy_params=params,
                start_date=_dt.fromisoformat(str(bt_block["start_date"])),
                end_date=_dt.fromisoformat(str(bt_block["end_date"])),
                initial_capital=float(bt_block["initial_capital"]),
                status="running", started_at=_dt.now(),
            )
            db.add(bt); db.commit(); db.refresh(bt)
            _persist_results(db, bt, results)
            bt.status = "completed"; bt.completed_at = _dt.now()
            bt.is_saved = True  # top performers of a job are kept
            db.commit()
            persisted += 1
        return persisted
    finally:
        db.close()


def _cmd_runs(args) -> int:
    from app.models.backtest import Backtest
    db = _runs_db()
    try:
        if args.runs_cmd == "prune":
            # Keep the best --keep runs per expert (by --metric, completed only); delete the
            # rest. Saved runs (is_saved) are ALWAYS kept and never counted against the budget.
            col = _METRIC_COL.get(args.metric)
            if col is None:
                sys.exit(f"ba2-test: unknown metric {args.metric!r}; use {sorted(_METRIC_COL)}")
            q = db.query(Backtest).filter(Backtest.status == "completed")
            if args.expert:
                q = q.filter(Backtest.expert_name == args.expert)
            rows = q.all()
            by_expert: dict = {}
            for r in rows:
                by_expert.setdefault(r.expert_name or "(none)", []).append(r)
            deleted = 0
            for expert, group in by_expert.items():
                keepers = [r for r in group if r.is_saved]
                cands = [r for r in group if not r.is_saved]
                cands.sort(key=lambda r: (getattr(r, col) if getattr(r, col) is not None else -1e9),
                           reverse=True)
                survivors = cands[: max(0, args.keep)]
                losers = cands[args.keep:]
                for r in losers:
                    db.delete(r)
                    deleted += 1
                print(f"{expert}: kept {len(survivors)} top + {len(keepers)} saved, "
                      f"deleted {len(losers)} (by {args.metric})")
            db.commit()
            print(f"-- pruned {deleted} run(s) total")
            return 0

        if args.runs_cmd == "stats":
            q = db.query(Backtest).filter(Backtest.status == "completed")
            if args.expert:
                q = q.filter(Backtest.expert_name == args.expert)
            if args.group is not None:
                q = q.filter(Backtest.optimization_id == args.group)
            rows = q.all()
            buckets: dict = {}
            key = (lambda r: r.optimization_id) if args.group is not None else (lambda r: r.expert_name or "(none)")
            for r in rows:
                buckets.setdefault(key(r), []).append(r)
            for k, group in sorted(buckets.items(), key=lambda kv: str(kv[0])):
                def _vals(c):
                    return [getattr(r, c) for r in group if getattr(r, c) is not None]
                shp = _vals("sharpe_ratio"); ret = _vals("total_return")
                best = max(shp) if shp else None
                avg = (sum(shp) / len(shp)) if shp else None
                label = ("opt#" + str(k)) if args.group is not None else str(k)
                print(f"{label}: n={len(group)} best_sharpe={best if best is None else round(best,2)} "
                      f"avg_sharpe={avg if avg is None else round(avg,2)} "
                      f"best_return={max(ret) if ret else None}")
            print(f"-- {len(rows)} run(s)")
            return 0

        if args.runs_cmd == "list":
            q = db.query(Backtest)
            if args.saved_only:
                q = q.filter(Backtest.is_saved == True)  # noqa: E712 (SQLAlchemy needs ==)
            if args.engine:
                q = q.filter(Backtest.engine_type == args.engine)
            if getattr(args, "expert", None):
                q = q.filter(Backtest.expert_name == args.expert)
            if getattr(args, "group", None) is not None:
                q = q.filter(Backtest.optimization_id == args.group)
            rows = q.order_by(Backtest.created_at.desc()).limit(args.limit).all()
            print(f"{'id':>5}  {'expert':<16} {'opt':>5} {'status':<10} {'ret%':>8} {'sharpe':>7} "
                  f"{'saved':<5} name")
            for r in rows:
                ret = f"{r.total_return:.2f}" if r.total_return is not None else "-"
                shp = f"{r.sharpe_ratio:.2f}" if r.sharpe_ratio is not None else "-"
                opt = str(r.optimization_id) if r.optimization_id is not None else "-"
                print(f"{r.id:>5}  {(r.expert_name or '-'):<16} {opt:>5} {(r.status or ''):<10} "
                      f"{ret:>8} {shp:>7} {('yes' if r.is_saved else 'no'):<5} {r.name}")
            print(f"-- {len(rows)} run(s)")
            return 0

        if args.runs_cmd == "save":
            r = db.query(Backtest).filter(Backtest.id == args.id).first()
            if r is None:
                sys.exit(f"ba2-test: run {args.id} not found")
            if args.name:
                r.name = args.name
            r.is_saved = True
            db.commit()
            print(f"saved run {r.id}: {r.name}")
            return 0

        if args.runs_cmd == "delete":
            r = db.query(Backtest).filter(Backtest.id == args.id).first()
            if r is None:
                sys.exit(f"ba2-test: run {args.id} not found")
            db.delete(r)
            db.commit()
            print(f"deleted run {args.id}")
            return 0

        if args.runs_cmd == "clear-unsaved":
            unsaved = db.query(Backtest).filter(Backtest.is_saved == False).all()  # noqa: E712
            n = len(unsaved)
            for r in unsaved:
                db.delete(r)
            db.commit()
            print(f"deleted {n} unsaved run(s)")
            return 0
        return 0
    finally:
        db.close()


def main(argv: "list | None" = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    _enter_backend()

    p = argparse.ArgumentParser(prog="ba2-test", description="BA2 Test Platform CLI.")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="Launch the API and/or the React frontend.")
    s.add_argument("--mode", default="both", choices=["both", "back", "front"],
                   help="What to start: both (default), back (API only), front (Vite UI only).")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8000, help="Backend API port (default 8000).")
    s.add_argument("--frontend-port", type=int, default=5173, help="Vite dev-server port (default 5173).")
    s.add_argument("--reload", action="store_true")

    # backtest: parse_known_args so the rest passes through to run_daily_backtest.
    sub.add_parser("backtest", help="Run a daily expert backtest (args forwarded to run_daily_backtest).",
                   add_help=False)

    fc = sub.add_parser("fetch-cache", help="Populate the as-of OHLCV cache.")
    fc.add_argument("--symbols", required=True, help="Comma-separated symbols.")
    fc.add_argument("--timeframes", default="1d", help="Comma-separated intervals (default 1d).")
    fc.add_argument("--start", required=True, help="ISO start date.")
    fc.add_argument("--end", required=True, help="ISO end date.")
    fc.add_argument("--provider", default="fmp", help="OHLCV provider (default fmp).")
    fc.add_argument("--workers", type=int, default=5)

    fs = sub.add_parser("fetch-screener", help="Build the screener-history cache for a range.")
    fs.add_argument("--settings-json", required=True, help="Path to a JSON file of screener settings.")
    fs.add_argument("--start", required=True, help="ISO start date.")
    fs.add_argument("--end", required=True, help="ISO end date.")
    fs.add_argument("--group", default="cli", help="Group label for the cached survivors.")
    fs.add_argument("--cache-db", required=True, help="Path to the screener-history SQLite cache.")
    fs.add_argument("--cadence-days", type=int, default=7, help="Days between scan dates (default 7).")

    cc = sub.add_parser("cache-clear", help="Clear cache (all, or one type).")
    cc.add_argument("--type", default=None, help="Cache type to clear (omit = all).")
    cc.add_argument("--before", default=None, help="Only clear entries older than this ISO date.")

    sub.add_parser("cache-usage", help="Show cache disk usage per type.")

    # runs: manage tracked backtest runs (the shared `backtests` results table).
    rp = sub.add_parser("runs", help="List / save / delete tracked backtest runs.")
    rsub = rp.add_subparsers(dest="runs_cmd", required=True)
    rl = rsub.add_parser("list", help="List tracked runs (newest first).")
    rl.add_argument("--limit", type=int, default=50)
    rl.add_argument("--saved-only", action="store_true", help="Only runs marked saved.")
    rl.add_argument("--engine", default=None, help="Filter by engine_type (ml/daily_expert).")
    rl.add_argument("--expert", default=None, help="Filter by expert_name.")
    rl.add_argument("--group", type=int, default=None, help="Filter by optimization_id.")
    rs = rsub.add_parser("save", help="Mark a run saved (survives clear-unsaved).")
    rs.add_argument("id", type=int)
    rs.add_argument("--name", default=None, help="Optionally rename the run.")
    rd = rsub.add_parser("delete", help="Delete one run by id.")
    rd.add_argument("id", type=int)
    rsub.add_parser("clear-unsaved", help="Delete all runs not marked saved.")
    rpr = rsub.add_parser("prune", help="Keep best N runs per expert (by metric); delete the rest.")
    rpr.add_argument("--keep", type=int, default=10, help="How many top runs to keep per expert.")
    rpr.add_argument("--metric", default="sharpe", help="Ranking metric (sharpe/calmar/return/...).")
    rpr.add_argument("--expert", default=None, help="Only prune this expert (else all).")
    rst = rsub.add_parser("stats", help="Per-expert (or per opt-job) summary stats.")
    rst.add_argument("--expert", default=None, help="Filter to one expert.")
    rst.add_argument("--group", type=int, default=None, help="Group by optimization_id (this job).")

    rep = sub.add_parser("report", help="Write an HTML summary of tracked backtests.")
    rep.add_argument("--out", default=None,
                     help="Output HTML path (default: <repo>/reports/ba2_backtest_report.html, "
                          "tracked in git so it syncs across machines).")

    op = sub.add_parser("optimize", help="Joint genetic optimization (expert + RM params + TP/SL).")
    op.add_argument("--expert", required=True, help="Expert class (FMPRating/FMPEarningsDrift/...).")
    op.add_argument("--universe", required=True, help="Comma-separated symbols.")
    op.add_argument("--start", required=True, help="ISO start date.")
    op.add_argument("--end", required=True, help="ISO end date.")
    op.add_argument("--fitness", default="sharpe_ratio", help="Fitness metric (default sharpe_ratio).")
    op.add_argument("--generations", type=int, default=6)
    op.add_argument("--population", type=int, default=10)
    op.add_argument("--parallel", type=int, default=4, help="Parallel trials (ThreadPoolExecutor).")
    op.add_argument("--early-stop", type=int, default=4)
    op.add_argument("--save-top", type=int, default=5,
                    help="Persist the top-N distinct param sets as saved Backtests (default 5).")
    op.add_argument("--seed", type=int, default=42, help="RNG seed (determinism).")
    op.add_argument("--initial-capital", type=float, default=10000.0)
    op.add_argument("--commission", type=float, default=1.0)
    op.add_argument("--slippage", type=float, default=0.0)
    op.add_argument("--fill-model", default="next_bar_open")
    op.add_argument("--interval", default="1d", help="Execution/fill interval (1d; 5min for intraday fills).")
    op.add_argument("--run-schedule", default="weekly", choices=["daily", "weekly"])
    op.add_argument("--run-schedule-day", default="monday")
    op.add_argument("--name", default=None)

    # Split out the backtest passthrough before full parsing.
    if argv and argv[0] == "backtest":
        return _cmd_backtest(argv[1:])

    args = p.parse_args(argv)
    return {
        "serve": lambda: _cmd_serve(args),
        "fetch-cache": lambda: _cmd_fetch_cache(args),
        "fetch-screener": lambda: _cmd_fetch_screener(args),
        "cache-usage": lambda: _cmd_cache_usage(args),
        "cache-clear": lambda: _cmd_cache_clear(args),
        "runs": lambda: _cmd_runs(args),
        "report": lambda: _cmd_report(args),
        "optimize": lambda: _cmd_optimize(args),
    }[args.cmd]()


if __name__ == "__main__":
    raise SystemExit(main())
