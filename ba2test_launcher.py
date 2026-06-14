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


def _cmd_serve(args) -> int:
    try:
        import uvicorn
    except ImportError:
        sys.exit("ba2-test: uvicorn not installed. Install backend/requirements.txt into this venv.")
    uvicorn.run("app.main:app", host=args.host, port=args.port, reload=args.reload)
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

    s = sub.add_parser("serve", help="Launch the FastAPI API.")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8000)
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
    }[args.cmd]()


if __name__ == "__main__":
    raise SystemExit(main())
