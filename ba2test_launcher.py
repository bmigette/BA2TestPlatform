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
    }[args.cmd]()


if __name__ == "__main__":
    raise SystemExit(main())
