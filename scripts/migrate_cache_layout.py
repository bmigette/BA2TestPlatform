#!/usr/bin/env python
"""Migrate the BA2 cache/data layout out of the code repos into ``BA2_HOME``.

Moves the legacy caches/artifacts (which used to live inside the repo tree or
under ~/Documents/ba2_trade_platform) into the locked single-root layout:

    BA2_HOME (default ~/Documents/ba2)
      common/
        cache/                      shared raw provider cache (OHLCV/asof/fmp_history)
        options/                    options-history cache
      test/
        db.sqlite                   test platform keys/app DB (ba2_common DB_FILE)
        dl_forecasting.db           test platform FastAPI app DB
        datasets/                   generated dataset CSVs
        trained_models/             saved model artifacts
        cache/jobs/                 per-job cache
        cache/news/                 news content files
        news_exports/               exported news JSON
      trade/
        db.sqlite                   live trade instance DB (was the legacy shared DB)
        screener/                   screener metric store + history db

DB placement: a DB is DATA, not a shared cache, so DBs are bucketed by owner —
the live trade DB -> trade/, the test platform DBs -> test/. Only shared raw
provider caches (OHLCV/fmp_history/screener/options) stay in common/cache. The
legacy shared ~/Documents/ba2_trade_platform/db.sqlite is the TRADE DB; it is
MOVED to trade/db.sqlite AND additionally COPIED to test/db.sqlite to seed the
test platform's keys (so test backtests have API keys post-migrate).

DRY-RUN by default — prints the planned moves/copies + sizes. Pass ``--apply``
to perform them. Idempotent: skips an op when the source is missing OR the
destination already exists. Never deletes a source if a move fails.

Run with the backend venv:

    cd BA2TestPlatform/backend && ./venv/bin/python ../scripts/migrate_cache_layout.py [--apply]
    # or from the repo root:
    backend/venv/bin/python scripts/migrate_cache_layout.py [--apply]
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import List, Optional, Tuple


def _human(nbytes: int) -> str:
    """Human-readable byte size."""
    size = float(nbytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024.0 or unit == "TB":
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


def _tree_size(path: Path) -> int:
    """Total bytes under a file or dir (best-effort)."""
    if path.is_file():
        try:
            return path.stat().st_size
        except OSError:
            return 0
    total = 0
    for f in path.rglob("*"):
        if f.is_file():
            try:
                total += f.stat().st_size
            except OSError:
                pass
    return total


def _resolve_new_paths() -> dict:
    """Resolve the NEW layout from ba2_common.config (single source of truth)."""
    try:
        from ba2_common import config as cfg
    except Exception as exc:  # pragma: no cover
        print(f"ERROR: cannot import ba2_common.config ({exc}). "
              "Run with the backend venv (./venv/bin/python).", file=sys.stderr)
        sys.exit(2)
    return {
        "BA2_HOME": Path(cfg.BA2_HOME),
        "COMMON_DIR": Path(cfg.COMMON_DIR),
        "TEST_DIR": Path(cfg.TEST_DIR),
        "TRADE_DIR": Path(cfg.TRADE_DIR),
        "CACHE_FOLDER": Path(cfg.CACHE_FOLDER),
        "DB_FILE": Path(cfg.DB_FILE),
        "SCREENER_STORE_DIR": Path(cfg.SCREENER_STORE_DIR),
        "SCREENER_HISTORY_DB": Path(cfg.SCREENER_HISTORY_DB),
        "OPTIONS_CACHE_DB": Path(cfg.OPTIONS_CACHE_DB),
    }


def _legacy_root() -> Path:
    """The legacy ~/Documents/ba2_trade_platform root (old common cache/DB home)."""
    return Path(os.path.expanduser("~")) / "Documents" / "ba2_trade_platform"


def _backend_dir() -> Path:
    """BA2TestPlatform/backend dir (scripts/ is at repo root)."""
    return Path(__file__).resolve().parents[1] / "backend"


def _build_moves(new: dict) -> List[Tuple[str, Path, Path, str]]:
    """Return the list of (label, src, dst, op) operations for the migration.

    ``op`` is "move" (relocate) or "copy" (duplicate, leaving the source in
    place). DBs are bucketed by owner: the legacy shared DB is the TRADE DB ->
    moved to trade/db.sqlite, and additionally COPIED to the test keys DB so the
    test platform has API keys post-migrate. Only shared raw provider caches stay
    in common/cache."""
    legacy = _legacy_root()
    backend = _backend_dir()
    trade_db = new["TRADE_DIR"] / "db.sqlite"
    # ORDER MATTERS: the sub-caches that live UNDER backend/datasets/cache (job
    # cache, news cache, options cache) are relocated to DIFFERENT destinations
    # than the wholesale `datasets` move, so they must be moved out FIRST — before
    # the parent `datasets` tree is moved — or they'd be swept into test/datasets.
    # Likewise the trade DB is COPIED to seed the test keys DB BEFORE it is moved,
    # so the copy still has a source to read.
    moves: List[Tuple[str, Path, Path, str]] = [
        # --- common bucket: shared provider cache ---
        ("legacy common cache", legacy / "cache", new["CACHE_FOLDER"], "move"),
        # --- DB placement (DATA, bucketed by owner) ---
        # The legacy shared DB is the LIVE trade DB. Seed the test keys DB from it
        # (copy) FIRST, then move the original into the trade/ bucket.
        ("seed test keys DB (copy of legacy trade DB)", legacy / "db.sqlite",
         new["DB_FILE"], "copy"),
        ("legacy trade DB", legacy / "db.sqlite", trade_db, "move"),
        ("legacy test app DB", backend / "dl_forecasting.db",
         new["TEST_DIR"] / "dl_forecasting.db", "move"),
        # --- test sub-caches (move BEFORE the parent datasets tree) ---
        ("job cache", backend / "datasets" / "cache" / "jobs",
         new["TEST_DIR"] / "cache" / "jobs", "move"),
        ("news cache", backend / "datasets" / "cache" / "news",
         new["TEST_DIR"] / "cache" / "news", "move"),
        # --- options cache -> common (also under backend/datasets/cache) ---
        ("options cache (legacy backend)", backend / "datasets" / "cache" / "options_cache.sqlite",
         new["OPTIONS_CACHE_DB"], "move"),
        # --- test bucket: BA2TestPlatform artifacts (were inside the repo) ---
        ("datasets", backend / "datasets", new["TEST_DIR"] / "datasets", "move"),
        ("trained_models", backend / "trained_models", new["TEST_DIR"] / "trained_models", "move"),
        ("news exports", backend / "news_exports", new["TEST_DIR"] / "news_exports", "move"),
        # --- trade bucket: screener caches ---
        ("screener store (legacy backend)", backend / "screener" / "metric_store",
         new["SCREENER_STORE_DIR"], "move"),
        ("screener history db (legacy backend)", backend / "screener" / "screener_history.sqlite",
         new["SCREENER_HISTORY_DB"], "move"),
    ]
    return moves


def _is_empty_dir(path: Path) -> bool:
    """True if ``path`` is a directory with no entries."""
    if not path.is_dir():
        return False
    try:
        next(path.iterdir())
        return False
    except StopIteration:
        return True
    except OSError:
        return False


def _plan(moves: List[Tuple[str, Path, Path, str]]) -> List[dict]:
    """Classify each op as do/skip with a reason + size (no I/O beyond stat).

    ``op`` ("move"/"copy") is preserved on the entry. An EMPTY destination
    directory (e.g. one auto-created by ``app.paths`` on import) is NOT treated as
    "already migrated": it is removed first so the real source data still lands.
    A NON-empty destination is left alone (idempotent)."""
    plan: List[dict] = []
    for label, src, dst, op in moves:
        entry = {"label": label, "src": src, "dst": dst, "op": op, "action": None,
                 "reason": "", "bytes": 0, "clear_empty_dst": False}
        if not src.exists():
            entry["action"] = "skip"
            entry["reason"] = "source missing"
        elif dst.exists() and not _is_empty_dir(dst):
            entry["action"] = "skip"
            entry["reason"] = "destination already exists"
        else:
            entry["action"] = op
            entry["bytes"] = _tree_size(src)
            entry["clear_empty_dst"] = _is_empty_dir(dst)
        plan.append(entry)
    return plan


def _apply_move(src: Path, dst: Path, clear_empty_dst: bool = False,
                op: str = "move") -> Optional[str]:
    """Perform one move/copy. Returns an error string on failure, else None.

    ``op`` is "move" (relocate) or "copy" (duplicate; source left in place — used
    to seed the test keys DB from the trade DB). If ``clear_empty_dst`` the
    (empty) destination dir is removed first so the op replaces it rather than
    nesting the source inside it. Creates parent dirs first; never deletes the
    source on failure (shutil.move/copy2 leave the source intact if they raise)."""
    try:
        if clear_empty_dst and _is_empty_dir(dst):
            dst.rmdir()
        dst.parent.mkdir(parents=True, exist_ok=True)
        if op == "copy":
            if src.is_dir():
                shutil.copytree(str(src), str(dst))
            else:
                shutil.copy2(str(src), str(dst))
        else:
            shutil.move(str(src), str(dst))
        return None
    except Exception as exc:  # noqa: BLE001
        return str(exc)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Migrate BA2 cache/data layout into BA2_HOME.")
    ap.add_argument("--apply", action="store_true",
                    help="Perform the moves (default is a dry-run preview).")
    args = ap.parse_args(argv)

    new = _resolve_new_paths()
    moves = _build_moves(new)
    plan = _plan(moves)

    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"=== BA2 cache-layout migration [{mode}] ===")
    print(f"BA2_HOME = {new['BA2_HOME']}")
    print(f"  common = {new['COMMON_DIR']}")
    print(f"  test   = {new['TEST_DIR']}")
    print(f"  trade  = {new['TRADE_DIR']}")
    print("")

    to_move = [p for p in plan if p["action"] in ("move", "copy")]
    skipped = [p for p in plan if p["action"] == "skip"]

    print("Planned operations:")
    if not to_move:
        print("  (none — nothing to migrate)")
    for p in to_move:
        verb = "COPY" if p["op"] == "copy" else "MOVE"
        print(f"  {verb}  {p['label']:<44} {_human(p['bytes']):>10}")
        print(f"        {p['src']}")
        print(f"     -> {p['dst']}")
    print("")
    print("Skipped:")
    for p in skipped:
        print(f"  SKIP  {p['label']:<44} ({p['reason']})")
    print("")

    total = sum(p["bytes"] for p in to_move)
    print(f"Total to move/copy: {len(to_move)} item(s), {_human(total)}")

    if not args.apply:
        print("")
        print("DRY-RUN only — no files were moved. Re-run with --apply to perform the migration.")
        return 0

    print("")
    print("Applying...")
    failures = 0
    for p in to_move:
        err = _apply_move(p["src"], p["dst"],
                          clear_empty_dst=p.get("clear_empty_dst", False),
                          op=p["op"])
        if err:
            failures += 1
            print(f"  FAILED  {p['label']}: {err} (source left intact)")
        else:
            verb = "copied" if p["op"] == "copy" else "moved"
            print(f"  {verb:<7} {p['label']} -> {p['dst']}")

    print("")
    print("=== Summary ===")
    print(f"  moved:   {len(to_move) - failures}")
    print(f"  failed:  {failures}")
    print(f"  skipped: {len(skipped)}")
    print("")
    print("IMPORTANT: restart any running BA2 instances (test API + live trade apps)")
    print("so they pick up the new cache/data locations.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
