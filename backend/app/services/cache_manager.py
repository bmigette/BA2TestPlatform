"""Cache discovery + usage scanner for the cache-management UI.

Single source of truth for every cache root. Normalizes the CWD path
inconsistency across the codebase by resolving every backend-relative root
against the backend dir, NOT the process CWD:

  - dataproviders/base.py:78   Path("backend/datasets/cache")  (CWD-relative)
  - app/api/datasets.py:738    Path("datasets")                (CWD-relative)
  - app/services/news_cache.py NewsCacheService(cache_dir="datasets/cache/news")
  - app/api/tools.py:244       NEWS_EXPORTS_DIR = Path("news_exports")
  - dataproviders/interfaces/MarketDataProviderInterface.py:23
        CACHE_FOLDER = os.getenv("CACHE_FOLDER", <backend>/cache)  (env-overridable)

The ``asof`` provider cache is a DIFFERENT root than the backend tree: it lives
under ``ba2_common.config.CACHE_FOLDER`` (default ~/Documents/ba2_trade_platform/cache,
NOT env-driven), with the native parquet time-series at ``<CACHE_FOLDER>/<provider>/``
and the provider_cache spill/SQLite at ``<CACHE_FOLDER>/datasets/cache``
(ba2_providers/cache/native_cache.py). We import that constant rather than guess.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# backend/ root = parents[2] from app/services/cache_manager.py
# (app/services/cache_manager.py -> app/services -> app -> backend)
BACKEND_DIR = Path(__file__).resolve().parents[2]

# CACHE_FOLDER as the backend provider layer sees it
# (MarketDataProviderInterface.py:23, env-overridable, default <backend>/cache).
CACHE_FOLDER = Path(os.getenv("CACHE_FOLDER", str(BACKEND_DIR / "cache")))


def _asof_roots() -> List[Path]:
    """Resolve the ba2_providers as_of cache roots from ba2_common.config.

    Returns both the native parquet time-series root (<CACHE_FOLDER>/<provider>/)
    and the provider_cache spill root (<CACHE_FOLDER>/datasets/cache). These live
    under ba2_common.config.CACHE_FOLDER, a DIFFERENT root than the backend's
    own <backend>/cache. Imported defensively so a backend without ba2_common
    installed still loads the scanner (empty asof type)."""
    try:
        from ba2_common.config import CACHE_FOLDER as ASOF_CACHE_FOLDER
    except Exception:
        return []
    base = Path(ASOF_CACHE_FOLDER)
    return [base, base / "datasets" / "cache"]


def _resolve(p: "str | Path") -> Path:
    """Resolve a backend-relative path against BACKEND_DIR (absolute passes through)."""
    p = Path(p)
    return p if p.is_absolute() else (BACKEND_DIR / p)


# Cache-type -> on-disk root(s) + metadata. Mirrors the cache_ui_scope contract.
# DESTRUCTIVE types (datasets, models) are excluded from "clean all".
CACHE_TYPES: Dict[str, Dict[str, Any]] = {
    "ohlcv":    {"roots": [CACHE_FOLDER],                      "destructive": False, "ttl_hours": 24},
    "jobs":     {"roots": [_resolve("datasets/cache/jobs")],  "destructive": False, "ttl_hours": None},
    "news":     {"roots": [_resolve("datasets/cache/news")],  "destructive": False, "ttl_hours": None, "db_backed": True},
    "datasets": {"roots": [_resolve("datasets")],             "destructive": True,  "ttl_hours": None},
    "models":   {"roots": [_resolve("trained_models")],       "destructive": True,  "ttl_hours": None},
    "exports":  {"roots": [_resolve("news_exports")],         "destructive": False, "ttl_hours": None},
    # ba2_providers as_of cache: parquet time-series + provider_cache spill, under
    # ba2_common.config.CACHE_FOLDER (NOT <backend>/cache). Resolved lazily.
    "asof":     {"roots": _asof_roots(),                      "destructive": False, "ttl_hours": None},
}


def _scan_dir(root: Path) -> Dict[str, Any]:
    """Return total bytes, file count, oldest/newest mtime (ISO UTC) for a tree."""
    total = 0
    count = 0
    oldest: Optional[float] = None
    newest: Optional[float] = None
    if not root.exists():
        return {"bytes": 0, "files": 0, "oldest": None, "newest": None, "exists": False}
    for f in root.rglob("*"):
        if f.is_file():
            try:
                st = f.stat()
            except OSError:
                continue
            total += st.st_size
            count += 1
            m = st.st_mtime
            oldest = m if oldest is None else min(oldest, m)
            newest = m if newest is None else max(newest, m)
    return {
        "bytes": total,
        "files": count,
        "exists": True,
        "oldest": datetime.fromtimestamp(oldest, tz=timezone.utc).isoformat() if oldest else None,
        "newest": datetime.fromtimestamp(newest, tz=timezone.utc).isoformat() if newest else None,
    }


def get_usage() -> Dict[str, Any]:
    """Per-type disk usage for every tracked cache (bytes, file count, oldest/newest
    mtime, destructive flag, TTL). News also reports DB row counts when available."""
    out: Dict[str, Any] = {}
    for name, cfg in CACHE_TYPES.items():
        agg: Dict[str, Any] = {
            "bytes": 0, "files": 0, "oldest": None, "newest": None, "exists": False,
            "destructive": cfg["destructive"], "ttl_hours": cfg["ttl_hours"],
        }
        for root in cfg["roots"]:
            s = _scan_dir(Path(root))
            agg["bytes"] += s["bytes"]
            agg["files"] += s["files"]
            agg["exists"] = agg["exists"] or s["exists"]
            for k in ("oldest", "newest"):
                if s[k] and (
                    agg[k] is None
                    or (k == "oldest" and s[k] < agg[k])
                    or (k == "newest" and s[k] > agg[k])
                ):
                    agg[k] = s[k]
        if cfg.get("db_backed") and name == "news":
            try:
                from app.services.news_cache import NewsCacheService
                agg["db_stats"] = NewsCacheService().get_cache_stats()  # news_cache.py:550
            except Exception:
                agg["db_stats"] = None
        out[name] = agg
    return out


def drill_down(cache_type: str) -> List[Dict[str, Any]]:
    """Per-item breakdown for one cache type (UI drill-down).

    ohlcv: per <SYMBOL>_<interval> file under each provider subfolder.
    news:  per-provider article counts from the DB stats.
    jobs/models: per task_id directory size.
    datasets/exports/asof: flat file listing.
    """
    cfg = CACHE_TYPES.get(cache_type)
    if not cfg:
        raise KeyError(cache_type)
    items: List[Dict[str, Any]] = []
    if cache_type == "ohlcv":
        # provider subfolders contain <SYMBOL>_<interval>.{csv,parquet}
        # (MarketDataProviderInterface.py:50 per-class subfolder; base.py file naming).
        for root in cfg["roots"]:
            root = Path(root)
            if not root.exists():
                continue
            for f in root.rglob("*"):
                if f.is_file() and f.suffix in (".csv", ".parquet"):
                    stem = f.stem  # SYMBOL_interval
                    sym, _, interval = stem.rpartition("_")
                    st = f.stat()
                    items.append({
                        "provider": f.parent.name,
                        "symbol": sym or stem,
                        "interval": interval,
                        "bytes": st.st_size,
                        "mtime": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
                        "stale": (datetime.now().timestamp() - st.st_mtime) > 24 * 3600,
                    })
    elif cache_type == "news":
        try:
            from app.services.news_cache import NewsCacheService
            stats = NewsCacheService().get_cache_stats()  # by_provider counts
        except Exception:
            stats = {}
        for prov, n in (stats.get("by_provider") or {}).items():
            items.append({"provider": prov, "articles": n})
    elif cache_type in ("jobs", "models"):
        for root in cfg["roots"]:
            root = Path(root)
            if not root.exists():
                continue
            for d in root.iterdir():
                if d.is_dir():
                    size = sum(x.stat().st_size for x in d.rglob("*") if x.is_file())
                    items.append({"task_id": d.name, "bytes": size})
    else:  # datasets, exports, asof — flat file listing
        for root in cfg["roots"]:
            root = Path(root)
            if not root.exists():
                continue
            for f in root.rglob("*"):
                if f.is_file():
                    st = f.stat()
                    items.append({
                        "name": str(f.relative_to(root)),
                        "bytes": st.st_size,
                        "mtime": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
                    })
    return items
