"""Resolve a screener-typed daily-backtest universe from the OFFLINE screener cache (Task 8).

A daily backtest may declare a *screener* universe instead of a static symbol list. The
point-in-time survivor set for each scan date is expensive to compute (a full as-of
``StockScreener`` pipeline with provider fetches), so it is built AHEAD of time via
``ba2-test fetch-screener`` into a ``ScreenerHistoryCache`` SQLite DB. At backtest time we
must NOT live-screen (slow, and a mid-run screen would not stay point-in-time) — we only
REPLAY what was cached.

:func:`resolve_screener_universe` therefore reads ONLY from the cache: it unions the cached
survivor symbols across every scan date in ``[start, end]`` for the given group. If NO scan
date in the range is cached it raises :class:`ScreenerCacheMiss` (fail-early,
``backend/CLAUDE.md``) telling the user to build the cache first — it never falls back to
``screened_universe_for_bar`` (which rebuilds/live-screens on a miss).
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

from app.services.screener_history_cache import (
    ScreenerHistoryCache,
    _scan_date_key,
    screen_config_hash,
)


class ScreenerCacheMiss(RuntimeError):
    """Raised when no cached scan date covers the requested range for the screener universe.

    Subclasses ``RuntimeError`` so it propagates out of the daily handler's generic
    ``except Exception`` and fails the run with a clear, actionable message (build the cache
    via ``ba2-test fetch-screener``) rather than silently running an empty universe.
    """


def _coerce_date_key(value: Any, field: str) -> str:
    """Canonicalise a range bound (ISO string OR datetime) to a ``YYYY-MM-DD`` cache key."""
    if isinstance(value, datetime):
        return _scan_date_key(value)
    try:
        return _scan_date_key(datetime.fromisoformat(str(value)))
    except (TypeError, ValueError) as e:  # noqa: BLE001
        raise ValueError(f"{field} is not a valid ISO date: {value!r} ({e})")


def _config_hash(screener_settings: Dict[str, Any]) -> str:
    """The cache hash-namespace for these settings, computed POST-coercion (read-only).

    Constructs a ``StockScreener`` purely to coerce the settings to their canonical types
    (its ``__init__`` does NO provider fetches), so the hash matches the one
    ``ba2-test fetch-screener`` wrote rows under. Falls back to hashing the raw settings if
    the provider package is unavailable (then the namespace matches only when the supplied
    settings are already coerced — true for a settings dict that round-tripped the cache).
    """
    try:
        from ba2_providers.StockScreener import StockScreener

        sc = StockScreener(screener_settings)
        return screen_config_hash(sc._settings)
    except Exception:  # noqa: BLE001 — provider not importable: hash the raw subset
        return screen_config_hash(screener_settings)


def resolve_screener_universe(
    screener_settings: Dict[str, Any],
    start: Any,
    end: Any,
    cache_db: str,
    group: str,
) -> List[str]:
    """Return the sorted union of cached survivor symbols across ``[start, end]`` scan dates.

    READ-ONLY: queries the offline ``ScreenerHistoryCache`` only — it does NOT call
    ``screened_universe_for_bar`` (which would rebuild/live-screen on a miss). The cache is
    scoped by ``group`` (the ``--group`` label used by ``ba2-test fetch-screener``) and by the
    config-hash namespace derived from ``screener_settings`` so different criteria never mix.

    Args:
        screener_settings: the screener criteria dict (the same shape fed to
            ``ba2-test fetch-screener --settings-json``).
        start, end: the backtest range bounds (ISO string or ``datetime``); inclusive.
        cache_db: path to the screener-history SQLite cache (``--cache-db``).
        group: the survivor group label (``--group``).

    Raises:
        ScreenerCacheMiss: if NO scan date in ``[start, end]`` is present in the cache for
            this group + config namespace — tells the user to build it via
            ``ba2-test fetch-screener``.
        ValueError: on an unparseable date bound.
    """
    start_key = _coerce_date_key(start, "start")
    end_key = _coerce_date_key(end, "end")
    cfg_hash = _config_hash(screener_settings)

    cache = ScreenerHistoryCache(cache_db)
    scan_dates = cache.cached_scan_dates(group, start_key, end_key, cfg_hash)
    if not scan_dates:
        raise ScreenerCacheMiss(
            f"No cached screener scan dates in [{start_key}, {end_key}] for group "
            f"'{group}' (config {cfg_hash[:8]}) in {cache_db}. Build the screener-history "
            f"cache first, e.g.: ba2-test fetch-screener --settings-json <file> "
            f"--start {start_key} --end {end_key} --group {group} --cache-db {cache_db}"
        )

    symbols: set[str] = set()
    for sd in scan_dates:
        for row in cache.survivors_for_key(sd, group, cfg_hash):
            sym = row.get("symbol")
            if sym:
                symbols.add(sym)
    return sorted(symbols)
