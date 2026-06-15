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


# Process-global memo of resolved screener unions, keyed by (cache_db, group, cfg_hash, start,
# end). The union is identical for every GA trial on a fixed cache + config + range, and the
# pool worker stays alive across trials, so this resolves the universe ~once per worker and
# shares it across the whole population (the OHLCV-memo pattern). Cleared by clear_universe_memo.
_UNIVERSE_MEMO: Dict[tuple, List[str]] = {}


def clear_universe_memo() -> None:
    """Drop the process-global screener-universe memo (tests / between distinct caches)."""
    _UNIVERSE_MEMO.clear()


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

    # Process-global memo: the screener union is param-INDEPENDENT (same for every GA trial on
    # a fixed cache + group + config + range), so resolve it ONCE per worker and reuse across the
    # whole population — exactly like the OHLCV memo. Re-resolving per trial was ~1.5s each
    # (one sqlite connection + query per scan date) — ~900s across a 600-trial optimization.
    memo_key = (str(cache_db), group, cfg_hash, start_key, end_key)
    hit = _UNIVERSE_MEMO.get(memo_key)
    if hit is not None:
        return list(hit)

    # ONE indexed query for the DISTINCT survivor union (was cached_scan_dates + a
    # survivors_for_key per date — a connection-open per scan date).
    cache = ScreenerHistoryCache(cache_db)
    symbols = cache.union_symbols(group, start_key, end_key, cfg_hash)
    if not symbols:
        raise ScreenerCacheMiss(
            f"No cached screener survivors in [{start_key}, {end_key}] for group "
            f"'{group}' (config {cfg_hash[:8]}) in {cache_db}. Build the screener-history "
            f"cache first, e.g.: ba2-test fetch-screener --settings-json <file> "
            f"--start {start_key} --end {end_key} --group {group} --cache-db {cache_db}"
        )

    _UNIVERSE_MEMO[memo_key] = symbols
    return list(symbols)
