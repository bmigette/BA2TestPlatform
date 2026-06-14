"""Grouped/labeled screener-history cache (Phase 3, Task 5).

Row identity: ``(symbol, scan_date, screen_config_hash)``. ``screen_config_hash`` is the
*machine* identity (sha256 of the result-affecting screener settings AFTER coercion).
``group_label`` (a.k.a. group/expert) is a *human* label (e.g. ``FactorRanker``,
``PennyMomentumTrader``, ``Weinstein-S2``). A config change yields a distinct hash
namespace so different criteria never mix; replaying a ``(scan_date, hash)`` returns the
exact survivor set with **zero provider fetches** (gate item 4: cache-once).

Storage is a dedicated SQLite ``screener_history`` table (Decision 7). The table is
self-created on construction (``executescript(_DDL)``) for standalone / test use; the host
DB schema lifecycle adds it via ``db_migrate/019_add_screener_history.py``. The DDL here
and the migration's CREATE TABLE are kept identical so both paths converge on one schema.

A backtest sweep over scan dates becomes a cheap cache replay: the Phase-4 engine calls
:func:`screened_universe_for_bar` per rebalance bar — first hit runs the as-of
``StockScreener`` pipeline and persists labeled survivors; every subsequent hit replays.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ba2_common.logger import logger

# Module-level import so :func:`screened_universe_for_bar` resolves ``StockScreener`` via
# this module's global namespace (tests patch it as
# ``screener_history_cache.StockScreener``). Editable install makes the package importable.
from ba2_providers.StockScreener import StockScreener


# The result-affecting screener settings: the 14 ``screener_*`` threshold keys + the
# Weinstein stage-2 flag + the sort metric + the universe mode. These — and ONLY these —
# decide which symbols survive, so they (post-coercion) define the cache hash namespace.
# Mirrors ``StockScreener._DEFAULTS`` (Decision 6).
_HASH_KEYS = [
    "screener_provider",
    "screener_market_cap_min",
    "screener_market_cap_max",
    "screener_volume_min",
    "screener_volume_max",
    "screener_float_min",
    "screener_float_max",
    "screener_price_min",
    "screener_price_max",
    "screener_relative_volume_min",
    "screener_price_drop_pct",
    "screener_price_drop_days",
    "screener_max_stocks",
    "screener_sort_metric",
    "screener_weinstein_stage2_only",
    "universe_mode",
]


def screen_config_hash(coerced_settings: Dict[str, Any]) -> str:
    """sha256 of canonical-JSON of the result-affecting settings subset (post-coercion).

    Deterministic (sorted keys, compact separators) so the same criteria always hash to
    the same namespace; a single threshold change flips the hash -> a distinct namespace.
    Missing keys map to ``None`` so a settings dict that omits a key hashes the same as one
    that passes that key's default through ``StockScreener`` coercion (callers should pass
    the post-coercion ``StockScreener._settings`` to bake defaults in).
    """
    subset = {k: coerced_settings.get(k) for k in _HASH_KEYS}
    canonical = json.dumps(subset, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# Kept byte-identical to db_migrate/019_add_screener_history.py so the standalone
# self-create and the host migration converge on one schema.
_DDL = """
CREATE TABLE IF NOT EXISTS screener_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    scan_date TEXT NOT NULL,
    group_label TEXT NOT NULL,
    screen_config_hash TEXT NOT NULL,
    rank INTEGER,
    sort_metric_value REAL,
    market_cap_as_of REAL,
    price_as_of REAL,
    relative_volume REAL,
    weinstein_stage INTEGER,
    price_drop_pct REAL,
    universe_mode TEXT,
    market_cap_source TEXT,
    float_approx INTEGER DEFAULT 1,
    created_at TEXT NOT NULL,
    UNIQUE(symbol, scan_date, screen_config_hash)
);
CREATE INDEX IF NOT EXISTS ix_scan_hash ON screener_history(scan_date, screen_config_hash);
"""


def _scan_date_key(scan_date: datetime) -> str:
    """Canonicalise a scan date to a ``YYYY-MM-DD`` row key (date granularity)."""
    return scan_date.strftime("%Y-%m-%d")


class ScreenerHistoryCache:
    """SQLite-backed grouped/labeled survivor cache for the as-of screener.

    One row per surviving ``(symbol, scan_date, screen_config_hash)``. ``write`` is
    idempotent via the UNIQUE key (INSERT OR REPLACE), so re-running a scan overwrites
    rather than duplicating. ``has``/``replay`` are read-only and never fetch.
    """

    def __init__(self, db_path: str):
        self._db_path = db_path
        self._lock = threading.Lock()
        with self._connect() as con:
            con.executescript(_DDL)

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self._db_path, timeout=30.0)
        con.row_factory = sqlite3.Row
        return con

    def has(self, scan_date: datetime, cfg_hash: str) -> bool:
        """True if any survivor row exists for ``(scan_date, cfg_hash)`` — replay-able."""
        with self._connect() as con:
            row = con.execute(
                "SELECT 1 FROM screener_history "
                "WHERE scan_date=? AND screen_config_hash=? LIMIT 1",
                (_scan_date_key(scan_date), cfg_hash),
            ).fetchone()
            return row is not None

    def replay(self, scan_date: datetime, cfg_hash: str) -> List[Dict[str, Any]]:
        """Return cached survivors (rank-ordered) for ``(scan_date, hash)`` — NO fetches."""
        with self._connect() as con:
            rows = con.execute(
                "SELECT * FROM screener_history "
                "WHERE scan_date=? AND screen_config_hash=? "
                "ORDER BY rank ASC",
                (_scan_date_key(scan_date), cfg_hash),
            ).fetchall()
        return [dict(r) for r in rows]

    def write(
        self,
        scan_date: datetime,
        cfg_hash: str,
        group_label: str,
        survivors: List[Dict[str, Any]],
        universe_mode: str,
    ) -> int:
        """Persist the survivor list (idempotent via the UNIQUE key). Returns row count."""
        now = datetime.now(timezone.utc).isoformat()
        sd = _scan_date_key(scan_date)
        written = 0
        with self._lock, self._connect() as con:
            for rank, s in enumerate(survivors):
                con.execute(
                    "INSERT OR REPLACE INTO screener_history "
                    "(symbol, scan_date, group_label, screen_config_hash, rank, "
                    " sort_metric_value, market_cap_as_of, price_as_of, relative_volume, "
                    " weinstein_stage, price_drop_pct, universe_mode, market_cap_source, "
                    " float_approx, created_at) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        s.get("symbol"),
                        sd,
                        group_label,
                        cfg_hash,
                        rank,
                        s.get("sort_metric_value"),
                        s.get("market_cap"),
                        s.get("price"),
                        s.get("relative_volume"),
                        s.get("weinstein_stage"),
                        s.get("price_drop_pct"),
                        universe_mode,
                        s.get("market_cap_source"),
                        1 if s.get("float_approx", True) else 0,
                        now,
                    ),
                )
                written += 1
            con.commit()
        logger.info(
            f"screener-history: wrote {written} survivors for {sd} "
            f"({group_label}, {cfg_hash[:8]})"
        )
        return written


def screened_universe_for_bar(
    settings: Dict[str, Any],
    scan_date: datetime,
    group_label: str,
    cache: "ScreenerHistoryCache",
    progress_callback=None,
) -> List[Dict[str, Any]]:
    """Replay-or-compute the screened universe for one rebalance bar.

    If ``(scan_date, screen_config_hash)`` is already cached, replay it with **no provider
    fetches** (gate item 4). Otherwise run the as-of ``StockScreener`` pipeline, annotate
    the cache-audit columns, persist the labeled survivors, and return them.

    The returned shape differs between the two paths by design: the compute path returns
    the raw ``StockScreener`` survivor dicts (the bar's universe, threaded onward); the
    replay path returns the persisted rows. Callers key off ``symbol`` (stable in both).
    """
    sc = StockScreener(settings, progress_callback=progress_callback, as_of=scan_date)
    cfg_hash = screen_config_hash(sc._settings)  # post-coercion: defaults baked in

    if cache.has(scan_date, cfg_hash):
        logger.info(
            f"screener-history: replay {_scan_date_key(scan_date)} {cfg_hash[:8]} (no fetch)"
        )
        return cache.replay(scan_date, cfg_hash)

    result = sc.screen()
    survivors = result["results"]

    # Annotate sort_metric_value (used for cache rank ordering) + cache-audit defaults.
    metric = sc._settings["screener_sort_metric"]
    for s in survivors:
        s["sort_metric_value"] = s.get(metric) if s.get(metric) is not None else s.get("market_cap")
        s.setdefault("market_cap_source", "shares_x_close")
        s.setdefault("float_approx", True)

    cache.write(scan_date, cfg_hash, group_label, survivors, sc._settings["universe_mode"])
    return survivors
