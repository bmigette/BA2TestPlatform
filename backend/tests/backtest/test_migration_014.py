"""Migration 014 gate (Phase 3 Task 5): add the `screener_history` table.

Verifies the migration:
  * applies on a FRESH DB (creates the table + ix_scan_hash index, returns True);
  * is idempotent (a second upgrade is a no-op returning False);
  * applies on an EXISTING populated DB WITHOUT touching existing tables/rows
    (it only adds a new table);
  * produces a schema byte-compatible with the cache service self-create
    (ScreenerHistoryCache can open the migrated DB and write/replay rows);
  * downgrade drops the table cleanly.

Pure-sqlite (no network, no app server) so it runs in the unit suite. Run from the
backend cwd (`app.*` import root):
    ./venv/bin/python -m pytest tests/backtest/test_migration_014.py -v
"""

import importlib.util
import os
import sqlite3
import tempfile

MIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "db_migrate",
    "014_add_screener_history.py",
)


def _load_migration():
    spec = importlib.util.spec_from_file_location("m014", MIG_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _table_exists(cur, name):
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,))
    return cur.fetchone() is not None


def _index_exists(cur, name):
    cur.execute("SELECT name FROM sqlite_master WHERE type='index' AND name=?", (name,))
    return cur.fetchone() is not None


def _cols(cur, table):
    cur.execute(f"PRAGMA table_info({table})")
    return [r[1] for r in cur.fetchall()]


def test_014_fresh_db_creates_table_and_index():
    m = _load_migration()
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "fresh.db")
        conn = sqlite3.connect(p)
        cur = conn.cursor()
        assert not _table_exists(cur, "screener_history")

        assert m.upgrade(cur, conn) is True
        assert _table_exists(cur, "screener_history")
        assert _index_exists(cur, "ix_scan_hash")

        cols = _cols(cur, "screener_history")
        for expected in (
            "symbol", "scan_date", "group_label", "screen_config_hash", "rank",
            "sort_metric_value", "market_cap_as_of", "price_as_of", "relative_volume",
            "weinstein_stage", "price_drop_pct", "universe_mode", "market_cap_source",
            "float_approx", "created_at",
        ):
            assert expected in cols, f"missing column {expected}"

        # Idempotent.
        assert m.upgrade(cur, conn) is False
        conn.close()


def test_014_existing_populated_db_no_data_loss():
    """Existing host tables/rows are untouched; only the new table is added."""
    m = _load_migration()
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "existing.db")
        conn = sqlite3.connect(p)
        cur = conn.cursor()
        # Stand in for a populated host DB.
        cur.execute("CREATE TABLE backtests (id INTEGER PRIMARY KEY, name TEXT)")
        cur.executemany(
            "INSERT INTO backtests (id, name) VALUES (?, ?)",
            [(1, "run-a"), (2, "run-b")],
        )
        conn.commit()

        assert m.upgrade(cur, conn) is True
        assert _table_exists(cur, "screener_history")

        # Existing data intact.
        cur.execute("SELECT id, name FROM backtests ORDER BY id")
        assert cur.fetchall() == [(1, "run-a"), (2, "run-b")]
        conn.close()


def test_014_schema_matches_cache_service():
    """The migrated table must be writable/replayable by the cache service (one schema)."""
    m = _load_migration()
    from datetime import datetime, timezone
    from app.services.screener_history_cache import ScreenerHistoryCache

    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "shared.db")
        conn = sqlite3.connect(p)
        cur = conn.cursor()
        assert m.upgrade(cur, conn) is True
        conn.close()

        # Service opens the migrated DB; its IF-NOT-EXISTS self-create is a no-op here.
        cache = ScreenerHistoryCache(p)
        sd = datetime(2020, 6, 30, tzinfo=timezone.utc)
        cache.write(sd, "hashX", "FactorRanker",
                    [{"symbol": "AAA", "market_cap": 5e9, "price": 50.0}], "broad")
        rows = cache.replay(sd, "hashX")
        assert [r["symbol"] for r in rows] == ["AAA"]


def test_014_downgrade_drops_table():
    m = _load_migration()
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "down.db")
        conn = sqlite3.connect(p)
        cur = conn.cursor()
        m.upgrade(cur, conn)
        assert _table_exists(cur, "screener_history")
        assert m.downgrade(cur, conn) is True
        assert not _table_exists(cur, "screener_history")
        assert not _index_exists(cur, "ix_scan_hash")
        conn.close()
