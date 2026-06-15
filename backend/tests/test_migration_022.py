"""
Test for migration 022: drop the retired `Strategy.rm_*` columns.

Builds a legacy `strategies` table containing a few `rm_*` columns alongside the
kept columns + one row, loads `db_migrate/022_drop_strategy_rm_columns.py` via
importlib, runs its migration against an open sqlite3 connection (the real runner
contract: `upgrade(cursor, conn)`), then asserts NO column starts with `rm_` and
the kept row (name, initial_tp_percent, id) survives.
"""

import importlib.util
import sqlite3
from pathlib import Path

MIGRATION_PATH = (
    Path(__file__).resolve().parent.parent
    / "db_migrate"
    / "022_drop_strategy_rm_columns.py"
)


def _load_migration():
    spec = importlib.util.spec_from_file_location(
        "migration_022", MIGRATION_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _table_columns(cursor, table):
    cursor.execute(f"PRAGMA table_info({table})")
    return [row[1] for row in cursor.fetchall()]


def _build_legacy_strategies(conn):
    """Create a legacy strategies table with rm_* + kept columns and one row."""
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE strategies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name VARCHAR(255) NOT NULL,
            description TEXT,
            required_fields JSON,
            entry_conditions JSON,
            buy_entry_conditions JSON,
            sell_entry_conditions JSON,
            exit_conditions JSON,
            initial_tp_percent FLOAT DEFAULT 5.0,
            initial_tp_optimize BOOLEAN DEFAULT 0,
            initial_tp_min FLOAT,
            initial_tp_max FLOAT,
            initial_tp_step FLOAT,
            initial_sl_percent FLOAT DEFAULT 2.0,
            initial_sl_optimize BOOLEAN DEFAULT 0,
            initial_sl_min FLOAT,
            initial_sl_max FLOAT,
            initial_sl_step FLOAT,
            rm_risk_per_trade_pct FLOAT DEFAULT 1.0,
            rm_risk_per_trade_pct_optimize BOOLEAN DEFAULT 0,
            rm_risk_per_trade_pct_min FLOAT,
            rm_risk_per_trade_pct_max FLOAT,
            rm_risk_per_trade_pct_step FLOAT,
            rm_max_concurrent_positions INTEGER DEFAULT 5,
            rm_max_concurrent_positions_optimize BOOLEAN DEFAULT 0,
            rm_max_concurrent_positions_min INTEGER,
            rm_max_concurrent_positions_max INTEGER,
            rm_max_concurrent_positions_step INTEGER,
            created_at DATETIME,
            updated_at DATETIME
        )
        """
    )
    cursor.execute(
        """
        INSERT INTO strategies (name, initial_tp_percent, rm_risk_per_trade_pct)
        VALUES (?, ?, ?)
        """,
        ("Legacy Strat", 7.5, 1.0),
    )
    conn.commit()


def test_migration_drops_all_rm_columns_and_keeps_data():
    conn = sqlite3.connect(":memory:")
    try:
        _build_legacy_strategies(conn)
        cursor = conn.cursor()

        # Sanity: rm_* columns exist before migration.
        before = _table_columns(cursor, "strategies")
        assert any(c.startswith("rm_") for c in before)

        migration = _load_migration()
        migration.upgrade(cursor, conn)

        after = _table_columns(cursor, "strategies")
        # No column starts with rm_ anymore.
        assert not any(c.startswith("rm_") for c in after), after
        # Kept columns survive.
        assert "name" in after
        assert "initial_tp_percent" in after
        assert "id" in after

        # The kept row (and its id/name/tp) survived.
        cursor.execute(
            "SELECT id, name, initial_tp_percent FROM strategies"
        )
        rows = cursor.fetchall()
        assert len(rows) == 1
        assert rows[0][0] == 1
        assert rows[0][1] == "Legacy Strat"
        assert rows[0][2] == 7.5
    finally:
        conn.close()


def test_migration_is_idempotent_noop_when_no_rm_columns():
    conn = sqlite3.connect(":memory:")
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE strategies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name VARCHAR(255) NOT NULL,
                initial_tp_percent FLOAT DEFAULT 5.0
            )
            """
        )
        cursor.execute(
            "INSERT INTO strategies (name, initial_tp_percent) VALUES (?, ?)",
            ("No RM", 5.0),
        )
        conn.commit()

        migration = _load_migration()
        result = migration.upgrade(cursor, conn)
        # No rm_* columns -> migration is a no-op (falsy return).
        assert not result

        after = _table_columns(cursor, "strategies")
        assert not any(c.startswith("rm_") for c in after)
        cursor.execute("SELECT name FROM strategies")
        assert cursor.fetchone()[0] == "No RM"
    finally:
        conn.close()
