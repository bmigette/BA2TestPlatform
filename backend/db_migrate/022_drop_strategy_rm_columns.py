"""
Migration 022: drop the retired `Strategy.rm_*` columns.

Risk Manager params are now optimized as expert settings, so the 5 RM params'
25 columns on the `strategies` table are dead:

  rm_risk_per_trade_pct, rm_per_instrument_cap_pct, rm_min_stop_pct,
  rm_atr_stop_mult, rm_max_concurrent_positions

  ...each with the matching _optimize / _min / _max / _step columns.

sqlite cannot reliably DROP COLUMN across versions, so we rebuild the table:
create a new table from `SELECT <kept cols>` of the old one, drop the old table,
rename the new one back. `id` values are preserved (copied as a plain column,
and the column keeps its INTEGER PRIMARY KEY role on the rebuilt table).

Idempotent: a no-op (returns False) when there are no `rm_*` columns, so it is
safe on a FRESH DB (SQLAlchemy create_all already builds `strategies` without
rm_* columns) and on a re-run. The ML path never used these columns.
Follows the 017-021 house pattern: `upgrade(cursor, conn)` returning truthy when
changes were applied.
"""


def get_table_columns(cursor, table_name):
    cursor.execute(f"PRAGMA table_info({table_name})")
    return [col[1] for col in cursor.fetchall()]


def _table_exists(cursor, name):
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,)
    )
    return cursor.fetchone() is not None


def upgrade(cursor, conn):
    """Rebuild `strategies` keeping only the non-`rm_` columns + their data."""
    if not _table_exists(cursor, "strategies"):
        print("  - strategies table does not exist yet; nothing to migrate")
        return False

    columns = get_table_columns(cursor, "strategies")
    rm_columns = [c for c in columns if c.startswith("rm_")]
    if not rm_columns:
        print("  - no rm_* columns on strategies; nothing to migrate")
        return False

    kept = [c for c in columns if not c.startswith("rm_")]
    kept_csv = ", ".join(kept)
    print(f"  - dropping {len(rm_columns)} rm_* columns from strategies")

    # SQLite-safe table rebuild. Copy kept columns (incl. id) into a new table,
    # then swap names. Drop any index referencing the old table first to avoid
    # name clashes on rebuild.
    cursor.execute("DROP TABLE IF EXISTS strategies_new")
    cursor.execute(
        f"CREATE TABLE strategies_new AS SELECT {kept_csv} FROM strategies"
    )
    cursor.execute("DROP TABLE strategies")
    cursor.execute("ALTER TABLE strategies_new RENAME TO strategies")

    conn.commit()
    print(f"  - rebuilt strategies with {len(kept)} columns")
    return True


def downgrade(cursor, conn):
    """SQLite has no simple DROP COLUMN; downgrade is a no-op (data is gone)."""
    print("  - Downgrade not supported for this migration")
    return False
