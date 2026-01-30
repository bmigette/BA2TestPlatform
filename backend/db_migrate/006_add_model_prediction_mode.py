"""
Migration 006: Add prediction_mode column to trained_models table

Stores the prediction mode used during training:
- "shift": Target shifted by prediction_horizon, single output (c_out=2)
- "multistep": Multi-label output for T+1 to T+prediction_horizon (c_out=N)
"""


def get_table_columns(cursor, table_name):
    """Get list of column names for a table."""
    cursor.execute(f"PRAGMA table_info({table_name})")
    return [col[1] for col in cursor.fetchall()]


def upgrade(cursor, conn):
    """Add prediction_mode column to trained_models table."""
    columns = get_table_columns(cursor, "trained_models")
    if "prediction_mode" not in columns:
        cursor.execute("ALTER TABLE trained_models ADD COLUMN prediction_mode TEXT DEFAULT 'shift'")
        conn.commit()
        print("  - Added prediction_mode column to trained_models table")
        return True
    else:
        print("  - prediction_mode column already exists")
        return False


def downgrade(cursor, conn):
    """SQLite doesn't support DROP COLUMN easily, so this is a no-op."""
    print("  - Downgrade not supported for this migration")
    return False
