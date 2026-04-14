"""
Add description field to backtests table for user/agent notes.
"""

from sqlalchemy import text


def migrate(engine):
    """Add description column to backtests table."""
    with engine.connect() as conn:
        # Check if column already exists
        result = conn.execute(text("PRAGMA table_info(backtests)"))
        columns = [row[1] for row in result]

        if 'description' not in columns:
            conn.execute(text("ALTER TABLE backtests ADD COLUMN description TEXT"))
            conn.commit()
            print("Added 'description' column to backtests table")
        else:
            print("Column 'description' already exists in backtests table")
