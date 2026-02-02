"""
Migration script to add buy_entry_conditions and sell_entry_conditions columns to strategies table.

Run with: ./venv/bin/python scripts/migrate_strategies_buy_sell.py
"""

import sqlite3
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets", "ba2ml.db")


def migrate():
    """Add buy_entry_conditions and sell_entry_conditions columns to strategies table."""

    if not os.path.exists(DB_PATH):
        print(f"Database not found at {DB_PATH}")
        print("Creating new database with init_db...")
        from app.models.database import init_db
        init_db()
        print("Database created successfully.")
        return

    print(f"Connecting to database: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Check if columns already exist
    cursor.execute("PRAGMA table_info(strategies)")
    columns = [col[1] for col in cursor.fetchall()]

    migrations_needed = []

    if 'buy_entry_conditions' not in columns:
        migrations_needed.append(('buy_entry_conditions', 'JSON'))

    if 'sell_entry_conditions' not in columns:
        migrations_needed.append(('sell_entry_conditions', 'JSON'))

    if not migrations_needed:
        print("No migrations needed - columns already exist.")
        conn.close()
        return

    print(f"Migrations needed: {[m[0] for m in migrations_needed]}")

    for col_name, col_type in migrations_needed:
        print(f"Adding column: {col_name} ({col_type})")
        cursor.execute(f"ALTER TABLE strategies ADD COLUMN {col_name} {col_type}")

    conn.commit()
    print("Migration completed successfully!")

    # Verify
    cursor.execute("PRAGMA table_info(strategies)")
    columns = [col[1] for col in cursor.fetchall()]
    print(f"Current columns: {columns}")

    conn.close()


if __name__ == "__main__":
    migrate()
