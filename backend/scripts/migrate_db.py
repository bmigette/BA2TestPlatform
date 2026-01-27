#!/usr/bin/env python3
"""
Database Migration Script

Run this script to apply pending database migrations.
Uses SQLite's ALTER TABLE for simple column additions.

Usage:
    ./venv/bin/python scripts/migrate_db.py
"""

import sqlite3
import os
import sys
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Database path
DB_PATH = os.getenv("DATABASE_PATH", "dl_forecasting.db")


def get_table_columns(cursor, table_name):
    """Get list of column names for a table."""
    cursor.execute(f"PRAGMA table_info({table_name})")
    return [col[1] for col in cursor.fetchall()]


def migration_001_add_dataset_task_id(cursor, conn):
    """Add task_id column to datasets table for background task tracking."""
    columns = get_table_columns(cursor, "datasets")
    if "task_id" not in columns:
        cursor.execute("ALTER TABLE datasets ADD COLUMN task_id VARCHAR(50)")
        conn.commit()
        print("  - Added task_id column to datasets table")
        return True
    else:
        print("  - task_id column already exists")
        return False


def migration_002_add_dataset_progress_message(cursor, conn):
    """Add progress_message column to datasets table for progress tracking."""
    columns = get_table_columns(cursor, "datasets")
    if "progress_message" not in columns:
        cursor.execute("ALTER TABLE datasets ADD COLUMN progress_message TEXT")
        conn.commit()
        print("  - Added progress_message column to datasets table")
        return True
    else:
        print("  - progress_message column already exists")
        return False


# List of migrations in order
MIGRATIONS = [
    ("001_add_dataset_task_id", migration_001_add_dataset_task_id),
    ("002_add_dataset_progress_message", migration_002_add_dataset_progress_message),
]


def run_migrations():
    """Run all pending migrations."""
    print(f"Database Migration Script")
    print(f"Database: {DB_PATH}")
    print(f"Time: {datetime.now().isoformat()}")
    print("-" * 50)

    if not os.path.exists(DB_PATH):
        print(f"Error: Database file not found: {DB_PATH}")
        print("Run the application first to create the database.")
        sys.exit(1)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    applied = 0
    skipped = 0

    for name, migration_func in MIGRATIONS:
        print(f"\nMigration: {name}")
        try:
            if migration_func(cursor, conn):
                applied += 1
            else:
                skipped += 1
        except Exception as e:
            print(f"  - ERROR: {e}")
            conn.rollback()
            sys.exit(1)

    conn.close()

    print("\n" + "-" * 50)
    print(f"Migrations complete: {applied} applied, {skipped} skipped")


if __name__ == "__main__":
    run_migrations()
