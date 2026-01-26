"""
Configuration helpers for news providers.

Provides get_app_setting function that retrieves settings from
environment variables or the database AppSetting table.
"""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def get_app_setting(key: str) -> Optional[str]:
    """
    Get application setting from environment or database.

    Checks environment variables first (with uppercase key and underscores),
    then falls back to the database AppSetting table.

    Args:
        key: Setting key (e.g., 'FMP_API_KEY', 'finnhub_api_key')

    Returns:
        Setting value or None if not found
    """
    # First try environment variable (uppercase with underscores)
    env_key = key.upper().replace('.', '_').replace('-', '_')
    env_value = os.getenv(env_key)
    if env_value:
        return env_value

    # Try to get from database AppSetting table
    try:
        from app.models.database import SessionLocal
        from app.models.app_setting import AppSetting

        with SessionLocal() as db:
            setting = db.query(AppSetting).filter(AppSetting.key == key).first()
            if setting:
                return setting.get_value()  # Handles decryption if needed
    except ImportError:
        logger.debug(f"Could not import database models for setting {key}")
    except Exception as e:
        logger.debug(f"Could not fetch setting {key} from database: {e}")

    return None
