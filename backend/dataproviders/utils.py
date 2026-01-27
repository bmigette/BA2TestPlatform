"""
Utility functions for data providers.
"""

import functools
import logging
from datetime import datetime, timedelta
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def validate_date_range(
    start_date: Optional[datetime],
    end_date: Optional[datetime],
    lookback_days: Optional[int] = None,
    max_days: Optional[int] = None
) -> Tuple[datetime, datetime]:
    """
    Validate and normalize a date range.

    Args:
        start_date: Start date (optional)
        end_date: End date (optional, defaults to now)
        lookback_days: Default lookback days if start_date is None
        max_days: Maximum allowed days in range (optional)

    Returns:
        Tuple of (start_date, end_date) as datetime objects
    """
    # Default end_date to now if not provided
    if end_date is None:
        end_date = datetime.now()

    # Default start_date based on lookback_days
    if start_date is None:
        if lookback_days is not None:
            start_date = end_date - timedelta(days=lookback_days)
        else:
            # Default to 30 days
            start_date = end_date - timedelta(days=30)

    # Ensure start_date is before end_date
    if start_date > end_date:
        start_date, end_date = end_date, start_date

    # Check max_days constraint
    if max_days is not None:
        days_diff = (end_date - start_date).days
        if days_diff > max_days:
            start_date = end_date - timedelta(days=max_days)

    return start_date, end_date


def validate_lookback_days(
    lookback_days: int,
    max_lookback: int = 365,
    min_lookback: int = 1
) -> int:
    """
    Validate lookback days parameter.

    Args:
        lookback_days: Number of days to look back
        max_lookback: Maximum allowed lookback (default: 365)
        min_lookback: Minimum allowed lookback (default: 1)

    Returns:
        Validated lookback days value

    Raises:
        ValueError: If lookback_days is out of range
    """
    if lookback_days < min_lookback:
        raise ValueError(f"lookback_days must be at least {min_lookback}")
    if lookback_days > max_lookback:
        raise ValueError(f"lookback_days cannot exceed {max_lookback}")
    return lookback_days


def calculate_date_range(
    end_date: datetime,
    lookback_days: int
) -> Tuple[datetime, datetime]:
    """
    Calculate a date range based on end_date and lookback_days.

    Args:
        end_date: End date for the range
        lookback_days: Number of days to look back from end_date

    Returns:
        Tuple of (start_date, end_date)
    """
    start_date = end_date - timedelta(days=lookback_days)
    return start_date, end_date


def log_provider_call(func):
    """
    Decorator to log data provider method calls.

    Logs the function name, arguments, and execution time.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        func_name = func.__name__
        class_name = args[0].__class__.__name__ if args else "Unknown"

        # Log the call
        logger.debug(f"{class_name}.{func_name} called with kwargs: {kwargs}")

        start_time = datetime.now()

        try:
            result = func(*args, **kwargs)
            elapsed = (datetime.now() - start_time).total_seconds()
            logger.debug(f"{class_name}.{func_name} completed in {elapsed:.2f}s")
            return result

        except Exception as e:
            elapsed = (datetime.now() - start_time).total_seconds()
            logger.error(
                f"{class_name}.{func_name} failed after {elapsed:.2f}s: {e}"
            )
            raise

    return wrapper
