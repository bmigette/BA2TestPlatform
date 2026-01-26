"""
Utility functions for data providers.
"""

import functools
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


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
