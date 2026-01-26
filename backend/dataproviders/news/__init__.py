"""
Market News Data Providers

Providers for company-specific and global market news.

Available Providers:
    - FMP: Financial Modeling Prep News API
    - AlphaVantage: Alpha Vantage News Sentiment API
    - Finnhub: Finnhub News API
    - Alpaca: Alpaca Markets News API

Removed Providers:
    - GoogleNewsProvider: Removed - web scraping is unreliable (consent pages, rate limiting)
    - AINewsProvider: Disabled - requires ModelFactory dependency
"""

import logging

logger = logging.getLogger(__name__)

# Base classes (always available)
from .base import (
    MarketNewsInterface,
    validate_date_range,
    validate_lookback_days,
    calculate_date_range
)

# Available providers (may fail if dependencies not installed)
FMPNewsProvider = None
AlphaVantageNewsProvider = None
FinnhubNewsProvider = None
AlpacaNewsProvider = None

try:
    from .FMPNewsProvider import FMPNewsProvider
except ImportError as e:
    logger.debug(f"FMPNewsProvider not available: {e}")

try:
    from .AlphaVantageNewsProvider import AlphaVantageNewsProvider
except ImportError as e:
    logger.debug(f"AlphaVantageNewsProvider not available: {e}")

try:
    from .FinnhubNewsProvider import FinnhubNewsProvider
except ImportError as e:
    logger.debug(f"FinnhubNewsProvider not available: {e}")

try:
    from .AlpacaNewsProvider import AlpacaNewsProvider
except ImportError as e:
    logger.debug(f"AlpacaNewsProvider not available: {e}")


def get_available_providers() -> list[str]:
    """Get list of available news providers."""
    providers = []
    if FMPNewsProvider is not None:
        providers.append("fmp")
    if AlphaVantageNewsProvider is not None:
        providers.append("alphavantage")
    if FinnhubNewsProvider is not None:
        providers.append("finnhub")
    if AlpacaNewsProvider is not None:
        providers.append("alpaca")
    return providers


__all__ = [
    "MarketNewsInterface",
    "validate_date_range",
    "validate_lookback_days",
    "calculate_date_range",
    "FMPNewsProvider",
    "AlphaVantageNewsProvider",
    "FinnhubNewsProvider",
    "AlpacaNewsProvider",
    "get_available_providers",
]
