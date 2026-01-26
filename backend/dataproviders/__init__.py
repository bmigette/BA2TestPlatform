"""
Data providers package for Deep Learning Financial Forecasting Platform.

This package provides access to various market data sources including:
- OHLCV data (Open, High, Low, Close, Volume)
- Fundamental data (financials, earnings, etc.)
- Technical indicators
- Macro economic data
- News and sentiment

Refactored from BA2TradePlatform to be a standalone, reusable package.
"""

from .base import MarketDataProviderInterface, MarketDataPoint
from .utils import log_provider_call

# Import OHLCV providers
try:
    from .ohlcv.YFinanceDataProvider import YFinanceDataProvider
except ImportError:
    YFinanceDataProvider = None

try:
    from .ohlcv.AlphaVantageOHLCVProvider import AlphaVantageOHLCVProvider
except ImportError:
    AlphaVantageOHLCVProvider = None

__all__ = [
    "MarketDataProviderInterface",
    "MarketDataPoint",
    "log_provider_call",
    "YFinanceDataProvider",
    "AlphaVantageOHLCVProvider",
]
