"""
OHLCV (Open, High, Low, Close, Volume) data providers.

Provides historical price data from various sources.
"""

import logging

logger = logging.getLogger(__name__)

try:
    from .YFinanceDataProvider import YFinanceDataProvider
except ImportError as e:
    logger.warning(f"Could not import YFinanceDataProvider: {e}")
    YFinanceDataProvider = None

try:
    from .AlphaVantageOHLCVProvider import AlphaVantageOHLCVProvider
except ImportError as e:
    logger.warning(f"Could not import AlphaVantageOHLCVProvider: {e}")
    AlphaVantageOHLCVProvider = None

try:
    from .PolygonOHLCVProvider import PolygonOHLCVProvider
except ImportError as e:
    logger.warning(f"Could not import PolygonOHLCVProvider: {e}")
    PolygonOHLCVProvider = None

try:
    from .EODHDOHLCVProvider import EODHDOHLCVProvider
except ImportError as e:
    logger.warning(f"Could not import EODHDOHLCVProvider: {e}")
    EODHDOHLCVProvider = None

# Note: Alpaca and FMP providers also available
# try:
#     from .AlpacaOHLCVProvider import AlpacaOHLCVProvider
# except ImportError:
#     AlpacaOHLCVProvider = None
#
# try:
#     from .FMPOHLCVProvider import FMPOHLCVProvider
# except ImportError:
#     FMPOHLCVProvider = None

__all__ = [
    'YFinanceDataProvider',
    'AlphaVantageOHLCVProvider',
    'PolygonOHLCVProvider',
    'EODHDOHLCVProvider',
    # 'AlpacaOHLCVProvider',
    # 'FMPOHLCVProvider'
]
