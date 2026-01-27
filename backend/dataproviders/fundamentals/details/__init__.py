"""
Company Fundamentals Details Providers

Providers for detailed financial statements (balance sheets, income statements, cash flow)

Available Providers:
    - AlphaVantage: Financial statements from Alpha Vantage API
    - YFinance: Financial statements from Yahoo Finance
    - FMP: Financial statements from Financial Modeling Prep API
"""

import logging

logger = logging.getLogger(__name__)

# Lazy imports to handle missing dependencies gracefully
# Each provider is imported in a try/except block so missing dependencies
# don't prevent other providers from being used

__all__ = []

# YFinance - most likely to be available (requires only yfinance)
try:
    from .YFinanceCompanyDetailsProvider import YFinanceCompanyDetailsProvider
    __all__.append("YFinanceCompanyDetailsProvider")
except ImportError as e:
    logger.debug(f"YFinanceCompanyDetailsProvider not available: {e}")
    YFinanceCompanyDetailsProvider = None

# AlphaVantage - requires alpha_vantage package
try:
    from .AlphaVantageCompanyDetailsProvider import AlphaVantageCompanyDetailsProvider
    __all__.append("AlphaVantageCompanyDetailsProvider")
except ImportError as e:
    logger.debug(f"AlphaVantageCompanyDetailsProvider not available: {e}")
    AlphaVantageCompanyDetailsProvider = None

# FMP - requires fmpsdk package
try:
    from .FMPCompanyDetailsProvider import FMPCompanyDetailsProvider
    __all__.append("FMPCompanyDetailsProvider")
except ImportError as e:
    logger.debug(f"FMPCompanyDetailsProvider not available: {e}")
    FMPCompanyDetailsProvider = None
