"""
Company Fundamentals Overview Providers

Providers for high-level company fundamentals (P/E ratio, market cap, EPS, etc.)

Available Providers:
    - AlphaVantage: Company overview from Alpha Vantage API
    - FMP: Company profile from Financial Modeling Prep API
"""

import logging

logger = logging.getLogger(__name__)

__all__ = []

# AlphaVantage - requires alpha_vantage package
try:
    from .AlphaVantageCompanyOverviewProvider import AlphaVantageCompanyOverviewProvider
    __all__.append("AlphaVantageCompanyOverviewProvider")
except ImportError as e:
    logger.debug(f"AlphaVantageCompanyOverviewProvider not available: {e}")
    AlphaVantageCompanyOverviewProvider = None

# FMP - requires fmpsdk package
try:
    from .FMPCompanyOverviewProvider import FMPCompanyOverviewProvider
    __all__.append("FMPCompanyOverviewProvider")
except ImportError as e:
    logger.debug(f"FMPCompanyOverviewProvider not available: {e}")
    FMPCompanyOverviewProvider = None
