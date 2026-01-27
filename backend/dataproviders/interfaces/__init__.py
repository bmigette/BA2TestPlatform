"""
Data Provider Interfaces Module

This module contains all interface definitions for data providers:
- Market data providers
- Fundamentals providers
- News providers
- Macro economics providers

Import interfaces from this module:
    from dataproviders.interfaces import (
        DataProviderInterface,
        MarketDataProviderInterface,
        CompanyFundamentalsDetailsInterface,
        # ... etc
    )
"""

# Data provider interfaces
from .DataProviderInterface import DataProviderInterface
from .MarketIndicatorsInterface import MarketIndicatorsInterface
from .CompanyFundamentalsOverviewInterface import CompanyFundamentalsOverviewInterface
from .CompanyFundamentalsDetailsInterface import CompanyFundamentalsDetailsInterface
from .MarketNewsInterface import MarketNewsInterface
from .MacroEconomicsInterface import MacroEconomicsInterface
from .CompanyInsiderInterface import CompanyInsiderInterface
from .MarketDataProviderInterface import MarketDataProviderInterface
from .SocialMediaDataProviderInterface import SocialMediaDataProviderInterface

# Types
from .types import TimeInterval, MarketDataPoint, InstrumentType, TimeHorizon


__all__ = [
    # Data provider interfaces
    "DataProviderInterface",
    "MarketIndicatorsInterface",
    "CompanyFundamentalsOverviewInterface",
    "CompanyFundamentalsDetailsInterface",
    "MarketNewsInterface",
    "MacroEconomicsInterface",
    "CompanyInsiderInterface",
    "MarketDataProviderInterface",
    "SocialMediaDataProviderInterface",
    # Types
    "TimeInterval",
    "MarketDataPoint",
    "InstrumentType",
    "TimeHorizon",
]
