"""
Company Fundamentals Data Providers

Contains providers for company fundamentals data, organized by detail level:
- overview/: High-level metrics (P/E, market cap, EPS, etc.)
- details/: Detailed financial statements (balance sheet, income, cashflow)

Available Providers:
- AlphaVantageCompanyOverviewProvider: Company overview from Alpha Vantage
- FMPCompanyOverviewProvider: Company overview from Financial Modeling Prep
- AlphaVantageCompanyDetailsProvider: Financial statements from Alpha Vantage
- YFinanceCompanyDetailsProvider: Financial statements from Yahoo Finance
- FMPCompanyDetailsProvider: Financial statements from Financial Modeling Prep

Note: Each provider is conditionally imported; missing dependencies will not
cause import errors for the entire module.
"""

# Import from submodules - they handle missing dependencies gracefully
from .overview import (
    AlphaVantageCompanyOverviewProvider,
    FMPCompanyOverviewProvider
)
from .details import (
    AlphaVantageCompanyDetailsProvider,
    YFinanceCompanyDetailsProvider,
    FMPCompanyDetailsProvider
)

# Build __all__ dynamically based on what was successfully imported
__all__ = []
if AlphaVantageCompanyOverviewProvider is not None:
    __all__.append("AlphaVantageCompanyOverviewProvider")
if FMPCompanyOverviewProvider is not None:
    __all__.append("FMPCompanyOverviewProvider")
if AlphaVantageCompanyDetailsProvider is not None:
    __all__.append("AlphaVantageCompanyDetailsProvider")
if YFinanceCompanyDetailsProvider is not None:
    __all__.append("YFinanceCompanyDetailsProvider")
if FMPCompanyDetailsProvider is not None:
    __all__.append("FMPCompanyDetailsProvider")
