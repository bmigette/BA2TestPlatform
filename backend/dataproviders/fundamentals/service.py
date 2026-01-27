"""
Unified Fundamentals Service

Provides a single interface for fetching financial statements from multiple
providers with fallback support and data normalization.
"""

import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Literal

from .models import (
    FinancialStatementResponse,
    BalanceSheetPeriod,
    IncomeStatementPeriod,
    CashFlowPeriod,
    EarningsPeriod,
    YFINANCE_BALANCE_SHEET_MAPPING,
    YFINANCE_INCOME_STATEMENT_MAPPING,
    YFINANCE_CASH_FLOW_MAPPING,
    FMP_BALANCE_SHEET_MAPPING,
    FMP_INCOME_STATEMENT_MAPPING,
    FMP_CASH_FLOW_MAPPING,
    apply_mapping,
)

logger = logging.getLogger(__name__)


class FundamentalsService:
    """
    Unified service for fetching financial statements from multiple providers.

    Supports provider priority ordering - tries providers in order and uses
    the first successful result. Can also merge data from multiple providers.
    """

    def __init__(self, providers: List[str] = None):
        """
        Initialize the fundamentals service.

        Args:
            providers: List of provider names in priority order.
                      Default: ['yfinance', 'fmp', 'alphavantage']
        """
        self.provider_priority = providers or ['yfinance', 'fmp', 'alphavantage']
        self._providers = {}
        self._initialize_providers()

    def _initialize_providers(self):
        """Initialize available providers."""
        # Lazy import to avoid circular dependencies
        for provider_name in self.provider_priority:
            try:
                if provider_name == 'yfinance':
                    from .details import YFinanceCompanyDetailsProvider
                    self._providers['yfinance'] = YFinanceCompanyDetailsProvider()
                elif provider_name == 'fmp':
                    from .details import FMPCompanyDetailsProvider
                    self._providers['fmp'] = FMPCompanyDetailsProvider()
                elif provider_name == 'alphavantage':
                    from .details import AlphaVantageCompanyDetailsProvider
                    self._providers['alphavantage'] = AlphaVantageCompanyDetailsProvider()
                logger.debug(f"Initialized provider: {provider_name}")
            except Exception as e:
                logger.warning(f"Failed to initialize provider {provider_name}: {e}")

    def get_balance_sheet(
        self,
        symbol: str,
        frequency: Literal["quarterly", "annual"] = "quarterly",
        end_date: datetime = None,
        start_date: Optional[datetime] = None,
        lookback_periods: Optional[int] = None,
    ) -> FinancialStatementResponse:
        """
        Get balance sheet data, trying providers in priority order.

        Args:
            symbol: Stock ticker symbol
            frequency: 'quarterly' or 'annual'
            end_date: End date for data range
            start_date: Start date (mutually exclusive with lookback_periods)
            lookback_periods: Number of periods to fetch

        Returns:
            Standardized FinancialStatementResponse
        """
        end_date = end_date or datetime.now()

        for provider_name in self.provider_priority:
            if provider_name not in self._providers:
                continue

            try:
                provider = self._providers[provider_name]
                result = provider.get_balance_sheet(
                    symbol=symbol,
                    frequency=frequency,
                    end_date=end_date,
                    start_date=start_date,
                    lookback_periods=lookback_periods,
                    format_type="dict"
                )

                if isinstance(result, dict) and not result.get("error"):
                    normalized = self._normalize_balance_sheet(result, provider_name)
                    logger.info(f"Got balance sheet for {symbol} from {provider_name}: {len(normalized.periods)} periods")
                    return normalized
                else:
                    logger.warning(f"Provider {provider_name} returned error for {symbol}: {result}")

            except Exception as e:
                logger.warning(f"Provider {provider_name} failed for {symbol}: {e}")
                continue

        # All providers failed
        return FinancialStatementResponse(
            symbol=symbol,
            provider="none",
            statement_type="balance_sheet",
            frequency=frequency,
            end_date=end_date.isoformat() if end_date else None,
            periods=[],
            period_count=0
        )

    def get_income_statement(
        self,
        symbol: str,
        frequency: Literal["quarterly", "annual"] = "quarterly",
        end_date: datetime = None,
        start_date: Optional[datetime] = None,
        lookback_periods: Optional[int] = None,
    ) -> FinancialStatementResponse:
        """Get income statement data, trying providers in priority order."""
        end_date = end_date or datetime.now()

        for provider_name in self.provider_priority:
            if provider_name not in self._providers:
                continue

            try:
                provider = self._providers[provider_name]
                result = provider.get_income_statement(
                    symbol=symbol,
                    frequency=frequency,
                    end_date=end_date,
                    start_date=start_date,
                    lookback_periods=lookback_periods,
                    format_type="dict"
                )

                if isinstance(result, dict) and not result.get("error"):
                    normalized = self._normalize_income_statement(result, provider_name)
                    logger.info(f"Got income statement for {symbol} from {provider_name}: {len(normalized.periods)} periods")
                    return normalized
                else:
                    logger.warning(f"Provider {provider_name} returned error for {symbol}: {result}")

            except Exception as e:
                logger.warning(f"Provider {provider_name} failed for {symbol}: {e}")
                continue

        return FinancialStatementResponse(
            symbol=symbol,
            provider="none",
            statement_type="income_statement",
            frequency=frequency,
            end_date=end_date.isoformat() if end_date else None,
            periods=[],
            period_count=0
        )

    def get_cash_flow(
        self,
        symbol: str,
        frequency: Literal["quarterly", "annual"] = "quarterly",
        end_date: datetime = None,
        start_date: Optional[datetime] = None,
        lookback_periods: Optional[int] = None,
    ) -> FinancialStatementResponse:
        """Get cash flow statement data, trying providers in priority order."""
        end_date = end_date or datetime.now()

        for provider_name in self.provider_priority:
            if provider_name not in self._providers:
                continue

            try:
                provider = self._providers[provider_name]
                result = provider.get_cashflow_statement(
                    symbol=symbol,
                    frequency=frequency,
                    end_date=end_date,
                    start_date=start_date,
                    lookback_periods=lookback_periods,
                    format_type="dict"
                )

                if isinstance(result, dict) and not result.get("error"):
                    normalized = self._normalize_cash_flow(result, provider_name)
                    logger.info(f"Got cash flow for {symbol} from {provider_name}: {len(normalized.periods)} periods")
                    return normalized
                else:
                    logger.warning(f"Provider {provider_name} returned error for {symbol}: {result}")

            except Exception as e:
                logger.warning(f"Provider {provider_name} failed for {symbol}: {e}")
                continue

        return FinancialStatementResponse(
            symbol=symbol,
            provider="none",
            statement_type="cash_flow",
            frequency=frequency,
            end_date=end_date.isoformat() if end_date else None,
            periods=[],
            period_count=0
        )

    def get_earnings(
        self,
        symbol: str,
        frequency: Literal["quarterly", "annual"] = "quarterly",
        end_date: datetime = None,
        lookback_periods: int = 8,
    ) -> FinancialStatementResponse:
        """Get earnings data, trying providers in priority order."""
        end_date = end_date or datetime.now()

        for provider_name in self.provider_priority:
            if provider_name not in self._providers:
                continue

            try:
                provider = self._providers[provider_name]

                # YFinance earnings is deprecated, try to get from income statement
                if provider_name == 'yfinance':
                    result = self._get_yfinance_earnings_from_income(
                        provider, symbol, frequency, end_date, lookback_periods
                    )
                else:
                    result = provider.get_past_earnings(
                        symbol=symbol,
                        frequency=frequency,
                        end_date=end_date,
                        lookback_periods=lookback_periods,
                        format_type="dict"
                    )

                if isinstance(result, dict) and not result.get("error"):
                    normalized = self._normalize_earnings(result, provider_name)
                    if normalized.periods:
                        logger.info(f"Got earnings for {symbol} from {provider_name}: {len(normalized.periods)} periods")
                        return normalized
                else:
                    logger.warning(f"Provider {provider_name} returned error for {symbol}: {result}")

            except Exception as e:
                logger.warning(f"Provider {provider_name} failed for {symbol}: {e}")
                continue

        return FinancialStatementResponse(
            symbol=symbol,
            provider="none",
            statement_type="earnings",
            frequency=frequency,
            end_date=end_date.isoformat() if end_date else None,
            periods=[],
            period_count=0
        )

    def _get_yfinance_earnings_from_income(
        self,
        provider,
        symbol: str,
        frequency: str,
        end_date: datetime,
        lookback_periods: int
    ) -> Dict[str, Any]:
        """
        Extract earnings data from YFinance income statement.

        YFinance deprecated the get_earnings method, so we extract EPS
        from the income statement instead.
        """
        try:
            result = provider.get_income_statement(
                symbol=symbol,
                frequency=frequency,
                end_date=end_date,
                lookback_periods=lookback_periods,
                format_type="dict"
            )

            if isinstance(result, dict) and "periods" in result:
                earnings = []
                for period in result.get("periods", []):
                    items = period.get("items", {})

                    # Extract EPS from income statement
                    basic_eps = items.get("Basic EPS", items.get("Diluted EPS", 0))

                    earnings.append({
                        "fiscal_date_ending": period.get("date", ""),
                        "reported_eps": float(basic_eps) if basic_eps else 0,
                        "estimated_eps": None,  # Not available from income statement
                        "surprise": None,
                        "surprise_percent": None
                    })

                return {
                    "symbol": symbol,
                    "frequency": frequency,
                    "earnings": earnings
                }
        except Exception as e:
            logger.warning(f"Failed to extract earnings from income statement: {e}")

        return {"error": "Failed to get earnings from income statement"}

    def _normalize_balance_sheet(
        self,
        result: Dict[str, Any],
        provider_name: str
    ) -> FinancialStatementResponse:
        """Normalize balance sheet data to standard format."""
        periods = []

        # Get the raw periods from the result
        raw_periods = result.get("periods", result.get("statements", []))

        for raw_period in raw_periods:
            if provider_name == "yfinance":
                # YFinance uses 'date' and 'items' structure
                fiscal_date = raw_period.get("date", "")
                items = raw_period.get("items", {})
                normalized = apply_mapping(items, YFINANCE_BALANCE_SHEET_MAPPING)
                normalized["fiscal_date"] = fiscal_date
            elif provider_name == "fmp":
                # FMP uses flat structure with fiscal_date_ending
                normalized = apply_mapping(raw_period, FMP_BALANCE_SHEET_MAPPING)
            else:
                # AlphaVantage and others - use as-is for now
                normalized = raw_period
                if "fiscalDateEnding" in normalized:
                    normalized["fiscal_date"] = normalized.pop("fiscalDateEnding")

            periods.append(normalized)

        return FinancialStatementResponse(
            symbol=result.get("symbol", ""),
            provider=provider_name,
            statement_type="balance_sheet",
            frequency=result.get("frequency", "quarterly"),
            start_date=result.get("start_date"),
            end_date=result.get("end_date"),
            periods=periods,
            period_count=len(periods)
        )

    def _normalize_income_statement(
        self,
        result: Dict[str, Any],
        provider_name: str
    ) -> FinancialStatementResponse:
        """Normalize income statement data to standard format."""
        periods = []

        raw_periods = result.get("periods", result.get("statements", []))

        for raw_period in raw_periods:
            if provider_name == "yfinance":
                fiscal_date = raw_period.get("date", "")
                items = raw_period.get("items", {})
                normalized = apply_mapping(items, YFINANCE_INCOME_STATEMENT_MAPPING)
                normalized["fiscal_date"] = fiscal_date
            elif provider_name == "fmp":
                normalized = apply_mapping(raw_period, FMP_INCOME_STATEMENT_MAPPING)
            else:
                normalized = raw_period
                if "fiscalDateEnding" in normalized:
                    normalized["fiscal_date"] = normalized.pop("fiscalDateEnding")

            periods.append(normalized)

        return FinancialStatementResponse(
            symbol=result.get("symbol", ""),
            provider=provider_name,
            statement_type="income_statement",
            frequency=result.get("frequency", "quarterly"),
            start_date=result.get("start_date"),
            end_date=result.get("end_date"),
            periods=periods,
            period_count=len(periods)
        )

    def _normalize_cash_flow(
        self,
        result: Dict[str, Any],
        provider_name: str
    ) -> FinancialStatementResponse:
        """Normalize cash flow data to standard format."""
        periods = []

        raw_periods = result.get("periods", result.get("statements", []))

        for raw_period in raw_periods:
            if provider_name == "yfinance":
                fiscal_date = raw_period.get("date", "")
                items = raw_period.get("items", {})
                normalized = apply_mapping(items, YFINANCE_CASH_FLOW_MAPPING)
                normalized["fiscal_date"] = fiscal_date
            elif provider_name == "fmp":
                normalized = apply_mapping(raw_period, FMP_CASH_FLOW_MAPPING)
            else:
                normalized = raw_period
                if "fiscalDateEnding" in normalized:
                    normalized["fiscal_date"] = normalized.pop("fiscalDateEnding")

            periods.append(normalized)

        return FinancialStatementResponse(
            symbol=result.get("symbol", ""),
            provider=provider_name,
            statement_type="cash_flow",
            frequency=result.get("frequency", "quarterly"),
            start_date=result.get("start_date"),
            end_date=result.get("end_date"),
            periods=periods,
            period_count=len(periods)
        )

    def _normalize_earnings(
        self,
        result: Dict[str, Any],
        provider_name: str
    ) -> FinancialStatementResponse:
        """Normalize earnings data to standard format."""
        periods = []

        raw_earnings = result.get("earnings", [])

        for raw_earning in raw_earnings:
            normalized = {
                "fiscal_date": raw_earning.get("fiscal_date_ending", raw_earning.get("fiscal_date", "")),
                "report_date": raw_earning.get("report_date"),
                "reported_eps": raw_earning.get("reported_eps"),
                "estimated_eps": raw_earning.get("estimated_eps"),
                "surprise": raw_earning.get("surprise"),
                "surprise_percent": raw_earning.get("surprise_percent"),
            }
            # Remove None values
            normalized = {k: v for k, v in normalized.items() if v is not None}
            periods.append(normalized)

        return FinancialStatementResponse(
            symbol=result.get("symbol", ""),
            provider=provider_name,
            statement_type="earnings",
            frequency=result.get("frequency", "quarterly"),
            end_date=result.get("end_date"),
            periods=periods,
            period_count=len(periods)
        )


def get_fundamentals_service(providers: List[str] = None) -> FundamentalsService:
    """Factory function to create a FundamentalsService instance."""
    return FundamentalsService(providers=providers)
