"""
YFinance Data Provider

Implementation of MarketDataProvider using Yahoo Finance as the data source.
Provides historical market data with smart caching strategy.
"""

from datetime import datetime
from typing import Optional
import pandas as pd
import yfinance as yf
import logging

from ..base import MarketDataProviderInterface
from ..utils import log_provider_call

logger = logging.getLogger(__name__)


class YFinanceDataProvider(MarketDataProviderInterface):
    """
    Yahoo Finance data provider implementation.
    
    Features:
    - Fetches historical OHLCV data from Yahoo Finance
    - Symbol-based caching (one CSV per symbol+interval)
    - Automatic cache refresh when data is older than 24 hours
    - Caches 15 years of historical data by default
    
    Cache Strategy:
    - File format: {SYMBOL}_{INTERVAL}.csv (e.g., AAPL_1d.csv)
    - Location: config.CACHE_FOLDER
    - Max age: 24 hours (configurable)
    - If cache exists and is fresh: use cached data
    - If cache is stale or missing: fetch from API and update cache
    
    Usage:
        from ba2_trade_platform.modules.dataproviders import YFinanceDataProvider
        
        provider = YFinanceDataProvider()
        
        # Get data as MarketDataPoint objects
        datapoints = provider.get_data(
            symbol='AAPL',
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 12, 31),
            interval='1d'
        )
        
        # Or get as DataFrame for analysis
        df = provider.get_dataframe(
            symbol='AAPL',
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 12, 31),
            interval='1d'
        )
    """
    
    def __init__(self):
        """
        Initialize YFinance data provider.
        
        Caching is automatically configured using config.CACHE_FOLDER.
        """
        super().__init__()
        logger.debug("YFinanceDataProvider initialized")
    
    def _get_ohlcv_data_impl(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = '1d'
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data from Yahoo Finance API (internal implementation).
        
        Args:
            symbol: Ticker symbol (e.g., 'AAPL', 'MSFT')
            start_date: Start date for data
            end_date: End date for data
            interval: Data interval ('1m', '5m', '15m', '30m', '1h', '1d', '1wk', '1mo')
        
        Returns:
            DataFrame with columns: Date, Open, High, Low, Close, Volume
        
        Raises:
            Exception: If data fetching fails
        """
        # SINGLE SHARED FETCH PATH: delegate the actual yfinance download to the COMMON
        # ba2_providers package provider so the test platform and the backtest use identical
        # fetch logic. This legacy class keeps only its caching/interface role.
        from ba2_providers.ohlcv.YFinanceDataProvider import YFinanceDataProvider as _CommonYF
        common = getattr(self, "_common_yf", None) or _CommonYF()
        self._common_yf = common
        logger.debug(
            f"Fetching {symbol} {start_date.date()}->{end_date.date()} interval {interval} "
            f"via common ba2_providers YFinance provider"
        )
        df = common._get_ohlcv_data_impl(symbol, start_date, end_date, interval=interval)
        if df is None or df.empty:
            raise Exception(f"No data returned for {symbol}")
        return df
    
    @log_provider_call
    def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Get the current/latest price for a symbol.
        
        This is a convenience method that fetches the most recent closing price.
        It will use cached data if available and fresh, otherwise fetch from API.
        
        Args:
            symbol: Ticker symbol (e.g., 'AAPL', 'MSFT')
        
        Returns:
            Current closing price, or None if failed
        """
        try:
            # Get last 2 days of data to ensure we have latest
            end_date = datetime.now()
            start_date = end_date - pd.Timedelta(days=2)
            
            datapoints = self.get_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                interval='1d'
            )
            
            if not datapoints:
                logger.warning(f"No data available for {symbol}")
                return None
            
            # Get the most recent data point
            latest = datapoints[-1]
            logger.debug(f"Current price for {symbol}: {latest.close}")
            
            return latest.close
            
        except Exception as e:
            logger.error(f"Failed to get current price for {symbol}: {e}", exc_info=True)
            return None
    
    def validate_symbol(self, symbol: str) -> bool:
        """
        Check if a symbol is valid and has data available.
        
        Args:
            symbol: Ticker symbol to validate
        
        Returns:
            True if symbol is valid and has data, False otherwise
        """
        try:
            # Try to fetch 1 day of data
            end_date = datetime.now()
            start_date = end_date - pd.Timedelta(days=5)  # Look back 5 days to account for weekends
            
            data = self._get_ohlcv_data_impl(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                interval='1d'
            )
            
            is_valid = data is not None and not data.empty
            
            if is_valid:
                logger.info(f"Symbol {symbol} is valid")
            else:
                logger.warning(f"Symbol {symbol} is invalid or has no data")
            
            return is_valid
            
        except Exception as e:
            logger.error(f"Symbol validation failed for {symbol}: {e}", exc_info=True)
            return False
    
    def get_provider_name(self) -> str:
        """Get the provider name."""
        return "yfinance"
    
    def get_supported_features(self) -> list[str]:
        """Get supported features of this provider."""
        return ["ohlcv", "intraday", "daily", "weekly", "monthly"]
    
    def validate_config(self) -> bool:
        """
        Validate provider configuration.
        
        Returns:
            bool: Always True (Yahoo Finance doesn't require API keys)
        """
        return True
