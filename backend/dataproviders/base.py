"""
Base classes and interfaces for data providers.

This module provides the core interfaces and base implementations
that all data providers should inherit from.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class MarketDataPoint:
    """
    Represents a single market data point with OHLCV data.
    """

    def __init__(
        self,
        timestamp: datetime,
        open_price: float,
        high: float,
        low: float,
        close: float,
        volume: float,
        adjusted_close: Optional[float] = None,
        **kwargs
    ):
        self.timestamp = timestamp
        self.open = open_price
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
        self.adjusted_close = adjusted_close or close

        # Store any additional fields
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self):
        return (
            f"MarketDataPoint(timestamp={self.timestamp}, "
            f"open={self.open:.2f}, high={self.high:.2f}, "
            f"low={self.low:.2f}, close={self.close:.2f}, "
            f"volume={self.volume:.0f})"
        )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'adjusted_close': self.adjusted_close
        }


class MarketDataProviderInterface(ABC):
    """
    Abstract base class for market data providers.

    All data providers should inherit from this class and implement
    the required methods. This provides a consistent interface for
    fetching OHLCV data across different sources.
    """

    def __init__(self):
        """Initialize the provider."""
        self.cache_folder = Path("backend/datasets/cache")
        self.cache_folder.mkdir(parents=True, exist_ok=True)
        self.cache_max_age_hours = 24

    @abstractmethod
    def _get_ohlcv_data_impl(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = '1d'
    ) -> pd.DataFrame:
        """
        Internal implementation to fetch OHLCV data.

        Must be implemented by subclasses.

        Args:
            symbol: Ticker symbol
            start_date: Start date for data
            end_date: End date for data
            interval: Data interval (e.g., '1d', '1h', '15m')

        Returns:
            DataFrame with columns: Date, Open, High, Low, Close, Volume
        """
        pass

    def get_ohlcv_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = '1d',
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Get OHLCV data with optional caching.

        Args:
            symbol: Ticker symbol
            start_date: Start date for data
            end_date: End date for data
            interval: Data interval
            use_cache: Whether to use cached data

        Returns:
            DataFrame with OHLCV data
        """
        # Check cache first if enabled
        if use_cache:
            cache_file = self.cache_folder / f"{symbol}_{interval}.csv"
            if cache_file.exists():
                # Check if cache is fresh
                cache_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
                if cache_age < timedelta(hours=self.cache_max_age_hours):
                    logger.debug(f"Using cached data for {symbol} (age: {cache_age})")
                    df = pd.read_csv(cache_file)
                    df['Date'] = pd.to_datetime(df['Date'])

                    # Handle timezone-aware comparison
                    # Convert dates to match DataFrame timezone (or make both naive)
                    if df['Date'].dt.tz is not None:
                        # DataFrame is timezone-aware, make comparison dates timezone-aware
                        start_dt = pd.Timestamp(start_date).tz_localize('UTC') if start_date.tzinfo is None else pd.Timestamp(start_date)
                        end_dt = pd.Timestamp(end_date).tz_localize('UTC') if end_date.tzinfo is None else pd.Timestamp(end_date)
                    else:
                        # DataFrame is timezone-naive
                        start_dt = pd.Timestamp(start_date).tz_localize(None) if hasattr(start_date, 'tzinfo') and start_date.tzinfo else pd.Timestamp(start_date)
                        end_dt = pd.Timestamp(end_date).tz_localize(None) if hasattr(end_date, 'tzinfo') and end_date.tzinfo else pd.Timestamp(end_date)

                    # Filter by date range
                    df = df[(df['Date'] >= start_dt) & (df['Date'] <= end_dt)]
                    return df

        # Fetch fresh data
        df = self._get_ohlcv_data_impl(symbol, start_date, end_date, interval)

        # Cache the data if caching is enabled
        if use_cache and not df.empty:
            cache_file = self.cache_folder / f"{symbol}_{interval}.csv"
            df.to_csv(cache_file, index=False)
            logger.debug(f"Cached data for {symbol} to {cache_file}")

        return df

    def get_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = '1d',
        use_cache: bool = True
    ) -> List[MarketDataPoint]:
        """
        Get market data as a list of MarketDataPoint objects.

        Args:
            symbol: Ticker symbol
            start_date: Start date for data
            end_date: End date for data
            interval: Data interval
            use_cache: Whether to use cached data

        Returns:
            List of MarketDataPoint objects
        """
        df = self.get_ohlcv_data(symbol, start_date, end_date, interval, use_cache)

        if df.empty:
            return []

        # Convert DataFrame to MarketDataPoint objects
        datapoints = []
        for _, row in df.iterrows():
            datapoint = MarketDataPoint(
                timestamp=row['Date'],
                open_price=row['Open'],
                high=row['High'],
                low=row['Low'],
                close=row['Close'],
                volume=row['Volume'],
                adjusted_close=row.get('Adjusted_Close', row['Close'])
            )
            datapoints.append(datapoint)

        return datapoints

    def get_dataframe(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = '1d',
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Get market data as a pandas DataFrame.

        Alias for get_ohlcv_data for backwards compatibility.
        """
        return self.get_ohlcv_data(symbol, start_date, end_date, interval, use_cache)

    def normalize_time_to_interval(self, dt: datetime, interval: str) -> datetime:
        """
        Normalize a datetime to the start of its interval boundary.

        Examples:
            - '1h': 2024-01-01 13:45:30 -> 2024-01-01 13:00:00
            - '15m': 2024-01-01 13:42:30 -> 2024-01-01 13:30:00
            - '1d': 2024-01-01 13:45:30 -> 2024-01-01 00:00:00
        """
        if interval == '1d':
            return dt.replace(hour=0, minute=0, second=0, microsecond=0)
        elif interval == '1h':
            return dt.replace(minute=0, second=0, microsecond=0)
        elif interval == '15m':
            minute = (dt.minute // 15) * 15
            return dt.replace(minute=minute, second=0, microsecond=0)
        elif interval == '30m':
            minute = (dt.minute // 30) * 30
            return dt.replace(minute=minute, second=0, microsecond=0)
        elif interval == '5m':
            minute = (dt.minute // 5) * 5
            return dt.replace(minute=minute, second=0, microsecond=0)
        elif interval == '1m':
            return dt.replace(second=0, microsecond=0)
        else:
            return dt

    @abstractmethod
    def get_provider_name(self) -> str:
        """Get the provider name (e.g., 'yfinance', 'alphavantage')."""
        pass

    @abstractmethod
    def get_supported_features(self) -> list[str]:
        """Get list of supported features."""
        pass

    @abstractmethod
    def validate_config(self) -> bool:
        """Validate provider configuration (e.g., API keys)."""
        pass
