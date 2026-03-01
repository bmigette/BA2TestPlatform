"""
FMP OHLCV Provider

Historical stock price data provider using Financial Modeling Prep API.
Provides OHLCV (Open, High, Low, Close, Volume) data for stocks.

API Documentation: 
- Intraday (1-30min): https://site.financialmodelingprep.com/developer/docs#intraday-30-min
- Historical Daily: https://site.financialmodelingprep.com/developer/docs#historical-price-eod-full
"""

from typing import Annotated, Optional, Dict, Any
from datetime import datetime, timedelta
import pandas as pd
import requests
import os
import logging

from ..base import MarketDataProviderInterface

# Configure logger
logger = logging.getLogger(__name__)


class FMPOHLCVProvider(MarketDataProviderInterface):
    """
    Financial Modeling Prep OHLCV data provider.
    
    Uses FMP API to retrieve historical stock price data.
    Supports multiple timeframes (1min, 5min, 15min, 30min, 1hour, 4hour, 1day).
    
    API Endpoints:
        - Intraday: /api/v3/historical-chart/{interval}/{symbol}
        - Daily: /api/v3/historical-price-full/{symbol}
    
    Requires:
        - FMP API key in app settings (FMP_API_KEY)
    """
    
    # Timeframe mapping: user format -> FMP API interval
    TIMEFRAME_MAP = {
        "1m": "1min",
        "1min": "1min",
        "5m": "5min",
        "5min": "5min",
        "15m": "15min",
        "15min": "15min",
        "30m": "30min",
        "30min": "30min",
        "1h": "1hour",
        "1hour": "1hour",
        "4h": "4hour",
        "4hour": "4hour",
        "1d": "daily",
        "1day": "daily",
    }
    
    # Base URL for FMP API
    BASE_URL = "https://financialmodelingprep.com/api/v3"

    # Maximum days per API request for each intraday interval.
    # FMP silently truncates responses that exceed ~5 000 bars.
    INTRADAY_CHUNK_DAYS: dict = {
        "1min":  30,
        "5min":  90,
        "15min": 180,
        "30min": 365,
        "1hour": 730,
        "4hour": 1825,
    }

    # Chunk size for daily data requests.
    # FMP's historical-price-full endpoint can silently cap long ranges;
    # fetching in 4-year chunks keeps each request well within any plan limit.
    DAILY_CHUNK_DAYS: int = 365 * 4

    # Tolerance (calendar days) when checking whether returned data covers
    # the requested range.  Gaps smaller than this are treated as market
    # holidays / weekends and ignored.
    COVERAGE_TOLERANCE_DAYS: int = 5

    def __init__(self):
        """Initialize FMP OHLCV provider with caching support."""
        # Call parent __init__ to set up caching
        super().__init__()

        # Get API key from environment variables
        self.api_key = os.getenv("FMP_API_KEY")

        if not self.api_key:
            logger.warning(
                "FMP API key not configured. "
                "Please set FMP_API_KEY environment variable."
            )
        else:
            logger.debug("Initialized FMPOHLCVProvider with caching")
    
    def _get_ohlcv_data_impl(
        self,
        symbol: Annotated[str, "Stock ticker symbol"],
        start_date: Annotated[datetime, "Start date for data"],
        end_date: Annotated[datetime, "End date for data"],
        interval: Annotated[str, "Data interval (1m, 5m, 15m, 30m, 1h, 4h, 1d)"] = "1d"
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data from FMP API (internal implementation).
        
        This method is called by the parent class's get_ohlcv_data() method
        when cache is invalid or disabled.
        
        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL', 'MSFT')
            start_date: Start date for data
            end_date: End date for data
            interval: Data interval (1m, 5m, 15m, 30m, 1h, 4h, 1d)
        
        Returns:
            DataFrame with columns: Date, Open, High, Low, Close, Volume
        
        Raises:
            ValueError: If interval not supported
            requests.HTTPError: If API request fails
        """
        # Map interval to FMP format
        if interval not in self.TIMEFRAME_MAP:
            raise ValueError(
                f"Interval '{interval}' not supported by FMP. "
                f"Supported intervals: {list(self.TIMEFRAME_MAP.keys())}"
            )
        
        fmp_interval = self.TIMEFRAME_MAP[interval]
        
        logger.debug(
            f"Fetching FMP OHLCV data for {symbol} from {start_date.date()} "
            f"to {end_date.date()} with interval {interval} (FMP: {fmp_interval})"
        )
        
        try:
            # Choose endpoint based on interval
            if fmp_interval == "daily":
                # Use historical-price-full for daily data
                df = self._fetch_daily_data(symbol, start_date, end_date)
            else:
                # Use historical-chart for intraday data
                df = self._fetch_intraday_data(symbol, start_date, end_date, fmp_interval)
            
            if df.empty:
                logger.warning(f"No data returned from FMP for {symbol}")
                return pd.DataFrame(columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
            
            logger.info(f"Retrieved {len(df)} bars from FMP for {symbol}")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to get FMP OHLCV data for {symbol}: {e}", exc_info=True)
            raise
    
    def _fetch_daily_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Fetch daily OHLCV data from FMP API in DAILY_CHUNK_DAYS chunks.

        FMP's historical-price-full endpoint may silently cap very long date
        ranges.  Chunking by DAILY_CHUNK_DAYS and checking actual coverage
        ensures the full requested range is retrieved without duplicates.

        Args:
            symbol: Stock ticker symbol
            start_date: Start date for data
            end_date: End date for data

        Returns:
            DataFrame with columns: Date, Open, High, Low, Close, Volume
        """
        chunks = []
        chunk_start = start_date

        while chunk_start < end_date:
            chunk_end = min(chunk_start + timedelta(days=self.DAILY_CHUNK_DAYS), end_date)
            next_start = chunk_end + timedelta(days=1)  # default advancement

            url = f"{self.BASE_URL}/historical-price-full/{symbol}"
            params = {
                "apikey": self.api_key,
                "from": chunk_start.strftime("%Y-%m-%d"),
                "to": chunk_end.strftime("%Y-%m-%d"),
            }

            logger.debug(
                f"FMP daily chunk {symbol}: "
                f"{chunk_start.date()} to {chunk_end.date()}"
            )

            try:
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()

                if "historical" not in data or not data["historical"]:
                    logger.debug(
                        f"  No data in chunk {chunk_start.date()} to {chunk_end.date()}"
                    )
                else:
                    chunk_df = pd.DataFrame(data["historical"])
                    chunk_df = chunk_df.rename(columns={
                        "date": "Date",
                        "open": "Open",
                        "high": "High",
                        "low": "Low",
                        "close": "Close",
                        "volume": "Volume",
                    })
                    chunk_df = chunk_df[["Date", "Open", "High", "Low", "Close", "Volume"]]
                    chunk_df["Date"] = pd.to_datetime(chunk_df["Date"], utc=True)
                    chunks.append(chunk_df)
                    logger.debug(f"  Got {len(chunk_df)} bars")

                    # Smart advancement: if FMP returned data significantly short
                    # of chunk_end, advance from actual data end to avoid missing
                    # bars or re-requesting already-covered dates.
                    actual_end = chunk_df["Date"].max().date()
                    chunk_end_date = chunk_end.date() if isinstance(chunk_end, datetime) else chunk_end
                    gap_days = (chunk_end_date - actual_end).days
                    if gap_days > self.COVERAGE_TOLERANCE_DAYS:
                        logger.debug(
                            f"  Coverage gap: data ends {actual_end}, "
                            f"chunk to {chunk_end_date} (gap={gap_days}d). "
                            f"Advancing from actual end."
                        )
                        next_start = chunk_df["Date"].max() + timedelta(days=1)

            except Exception as e:
                logger.warning(
                    f"FMP daily chunk {chunk_start.date()}-{chunk_end.date()} failed: {e}"
                )

            chunk_start = next_start

        if not chunks:
            return pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close", "Volume"])

        df = pd.concat(chunks, ignore_index=True)
        df = df.drop_duplicates(subset=["Date"])

        # Filter to exact requested range
        start_ts = (
            pd.Timestamp(start_date).tz_localize("UTC")
            if pd.Timestamp(start_date).tz is None
            else pd.Timestamp(start_date).tz_convert("UTC")
        )
        end_ts = (
            pd.Timestamp(end_date).tz_localize("UTC")
            if pd.Timestamp(end_date).tz is None
            else pd.Timestamp(end_date).tz_convert("UTC")
        )
        df = df[(df["Date"] >= start_ts) & (df["Date"] <= end_ts)]

        df = df.sort_values("Date").reset_index(drop=True)
        logger.info(
            f"FMP daily {symbol}: {len(df)} total bars "
            f"({len(chunks)} chunk requests)"
        )
        return df
    
    def _fetch_intraday_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        fmp_interval: str
    ) -> pd.DataFrame:
        """
        Fetch intraday OHLCV data from FMP API, chunking long date ranges to avoid
        silent API truncation.

        FMP's /historical-chart endpoint silently truncates responses beyond ~5 000
        bars. Chunking by INTRADAY_CHUNK_DAYS ensures complete data is retrieved
        across any requested date range.

        Args:
            symbol: Stock ticker symbol
            start_date: Start date for data
            end_date: End date for data
            fmp_interval: FMP interval string (1min, 5min, 15min, 30min, 1hour, 4hour)

        Returns:
            DataFrame with columns: Date, Open, High, Low, Close, Volume
        """
        chunk_days = self.INTRADAY_CHUNK_DAYS.get(fmp_interval, 90)
        chunks = []
        chunk_start = start_date

        while chunk_start < end_date:
            chunk_end = min(chunk_start + timedelta(days=chunk_days), end_date)
            next_start = chunk_end + timedelta(days=1)  # default advancement

            url = f"{self.BASE_URL}/historical-chart/{fmp_interval}/{symbol}"
            params = {
                "apikey": self.api_key,
                "from": chunk_start.strftime("%Y-%m-%d"),
                "to": chunk_end.strftime("%Y-%m-%d"),
            }

            logger.debug(
                f"FMP intraday chunk {symbol}/{fmp_interval}: "
                f"{chunk_start.date()} to {chunk_end.date()}"
            )

            try:
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()

                if data and isinstance(data, list):
                    chunk_df = pd.DataFrame(data)
                    chunk_df = chunk_df.rename(columns={
                        "date": "Date",
                        "open": "Open",
                        "high": "High",
                        "low": "Low",
                        "close": "Close",
                        "volume": "Volume",
                    })
                    chunk_df = chunk_df[["Date", "Open", "High", "Low", "Close", "Volume"]]
                    chunk_df["Date"] = pd.to_datetime(chunk_df["Date"], utc=True)
                    chunks.append(chunk_df)
                    logger.debug(f"  Got {len(chunk_df)} bars")

                    # Smart advancement: if FMP returned data significantly short
                    # of chunk_end, advance from actual data end rather than
                    # chunk_end to avoid re-requesting dates or missing bars.
                    actual_end = chunk_df["Date"].max().date()
                    chunk_end_date = chunk_end.date() if isinstance(chunk_end, datetime) else chunk_end
                    gap_days = (chunk_end_date - actual_end).days
                    if gap_days > self.COVERAGE_TOLERANCE_DAYS:
                        logger.debug(
                            f"  Coverage gap: data ends {actual_end}, "
                            f"chunk to {chunk_end_date} (gap={gap_days}d). "
                            f"Advancing from actual end."
                        )
                        next_start = chunk_df["Date"].max() + timedelta(days=1)

            except Exception as e:
                logger.warning(
                    f"FMP intraday chunk {chunk_start.date()}-{chunk_end.date()} "
                    f"failed: {e}"
                )

            chunk_start = next_start

        if not chunks:
            return pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close", "Volume"])

        df = pd.concat(chunks, ignore_index=True)
        df = df.drop_duplicates(subset=["Date"])

        # Filter to exact requested range
        start_ts = (
            pd.Timestamp(start_date).tz_localize("UTC")
            if pd.Timestamp(start_date).tz is None
            else pd.Timestamp(start_date).tz_convert("UTC")
        )
        end_ts = (
            pd.Timestamp(end_date).tz_localize("UTC")
            if pd.Timestamp(end_date).tz is None
            else pd.Timestamp(end_date).tz_convert("UTC")
        )
        df = df[(df["Date"] >= start_ts) & (df["Date"] <= end_ts)]

        df = df.sort_values("Date").reset_index(drop=True)
        logger.info(
            f"FMP intraday {symbol}/{fmp_interval}: {len(df)} total bars "
            f"({len(chunks)} chunk requests)"
        )
        return df

    def get_provider_name(self) -> str:
        """Get the provider name."""
        return "fmp"
    
    def get_supported_features(self) -> list[str]:
        """Get supported features of this provider."""
        return ["ohlcv", "intraday", "daily"]
    
    def validate_config(self) -> bool:
        """
        Validate provider configuration.
        
        Returns:
            bool: True if configuration is valid
        """
        # FMP API key validated by FMP common module
        return True

