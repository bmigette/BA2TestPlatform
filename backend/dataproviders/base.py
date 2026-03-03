"""
Base classes and interfaces for data providers.

This module provides the core interfaces and base implementations
that all data providers should inherit from.
"""

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Callable, Optional, List, Dict, Any
import threading
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
        self.cache_folder = Path("datasets/cache/ohlcv")
        self.cache_folder.mkdir(parents=True, exist_ok=True)
        self.cache_max_age_hours = 24

    def _get_cache_file(self, symbol: str, interval: str) -> Path:
        """
        Return the per-provider cache file path, creating the directory if needed.

        Args:
            symbol: Ticker symbol (e.g., 'AAPL').
            interval: Data interval string (e.g., '1d', '1h').

        Returns:
            Path to the cache CSV file under cache_folder/<provider_name>/.
        """
        provider_dir = self.cache_folder / self.get_provider_name()
        provider_dir.mkdir(parents=True, exist_ok=True)
        return provider_dir / f"{symbol}_{interval}.csv"

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
        use_cache: bool = True,
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Get OHLCV data with optional caching.

        Args:
            symbol: Ticker symbol
            start_date: Start date for data
            end_date: End date for data
            interval: Data interval
            use_cache: Whether to use/write cached data
            force_refresh: If True, bypass reading from cache but still write
                           fresh data to cache (useful for cache prefetch jobs)

        Returns:
            DataFrame with OHLCV data
        """
        # Check cache first if enabled and not forcing a refresh
        if use_cache and not force_refresh:
            cache_file = self._get_cache_file(symbol, interval)
            if cache_file.exists():
                # Check if cache is fresh
                cache_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
                if cache_age < timedelta(hours=self.cache_max_age_hours):
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

                    # Check if cache covers the requested date range
                    cache_max_date = df['Date'].max()
                    cache_min_date = df['Date'].min()

                    # If requested end date is beyond cache max (with 1 day tolerance for market closed days),
                    # or requested start date is before cache min, fetch fresh data
                    if end_dt > cache_max_date + pd.Timedelta(days=3) or start_dt < cache_min_date - pd.Timedelta(days=3):
                        logger.debug(f"Cache for {symbol} doesn't cover requested range ({start_dt} to {end_dt}), cache has ({cache_min_date} to {cache_max_date}). Fetching fresh data.")
                    else:
                        logger.debug(f"Using cached data for {symbol} (age: {cache_age})")
                        # Filter by date range
                        df = df[(df['Date'] >= start_dt) & (df['Date'] <= end_dt)]
                        return df

        # Fetch fresh data
        df = self._get_ohlcv_data_impl(symbol, start_date, end_date, interval)

        # Cache the data if caching is enabled (also when force_refresh bypassed reading)
        if (use_cache or force_refresh) and not df.empty:
            cache_file = self._get_cache_file(symbol, interval)
            df.to_csv(cache_file, index=False)
            logger.debug(f"Cached data for {symbol} to {cache_file}")

        return df

    def extend_ohlcv_cache(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = '1d',
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> pd.DataFrame:
        """
        Fetch and cache OHLCV data for the requested range using a two-phase approach.

        Phase 1 – Gap filling (parallel, 5 workers):
            Scans existing cache for internal gaps > 5 calendar days that overlap
            the requested range, then fetches all gaps concurrently.

        Phase 2 – Range extension (sequential):
            Fetches any head/tail portions that fall outside the existing cached range.

        Args:
            symbol: Ticker symbol
            start_date: Desired range start (inclusive)
            end_date: Desired range end (inclusive)
            interval: Data interval ('1d', '1h', etc.)
            progress_callback: Optional callable(pct, message) for progress reporting.

        Returns:
            Full merged DataFrame (existing + newly fetched), saved to cache.
        """

        def _report(pct: float, msg: str) -> None:
            logger.info(msg)
            if progress_callback:
                progress_callback(pct, msg)

        def _to_naive_ts(dt: datetime) -> pd.Timestamp:
            ts = pd.Timestamp(dt)
            if ts.tzinfo is not None:
                ts = ts.tz_convert('UTC').tz_localize(None)
            return ts

        cache_file = self._get_cache_file(symbol, interval)
        start_ts = _to_naive_ts(start_date)
        end_ts = _to_naive_ts(end_date)

        # ------------------------------------------------------------------ #
        # Load existing cache                                                  #
        # ------------------------------------------------------------------ #
        existing = pd.DataFrame()
        if cache_file.exists():
            try:
                existing = pd.read_csv(cache_file)
                existing['Date'] = pd.to_datetime(existing['Date']).dt.tz_localize(None)
            except Exception as e:
                logger.warning(f"Could not read existing cache {cache_file}: {e}")
                existing = pd.DataFrame()

        if existing.empty:
            # No cache — fetch the full requested range in one shot
            _report(5.0, f"{symbol}/{interval}: No cache, fetching full range "
                         f"{start_date.date()} → {end_date.date()}")
            new_data = self._get_ohlcv_data_impl(symbol, start_date, end_date, interval)
            if not new_data.empty:
                new_data['Date'] = pd.to_datetime(new_data['Date']).dt.tz_localize(None)
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                new_data.to_csv(cache_file, index=False)
                logger.info(f"Saved {len(new_data)} rows to {cache_file}")
            _report(100.0, f"{symbol}/{interval}: Done — {len(new_data)} rows")
            return new_data

        existing = existing.sort_values('Date').reset_index(drop=True)

        # ------------------------------------------------------------------ #
        # Phase 1 – Identify internal gaps within the requested range          #
        # ------------------------------------------------------------------ #
        gap_threshold = pd.Timedelta('5 days')
        diffs = existing['Date'].diff()
        gaps_to_fill: List[tuple] = []
        for idx in diffs[diffs > gap_threshold].index:
            gap_start_dt = existing.loc[idx - 1, 'Date']
            gap_end_dt = existing.loc[idx, 'Date']
            # Only fill gaps that overlap with [start_ts, end_ts]
            if gap_end_dt > start_ts and gap_start_dt < end_ts:
                gaps_to_fill.append((gap_start_dt.to_pydatetime(), gap_end_dt.to_pydatetime()))

        _report(5.0, f"{symbol}/{interval}: Found {len(gaps_to_fill)} gap(s) to fill")

        # ------------------------------------------------------------------ #
        # Phase 1 – Fetch all gaps concurrently (max 5 workers)               #
        # ------------------------------------------------------------------ #
        gap_pieces: List[pd.DataFrame] = []
        if gaps_to_fill:
            completed_gaps = [0]
            total_gaps = len(gaps_to_fill)
            gap_lock = threading.Lock()

            def fetch_gap(gs: datetime, ge: datetime) -> pd.DataFrame:
                try:
                    data = self._get_ohlcv_data_impl(symbol, gs, ge, interval)
                    if not data.empty:
                        data['Date'] = pd.to_datetime(data['Date']).dt.tz_localize(None)
                        logger.info(f"  Gap {gs.date()}→{ge.date()}: {len(data)} bars")
                        return data
                except Exception as exc:
                    logger.warning(f"  Failed to fill gap {gs.date()}→{ge.date()}: {exc}")
                return pd.DataFrame()

            with ThreadPoolExecutor(max_workers=5) as executor:
                future_to_gap = {
                    executor.submit(fetch_gap, gs, ge): (gs, ge)
                    for gs, ge in gaps_to_fill
                }
                for future in as_completed(future_to_gap):
                    data = future.result()
                    if not data.empty:
                        gap_pieces.append(data)
                    with gap_lock:
                        completed_gaps[0] += 1
                        pct = 5.0 + (completed_gaps[0] / total_gaps) * 75.0
                        _report(pct, f"{symbol}/{interval}: Filled {completed_gaps[0]}/{total_gaps} gaps")

        # Merge existing + gap fills
        all_pieces = [existing] + gap_pieces
        merged = (
            pd.concat(all_pieces, ignore_index=True)
              .drop_duplicates(subset=['Date'])
              .sort_values('Date')
              .reset_index(drop=True)
        )
        cache_min = merged['Date'].min()
        cache_max = merged['Date'].max()

        # ------------------------------------------------------------------ #
        # Phase 2 – Extend head/tail to cover the full requested range        #
        # ------------------------------------------------------------------ #
        extension_pieces = [merged]

        if start_ts < cache_min:
            _report(82.0, f"{symbol}/{interval}: Extending left: "
                          f"{start_date.date()} → {cache_min.date()}")
            left = self._get_ohlcv_data_impl(
                symbol, start_date, cache_min.to_pydatetime(), interval
            )
            if not left.empty:
                left['Date'] = pd.to_datetime(left['Date']).dt.tz_localize(None)
                extension_pieces.append(left)
            _report(90.0, f"{symbol}/{interval}: Left extension done")

        if end_ts > cache_max:
            _report(92.0, f"{symbol}/{interval}: Extending right: "
                          f"{cache_max.date()} → {end_date.date()}")
            right = self._get_ohlcv_data_impl(
                symbol, cache_max.to_pydatetime(), end_date, interval
            )
            if not right.empty:
                right['Date'] = pd.to_datetime(right['Date']).dt.tz_localize(None)
                extension_pieces.append(right)
            _report(98.0, f"{symbol}/{interval}: Right extension done")

        final = (
            pd.concat(extension_pieces, ignore_index=True)
              .drop_duplicates(subset=['Date'])
              .sort_values('Date')
              .reset_index(drop=True)
        )

        if not final.empty:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            final.to_csv(cache_file, index=False)
            logger.info(f"Saved {len(final)} rows to {cache_file}")

        _report(100.0, f"{symbol}/{interval}: Done — {len(final)} rows total")
        return final

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
