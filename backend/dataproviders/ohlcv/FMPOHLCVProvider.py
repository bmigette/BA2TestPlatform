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
import time
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

    # Maximum calendar days returned by FMP per API call for each intraday interval.
    # Source: FMP developer docs (empirically verified).
    #
    # _fetch_intraday_data uses these values to drive backward stepping:
    # each iteration steps back (max_days - 1) days from the current chunk_end.
    # The -1 day safety margin creates a 1-day overlap between consecutive
    # windows, which drop_duplicates removes to ensure no data is missed.
    INTRADAY_CHUNK_DAYS: dict = {
        "1min":   3,
        "5min":  10,
        "15min": 45,
        "30min": 30,
        "1hour": 90,
        "4hour": 60,   # not in official docs; conservative estimate
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
    
    # Backoff delays (seconds) for consecutive retries: 15s, 30s, 60s.
    # Total attempts = len + 1 (initial + one per delay).
    RATE_LIMIT_RETRY_DELAYS: tuple = (15, 30, 60)

    def _fmp_get(self, url: str, params: dict) -> requests.Response:
        """
        Perform a GET request to the FMP API with exponential backoff retries.

        FMP returns rate-limit errors as HTTP 200 with a JSON body like
        {"Error Message": "Limit Reach."} rather than a 429 status.  This
        method detects both HTTP errors and FMP JSON error responses, retrying
        with increasing delays (15s, 30s, 60s).

        Args:
            url: Full endpoint URL
            params: Query parameters (including apikey)

        Returns:
            Response object whose JSON content is a valid (non-error) payload

        Raises:
            RuntimeError after all retries are exhausted
        """
        total_attempts = len(self.RATE_LIMIT_RETRY_DELAYS) + 1
        last_exc: Exception = RuntimeError("No attempts made")
        for attempt in range(total_attempts):
            if attempt > 0:
                delay = self.RATE_LIMIT_RETRY_DELAYS[attempt - 1]
                logger.warning(
                    f"FMP request failed, retrying in {delay}s "
                    f"(attempt {attempt + 1}/{total_attempts})..."
                )
                time.sleep(delay)
            try:
                resp = requests.get(url, params=params, timeout=30)
                resp.raise_for_status()
                # FMP returns rate-limit / API errors as HTTP 200 with a JSON dict.
                # Detect and raise so the retry loop handles them.
                try:
                    payload = resp.json()
                    if isinstance(payload, dict):
                        err = (payload.get("Error Message")
                               or payload.get("message")
                               or payload.get("error"))
                        if err:
                            raise RuntimeError(f"FMP API error: {err}")
                except (ValueError, KeyError):
                    pass  # Not JSON or no error key – let the caller handle it
                return resp
            except Exception as exc:
                last_exc = exc
        raise RuntimeError(
            f"FMP request failed after {total_attempts} attempts: {last_exc}"
        )

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
        # SINGLE SHARED FETCH PATH: delegate the actual FMP fetch to the COMMON ba2_providers
        # package provider so every code path (backtest as-of cache AND fetch-cache) uses
        # identical fetch logic — including the chunked intraday paging that works around FMP's
        # ~624-bar/call cap. This test-platform provider keeps only its caching/interface role
        # (base.py _get_cache_file / extend_ohlcv_cache); it no longer carries a duplicate fetch.
        from ba2_providers.ohlcv.FMPOHLCVProvider import FMPOHLCVProvider as _CommonFMP
        common = getattr(self, "_common_fmp", None) or _CommonFMP()
        self._common_fmp = common
        logger.debug(
            f"Fetching FMP OHLCV for {symbol} {start_date.date()}->{end_date.date()} "
            f"interval {interval} via common ba2_providers provider"
        )
        try:
            df = common._get_ohlcv_data_impl(symbol, start_date, end_date, interval=interval)
        except Exception as e:
            logger.error(f"Failed to get FMP OHLCV data for {symbol}: {e}", exc_info=True)
            raise
        if df is None or df.empty:
            logger.warning(f"No data returned from FMP (common provider) for {symbol}")
            return pd.DataFrame(columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
        logger.info(f"Retrieved {len(df)} bars from FMP (common provider) for {symbol}")
        return df
    
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
                response = self._fmp_get(url, params)
                data = response.json()

                # _fmp_get already detects FMP JSON errors (e.g. rate-limit) and raises.
                # If we still get a dict here, it's an unexpected non-error structure.
                if not isinstance(data, dict) or "historical" not in data or not data["historical"]:
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
        Fetch intraday OHLCV data from FMP API using backward stepping.

        FMP's /historical-chart endpoint ignores the 'from' parameter and always
        returns the last ~1170 bars ending at 'to', regardless of the requested
        range.  Forward chunking therefore leaves large gaps (e.g. 27-day gaps for
        1min with 30-day chunks).

        This method steps backward from end_date: each response's oldest bar
        becomes the next chunk's 'to' date (minus 1 minute), until the oldest bar
        is at or before start_date or FMP returns no data.

        Args:
            symbol: Stock ticker symbol
            start_date: Start date for data (inclusive)
            end_date: End date for data (inclusive)
            fmp_interval: FMP interval string (1min, 5min, 15min, 30min, 1hour, 4hour)

        Returns:
            DataFrame with columns: Date, Open, High, Low, Close, Volume
        """
        max_days = self.INTRADAY_CHUNK_DAYS.get(fmp_interval, 3)
        # Step backward by (max_days - 1) each iteration: the -1 day safety
        # margin creates a 1-day overlap between consecutive windows so no bar
        # is ever missed at a day boundary.  drop_duplicates removes the overlap.
        step_days = max(1, max_days - 1)

        chunks = []
        chunk_end = end_date
        prev_oldest = None
        consecutive_failures = 0
        consecutive_empty = 0
        # After this many back-to-back API failures (exceptions) we give up
        MAX_CONSECUTIVE_FAILURES = 3
        # After this many consecutive empty windows we give up stepping backward.
        # At step_days=2 per window, 20 windows = 40 calendar days maximum void.
        # This lets the backward stepper skip over FMP data voids (e.g. the
        # monthly ~10-18 day gaps in old historical data) without stopping early.
        MAX_CONSECUTIVE_EMPTY = 20

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

        url = f"{self.BASE_URL}/historical-chart/{fmp_interval}/{symbol}"

        while True:
            chunk_end_dt = chunk_end if isinstance(chunk_end, datetime) else chunk_end
            params = {
                "apikey": self.api_key,
                "from": (chunk_end_dt - timedelta(days=max_days)).strftime("%Y-%m-%d"),
                "to":   chunk_end_dt.strftime("%Y-%m-%d"),
            }

            logger.debug(
                f"FMP intraday backward {symbol}/{fmp_interval}: "
                f"to={chunk_end_dt.date()}"
            )

            try:
                response = self._fmp_get(url, params)
                data = response.json()

                if not data or not isinstance(data, list):
                    # FMP returned empty for this window.  This can mean either
                    # (a) a temporary data void in FMP's historical records, or
                    # (b) we've stepped past the symbol's listing date.
                    # We step backward anyway and only stop after
                    # MAX_CONSECUTIVE_EMPTY consecutive empty windows, which
                    # lets us skip over voids of up to step_days*MAX windows.
                    consecutive_empty += 1
                    logger.warning(
                        f"FMP no data for {symbol}/{fmp_interval} to={chunk_end_dt.date()} "
                        f"— stepping back through void "
                        f"({consecutive_empty}/{MAX_CONSECUTIVE_EMPTY})"
                    )
                    if consecutive_empty >= MAX_CONSECUTIVE_EMPTY:
                        logger.warning(
                            f"FMP {symbol}/{fmp_interval}: gave up after "
                            f"{MAX_CONSECUTIVE_EMPTY} consecutive empty windows"
                        )
                        break
                    chunk_end = chunk_end_dt - timedelta(days=step_days)
                    continue

                consecutive_empty = 0  # reset on any successful (non-empty) response
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

                oldest = chunk_df["Date"].min()
                newest = chunk_df["Date"].max()
                chunks.append(chunk_df)
                consecutive_failures = 0  # reset on success
                logger.debug(
                    f"  Got {len(chunk_df)} bars: {oldest.date()} -> {newest.date()}"
                )

                # Guard against infinite loop (same window returned twice)
                if prev_oldest is not None and oldest.date() >= prev_oldest.date():
                    logger.debug("  Oldest date did not advance, stopping")
                    break
                prev_oldest = oldest

                # Stop once we have covered start_date (within tolerance)
                if oldest <= start_ts + pd.Timedelta(days=self.COVERAGE_TOLERANCE_DAYS):
                    logger.debug("  Covered full range, stopping")
                    break

                # Step backward by (max_days - 1): fixed, predictable, no day-by-day crawl
                chunk_end = chunk_end_dt - timedelta(days=step_days)

            except Exception as e:
                consecutive_failures += 1
                logger.warning(
                    f"FMP intraday backward chunk to={chunk_end_dt.date()} failed "
                    f"({consecutive_failures}/{MAX_CONSECUTIVE_FAILURES}): {e}"
                )
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    logger.warning("Too many consecutive failures, stopping backward stepping")
                    break
                # Don't advance chunk_end — retry the same window next iteration
                # (_fmp_get already waited for RATE_LIMIT_RETRY_DELAY per attempt)

        if not chunks:
            return pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close", "Volume"])

        df = pd.concat(chunks, ignore_index=True)
        df = df.drop_duplicates(subset=["Date"])
        df = df[(df["Date"] >= start_ts) & (df["Date"] <= end_ts)]
        df = df.sort_values("Date").reset_index(drop=True)

        logger.info(
            f"FMP intraday {symbol}/{fmp_interval}: {len(df)} total bars "
            f"({len(chunks)} backward chunk requests)"
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

