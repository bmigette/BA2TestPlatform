"""Adapter: source OHLCV for the ML dataset builder THROUGH ``ba2_providers``.

The dataset builder consumes ``provider.get_data(...) -> List[MarketDataPoint]``
(``app/api/datasets.py:223-244``; ``MarketDataPoint`` defined in
``dataproviders/base.py:23``). This adapter satisfies that exact contract while
routing the fetch through ``ba2_providers``' uniform provider + as_of cache, so
the two consumers share one cache: experts read point-in-time slices (Phases
1-4) and ML training reads a materialized feature/target matrix.

Selected only when ``OHLCV_SOURCE=ba2_providers`` (default ``legacy``). The
``as_of`` snapshot is ``end_date`` (inclusive close), which gives the
reproducible / leak-free view known at the dataset's end (design section 3).

Re-plan grounding (verified against the merged Phase-1/2 code):
  * There is NO ``provider.get(...)`` method and NO dict/list-of-bars return.
    The real seam is
    ``get_provider('ohlcv', name).get_ohlcv_data(symbol, start_date=None,
    end_date=None, interval='1d', use_cache=True, max_cache_age_hours=24,
    lookback_days=30) -> pandas.DataFrame`` with columns
    ``Date, Open, High, Low, Close, Volume`` (Date coerced to datetime).
    (ba2_common ``core/interfaces/MarketDataProviderInterface.py:579``; uniform
    alias ``ba2_providers/cache/cached_get.py:23`` ``ohlcv_get`` just forwards
    ``end_date=as_of, lookback_days=lookback``.)
  * ``MarketDataPoint.__init__(timestamp, open_price, high, low, close, volume,
    adjusted_close=None, **kwargs)`` -- the OHLC-open kwarg is ``open_price=``
    (NOT ``open=``) and there is NO ``symbol=`` kwarg. Attributes after
    construction are ``.timestamp/.open/.high/.low/.close/.volume`` (the names
    the builder reads at ``datasets.py:242``).
  * Registry key is lowercase: ``get_provider('ohlcv', 'fmp')`` /
    ``'yfinance'`` / ``'alpaca'`` / ``'alphavantage'``
    (``ba2_providers/__init__.py``).

NOTE on cache topology (cutover documentation): ``get_ohlcv_data`` uses its own
legacy per-class CSV cache under ``ba2_common.config.CACHE_FOLDER`` (i.e.
``~/Documents/ba2_trade_platform/cache/<ClassName>/<SYMBOL>_<interval>.csv``),
NOT the native parquet/SQLite as_of store (whose ``read/write_timeseries`` have
no callers yet). So re-sourcing through ba2_providers yields a SHARED
legacy-style CSV cache (one cache for experts + ML), not a parquet as_of store,
unless additional wiring lands later (out of Phase-5 scope).
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import List, Optional

from dataproviders.base import MarketDataPoint  # reuse the existing contract type

logger = logging.getLogger(__name__)


class BA2ProvidersOHLCVAdapter:
    """Routes OHLCV fetches through ``ba2_providers`` while returning the
    ``List[MarketDataPoint]`` contract the dataset builder already consumes.

    The underlying ``ba2_providers`` provider is created lazily so importing this
    module (and ``app.api.datasets``) never fails just because ``ba2_providers``
    is missing or a provider constructor raises (e.g. FMP requiring an API key).
    """

    def __init__(self, provider_name: str = "fmp"):
        self._provider_name = (provider_name or "fmp").lower()
        self._provider = None  # lazy: created on first use, see _get_provider()
        # Eagerly resolve so the caller (get_ohlcv_provider) can catch failures
        # and fall back to legacy BEFORE any dataset build is attempted.
        self._get_provider()

    def _get_provider(self):
        if self._provider is None:
            # Lazy import so a legacy-only install without ba2_providers still
            # loads datasets.py / this module.
            from ba2_providers import get_provider
            self._provider = get_provider("ohlcv", self._provider_name)
        return self._provider

    def get_data(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        interval: str = "1d",
        use_cache: bool = True,
    ) -> List[MarketDataPoint]:
        """Same signature/return as ``MarketDataProviderInterface.get_data`` so
        ``_build_dataset_in_background`` is untouched.

        Mapping (Phase-1/2 contract): ``as_of -> end_date`` (inclusive close);
        the requested span ``[start_date, end_date]`` becomes ``lookback_days``.
        ``get_ohlcv_data`` itself re-applies the ``start_date``/``end_date``
        window filter, so passing both keeps parity with the legacy path.
        """
        provider = self._get_provider()

        # Translate the [start, end] span into a lookback. The legacy builder
        # already widened start_date by the warmup period before calling, so we
        # preserve that span here. lookback_days must be >= 1.
        lookback_days = 30
        if start_date is not None and end_date is not None:
            lookback_days = max((end_date - start_date).days, 1)

        df = provider.get_ohlcv_data(
            symbol,
            start_date=start_date,
            end_date=end_date,
            interval=interval,
            use_cache=use_cache,
            lookback_days=lookback_days,
        )

        if df is None or len(df) == 0:
            return []

        # Defensive: tolerate either capitalized (Date/Open/...) or lowercase
        # column names. The documented contract is capitalized.
        cols = {c.lower(): c for c in df.columns}

        def col(name: str) -> str:
            return cols.get(name.lower(), name)

        c_date = col("Date")
        c_open = col("Open")
        c_high = col("High")
        c_low = col("Low")
        c_close = col("Close")
        c_volume = col("Volume")

        points: List[MarketDataPoint] = []
        for row in df.itertuples(index=False):
            d = row._asdict()
            points.append(
                MarketDataPoint(
                    timestamp=d[c_date],
                    open_price=float(d[c_open]),
                    high=float(d[c_high]),
                    low=float(d[c_low]),
                    close=float(d[c_close]),
                    volume=float(d[c_volume]),
                )
            )
        return points
