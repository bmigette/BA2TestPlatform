"""As-of OHLCV price source for the backtest — the "time machine" backing store.

This is the ONLY place that knows "the current backtest bar". The engine advances
the virtual clock via ``set_clock(as_of)`` once per simulated day; ``BacktestAccount.
_get_instrument_current_price_impl`` and the fill engine delegate every price lookup
here. Because all prices come from a pre-loaded, date-keyed bar store, a run is
hermetic and reproducible (no per-call network, no wall-clock dependence).

Backing store: each symbol's bounded daily history is pulled ONCE via the injected
ba2_providers OHLCV provider (``get_ohlcv_data`` -> pandas DataFrame with columns
``Date, Open, High, Low, Close, Volume``), normalised to::

    self._bars[symbol] = {date(YYYY-MM-DD): {"open","high","low","close","volume"}, ...}

so a bar lookup is O(1) by date. Daily bars are keyed by calendar ``date`` (the
timestamp's time component is dropped), which makes ``close_at(symbol, as_of)`` robust
to whatever time-of-day the virtual clock carries.

Verified against the installed ba2_providers OHLCV provider:
  * public method = ``get_ohlcv_data(symbol, start_date=, end_date=, interval=, ...)``
    -> pandas.DataFrame with columns ``Date, Open, High, Low, Close, Volume``.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from typing import Any, Dict, List, Optional


@lru_cache(maxsize=16)
def _is_intraday(interval: str) -> bool:
    """True for sub-daily bar intervals (1m/5m/15m/30m/1h/...). Daily and coarser
    (1d/1wk/1mo) are False — those keep calendar-date bar keys. Cached: the interval is
    constant for a run but this was called ~200k×/backtest (per price lookup)."""
    iv = (interval or "1d").lower()
    return iv.endswith("m") or iv.endswith("h") or iv.endswith("min")


@lru_cache(maxsize=4096)
def _to_datetime_cached(d: Any) -> datetime:
    return _to_datetime_impl(d)


def _to_datetime(d: Any) -> datetime:
    """Parse a datetime/date/Timestamp/ISO-string to a tz-naive UTC ``datetime``. Hot path
    (~200k calls/backtest, mostly the SAME clock value re-converted once per symbol per bar) —
    route hashable inputs through an LRU cache; fall back to the impl for unhashable ones."""
    try:
        return _to_datetime_cached(d)
    except TypeError:
        return _to_datetime_impl(d)


def _to_datetime_impl(d: Any) -> datetime:
    """Parse a datetime/date/Timestamp/ISO-string to a tz-naive UTC ``datetime``."""
    if isinstance(d, datetime):
        dt = d
    elif isinstance(d, date):
        dt = datetime(d.year, d.month, d.day)
    elif hasattr(d, "to_pydatetime"):           # pandas.Timestamp
        dt = d.to_pydatetime()
    elif isinstance(d, str):
        dt = datetime.fromisoformat(d)
    else:
        raise TypeError(f"Cannot normalise {d!r} ({type(d)}) to a datetime")
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def _norm(d: Any, interval: str = "1d") -> Any:
    """Normalise a datetime/date/Timestamp/ISO-string to the bar-store key.

    The key type depends on the execution interval (so daily backtests keep their
    historical behaviour and the same code serves intraday):
      * daily / coarser (1d, 1wk, 1mo) -> calendar ``date`` (time/tz dropped), as before.
      * intraday (1m..1h)              -> tz-naive UTC ``datetime`` (the full bar
                                          timestamp), so multiple bars per day are distinct.
    All keys within one run share a type because the source carries one interval.
    Raises loudly on an unparseable value rather than silently returning a wrong key.
    """
    if _is_intraday(interval):
        return _to_datetime(d)
    # Daily path — unchanged from the original date-keyed behaviour.
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    if hasattr(d, "date") and callable(getattr(d, "date")):
        return d.date()
    if isinstance(d, str):
        return datetime.fromisoformat(d).date()
    raise TypeError(f"Cannot normalise {d!r} ({type(d)}) to a date key")


def _bar_from_row(row: Dict[str, Any]) -> Dict[str, float]:
    """Map one OHLCV DataFrame row (dict) to a lowercase {open,high,low,close,volume}.

    Accepts either the provider's canonical capitalised columns (Open/High/Low/Close/
    Volume) or already-lowercased keys (so a hand-built fixture works too). Fails
    loudly if a required field is missing.
    """
    def pick(*names: str) -> Any:
        for n in names:
            if n in row and row[n] is not None:
                return row[n]
        raise KeyError(f"OHLCV row missing any of {names}: keys={list(row)}")

    return {
        "open": float(pick("Open", "open")),
        "high": float(pick("High", "high")),
        "low": float(pick("Low", "low")),
        "close": float(pick("Close", "close")),
        "volume": float(pick("Volume", "volume")) if ("Volume" in row or "volume" in row) else 0.0,
    }


class AsOfPriceSource:
    """A virtual-clock, date-indexed OHLCV store driving every backtest price lookup."""

    def __init__(self, ohlcv_provider: Any, interval: str = "1d"):
        self._ohlcv = ohlcv_provider          # ba2_providers OHLCV provider (or None for pre-seeded fixtures)
        self._interval = interval
        self._clock: Optional[datetime] = None
        # symbol -> {bar_key -> bar dict}. The key is a calendar ``date`` for daily/
        # coarser intervals and a tz-naive UTC ``datetime`` for intraday (see ``_norm``).
        self._bars: Dict[str, Dict[Any, Dict[str, float]]] = {}

    @property
    def interval(self) -> str:
        return self._interval

    @property
    def is_intraday(self) -> bool:
        return _is_intraday(self._interval)

    # ---- virtual clock -----------------------------------------------------
    def set_clock(self, as_of: datetime) -> None:
        """Advance the virtual clock to ``as_of`` (engine calls this once per bar)."""
        self._clock = as_of

    def now(self) -> datetime:
        if self._clock is None:
            raise RuntimeError(
                "AsOfPriceSource clock not set; the engine must call set_clock() per bar"
            )
        return self._clock

    def current(self) -> Optional[datetime]:
        """The current as_of clock, or None if not set yet (None-safe; never raises).

        Used by ``AsOfClampedOHLCVProvider`` to cap indicator/ATR fetches at the bar.
        """
        return self._clock

    # ---- loading -----------------------------------------------------------
    def preload(
        self,
        symbols: List[str],
        start: datetime,
        end: datetime,
        warmup_days: int,
    ) -> None:
        """Pull each symbol's bounded history once and index it by date.

        Native-cache contract: ONE fetch per symbol (a [start - warmup, end] window),
        then served sliced from memory. ``warmup_days`` extends the start backwards so
        indicators (e.g. ATR-14) have enough lookback before the first trading day.
        """
        if self._ohlcv is None:
            raise RuntimeError(
                "AsOfPriceSource.preload called with no OHLCV provider; either inject a "
                "provider or pre-seed bars via load_bars() (fixtures/tests)."
            )
        fetch_start = start - timedelta(days=warmup_days)
        for sym in symbols:
            df = self._ohlcv.get_ohlcv_data(
                sym,
                start_date=fetch_start,
                end_date=end,
                interval=self._interval,
            )
            self.load_bars_df(sym, df)  # vectorized build (avoids per-row dict + _norm loop)

    def load_bars(self, symbol: str, rows: List[Dict[str, Any]]) -> None:
        """Index a list of OHLCV row dicts for ``symbol`` by calendar date.

        Used by ``preload`` and directly by fixtures/tests (hand-built bar series).
        Each row must carry a date (``Date``/``date``) and OHLC(V) fields.
        """
        indexed: Dict[Any, Dict[str, float]] = {}
        for row in rows:
            d = _norm(row.get("Date", row.get("date")), self._interval)
            indexed[d] = _bar_from_row(row)
        self._bars[symbol] = indexed

    def load_bars_df(self, symbol: str, df: Any) -> None:
        """VECTORIZED index build straight from a pandas OHLCV DataFrame (the hot preload path).

        Avoids the per-row ``to_dict("records")`` + ``_norm`` + ``_bar_from_row`` Python loop over
        ~every bar — the dominant cold-load cost for large intraday runs (1.8M rows for 30 syms ×
        3yr × 5min). Date keys + OHLCV columns are converted in bulk via pandas/numpy; only the
        final per-key bar dict is built in a comprehension (the dict-of-dicts store is unchanged,
        so all lookups/semantics are identical to ``load_bars``)."""
        if df is None or len(df) == 0:
            self._bars[symbol] = {}
            return
        import numpy as np
        import pandas as pd
        dcol = "Date" if "Date" in df.columns else "date"
        dates = pd.to_datetime(df[dcol])
        if _is_intraday(self._interval):
            # tz-naive UTC datetime keys (identical to _norm's intraday path).
            if getattr(dates.dt, "tz", None) is not None:
                dates = dates.dt.tz_convert("UTC").dt.tz_localize(None)
            keys = list(dates.dt.to_pydatetime())
        else:
            keys = list(dates.dt.date)
        o = df["Open"].to_numpy(dtype=float)
        h = df["High"].to_numpy(dtype=float)
        low = df["Low"].to_numpy(dtype=float)
        c = df["Close"].to_numpy(dtype=float)
        v = (df["Volume"].to_numpy(dtype=float) if "Volume" in df.columns
             else np.zeros(len(df), dtype=float))
        self._bars[symbol] = {
            keys[i]: {"open": o[i], "high": h[i], "low": low[i], "close": c[i], "volume": v[i]}
            for i in range(len(keys))
        }

    # ---- queries -----------------------------------------------------------
    def has_symbol(self, symbol: str) -> bool:
        return symbol in self._bars and len(self._bars[symbol]) > 0

    def bar_at(self, symbol: str, as_of: Optional[datetime] = None) -> Optional[Dict[str, float]]:
        """The bar for ``symbol`` on the as-of bar (or current clock bar), or None."""
        d = _norm(as_of if as_of is not None else self.now(), self._interval)
        return self._bars.get(symbol, {}).get(d)

    def close_at(self, symbol: str, as_of: Optional[datetime] = None) -> Optional[float]:
        """Close price for ``symbol`` on the as-of day (or current clock day), or None."""
        bar = self.bar_at(symbol, as_of)
        return float(bar["close"]) if bar is not None else None

    def next_bar(self, symbol: str, after: datetime) -> Optional[Dict[str, float]]:
        """The NEXT trading bar strictly after ``after`` (for next-bar fills)."""
        cutoff = _norm(after, self._interval)
        cand = [d for d in self._bars.get(symbol, {}) if d > cutoff]
        if not cand:
            return None
        return self._bars[symbol][min(cand)]

    def next_bar_date(self, symbol: str, after: datetime) -> Optional[Any]:
        """The key of the next trading bar strictly after ``after`` (date or datetime), or None."""
        cutoff = _norm(after, self._interval)
        cand = [d for d in self._bars.get(symbol, {}) if d > cutoff]
        return min(cand) if cand else None

    def all_dates(self) -> List[Any]:
        """Sorted union of all bar keys across every loaded symbol (the trading clock).

        Keys are ``date`` for daily/coarser intervals and ``datetime`` for intraday.
        """
        seen: set = set()
        for bars in self._bars.values():
            seen.update(bars.keys())
        return sorted(seen)


def _to_utc(d: Any) -> datetime:
    """Normalise a datetime/date/Timestamp/ISO-string to a tz-aware UTC datetime."""
    if isinstance(d, datetime):
        return d if d.tzinfo is not None else d.replace(tzinfo=timezone.utc)
    if isinstance(d, date):
        return datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
    if hasattr(d, "to_pydatetime"):  # pandas.Timestamp
        dd = d.to_pydatetime()
        return dd if dd.tzinfo is not None else dd.replace(tzinfo=timezone.utc)
    if isinstance(d, str):
        dd = datetime.fromisoformat(d)
        return dd if dd.tzinfo is not None else dd.replace(tzinfo=timezone.utc)
    raise TypeError(f"Cannot normalise {d!r} ({type(d)}) to a datetime")


class AsOfClampedOHLCVProvider:
    """Backtest-only OHLCV wrapper that caps every ``get_ohlcv_data(end_date=...)`` at the
    price source's current as_of clock.

    The indicator calc (``PandasIndicatorCalc``) and ATR sizing (``position_sizing.
    get_latest_atr``) fetch with ``end_date=datetime.now()`` (wall clock). In a backtest that
    would pull bars from AFTER the simulated bar, leaking future data into the (causal)
    indicator/ATR used for rule conditions and position sizing. Wrapping the OHLCV provider
    here clamps any future end_date down to the current bar, so the indicator path is
    as_of-correct regardless of what end_date the caller requests. The LIVE path uses the
    unwrapped provider (its clock is wall-time, which is correct), so this is backtest-scoped.

    Everything other than ``get_ohlcv_data`` is delegated to the inner provider.
    """

    def __init__(self, inner: Any, price_source: AsOfPriceSource):
        self._inner = inner
        self._ps = price_source

    def get_ohlcv_data(self, symbol, start_date=None, end_date=None, interval="1d", **kwargs):
        asof = self._ps.current()
        if asof is not None and (end_date is None or _to_utc(end_date) > _to_utc(asof)):
            end_date = asof  # cap the fetch at the current backtest bar
        return self._inner.get_ohlcv_data(
            symbol, start_date=start_date, end_date=end_date, interval=interval, **kwargs
        )

    def __getattr__(self, name):  # delegate every other attribute/method to the inner provider
        return getattr(self._inner, name)


def _df_to_rows(df: Any) -> List[Dict[str, Any]]:
    """Convert a pandas OHLCV DataFrame (Date/Open/High/Low/Close/Volume) to row dicts.

    Kept here (not in load_bars) so load_bars stays pandas-free for fixtures/tests.
    """
    if df is None or len(df) == 0:
        return []
    # to_dict("records") yields one dict per row with the column names as keys.
    return df.to_dict("records")


# ---------------------------------------------------------------------------
# In-memory OHLCV memo (shared across every trial a worker process runs)
# ---------------------------------------------------------------------------
# Process-global cache of each symbol's FULL bounded OHLCV series. Key:
# (symbol, interval, bounds_start_iso, bounds_end_iso). Value: (DataFrame, dates_ndarray).
# A backtest asks for the same symbol's bars on EVERY bar (price_at_date) plus indicator/ATR
# lookbacks — all sub-ranges of one fixed window. Re-reading the disk cache + re-parsing dates
# per call dominated runtime (~370s of an 836s 6-month run). Caching the full series here, at
# MODULE level, means it is paid ~once per worker and reused across the whole GA population
# (the pool workers stay alive across trials), not once per call.
_FULL_SERIES_MEMO: Dict[tuple, Any] = {}


def clear_ohlcv_memo() -> None:
    """Drop the process-global OHLCV memo (tests / between distinct universes)."""
    _FULL_SERIES_MEMO.clear()


class MemoizedOHLCVProvider:
    """Wrap an OHLCV provider so each symbol's full [bounds] series is fetched ONCE per worker
    process and every ``get_ohlcv_data`` call is served by an in-memory date-range slice.

    ``bounds`` is the widest window the run needs (start - warmup .. end). The first request for
    a symbol fetches+parses that window once (module memo, shared across trials); every later
    request — same symbol, any sub-range, any later trial in the population — is an O(log n)
    ``searchsorted`` slice with no disk read and no date re-parsing. Non-OHLCV attributes are
    delegated to the inner provider unchanged.
    """

    def __init__(self, inner: Any, bounds_start: Any, bounds_end: Any, interval: str = "1d"):
        self._inner = inner
        self._bs = bounds_start
        self._be = bounds_end
        self._interval = interval

    def _full(self, symbol: str, interval: str):
        import numpy as np
        import pandas as pd

        key = (symbol, interval, _to_utc(self._bs).isoformat(), _to_utc(self._be).isoformat())
        hit = _FULL_SERIES_MEMO.get(key)
        if hit is None:
            df = self._inner.get_ohlcv_data(
                symbol, start_date=self._bs, end_date=self._be, interval=interval
            )
            if df is None or len(df) == 0:
                import pandas as _pd
                df = _pd.DataFrame() if df is None else df
                dates = np.array([], dtype="datetime64[ns]")
            else:
                df = df.reset_index(drop=True)
                dates = (
                    pd.to_datetime(df["Date"], utc=True).dt.tz_localize(None).values
                ).astype("datetime64[ns]")
                order = np.argsort(dates, kind="stable")
                df = df.iloc[order].reset_index(drop=True)
                dates = dates[order]
            hit = (df, dates)
            _FULL_SERIES_MEMO[key] = hit
        return hit

    def get_ohlcv_data(self, symbol, start_date=None, end_date=None, interval="1d", **kwargs):
        import numpy as np

        df, dates = self._full(symbol, interval)
        if len(df) == 0:
            return df
        lo = 0
        hi = len(df)
        if start_date is not None:
            s = np.datetime64(_to_utc(start_date).replace(tzinfo=None))
            lo = int(np.searchsorted(dates, s, side="left"))
        if end_date is not None:
            e = np.datetime64(_to_utc(end_date).replace(tzinfo=None))
            hi = int(np.searchsorted(dates, e, side="right"))
        return df.iloc[lo:hi].reset_index(drop=True)

    def __getattr__(self, name):
        return getattr(self._inner, name)
