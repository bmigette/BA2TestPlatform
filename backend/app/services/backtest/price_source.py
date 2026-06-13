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

from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional


def _norm(d: Any) -> date:
    """Normalise a datetime/date/Timestamp/ISO-string to a calendar ``date`` key.

    Daily bars are anchored to the calendar day; the time component (and timezone)
    of the virtual clock is irrelevant for a daily backtest. Raises loudly on an
    unparseable value rather than silently returning a wrong key.
    """
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    # pandas.Timestamp has a .date() method (and is not a datetime subclass in all
    # versions of the comparison above on some builds), and ISO strings are parsed.
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
        # symbol -> {date -> bar dict}
        self._bars: Dict[str, Dict[date, Dict[str, float]]] = {}

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
            self.load_bars(sym, _df_to_rows(df))

    def load_bars(self, symbol: str, rows: List[Dict[str, Any]]) -> None:
        """Index a list of OHLCV row dicts for ``symbol`` by calendar date.

        Used by ``preload`` and directly by fixtures/tests (hand-built bar series).
        Each row must carry a date (``Date``/``date``) and OHLC(V) fields.
        """
        indexed: Dict[date, Dict[str, float]] = {}
        for row in rows:
            d = _norm(row.get("Date", row.get("date")))
            indexed[d] = _bar_from_row(row)
        self._bars[symbol] = indexed

    # ---- queries -----------------------------------------------------------
    def has_symbol(self, symbol: str) -> bool:
        return symbol in self._bars and len(self._bars[symbol]) > 0

    def bar_at(self, symbol: str, as_of: Optional[datetime] = None) -> Optional[Dict[str, float]]:
        """The bar for ``symbol`` on the as-of day (or current clock day), or None."""
        d = _norm(as_of if as_of is not None else self.now())
        return self._bars.get(symbol, {}).get(d)

    def close_at(self, symbol: str, as_of: Optional[datetime] = None) -> Optional[float]:
        """Close price for ``symbol`` on the as-of day (or current clock day), or None."""
        bar = self.bar_at(symbol, as_of)
        return float(bar["close"]) if bar is not None else None

    def next_bar(self, symbol: str, after: datetime) -> Optional[Dict[str, float]]:
        """The NEXT trading bar strictly after ``after`` (for next-bar fills)."""
        cutoff = _norm(after)
        cand = [d for d in self._bars.get(symbol, {}) if d > cutoff]
        if not cand:
            return None
        return self._bars[symbol][min(cand)]

    def next_bar_date(self, symbol: str, after: datetime) -> Optional[date]:
        """The date of the next trading bar strictly after ``after`` (or None)."""
        cutoff = _norm(after)
        cand = [d for d in self._bars.get(symbol, {}) if d > cutoff]
        return min(cand) if cand else None

    def all_dates(self) -> List[date]:
        """Sorted union of all bar dates across every loaded symbol (the trading clock)."""
        seen: set[date] = set()
        for bars in self._bars.values():
            seen.update(bars.keys())
        return sorted(seen)


def _df_to_rows(df: Any) -> List[Dict[str, Any]]:
    """Convert a pandas OHLCV DataFrame (Date/Open/High/Low/Close/Volume) to row dicts.

    Kept here (not in load_bars) so load_bars stays pandas-free for fixtures/tests.
    """
    if df is None or len(df) == 0:
        return []
    # to_dict("records") yields one dict per row with the column names as keys.
    return df.to_dict("records")
