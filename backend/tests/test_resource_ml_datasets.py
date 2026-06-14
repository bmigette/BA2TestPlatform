"""Phase 5 Task 5 tests: BA2ProvidersOHLCVAdapter + flag-aware get_ohlcv_provider.

These exercise the re-source seam WITHOUT network: the underlying ba2_providers
provider is faked to return the documented get_ohlcv_data DataFrame contract
(columns Date, Open, High, Low, Close, Volume). The adapter must turn that into
the existing List[MarketDataPoint] contract the dataset builder consumes
(.timestamp/.open/.high/.low/.close/.volume), so _build_dataset_in_background is
untouched. The byte-equality build test is added in Task 7.
"""
from datetime import datetime

import pandas as pd
import pytest


def _fake_df():
    return pd.DataFrame(
        {
            "Date": [datetime(2020, 1, 2), datetime(2020, 1, 3)],
            "Open": [1.0, 1.5],
            "High": [2.0, 2.5],
            "Low": [0.5, 1.0],
            "Close": [1.5, 2.0],
            "Volume": [100, 200],
        }
    )


class _FakeProvider:
    """Mimics a ba2_providers OHLCV provider: get_ohlcv_data -> DataFrame."""

    def __init__(self, df=None):
        self._df = df if df is not None else _fake_df()
        self.calls = []

    def get_ohlcv_data(self, symbol, start_date=None, end_date=None,
                       interval="1d", use_cache=True, max_cache_age_hours=24,
                       lookback_days=30):
        self.calls.append(
            dict(symbol=symbol, start_date=start_date, end_date=end_date,
                 interval=interval, use_cache=use_cache, lookback_days=lookback_days)
        )
        return self._df


def _make_adapter(provider):
    """Build an adapter bound to a fake provider, bypassing the real ba2_providers
    import in __init__."""
    import dataproviders.ba2providers_adapter as mod

    adapter = mod.BA2ProvidersOHLCVAdapter.__new__(mod.BA2ProvidersOHLCVAdapter)
    adapter._provider_name = "fmp"
    adapter._provider = provider
    return adapter


def test_adapter_maps_dataframe_rows_to_marketdatapoints():
    fake = _FakeProvider()
    adapter = _make_adapter(fake)
    pts = adapter.get_data(
        "AAPL", datetime(2020, 1, 1), datetime(2020, 1, 5), "1d"
    )
    assert len(pts) == 2
    # Attribute names the builder reads at datasets.py:242
    assert pts[0].timestamp == datetime(2020, 1, 2)
    assert pts[0].open == 1.0
    assert pts[0].high == 2.0
    assert pts[0].low == 0.5
    assert pts[0].close == 1.5
    assert pts[0].volume == 100.0
    assert pts[1].close == 2.0 and pts[1].volume == 200.0


def test_adapter_maps_as_of_to_end_date_and_lookback():
    fake = _FakeProvider()
    adapter = _make_adapter(fake)
    start = datetime(2020, 1, 1)
    end = datetime(2020, 1, 11)
    adapter.get_data("AAPL", start, end, "1d")
    call = fake.calls[-1]
    # as_of == end_date (inclusive close)
    assert call["end_date"] == end
    assert call["start_date"] == start
    # span -> lookback_days, min 1
    assert call["lookback_days"] == 10
    assert call["interval"] == "1d"


def test_adapter_empty_dataframe_returns_empty_list():
    fake = _FakeProvider(df=_fake_df().iloc[0:0])
    adapter = _make_adapter(fake)
    assert adapter.get_data("AAPL", datetime(2020, 1, 1), datetime(2020, 1, 5)) == []


def test_adapter_tolerates_lowercase_columns():
    df = _fake_df().rename(
        columns={"Date": "date", "Open": "open", "High": "high",
                 "Low": "low", "Close": "close", "Volume": "volume"}
    )
    adapter = _make_adapter(_FakeProvider(df=df))
    pts = adapter.get_data("AAPL", datetime(2020, 1, 1), datetime(2020, 1, 5))
    assert len(pts) == 2 and pts[0].close == 1.5


def test_get_ohlcv_provider_defaults_to_legacy(monkeypatch):
    monkeypatch.delenv("OHLCV_SOURCE", raising=False)
    from app.api.datasets import get_ohlcv_provider
    from dataproviders.ohlcv.YFinanceDataProvider import YFinanceDataProvider
    from dataproviders.ohlcv.FMPOHLCVProvider import FMPOHLCVProvider

    assert isinstance(get_ohlcv_provider("yfinance"), YFinanceDataProvider)
    assert isinstance(get_ohlcv_provider("yf"), YFinanceDataProvider)
    assert isinstance(get_ohlcv_provider("fmp"), FMPOHLCVProvider)


def test_get_ohlcv_provider_legacy_explicit(monkeypatch):
    monkeypatch.setenv("OHLCV_SOURCE", "legacy")
    from app.api.datasets import get_ohlcv_provider
    from dataproviders.ohlcv.YFinanceDataProvider import YFinanceDataProvider

    assert isinstance(get_ohlcv_provider("yfinance"), YFinanceDataProvider)


def test_get_ohlcv_provider_ba2_returns_adapter(monkeypatch):
    """With the flag on and a constructible provider, the factory returns the
    adapter (not the legacy class)."""
    monkeypatch.setenv("OHLCV_SOURCE", "ba2_providers")
    import dataproviders.ba2providers_adapter as mod

    # Stub the eager provider resolution so no real ba2_providers/network is hit.
    monkeypatch.setattr(
        mod.BA2ProvidersOHLCVAdapter, "_get_provider",
        lambda self: _FakeProvider(), raising=True,
    )
    from app.api.datasets import get_ohlcv_provider

    prov = get_ohlcv_provider("fmp")
    assert isinstance(prov, mod.BA2ProvidersOHLCVAdapter)
    # and it satisfies the same get_data contract
    pts = prov.get_data("AAPL", datetime(2020, 1, 1), datetime(2020, 1, 5))
    assert len(pts) == 2 and pts[0].close == 1.5


def test_get_ohlcv_provider_ba2_falls_back_to_legacy_on_failure(monkeypatch):
    """If the ba2_providers path raises during construction, the factory must
    fall back to the legacy provider rather than blow up the dataset build."""
    monkeypatch.setenv("OHLCV_SOURCE", "ba2_providers")
    import dataproviders.ba2providers_adapter as mod

    def boom(self):
        raise RuntimeError("FMP_API_KEY missing")

    monkeypatch.setattr(mod.BA2ProvidersOHLCVAdapter, "_get_provider", boom,
                        raising=True)
    from app.api.datasets import get_ohlcv_provider
    from dataproviders.ohlcv.YFinanceDataProvider import YFinanceDataProvider

    prov = get_ohlcv_provider("yfinance")
    assert isinstance(prov, YFinanceDataProvider)
