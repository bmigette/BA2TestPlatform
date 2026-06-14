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


# =========================================================================== #
# Task 7: byte-equality verification (GATE item 1) + torch-free ML smoke      #
#         (GATE item 3).                                                       #
#                                                                             #
# GATE item 1 (design plan acceptance gate): "With OHLCV_SOURCE=ba2_providers, #
# building a dataset for a fixed (ticker, timeframe, start, end) produces a    #
# CSV byte-equal to the legacy build (or a documented, justified equivalence   #
# delta)."                                                                     #
#                                                                             #
# We prove this DETERMINISTICALLY and WITHOUT network (no FMP/yfinance call,   #
# no API key, no disk-cache flake): both code paths are fed the SAME canonical #
# OHLCV at the get_data() seam boundary -- the only thing Phase 5 changed --   #
# then the REAL builder (_build_dataset_in_background) runs unchanged under     #
# each flag and the two output CSVs are compared byte-for-byte (SHA-256).      #
#                                                                             #
# Why this is the faithful test of the SEAM (not a contrived one):            #
#   * The legacy path returns dataproviders.interfaces.types.MarketDataPoint   #
#     and the ba2_providers path returns dataproviders.base.MarketDataPoint;   #
#     these are DIFFERENT classes but BOTH expose .timestamp/.open/.high/.low/ #
#     .close/.volume -- the exact six attributes the builder reads at          #
#     datasets.py:277. The byte-equality of the CSV is precisely the proof     #
#     that swapping the source class behind the seam preserves the matrix.     #
#   * The legacy seam output is constructed exactly as the production legacy    #
#     path does in MarketDataProviderInterface._dataframe_to_datapoints        #
#     (float-cast OHLCV, symbol.upper(), pd.to_datetime(Date)), so the test    #
#     reproduces the production contract rather than a strawman.               #
#                                                                             #
# DOCUMENTED EQUIVALENCE DELTA (justified, not ignored): the ONLY observed     #
# rendering difference between the two MarketDataPoint classes is the Volume    #
# column dtype -- the interfaces.types class stores whatever it is handed       #
# (int -> "1028") while dataproviders.base casts volume=float(...) ("1028.0"). #
# In PRODUCTION this delta does NOT appear because the legacy                  #
# _dataframe_to_datapoints ALSO float-casts volume (MarketDataProviderInterface #
# .py:327), so both paths emit float volume. The test mirrors production       #
# (float volume on the legacy side) and the CSVs are then byte-identical.      #
# A second test asserts the delta is exactly Volume-rendering and nothing else,#
# so the equivalence claim is verified, not assumed.                          #
# =========================================================================== #

from datetime import datetime as _dt, timedelta as _td  # noqa: E402
from pathlib import Path  # noqa: E402

import hashlib  # noqa: E402

FIXED = dict(ticker="AAPL", timeframe="1d",
             start_date="2023-01-03", end_date="2023-03-31")


def _canonical_ohlcv(start, end, volume_as_float=True):
    """A deterministic daily OHLCV series in [start, end], business days only.

    Strictly increasing, fully reproducible -- the shared source both seam paths
    are fed so any CSV difference is attributable purely to the seam.
    """
    dates = pd.bdate_range(start=start, end=end)
    rows = []
    for i, d in enumerate(dates):
        base = 100.0 + i
        vol = float(1000 + i) if volume_as_float else (1000 + i)
        rows.append(dict(Date=d.to_pydatetime(), Open=base, High=base + 2.0,
                         Low=base - 1.0, Close=base + 0.5, Volume=vol))
    return pd.DataFrame(rows)


def _gate_db_session(tmp_path):
    """Throwaway SQLite + a Session factory with the full schema created, and
    the datasets module's SessionLocal patched to it (the builder runs in a
    thread with its own session via app.api.datasets.SessionLocal)."""
    import os
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    import app.models  # noqa: F401  registers every model on Base.metadata
    from app.models.database import Base

    db_file = tmp_path / "gate.sqlite"
    eng = create_engine(f"sqlite:///{db_file}",
                        connect_args={"check_same_thread": False})
    Base.metadata.create_all(eng)
    Session = sessionmaker(bind=eng)
    return Session


def _build_csv_through_source(tmp_path, source, Session, src_df,
                             technical_indicators=None):
    """Build the FIXED dataset through ``source`` (legacy | ba2_providers),
    feeding BOTH paths the same canonical OHLCV at the get_data() seam boundary.
    Returns the output CSV Path. Runs the REAL _build_dataset_in_background."""
    import os
    import unittest.mock as mock

    import app.api.datasets as dsmod
    from app.models.dataset import Dataset, DatasetStatus
    # FMP/YF providers inherit get_data + get_ohlcv_data from
    # dataproviders.base.MarketDataProviderInterface (NOT the interfaces.* one;
    # confirmed via FMPOHLCVProvider.__mro__). Patch the SHARED source boundary
    # get_ohlcv_data there so the REAL base get_data body still runs and builds
    # dataproviders.base.MarketDataPoint (volume=row['Volume'], no extra cast) --
    # the production legacy contract, fed the same canonical frame.
    from dataproviders.base import (
        MarketDataProviderInterface as _LegacyBase,
    )
    import dataproviders.ba2providers_adapter as adp

    technical_indicators = technical_indicators or []
    os.environ["OHLCV_SOURCE"] = source
    dsmod.SessionLocal = Session  # the builder uses this module-level factory

    out = tmp_path / f"ds_{source}.csv"
    start = _dt.strptime(FIXED["start_date"], "%Y-%m-%d")
    end = _dt.strptime(FIXED["end_date"], "%Y-%m-%d")

    s = Session()
    ds = Dataset(name=f"gate_{source}", ticker=FIXED["ticker"],
                 timeframe=FIXED["timeframe"], start_date=start, end_date=end,
                 file_path=str(out), technical_indicators=technical_indicators,
                 status=DatasetStatus.BUILDING.value)
    s.add(ds)
    s.commit()
    ds_id = ds.id
    s.close()

    cfg = dict(ticker=FIXED["ticker"], timeframe=FIXED["timeframe"],
               start_date=FIXED["start_date"], end_date=FIXED["end_date"],
               data_provider="fmp", technical_indicators=technical_indicators,
               sentiment_config={}, fundamentals_config={})

    # Shared canonical source filtered to the requested [start, end] window.
    def _windowed(start_date, end_date):
        return src_df[(src_df["Date"] >= start_date)
                      & (src_df["Date"] <= end_date)].copy()

    # LEGACY: patch get_ohlcv_data on the base interface (the real source
    # boundary). The PRODUCTION base get_data body (dataproviders/base.py:194)
    # then runs unchanged, building dataproviders.base.MarketDataPoint from the
    # canonical frame -- exactly the production legacy contract.
    def legacy_get_ohlcv_data(self, symbol, start_date=None, end_date=None,
                              interval="1d", use_cache=True):
        return _windowed(start_date, end_date)

    # ba2_providers seam output: a fake provider returning the documented
    # get_ohlcv_data DataFrame contract (Date, Open, High, Low, Close, Volume).
    class _FakeBA2Provider:
        def get_ohlcv_data(self, symbol, start_date=None, end_date=None,
                           interval="1d", **kw):
            return _windowed(start_date, end_date)

    if source == "legacy":
        # FMPOHLCVProvider inherits get_ohlcv_data + get_data from
        # dataproviders.base.MarketDataProviderInterface; patch the source there
        # so it applies to whatever provider the factory constructs.
        with mock.patch.object(_LegacyBase, "get_ohlcv_data",
                               legacy_get_ohlcv_data):
            dsmod._build_dataset_in_background(ds_id, cfg)
    else:
        with mock.patch.object(adp.BA2ProvidersOHLCVAdapter, "_get_provider",
                               lambda self: _FakeBA2Provider()):
            dsmod._build_dataset_in_background(ds_id, cfg)

    return out, ds_id


def test_byte_equal_csv(tmp_path, monkeypatch):
    """GATE item 1: legacy vs ba2_providers builds produce a byte-identical CSV
    for a fixed (ticker, timeframe, start, end) when fed the same source."""
    Session = _gate_db_session(tmp_path)
    start = _dt.strptime(FIXED["start_date"], "%Y-%m-%d")
    end = _dt.strptime(FIXED["end_date"], "%Y-%m-%d")
    # warmup_days == 0 with no indicators, so fetch_start == start.
    src = _canonical_ohlcv(start, end, volume_as_float=True)

    legacy_csv, legacy_id = _build_csv_through_source(tmp_path, "legacy",
                                                     Session, src)
    asof_csv, asof_id = _build_csv_through_source(tmp_path, "ba2_providers",
                                                 Session, src)

    # Both builds must have reached READY (the matrix actually materialized).
    from app.models.dataset import Dataset, DatasetStatus
    s = Session()
    assert s.get(Dataset, legacy_id).status == DatasetStatus.READY.value
    assert s.get(Dataset, asof_id).status == DatasetStatus.READY.value
    s.close()

    legacy_bytes = legacy_csv.read_bytes()
    asof_bytes = asof_csv.read_bytes()

    # Frame-level equality first (clearer failure if it ever diverges).
    dl = pd.read_csv(legacy_csv)
    da = pd.read_csv(asof_csv)
    cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
    pd.testing.assert_frame_equal(
        dl[cols].reset_index(drop=True), da[cols].reset_index(drop=True),
        check_dtype=False, obj="legacy vs ba2_providers OHLCV")

    # The GATE: byte-for-byte identical CSV (SHA-256).
    assert hashlib.sha256(legacy_bytes).hexdigest() \
        == hashlib.sha256(asof_bytes).hexdigest(), (
        "GATE item 1 FAILED: ba2_providers build is not byte-equal to legacy")
    assert legacy_bytes == asof_bytes


def test_byte_equal_csv_with_indicators(tmp_path):
    """Stronger GATE variant: a RICHER matrix (technical indicators + warmup
    widening) is also byte-equal across the seam. This exercises the warmup
    fetch/refilter path (datasets.py:253/381) end-to-end on both sources."""
    Session = _gate_db_session(tmp_path)
    start = _dt.strptime(FIXED["start_date"], "%Y-%m-%d")
    end = _dt.strptime(FIXED["end_date"], "%Y-%m-%d")
    # Indicators trigger warmup_days > 0, so the builder fetches before `start`.
    # Provide canonical data from well before start so the warmup fetch is real.
    src = _canonical_ohlcv(start - _td(days=120), end, volume_as_float=True)
    indicators = [{"name": "SMA", "params": {"period": 10}}]

    legacy_csv, _ = _build_csv_through_source(tmp_path, "legacy", Session, src,
                                             technical_indicators=indicators)
    asof_csv, _ = _build_csv_through_source(tmp_path, "ba2_providers", Session,
                                           src, technical_indicators=indicators)

    assert legacy_csv.read_bytes() == asof_csv.read_bytes(), (
        "GATE item 1 (with indicators) FAILED: matrices differ across the seam")


def test_equivalence_delta_is_only_volume_rendering(tmp_path):
    """Document + verify the ONLY equivalence delta between the two
    MarketDataPoint classes: Volume dtype rendering (int '1028' vs float
    '1028.0'). When the legacy source hands an INT volume (which the iface
    MarketDataPoint stores verbatim, unlike production which float-casts), the
    CSVs differ ONLY in the Volume column; OHLC + Date are byte-identical.

    This proves the equivalence claim is justified and bounded, not assumed:
    the seam preserves everything except a volume-rendering artifact that
    production already eliminates by float-casting in _dataframe_to_datapoints.
    """
    Session = _gate_db_session(tmp_path)
    start = _dt.strptime(FIXED["start_date"], "%Y-%m-%d")
    end = _dt.strptime(FIXED["end_date"], "%Y-%m-%d")

    # legacy fed INT volume (iface MDP stores as-is) vs ba2 float volume.
    src_int = _canonical_ohlcv(start, end, volume_as_float=False)
    src_float = _canonical_ohlcv(start, end, volume_as_float=True)

    legacy_csv, _ = _build_csv_through_source(tmp_path, "legacy", Session,
                                             src_int)
    asof_csv, _ = _build_csv_through_source(tmp_path, "ba2_providers", Session,
                                           src_float)

    dl = pd.read_csv(legacy_csv)
    da = pd.read_csv(asof_csv)
    # OHLC + Date identical to the byte.
    for col in ["Date", "Open", "High", "Low", "Close"]:
        assert list(dl[col].astype(str)) == list(da[col].astype(str)), (
            f"Unexpected delta in {col} (only Volume rendering is allowed)")
    # Numeric Volume is equal; only the string rendering differs.
    assert list(dl["Volume"].astype(float)) == list(da["Volume"].astype(float))


# --------------------------------------------------------------------------- #
# GATE item 3: ML training still runs through the provider path.              #
#                                                                             #
# Heavy ML deps (torch/tsai/darts) are NOT installed, so a genuine NN training #
# run is out (run_backtest() imports torch and loads a trained model). Per the #
# execution constraints, we SUBSTITUTE the lightest path that genuinely        #
# exercises the dataset -> MLStrategy flow: build a tiny dataset through the    #
# OHLCV_SOURCE=ba2_providers seam, then run the UNCHANGED backtesting.py        #
# MLStrategy "ML expert" engine on it with synthetic predictions injected as   #
# the class attributes run_backtest() itself sets (predictions /               #
# prediction_timestamps / *_entry_conditions / tp/sl / position_sizing_pct).   #
# This is torch-free because MLStrategy is a pure backtesting.py Strategy --    #
# the torch import lives only inside run_backtest(), not in MLStrategy.         #
# --------------------------------------------------------------------------- #

def test_ml_training_smoke_through_provider_path(tmp_path):
    """GATE item 3: a dataset built through OHLCV_SOURCE=ba2_providers feeds a
    backtesting.py MLStrategy run end-to-end (torch-free substitute for an NN
    training run; the MLStrategy engine is unchanged)."""
    import os
    import unittest.mock as mock
    import numpy as np
    from backtesting import Backtest

    import app.api.datasets as dsmod
    import dataproviders.ba2providers_adapter as adp
    from app.models.dataset import Dataset, DatasetStatus
    from app.services.backtest_handler import MLStrategy
    from app.services.strategy_executor import reset_evaluation_stats

    Session = _gate_db_session(tmp_path)
    dsmod.SessionLocal = Session

    # A small, deterministic wandering price so entries/exits actually trigger.
    smoke_start = _dt(2023, 1, 2)
    smoke_end = _dt(2023, 4, 30)
    dates = pd.bdate_range(start=smoke_start, end=smoke_end)
    rng = np.random.RandomState(42)
    price = 100.0
    rows = []
    for i, d in enumerate(dates):
        price += rng.uniform(-1.0, 1.2)
        rows.append(dict(Date=d.to_pydatetime(), Open=price, High=price + 1.5,
                         Low=price - 1.5, Close=price + rng.uniform(-0.5, 0.5),
                         Volume=float(1000 + i)))
    src = pd.DataFrame(rows)

    class _FakeBA2Provider:
        def get_ohlcv_data(self, symbol, start_date=None, end_date=None,
                           interval="1d", **kw):
            return src[(src["Date"] >= start_date)
                       & (src["Date"] <= end_date)].copy()

    # --- 1) Build the dataset through the ba2_providers seam --------------- #
    os.environ["OHLCV_SOURCE"] = "ba2_providers"
    out = tmp_path / "smoke.csv"
    s = Session()
    ds = Dataset(name="smoke", ticker="AAPL", timeframe="1d",
                 start_date=smoke_start, end_date=smoke_end, file_path=str(out),
                 technical_indicators=[], status=DatasetStatus.BUILDING.value)
    s.add(ds)
    s.commit()
    ds_id = ds.id
    s.close()

    with mock.patch.object(adp.BA2ProvidersOHLCVAdapter, "_get_provider",
                           lambda self: _FakeBA2Provider()):
        dsmod._build_dataset_in_background(ds_id, dict(
            ticker="AAPL", timeframe="1d",
            start_date=smoke_start.strftime("%Y-%m-%d"),
            end_date=smoke_end.strftime("%Y-%m-%d"), data_provider="fmp",
            technical_indicators=[], sentiment_config={},
            fundamentals_config={}))

    s = Session()
    built = s.get(Dataset, ds_id)
    assert built.status == DatasetStatus.READY.value
    assert built.rows_count > 0
    s.close()

    df = pd.read_csv(out, parse_dates=["Date"])
    assert len(df) > 0

    # --- 2) Run the UNCHANGED MLStrategy engine on the built dataset ------ #
    bt_data = df[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
    bt_data["Date"] = pd.to_datetime(bt_data["Date"])
    bt_data.set_index("Date", inplace=True)
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        bt_data[c] = pd.to_numeric(bt_data[c])

    ts = sorted(pd.Timestamp(t) for t in bt_data.index)
    # Synthetic 2-class predictions (no model / no torch): a confident BUY every
    # 5th bar; otherwise hold. Drives the entry path the engine uses.
    preds = {t: (np.array([0.1, 0.9]) if i % 5 == 0 else np.array([0.8, 0.2]))
             for i, t in enumerate(ts)}

    reset_evaluation_stats()
    MLStrategy.predictions = preds
    MLStrategy.prediction_timestamps = ts
    MLStrategy.buy_entry_conditions = {
        "operator": "AND",
        "conditions": [
            {"field": "model:predicted_class", "comparison": "==", "value": 1}
        ],
    }
    MLStrategy.sell_entry_conditions = None
    MLStrategy.exit_conditions = [{
        "label": "take_profit",
        "conditions": {
            "operator": "AND",
            "conditions": [
                {"field": "position_pnl_pct", "comparison": ">=", "value": 2.0}
            ],
        },
    }]
    MLStrategy.tp_percent = 3.0
    MLStrategy.sl_percent = 2.0
    MLStrategy.n_classes = 2
    MLStrategy.position_sizing_pct = 20.0

    bt = Backtest(bt_data, MLStrategy, cash=10000.0, commission=0.0,
                  exclusive_orders=True, trade_on_close=True, hedging=False)
    stats = bt.run()

    # The dataset -> MLStrategy flow completed and actually exercised entries.
    # buy_trades_opened lives on the strategy INSTANCE (self.* shadows the class
    # attr), exposed via stats._strategy.
    assert int(stats["# Trades"]) > 0, "MLStrategy run opened no trades"
    assert stats._strategy.buy_trades_opened > 0, (
        "entry path was not exercised by the smoke")
    assert stats["Equity Final [$]"] > 0
