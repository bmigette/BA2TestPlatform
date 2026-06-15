"""Task 6: backtest list filters (expert/optimization_id/saved) + create accepts engine/expert/universe.

Uses the shared conftest ``client``/``db`` fixtures (a throwaway SQLite gate engine with
``queue_task`` stubbed). The ``db`` and ``client`` fixtures bind to the SAME gate engine, so
rows seeded through ``db`` are visible to the route under test.
"""
from __future__ import annotations

from datetime import datetime

import pytest


def _seed_backtest(db, *, name, expert_name, optimization_id, is_saved=False, engine_type="daily_expert"):
    from app.models.backtest import Backtest

    bt = Backtest(
        name=name,
        expert_name=expert_name,
        optimization_id=optimization_id,
        is_saved=is_saved,
        engine_type=engine_type,
        start_date=datetime(2020, 1, 1),
        end_date=datetime(2020, 6, 1),
        initial_capital=10000.0,
        status="completed",
    )
    db.add(bt)
    db.commit()
    db.refresh(bt)
    return bt


@pytest.fixture
def seeded(db):
    """Three rows: (FMPRating, opt None), (FMPRating, opt 5), (FMPEarningsDrift, opt 5)."""
    a = _seed_backtest(db, name="rating-none", expert_name="FMPRating", optimization_id=None)
    b = _seed_backtest(db, name="rating-opt5", expert_name="FMPRating", optimization_id=5, is_saved=True)
    c = _seed_backtest(db, name="drift-opt5", expert_name="FMPEarningsDrift", optimization_id=5)
    return {"a": a, "b": b, "c": c}


# --- Part A: list filters ---------------------------------------------------
def test_list_filter_by_expert(client, seeded):
    resp = client.get("/api/backtests?expert=FMPRating")
    assert resp.status_code == 200, resp.text
    items = resp.json()["backtests"]
    names = {i["name"] for i in items}
    assert names == {"rating-none", "rating-opt5"}
    # every item exposes expert_name + optimization_id
    for i in items:
        assert i["expertName"] == "FMPRating"
        assert "optimizationId" in i


def test_list_filter_by_optimization_id(client, seeded):
    resp = client.get("/api/backtests?optimization_id=5")
    assert resp.status_code == 200, resp.text
    items = resp.json()["backtests"]
    names = {i["name"] for i in items}
    assert names == {"rating-opt5", "drift-opt5"}
    for i in items:
        assert i["optimizationId"] == 5


def test_list_filter_by_saved(client, seeded):
    resp = client.get("/api/backtests?saved=true")
    assert resp.status_code == 200, resp.text
    items = resp.json()["backtests"]
    names = {i["name"] for i in items}
    assert names == {"rating-opt5"}


def test_list_no_filter_returns_all(client, seeded):
    resp = client.get("/api/backtests")
    assert resp.status_code == 200, resp.text
    items = resp.json()["backtests"]
    names = {i["name"] for i in items}
    assert {"rating-none", "rating-opt5", "drift-opt5"} <= names


# --- Part B: create accepts engine/expert/universe --------------------------
def test_create_daily_expert_static_universe(client, db):
    """A daily_expert create with a supported expert + static universe returns a queued backtest."""
    payload = {
        "name": "daily-create-test",
        "engine": "daily_expert",
        "expert": {"class": "FMPRating", "settings": {}},
        "universe": {"mode": "static", "symbols": ["AAPL", "MSFT"]},
        "start_date": "2020-01-01",
        "end_date": "2020-06-01",
        "initial_capital": 10000.0,
        "commission": 1.0,
        "slippage": 5.0,
        "fill_model": "next_bar_open",
        "seed": 42,
    }
    resp = client.post("/api/backtests", json=payload)
    assert resp.status_code in (200, 201), resp.text
    body = resp.json()
    assert body["expertName"] == "FMPRating"
    assert body["engineType"] == "daily_expert"


def test_create_daily_expert_rejects_unknown_expert(client):
    payload = {
        "name": "bad-expert",
        "engine": "daily_expert",
        "expert": {"class": "NotARealExpert"},
        "universe": {"mode": "static", "symbols": ["AAPL"]},
        "start_date": "2020-01-01",
        "end_date": "2020-06-01",
        "initial_capital": 10000.0,
        "commission": 1.0,
        "slippage": 5.0,
        "fill_model": "next_bar_open",
        "seed": 42,
    }
    resp = client.post("/api/backtests", json=payload)
    assert resp.status_code == 400, resp.text


def test_create_daily_expert_rejects_empty_universe(client):
    payload = {
        "name": "empty-universe",
        "engine": "daily_expert",
        "expert": {"class": "FMPRating"},
        "universe": {"mode": "static", "symbols": []},
        "start_date": "2020-01-01",
        "end_date": "2020-06-01",
        "initial_capital": 10000.0,
        "commission": 1.0,
        "slippage": 5.0,
        "fill_model": "next_bar_open",
        "seed": 42,
    }
    resp = client.post("/api/backtests", json=payload)
    assert resp.status_code == 400, resp.text


def test_create_ml_still_works(client, db, monkeypatch):
    """The existing ML create path must keep working byte-for-byte (engine defaults to 'ml')."""
    from app.models.model import TrainedModel
    from app.models.dataset import Dataset

    model = TrainedModel(model_id="mdl-test123", name="m1", model_type="LSTM", status="completed")
    db.add(model)

    def _ds(name):
        return Dataset(
            name=name,
            ticker="AAPL",
            timeframe="1d",
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2020, 6, 1),
            file_path=f"/tmp/{name}.csv",
        )

    pred = _ds("pred-ds")
    exe = _ds("exec-ds")
    db.add(pred)
    db.add(exe)
    db.commit()
    db.refresh(model)
    db.refresh(pred)
    db.refresh(exe)

    payload = {
        "name": "ml-create-test",
        "model_id": "mdl-test123",
        "prediction_dataset_id": pred.id,
        "execution_dataset_id": exe.id,
        "start_date": "2020-01-01",
        "end_date": "2020-06-01",
        "initial_capital": 10000.0,
    }
    resp = client.post("/api/backtests", json=payload)
    assert resp.status_code in (200, 201), resp.text
    body = resp.json()
    assert body["engineType"] == "ml"
