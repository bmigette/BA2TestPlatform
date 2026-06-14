"""Phase 4 Task 8 companion: classic-RM optimize fields round-trip through the
strategy create/update endpoints so the new UI RM controls are load-bearing.

The POST /api/strategies/{id}/optimize route builds rm_params from the Strategy's
rm_* columns (_rm_cfg_from_strategy). Those columns are only populated when the
create/update endpoints accept the rm_* request fields — this test proves they do,
and that the persisted values flow back through to the folded optimization_config.

Self-contained (does NOT depend on a conftest): a throwaway sqlite DATABASE_URL is
set at module import time BEFORE any app import, mirroring tests/test_optimize_route.py.

Run from the backend dir:
    ./venv/bin/python -m pytest tests/test_strategy_rm_persist.py -v
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TEST_DB_PATH = os.path.join(os.path.dirname(__file__), "test_strategy_rm_persist.db")
if os.path.exists(TEST_DB_PATH):
    os.remove(TEST_DB_PATH)
os.environ["DATABASE_URL"] = f"sqlite:///{TEST_DB_PATH}"

import pytest


@pytest.fixture(scope="module")
def test_db():
    from app.models.database import engine, Base, SessionLocal

    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    yield db
    db.close()
    engine.dispose()
    if os.path.exists(TEST_DB_PATH):
        try:
            os.remove(TEST_DB_PATH)
        except PermissionError:
            pass


@pytest.fixture(scope="module")
def client(test_db):
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as c:
        yield c


def test_create_persists_rm_optimize_fields(client):
    payload = {
        "name": "rm-create-test",
        "initial_tp_percent": 5.0,
        "initial_sl_percent": 2.0,
        # float RM group marked optimize with explicit bounds
        "rm_risk_per_trade_pct": 1.5,
        "rm_risk_per_trade_pct_optimize": True,
        "rm_risk_per_trade_pct_min": 0.5,
        "rm_risk_per_trade_pct_max": 3.0,
        "rm_risk_per_trade_pct_step": 0.25,
        # integer RM group marked optimize
        "rm_max_concurrent_positions": 5,
        "rm_max_concurrent_positions_optimize": True,
        "rm_max_concurrent_positions_min": 1,
        "rm_max_concurrent_positions_max": 10,
        "rm_max_concurrent_positions_step": 1,
    }
    r = client.post("/api/strategies", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    # camelCase RM keys are surfaced in to_dict() for the UI to bind
    assert body["rmRiskPerTradePct"] == 1.5
    assert body["rmRiskPerTradePctOptimize"] is True
    assert body["rmRiskPerTradePctMin"] == 0.5
    assert body["rmMaxConcurrentPositionsOptimize"] is True
    assert body["rmMaxConcurrentPositionsMax"] == 10
    # A param the request omitted keeps the model default (baseline RM present)
    assert body["rmPerInstrumentCapPct"] == 20.0


def test_update_persists_and_optimize_route_folds_rm(client):
    # Create a baseline strategy with no RM optimize
    r = client.post(
        "/api/strategies",
        json={"name": "rm-update-test", "initial_tp_percent": 5.0, "initial_sl_percent": 2.0},
    )
    assert r.status_code == 200, r.text
    sid = r.json()["id"]
    assert r.json()["rmAtrStopMultOptimize"] is False

    # Update to mark the ATR-stop RM group optimizable
    r = client.put(
        f"/api/strategies/{sid}",
        json={
            "rm_atr_stop_mult": 2.0,
            "rm_atr_stop_mult_optimize": True,
            "rm_atr_stop_mult_min": 1.0,
            "rm_atr_stop_mult_max": 4.0,
            "rm_atr_stop_mult_step": 0.5,
        },
    )
    assert r.status_code == 200, r.text
    assert r.json()["rmAtrStopMultOptimize"] is True
    assert r.json()["rmAtrStopMultMin"] == 1.0

    # The optimize route must fold the persisted RM config into optimization_config
    opt_payload = {
        "fitness_metric": "sharpe",
        "optimization_type": "genetic",
        "optimization_config": {
            "populationSize": 8,
            "generations": 3,
            "crossoverProb": 0.7,
            "mutationProb": 0.2,
            "earlyStoppingGenerations": 10,
            "elitismPercent": 10.0,
            "seed": 42,
            "backtest": {
                "engine": "daily",
                "start_date": "2020-01-01",
                "end_date": "2021-01-01",
                "initial_capital": 10000.0,
            },
        },
    }
    r = client.post(f"/api/strategies/{sid}/optimize", json=opt_payload)
    assert r.status_code == 200, r.text
    rm = r.json()["optimizationConfig"]["rm_params"]
    assert rm["atr_stop_mult"]["optimize"] is True
    assert rm["atr_stop_mult"]["min"] == 1.0
    assert rm["atr_stop_mult"]["max"] == 4.0
    assert rm["atr_stop_mult"]["step"] == 0.5
