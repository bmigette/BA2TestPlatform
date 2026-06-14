"""Tests for the cache-management API (/api/cache).

Task 1 covers the usage scanner + drill-down endpoints. Deletion-endpoint
tests are added in a later task.
"""
import importlib

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def seeded_cache(tmp_path, monkeypatch):
    """Point every cache root at a throwaway tree and seed a few files."""
    monkeypatch.setenv("CACHE_FOLDER", str(tmp_path / "cache"))
    from app.services import cache_manager
    importlib.reload(cache_manager)  # re-read CACHE_FOLDER + rebuild CACHE_TYPES

    # seed ohlcv (provider subfolder + SYMBOL_interval file)
    ohlcv = tmp_path / "cache" / "FMPOHLCVProvider"
    ohlcv.mkdir(parents=True)
    (ohlcv / "AAPL_1d.csv").write_text("Date,Open\n2020-01-01,1\n")

    # seed datasets (destructive type) — override its root onto the temp tree
    cache_manager.CACHE_TYPES["datasets"]["roots"] = [tmp_path / "datasets"]
    (tmp_path / "datasets").mkdir()
    (tmp_path / "datasets" / "ds1.csv").write_text("x\n1\n")

    return cache_manager


def test_usage_reports_per_type(seeded_cache):
    from app.main import app
    client = TestClient(app)
    r = client.get("/api/cache/usage")
    assert r.status_code == 200
    types = r.json()["types"]
    assert types["ohlcv"]["files"] >= 1
    assert types["ohlcv"]["bytes"] > 0
    assert types["datasets"]["destructive"] is True
    # all expected types present
    for name in ("ohlcv", "jobs", "news", "datasets", "models", "exports", "asof"):
        assert name in types


def test_usage_reports_mtime_and_ttl(seeded_cache):
    from app.main import app
    client = TestClient(app)
    types = client.get("/api/cache/usage").json()["types"]
    # ohlcv has a 24h TTL and a populated newest mtime
    assert types["ohlcv"]["ttl_hours"] == 24
    assert types["ohlcv"]["newest"] is not None
    # non-destructive flag on ohlcv
    assert types["ohlcv"]["destructive"] is False


def test_drill_down_ohlcv(seeded_cache):
    from app.main import app
    client = TestClient(app)
    r = client.get("/api/cache/usage/ohlcv")
    assert r.status_code == 200
    items = r.json()["items"]
    assert any(
        it["symbol"] == "AAPL" and it["interval"] == "1d" and it["provider"] == "FMPOHLCVProvider"
        for it in items
    )


def test_drill_down_datasets_flat_listing(seeded_cache):
    from app.main import app
    client = TestClient(app)
    r = client.get("/api/cache/usage/datasets")
    assert r.status_code == 200
    items = r.json()["items"]
    assert any(it["name"] == "ds1.csv" and it["bytes"] > 0 for it in items)


def test_drill_down_unknown_type_404(seeded_cache):
    from app.main import app
    client = TestClient(app)
    assert client.get("/api/cache/usage/bogus").status_code == 404
