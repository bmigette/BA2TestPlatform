"""Phase 2 Task 5: results conversion (``build_results``) unit tests.

Feeds ``build_results`` a hand-made account STUB (no DB, no network) with a known equity
curve + known trades, and asserts the reused ``Backtest`` metric columns + the
``equity_curve``/``drawdown_curve``/``trades`` blobs come out with the right values and the
field names ``Backtest.to_dict()`` / ``_transform_trades_for_frontend`` consume.

Run from the backend dir:
    ./venv/bin/python -m pytest tests/backtest/test_results_metrics.py -v
"""
from __future__ import annotations

import math
from datetime import datetime

import pytest

from app.services.backtest.results import build_results


class _AccountStub:
    """Minimal stand-in: ``build_results`` only calls these two methods."""

    def __init__(self, snapshots, trades):
        self._snaps = snapshots
        self._trades = trades

    def get_balance_history(self):
        return self._snaps

    def get_filled_trades(self):
        return self._trades


def _snap(d, nlv, cash=0.0):
    return {"date": d, "net_liquidating_value": nlv, "cash_balance": cash, "equity_value": nlv - cash}


# 100k -> 110k -> 105k equity curve (a 10% gain then a ~4.5% dip from the 110 peak).
SNAPS = [
    _snap(datetime(2024, 1, 2), 100_000.0),
    _snap(datetime(2024, 1, 3), 110_000.0),
    _snap(datetime(2024, 1, 4), 105_000.0),
]

# 3 trades: 2 winners (+500, +300), 1 loser (-200).
TRADES = [
    {"symbol": "AAA", "side": "buy", "pnl": 500.0, "pnl_pct": 5.0, "bars_held": 2,
     "date": datetime(2024, 1, 2), "price": 10.0, "qty": 50},
    {"symbol": "BBB", "side": "buy", "pnl": 300.0, "pnl_pct": 3.0, "bars_held": 1,
     "date": datetime(2024, 1, 3), "price": 20.0, "qty": 20},
    {"symbol": "CCC", "side": "sell", "pnl": -200.0, "pnl_pct": -2.0, "bars_held": 3,
     "date": datetime(2024, 1, 4), "price": 30.0, "qty": 10},
]

CONFIG = {"initial_capital": 100_000.0}


def _results():
    return build_results(_AccountStub(SNAPS, TRADES), CONFIG)


def test_total_return_and_final_equity():
    r = _results()
    # 100k -> 105k = +5%.
    assert r["total_return"] == pytest.approx(5.0)
    assert r["final_equity"] == pytest.approx(105_000.0)
    assert r["equity_peak"] == pytest.approx(110_000.0)


def test_max_drawdown_reflects_110_to_105_dip():
    r = _results()
    # peak 110k, trough 105k -> drawdown = (105-110)/110 = -4.5454...%.
    assert r["max_drawdown"] == pytest.approx(-4.55, abs=0.01)
    assert r["max_drawdown"] < 0


def test_trade_quality_metrics():
    r = _results()
    assert r["total_trades"] == 3
    assert r["winning_trades"] == 2
    assert r["losing_trades"] == 1
    # 2 of 3 winners -> 66.67%.
    assert r["win_rate"] == pytest.approx(66.67, abs=0.01)
    # profit factor = gross_profit / gross_loss = 800 / 200 = 4.0, finite & under the cap.
    assert r["profit_factor"] == pytest.approx(4.0)
    assert r["profit_factor"] <= 999.99
    assert math.isfinite(r["profit_factor"])
    # best/worst by pnl_pct.
    assert r["best_trade"] == pytest.approx(5.0)
    assert r["worst_trade"] == pytest.approx(-2.0)


def test_profit_factor_capped_when_no_losers():
    """All-winners run -> profit factor would be Inf; must be capped at 999.99 (finite)."""
    winners_only = [TRADES[0], TRADES[1]]
    r = build_results(_AccountStub(SNAPS, winners_only), CONFIG)
    assert r["profit_factor"] == pytest.approx(999.99)
    assert math.isfinite(r["profit_factor"])


def test_curves_present_with_frontend_field_names():
    r = _results()
    # equity curve: one point per snapshot, {date, equity}.
    assert len(r["equity_curve"]) == 3
    assert set(r["equity_curve"][0].keys()) == {"date", "equity"}
    assert r["equity_curve"][0]["equity"] == pytest.approx(100_000.0)
    # drawdown curve: one point per snapshot, {date, drawdown}.
    assert len(r["drawdown_curve"]) == 3
    assert set(r["drawdown_curve"][0].keys()) == {"date", "drawdown"}
    assert r["drawdown_curve"][0]["drawdown"] == pytest.approx(0.0)  # at the first peak
    # trades carry the names _transform_trades_for_frontend reads.
    assert len(r["trades"]) == 3
    t = r["trades"][0]
    for k in ("entry_time", "exit_time", "direction", "entry_price", "exit_price",
              "size", "pnl", "pnl_pct", "bars_held", "exit_reason"):
        assert k in t
    # direction normalised to buy/sell (the vocabulary the model maps to long/short).
    assert t["direction"] in ("buy", "sell")
    assert r["trades"][2]["direction"] == "sell"


def test_all_metrics_finite_and_present():
    r = _results()
    expected_keys = {
        "total_trades", "winning_trades", "losing_trades", "win_rate",
        "total_return", "annualized_return", "buy_hold_return",
        "sharpe_ratio", "sortino_ratio", "calmar_ratio", "volatility",
        "max_drawdown", "avg_drawdown", "max_drawdown_duration",
        "profit_factor", "expectancy", "sqn", "avg_trade", "best_trade", "worst_trade",
        "avg_trade_duration", "exposure_time", "final_equity", "equity_peak",
    }
    assert expected_keys.issubset(set(r.keys()))
    for k in expected_keys:
        v = r[k]
        assert v is not None
        assert math.isfinite(float(v)), f"{k} is not finite: {v}"


def test_empty_run_is_safe():
    """No snapshots / no trades -> all-zero metrics, equity defaults to initial capital."""
    r = build_results(_AccountStub([], []), CONFIG)
    assert r["total_trades"] == 0
    assert r["total_return"] == pytest.approx(0.0)
    assert r["final_equity"] == pytest.approx(100_000.0)
    assert r["equity_curve"] == []
    assert r["drawdown_curve"] == []
    assert r["trades"] == []
    assert r["profit_factor"] == pytest.approx(0.0)
