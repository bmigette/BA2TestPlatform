"""Phase 1 Task 6: bar-based single-leg option leg fills + premium marking.

The fill engine FILLS a single-leg option order off the cached premium bar for its
contract (per ``fill_model``), exactly mirroring how the equity branch chooses its
fill bar (``next_bar_open`` -> the contract's next bar open; ``same_bar_close`` -> the
current bar close). Open option positions are MARKED each bar at the current premium
close x open qty x multiplier (100), so the equity curve includes options.

Equity fill/marking behaviour is untouched: this only adds an OPTION-only branch and
guards the equity branch to skip OPTION orders.

Run from the backend dir:
    ./venv/bin/python -m pytest tests/backtest/test_option_fills.py -q
"""
from __future__ import annotations

from datetime import date, datetime

import pytest

from ba2_common.core.types import OrderDirection, OrderStatus

CFG = {
    "starting_cash": 100_000.0,
    "commission_per_trade": 1.0,
    "slippage_bps": 0.0,  # no slippage -> fill premium == bar open exactly (deterministic)
    "fill_model": "next_bar_open",
}

_OCC = "AAPL240315C00180000"

# Two underlying bars so there IS a "next bar" relative to the submit/clock bar.
_AAPL_BARS = [
    {"Date": datetime(2024, 3, 5), "Open": 180, "High": 182, "Low": 178, "Close": 181, "Volume": 1000},
    {"Date": datetime(2024, 3, 6), "Open": 181, "High": 184, "Low": 180, "Close": 183, "Volume": 1100},
]


def _seed_cache(db_path: str) -> None:
    """Seed the CALL chain (2024-03-01) AND a premium bar for the contract on 2024-03-06.

    The fill bar (next_bar_open relative to the 2024-03-05 clock) is 2024-03-06: the
    order fills at that bar's OPEN premium (4.0). The same bar's CLOSE (4.5) is what
    the per-bar marking values the open position at.
    """
    from app.services.backtest.options_cache import OptionsHistoryCache

    cache = OptionsHistoryCache(db_path)
    cache.write_chain_rows(
        "AAPL",
        "2024-03-01",
        [
            {
                "occ_symbol": _OCC,
                "option_type": "call",
                "strike": 180.0,
                "expiry": "2024-03-15",
                "bid": 3.0,
                "ask": 3.2,
                "last": 3.1,
                "iv": 0.25,
            },
        ],
    )
    cache.write_bar_rows(
        [
            {
                "occ_symbol": _OCC,
                "date": "2024-03-06",
                "open": 4.0,
                "high": 4.8,
                "low": 3.9,
                "close": 4.5,
                "volume": 500,
                "underlying": "AAPL",
                "option_type": "call",
                "strike": 180.0,
                "expiry": "2024-03-15",
            },
        ]
    )


def _make_price_source(clock: datetime):
    from app.services.backtest.price_source import AsOfPriceSource

    ps = AsOfPriceSource(ohlcv_provider=None)  # no provider; bars pre-seeded
    ps.load_bars("AAPL", _AAPL_BARS)
    ps.set_clock(clock)
    return ps


@pytest.fixture
def options_account(tmp_path):
    """A BacktestAccount wired to a HistoricalOptionsProvider over a seeded temp cache.

    Mirrors the Task-4 ``options_account`` construction (fresh per-run trading DB + seam
    wiring + seeded account definition + injected options provider). The price source is
    returned with the account so the test can step the clock to the fill bar.
    """
    from app.services.backtest.backtest_db import (
        backtest_trading_db,
        seed_account_definition,
    )
    from app.services.backtest.seam_wiring import wire_backtest_seams
    from app.services.backtest.backtest_account import BacktestAccount
    from app.services.backtest.options_provider import HistoricalOptionsProvider

    cache_db = str(tmp_path / "options_cache.sqlite")
    _seed_cache(cache_db)
    provider = HistoricalOptionsProvider(cache_db)

    wire_backtest_seams()
    ctx = backtest_trading_db("optfills")
    ctx.__enter__()
    seed_account_definition(1, CFG)
    ps = _make_price_source(datetime(2024, 3, 5))
    acct = BacktestAccount(1, ps, CFG, options_provider=provider)
    wire_backtest_seams().register_account(1, acct)
    try:
        yield acct, ps
    finally:
        ctx.__exit__(None, None, None)


def _submit_market_buy_call(acct):
    from ba2_common.core.option_types import OptionLeg

    leg = OptionLeg(
        contract_symbol=_OCC,
        side=OrderDirection.BUY,
        position_intent="buy_to_open",
        underlying="AAPL",
    )
    return acct.submit_option_order(
        legs=[leg], quantity=1, order_type="market", option_strategy="long_call"
    )


def test_single_leg_market_call_fills_at_next_bar_open_premium(options_account):
    acct, ps = options_account
    order = _submit_market_buy_call(acct)
    assert order is not None
    assert order.status == OrderStatus.ACCEPTED  # staged, not yet filled on submit bar

    # next_bar_open convention (mirrors the equity branch): with the clock on the submit
    # bar (2024-03-05), refresh_orders fills at the NEXT bar (2024-03-06) open premium.
    acct.refresh_orders()
    acct.refresh_transactions()

    filled = acct.get_order(order.id)
    assert filled.status == OrderStatus.FILLED
    # next_bar_open with zero slippage -> the fill bar's OPEN premium (4.0), per share.
    assert filled.open_price == pytest.approx(4.0)
    assert filled.filled_qty == 1

    # the held option position shows up with qty 1 on the contract
    positions = acct.get_option_positions()
    assert len(positions) == 1
    assert positions[0].contract_symbol == _OCC
    assert positions[0].quantity == 1


def test_option_does_not_fill_on_entry_bar(options_account):
    """No look-ahead: with the clock on the LAST bar there is no next bar, so no fill."""
    acct, ps = options_account
    order = _submit_market_buy_call(acct)
    ps.set_clock(datetime(2024, 3, 6))  # last bar -> next_bar is None
    acct.refresh_orders()
    assert acct.get_order(order.id).status == OrderStatus.ACCEPTED


def test_option_fill_marks_premium_times_multiplier_in_equity(options_account):
    acct, ps = options_account
    cash_before = acct.get_balance()

    _submit_market_buy_call(acct)
    acct.refresh_orders()  # fills at 2024-03-06 open premium (4.0)
    acct.refresh_transactions()

    commission = CFG["commission_per_trade"]
    # Bought 1 contract @ 4.0 premium x 100 multiplier = $400 debit, + $1 commission.
    assert acct.get_balance() == pytest.approx(cash_before - 4.0 * 100 - commission)

    # snapshot_equity marks the OPEN option at the current bar's CLOSE premium. Step the
    # clock to the day that HAS the premium bar (2024-03-06): close = 4.5.
    ps.set_clock(datetime(2024, 3, 6))
    snap = acct.snapshot_equity(datetime(2024, 3, 6))
    assert snap["equity_value"] == pytest.approx(4.5 * 100)
    assert snap["net_liquidating_value"] == pytest.approx(acct.get_balance() + 4.5 * 100)


def test_option_position_value_falls_back_to_open_price_without_bar(options_account):
    """No premium bar on the marking day -> value the open option at its fill premium."""
    acct, ps = options_account
    _submit_market_buy_call(acct)
    acct.refresh_orders()  # fills at 2024-03-06 open premium (4.0)
    acct.refresh_transactions()

    # Step to a day with NO premium bar for the contract: marking falls back to open_price (4.0).
    ps.load_bars(
        "AAPL",
        _AAPL_BARS
        + [{"Date": datetime(2024, 3, 7), "Open": 183, "High": 185, "Low": 182, "Close": 184, "Volume": 900}],
    )
    ps.set_clock(datetime(2024, 3, 7))
    snap = acct.snapshot_equity(datetime(2024, 3, 7))
    assert snap["equity_value"] == pytest.approx(4.0 * 100)
