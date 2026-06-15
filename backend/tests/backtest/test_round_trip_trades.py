"""Round-trip pairing in ``BacktestAccount.get_round_trip_trades`` (realised-P&L rows).

``get_round_trip_trades`` groups FILLED orders by ``transaction_id`` and emits ONE
round-trip row per transaction so trade-quality metrics (win_rate / profit_factor /
expectancy / exit_reason) have realised P&L to work with.

These tests prove the SIDE-based entry/exit classification against a real
``BacktestAccount`` over a fresh per-run backtest DB (no network, no provider). Orders are
filled directly via ``_apply_fill`` (which stamps the simulated fill bar into
``_fill_dates`` and updates the cash/position ledger), so we control side / fill time /
``depends_on_order`` exactly.

The classification under test:

  * the OPENING order is the EARLIEST-filled order in the transaction; its side is the
    ``opening_side``;
  * ENTRIES are same-side orders (open + rebalance ADDs), EXITS are opposite-side orders
    (plain rebalance/stop sells AND dependent TP/SL/OCO legs — both are closers);
  * a transaction closed by a PLAIN market sell (``depends_on_order is None``, opposite
    side) must produce a real round-trip (exit_reason ``"exit"``), NOT fall through to the
    ``open_at_end`` mark-to-market branch (the pre-fix bug: a plain sell was mis-read as the
    entry, leaving no exits).

Run from the backend dir:
    ./venv/bin/python -m pytest tests/backtest/test_round_trip_trades.py -v
"""
from __future__ import annotations

from datetime import datetime

import pytest

CFG = {
    "starting_cash": 100_000.0,
    "commission_per_trade": 0.0,
    "slippage_bps": 0.0,
    "fill_model": "next_bar_open",
}

D1 = datetime(2024, 1, 2)
D2 = datetime(2024, 1, 3)
D3 = datetime(2024, 1, 4)
D4 = datetime(2024, 1, 5)


def _bars(symbol, last_close):
    return [
        {"Date": d, "Open": 100, "High": 200, "Low": 50, "Close": c, "Volume": 1000}
        for (d, c) in [(D1, 100), (D2, 100), (D3, 100), (D4, last_close)]
    ]


def _acct(symbol="AAPL", last_close=100.0, account_id=1, cfg=CFG):
    """Build a wired BacktestAccount over a fresh per-run backtest DB. Caller closes ctx."""
    from app.services.backtest.backtest_db import (
        backtest_trading_db,
        seed_account_definition,
    )
    from app.services.backtest.seam_wiring import wire_backtest_seams
    from app.services.backtest.backtest_account import BacktestAccount
    from app.services.backtest.price_source import AsOfPriceSource

    wire_backtest_seams()
    ctx = backtest_trading_db(f"round-trip-{account_id}")
    ctx.__enter__()
    seed_account_definition(account_id, cfg)
    ps = AsOfPriceSource(ohlcv_provider=None)
    ps.load_bars(symbol, _bars(symbol, last_close))
    acct = BacktestAccount(account_id, ps, cfg)
    wire_backtest_seams().register_account(account_id, acct)
    ps.set_clock(D1)
    return acct, ctx, ps


def _open_entry(acct, symbol="AAPL", qty=10, side=None):
    """Submit a MARKET entry (auto-creates the transaction); return (broker_id, txn_id)."""
    from ba2_common.core.models import TradingOrder
    from ba2_common.core.types import OrderType, OrderStatus, OrderDirection

    side = side or OrderDirection.BUY
    o = TradingOrder(
        account_id=acct.id,
        symbol=symbol,
        quantity=qty,
        side=side,
        order_type=OrderType.MARKET,
        status=OrderStatus.NEW,
        comment="rt-entry",
    )
    acct.submit_order(o)  # auto-creates a WAITING transaction
    persisted = acct.get_order(o.broker_order_id)
    return o.broker_order_id, persisted.transaction_id


def _attach_order(acct, txn_id, side, qty=10, symbol="AAPL", depends_on=None,
                  order_type=None, limit=None, stop=None):
    """Persist a sibling order on an existing transaction; return its broker_order_id."""
    from ba2_common.core.models import TradingOrder
    from ba2_common.core.types import OrderType, OrderStatus
    from ba2_common.core.db import add_instance

    bid = acct._next_broker_id()
    o = TradingOrder(
        account_id=acct.id,
        symbol=symbol,
        quantity=qty,
        side=side,
        order_type=order_type or OrderType.MARKET,
        limit_price=limit,
        stop_price=stop,
        transaction_id=txn_id,
        status=OrderStatus.NEW,
        depends_on_order=depends_on,
        broker_order_id=bid,
        comment="rt-leg",
    )
    add_instance(o)
    return bid


def _fill(acct, broker_id, px, as_of):
    """Fill a working order at ``px`` on bar ``as_of`` (stamps _fill_dates + ledger)."""
    o = acct.get_order(broker_id)
    acct._apply_fill(o, px, as_of)


# ---------------------------------------------------------------------------
# Plain-sell closer (the FactorRanker rebalance/stop case) — the bug
# ---------------------------------------------------------------------------
def test_plain_market_sell_closes_as_round_trip_not_open_at_end():
    """A LONG closed by a PLAIN market sell (depends_on_order None, opposite side) yields
    ONE round-trip with entry=buy / exit=sell / exit_reason 'exit' — NOT 'open_at_end'."""
    from ba2_common.core.types import OrderDirection

    acct, ctx, ps = _acct(account_id=101)
    try:
        buy_bid, txn = _open_entry(acct, qty=10, side=OrderDirection.BUY)
        _fill(acct, buy_bid, 100.0, D2)  # entry buy fills @100

        sell_bid = _attach_order(acct, txn, OrderDirection.SELL, qty=10)  # plain closer
        _fill(acct, sell_bid, 120.0, D3)  # closing sell fills @120

        rts = acct.get_round_trip_trades()
        assert len(rts) == 1
        t = rts[0]
        assert t["exit_reason"] == "exit"
        assert t["exit_reason"] != "open_at_end"
        assert t["direction"] == "buy"
        assert t["entry_price"] == pytest.approx(100.0)  # the BUY, not the sell
        assert t["exit_price"] == pytest.approx(120.0)   # the plain SELL
        assert t["size"] == pytest.approx(10.0)
        assert t["pnl"] == pytest.approx(200.0)  # (120-100)*10 long, 0 commission
        assert t["pnl"] > 0
    finally:
        ctx.__exit__(None, None, None)


def test_plain_sell_at_loss_counts_as_losing_trade():
    """Bought then sold LOWER via a plain sell -> pnl < 0 so win_rate reflects the loss."""
    from ba2_common.core.types import OrderDirection

    acct, ctx, ps = _acct(account_id=102)
    try:
        buy_bid, txn = _open_entry(acct, qty=10, side=OrderDirection.BUY)
        _fill(acct, buy_bid, 100.0, D2)

        sell_bid = _attach_order(acct, txn, OrderDirection.SELL, qty=10)
        _fill(acct, sell_bid, 80.0, D3)  # sold at a loss

        rts = acct.get_round_trip_trades()
        assert len(rts) == 1
        t = rts[0]
        assert t["exit_reason"] == "exit"
        assert t["entry_price"] == pytest.approx(100.0)
        assert t["exit_price"] == pytest.approx(80.0)
        assert t["pnl"] == pytest.approx(-200.0)
        assert t["pnl"] < 0  # a losing trade
    finally:
        ctx.__exit__(None, None, None)


def test_two_buys_then_one_sell_weighted_avg_entry():
    """ADD case: two buys (10 @100, 10 @110) then a single sell of 20 -> entry_px is the
    qty-weighted average of the two buys (105); exit is the sell."""
    from ba2_common.core.types import OrderDirection

    acct, ctx, ps = _acct(account_id=103)
    try:
        buy1_bid, txn = _open_entry(acct, qty=10, side=OrderDirection.BUY)
        _fill(acct, buy1_bid, 100.0, D2)  # first buy @100

        buy2_bid = _attach_order(acct, txn, OrderDirection.BUY, qty=10)  # rebalance ADD
        _fill(acct, buy2_bid, 110.0, D3)  # second buy @110

        sell_bid = _attach_order(acct, txn, OrderDirection.SELL, qty=20)
        _fill(acct, sell_bid, 130.0, D4)  # close all 20 @130

        rts = acct.get_round_trip_trades()
        assert len(rts) == 1
        t = rts[0]
        assert t["direction"] == "buy"
        assert t["entry_price"] == pytest.approx(105.0)  # (10*100 + 10*110)/20
        assert t["exit_price"] == pytest.approx(130.0)
        assert t["size"] == pytest.approx(20.0)
        assert t["pnl"] == pytest.approx((130.0 - 105.0) * 20.0)  # +500
        assert t["exit_reason"] == "exit"
    finally:
        ctx.__exit__(None, None, None)


# ---------------------------------------------------------------------------
# REGRESSION: dependent OCO/TP/SL legs must still pair + classify as before
# ---------------------------------------------------------------------------
def test_regression_tp_leg_closes_as_take_profit():
    """A dependent SELL_LIMIT (TP) leg still pairs as the exit with exit_reason
    'take_profit' and the same realised pnl as before the side-based rewrite."""
    from ba2_common.core.types import OrderDirection, OrderType

    acct, ctx, ps = _acct(account_id=104)
    try:
        buy_bid, txn = _open_entry(acct, qty=10, side=OrderDirection.BUY)
        _fill(acct, buy_bid, 100.0, D2)
        entry_db_id = acct.get_order(buy_bid).id

        tp_bid = _attach_order(
            acct, txn, OrderDirection.SELL, qty=10,
            depends_on=entry_db_id, order_type=OrderType.SELL_LIMIT, limit=130.0,
        )
        _fill(acct, tp_bid, 130.0, D3)  # TP fills @ limit

        rts = acct.get_round_trip_trades()
        assert len(rts) == 1
        t = rts[0]
        assert t["exit_reason"] == "take_profit"
        assert t["direction"] == "buy"
        assert t["entry_price"] == pytest.approx(100.0)
        assert t["exit_price"] == pytest.approx(130.0)
        assert t["pnl"] == pytest.approx(300.0)
    finally:
        ctx.__exit__(None, None, None)


def test_regression_sl_leg_closes_as_stop_loss():
    """A dependent SELL_STOP (SL) leg still classifies as 'stop_loss'."""
    from ba2_common.core.types import OrderDirection, OrderType

    acct, ctx, ps = _acct(account_id=105)
    try:
        buy_bid, txn = _open_entry(acct, qty=10, side=OrderDirection.BUY)
        _fill(acct, buy_bid, 100.0, D2)
        entry_db_id = acct.get_order(buy_bid).id

        sl_bid = _attach_order(
            acct, txn, OrderDirection.SELL, qty=10,
            depends_on=entry_db_id, order_type=OrderType.SELL_STOP, stop=90.0,
        )
        _fill(acct, sl_bid, 90.0, D3)

        rts = acct.get_round_trip_trades()
        assert len(rts) == 1
        t = rts[0]
        assert t["exit_reason"] == "stop_loss"
        assert t["entry_price"] == pytest.approx(100.0)
        assert t["exit_price"] == pytest.approx(90.0)
        assert t["pnl"] == pytest.approx(-100.0)
    finally:
        ctx.__exit__(None, None, None)


# ---------------------------------------------------------------------------
# Still-open transaction must KEEP the open_at_end mark-to-market branch
# ---------------------------------------------------------------------------
def test_open_transaction_marks_to_market_open_at_end():
    """An entry with NO closing fill is marked-to-market at the last price
    (exit_reason 'open_at_end'), with entry = the buy."""
    from ba2_common.core.types import OrderDirection

    acct, ctx, ps = _acct(account_id=106, last_close=150.0)
    try:
        buy_bid, txn = _open_entry(acct, qty=10, side=OrderDirection.BUY)
        _fill(acct, buy_bid, 100.0, D2)
        ps.set_clock(D4)  # last bar -> close 150

        rts = acct.get_round_trip_trades()
        assert len(rts) == 1
        t = rts[0]
        assert t["exit_reason"] == "open_at_end"
        assert t["entry_price"] == pytest.approx(100.0)
        assert t["exit_price"] == pytest.approx(150.0)  # marked to last close
    finally:
        ctx.__exit__(None, None, None)
