"""``BacktestAccount`` — a simulated broker implementing the live ``AccountInterface``.

The whole point of the design (§5) is to reuse the live decision/sizing/order code
unchanged: the real expert -> ``Recommendation`` -> ``TradeConditions`` -> classic
``TradeRiskManagement`` -> ``position_sizing`` -> ``account.submit_order()`` path runs
against THIS simulated broker. ``BacktestAccount`` therefore inherits ALL of
``AccountInterface``'s concrete orchestration (``submit_order`` validation/persistence,
``refresh_transactions`` lifecycle, ``close_transaction*``, the ``_validate_*`` helpers,
wash-trade locks) and only implements the broker-specific abstracts.

Equities-only v1: it inherits ``AccountInterface`` (NOT ``OptionsAccountInterface``).

This module (Phase 2 Task 2) implements:
  * the in-memory ledger (cash / signed positions / equity snapshots),
  * the 12 ``ReadOnlyAccountInterface`` abstracts + ``get_settings_definitions``,
  * the price-cache override (the critical gotcha — see ``get_instrument_current_price``),
  * ``snapshot_equity`` (the engine calls it per bar to build the equity curve).

The 6 trading abstracts (``_submit_order_impl``, ``cancel_order``, ``modify_order``,
``adjust_tp``/``adjust_sl``/``adjust_tp_sl``) plus the ``refresh_orders`` FILL ENGINE
implement the full per-bar fill / TP-SL / OCO engine (Phase 2 Task 3). All 18 abstracts
are concrete so the class instantiates (``__abstractmethods__`` is empty).

The fill engine (``refresh_orders``) is the heart of the simulator. Each invocation:
  1. ACTIVATES dependent WAITING_TRIGGER legs whose parent order has reached its trigger
     status (the inherited ``submit_order`` stages TP/SL/OCO legs as WAITING_TRIGGER with
     ``depends_on_order``/``depends_order_status_trigger`` exactly like AlpacaAccount);
  2. EVALUATES every working order against the chosen bar — MARKET fills at next-bar
     open (±slippage), LIMIT fills only when the bar's range crosses the limit, STOP
     triggers when the bar's range crosses the stop (then fills at stop ±slippage);
  3. APPLIES fills to the cash/position ledger (commission charged per fill);
  4. CANCELS the OCO sibling when one OCO leg fills (so the transaction closes on the
     first leg and the other does not also execute).

Transaction lifecycle (WAITING->OPENED->CLOSED) is NOT re-implemented here: the inherited
``refresh_transactions`` derives it from order states. The engine calls
``refresh_orders()`` then ``refresh_transactions()`` per bar. ``refresh_transactions``
recognises a TP/SL close via ``"OCO-" in comment`` or ``order_type == OrderType.OCO`` on a
filled dependent leg, so our legs MUST carry that marker.

Field/enum names verified against the installed ba2_common:
  * TradingOrder cols: id, account_id, symbol, quantity, side (OrderDirection),
    order_type (OrderType), status (OrderStatus), filled_qty, open_price, limit_price,
    stop_price, broker_order_id, depends_on_order, depends_order_status_trigger,
    transaction_id, comment, created_at, ...
  * OrderStatus has classmethods get_terminal_statuses()/get_executed_statuses()/
    get_active_statuses()/get_unfilled_statuses() (NOT get_open_order_statuses — that one
    does not exist). WAITING_TRIGGER is in get_active_statuses() but NOT get_unfilled_statuses().
  * AccountDefinition cols: id, name, provider, description.
  * Transaction has NO entry_order_id column; the market-entry order is the TradingOrder
    with transaction_id == txn.id AND depends_on_order IS NULL.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ba2_common.core.interfaces.AccountInterface import AccountInterface
from ba2_common.core.models import TradingOrder, Transaction
from ba2_common.core.types import OrderStatus, OrderType, OrderDirection, OrderOpenType
from ba2_common.core.db import get_db, get_instance, add_instance, update_instance

from .price_source import AsOfPriceSource


class _AttrDict(dict):
    """A dict whose keys are also attribute-accessible.

    Needed because the inherited ``_validate_position_size_limits`` reads
    ``account_info.equity`` (attribute access) while other callers use
    ``account_info["equity"]``. Supporting both keeps the inherited code working.
    """

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


@dataclass
class _Position:
    """In-memory ledger position. ``qty`` is signed: positive long, negative short."""

    symbol: str
    qty: float = 0.0
    avg_price: float = 0.0
    realized_pl: float = 0.0


class BacktestAccount(AccountInterface):
    """Simulated broker for daily multi-asset backtests."""

    # Class-level capability flags (mirror the live account contract).
    supports_trading = True
    supports_options = False

    def __init__(self, id: int, price_source: AsOfPriceSource, settings: Dict[str, Any]):
        # ReadOnlyAccountInterface.__init__ registers self.id in the _GLOBAL_PRICE_CACHE.
        super().__init__(id)
        self._price = price_source
        # Resolved config dict (validated fail-early by the engine before the run):
        #   starting_cash, commission_per_trade, slippage_bps, fill_model.
        self._cfg = settings
        self._cash: float = float(settings["starting_cash"])
        # symbol -> signed-position ledger.
        self._positions: Dict[str, _Position] = {}
        # The equity curve: one snapshot per simulated bar (engine appends via snapshot_equity).
        self._equity_snapshots: List[Dict[str, Any]] = []
        # Monotonic synthetic broker-order-id counter.
        self._broker_seq = 0
        # order-id -> SIMULATED fill date (the virtual bar an order filled on). The
        # TradingOrder row's ``created_at`` is stamped by the DB with wall-clock
        # ``datetime.now()`` at row creation, which is NON-deterministic across runs; the
        # filled-trade history must use the SIMULATED clock instead so two identical runs
        # produce a byte-identical trade list (the reproducibility gate). Populated in
        # ``_apply_fill`` and read by ``_order_to_trade``.
        self._fill_dates: Dict[int, datetime] = {}

    # ======================================================================
    # Settings
    # ======================================================================
    @classmethod
    def get_settings_definitions(cls) -> Dict[str, Any]:
        """Account settings schema. No defaults: the engine validates fail-early.

        (``get_settings_definitions`` is resolved-non-abstract on the MRO so it does not
        block instantiation, but we implement it for a proper settings surface.)
        """
        return {
            "starting_cash": {
                "type": "float",
                "required": True,
                "description": "Initial simulated cash",
            },
            "commission_per_trade": {
                "type": "float",
                "required": True,
                "description": "Flat $ commission applied per fill",
            },
            "slippage_bps": {
                "type": "float",
                "required": True,
                "description": "Slippage in basis points applied to market/stop fills (worsening)",
            },
            "fill_model": {
                "type": "str",
                "required": True,
                "description": "Fill model: 'next_bar_open' (default) | 'same_bar_close'",
            },
        }

    # ======================================================================
    # Ledger internals
    # ======================================================================
    def _open_positions_mtm(self) -> float:
        """Mark-to-market value of all open positions at the current bar's close.

        Signed value (long positions positive, short positions negative). A symbol
        with no price at the current bar contributes 0 (it cannot be valued today).
        """
        total = 0.0
        for p in self._positions.values():
            if p.qty == 0:
                continue
            px = self._price.close_at(p.symbol)
            if px is not None:
                total += p.qty * px
        return total

    def equity(self) -> float:
        """Net liquidating value = cash + mark-to-market of open positions."""
        return self._cash + self._open_positions_mtm()

    def snapshot_equity(self, as_of: datetime) -> Dict[str, Any]:
        """Append an equity-curve snapshot for ``as_of`` and return it.

        The engine calls this once per bar (after fills/transactions are rolled). Keys
        match ``ReadOnlyAccountInterface.get_balance_history``'s documented contract
        (date / net_liquidating_value / cash_balance / equity_value).
        """
        equity_value = self._open_positions_mtm()
        nlv = self._cash + equity_value
        snap = {
            "date": as_of,
            "net_liquidating_value": nlv,
            "cash_balance": self._cash,
            "equity_value": equity_value,
        }
        self._equity_snapshots.append(snap)
        return snap

    def _update_position(self, symbol: str, signed_qty: float, fill_px: float) -> None:
        """Apply a signed fill to the ledger.

        Increasing (same-sign) exposure updates the weighted-average price; reducing or
        flipping realises P&L on the closed portion. ``signed_qty`` is +buy / -sell.
        """
        pos = self._positions.get(symbol)
        if pos is None:
            pos = _Position(symbol=symbol)
            self._positions[symbol] = pos

        old_qty = pos.qty
        new_qty = old_qty + signed_qty

        if old_qty == 0 or (old_qty > 0) == (signed_qty > 0):
            # Opening or increasing in the same direction -> weighted-average price.
            total_cost = pos.avg_price * abs(old_qty) + fill_px * abs(signed_qty)
            denom = abs(new_qty)
            pos.avg_price = (total_cost / denom) if denom > 0 else 0.0
        else:
            # Reducing / closing / flipping -> realise P&L on the closed quantity.
            closing_qty = min(abs(signed_qty), abs(old_qty))
            direction = 1.0 if old_qty > 0 else -1.0
            pos.realized_pl += (fill_px - pos.avg_price) * closing_qty * direction
            if abs(signed_qty) > abs(old_qty):
                # Flipped through zero -> the remainder opens a new position at fill price.
                pos.avg_price = fill_px
            # If fully or partially closed without flipping, avg_price is unchanged.

        pos.qty = new_qty
        if pos.qty == 0:
            pos.avg_price = 0.0

    # ======================================================================
    # ReadOnlyAccountInterface abstracts (12)
    # ======================================================================
    def get_balance(self) -> Optional[float]:
        """Current cash balance (the simulated cash ledger)."""
        return self._cash

    def get_account_info(self) -> Dict[str, Any]:
        """Account info dict; exposes ``.equity`` (read by _validate_position_size_limits)."""
        eq = self.equity()
        return _AttrDict(
            {
                "balance": self._cash,
                "cash": self._cash,
                "equity": eq,
                "buying_power": max(self._cash, 0.0),
            }
        )

    def get_positions(self) -> Any:
        """List of open ledger positions (non-zero qty)."""
        out: List[_AttrDict] = []
        for p in self._positions.values():
            if p.qty == 0:
                continue
            cur = self._price.close_at(p.symbol)
            out.append(
                _AttrDict(
                    {
                        "symbol": p.symbol,
                        "qty": p.qty,
                        "quantity": p.qty,
                        "avg_price": p.avg_price,
                        "average_price": p.avg_price,
                        "current_price": cur,
                        "unrealized_pl": (None if cur is None else (cur - p.avg_price) * p.qty),
                        "realized_pl": p.realized_pl,
                    }
                )
            )
        return out

    def get_orders(self, status: Optional[Any] = None) -> Any:
        """Query ``TradingOrder`` rows for this account from the backtest DB.

        ``status`` filters by OrderStatus when provided (ALL / None returns everything).
        """
        from sqlmodel import select, Session

        with Session(get_db().bind) as session:
            stmt = select(TradingOrder).where(TradingOrder.account_id == self.id)
            if status is not None and status != OrderStatus.ALL:
                stmt = stmt.where(TradingOrder.status == status)
            return list(session.exec(stmt).all())

    def get_order(self, order_id: str) -> Any:
        """Look up an order by broker_order_id, then by numeric PK as a fallback."""
        from sqlmodel import select, Session

        with Session(get_db().bind) as session:
            row = session.exec(
                select(TradingOrder).where(TradingOrder.broker_order_id == str(order_id))
            ).first()
            if row is None and str(order_id).isdigit():
                row = session.get(TradingOrder, int(order_id))
            return row

    def symbols_exist(self, symbols: List[str]) -> Dict[str, bool]:
        """A symbol "exists" iff the backtest price store has bars for it."""
        return {s: self._price.has_symbol(s) for s in symbols}

    def _get_instrument_current_price_impl(self, symbol_or_symbols, price_type: str = "bid"):
        """The time machine: the as-of bar's close for the symbol(s).

        Single symbol -> float (raises if unavailable, per the live no-fallback rule).
        List -> {symbol: price-or-None}.
        """
        if isinstance(symbol_or_symbols, (list, tuple, set)):
            return {s: self._price.close_at(s) for s in symbol_or_symbols}
        px = self._price.close_at(symbol_or_symbols)
        if px is None:
            raise ValueError(
                f"No backtest price for {symbol_or_symbols} at {self._price.now()}"
            )
        return px

    def refresh_positions(self) -> bool:
        """No-op: the ledger is local and always current. Returns True."""
        return True

    def refresh_orders(self) -> bool:
        """Per-bar fill engine (THE core of the simulator).

        Called by the engine once per simulated bar (after ``set_clock``). Steps:

          1. ACTIVATE dependent WAITING_TRIGGER legs whose parent reached its trigger
             status — they become ACCEPTED (live) so they can fill on later bars.
          2. EVALUATE every working order against the chosen bar and FILL it if triggered:
             MARKET -> next-bar open (±slippage); LIMIT -> only if the bar crosses the
             limit; STOP -> only if the bar crosses the stop (then fills at stop ±slippage).
          3. CANCEL the OCO sibling when one OCO/TP/SL leg fills (first-leg-wins close).

        Activation runs first so a leg whose parent filled on THIS same bar (a same-bar
        MARKET entry) can be evaluated against the next bar on the following call — never
        on the entry bar (no look-ahead within a bar). Returns True.
        """
        as_of = self._price.now()
        self._activate_triggered_dependents()

        active = OrderStatus.get_active_statuses()
        # Re-read AFTER activation so newly-activated legs are seen this bar.
        working = [
            o
            for o in self.get_orders()
            if o.status in active and o.status != OrderStatus.WAITING_TRIGGER
        ]
        for o in working:
            fill_px = self._evaluate_fill(o, as_of)
            if fill_px is None:
                continue
            self._apply_fill(o, fill_px, as_of)
            self._cancel_oco_sibling(o)
        return True

    def _activate_triggered_dependents(self) -> None:
        """Promote WAITING_TRIGGER legs to ACCEPTED once their parent hits the trigger.

        A leg created by ``adjust_tp``/``adjust_sl``/``adjust_tp_sl`` waits with
        ``depends_on_order`` = the entry order id and ``depends_order_status_trigger`` =
        FILLED. When the parent reaches that status the leg goes live (ACCEPTED) so the
        fill engine evaluates it. Legs with no parent / unmet trigger are left waiting.
        """
        waiting = [o for o in self.get_orders() if o.status == OrderStatus.WAITING_TRIGGER]
        for leg in waiting:
            if leg.depends_on_order is None:
                continue
            parent = get_instance(TradingOrder, leg.depends_on_order)
            if parent is None:
                continue
            trigger = leg.depends_order_status_trigger or OrderStatus.FILLED
            if parent.status == trigger:
                leg.status = OrderStatus.ACCEPTED
                update_instance(leg)

    def refresh_transactions(self) -> bool:
        """Roll order state into transactions, then fix CLOSED transactions' ``close_date``.

        The inherited lifecycle closes a transaction via ``close_transaction_with_logging``,
        which stamps ``close_date = datetime.now(timezone.utc)`` (WALL clock). In a backtest the
        simulated clock is years off wall time, so a wall-clock close_date would corrupt any
        as-of date math (e.g. the days-since-last-close cooldown condition). After the inherited
        roll we re-stamp the ``close_date`` of every transaction CLOSED on THIS bar to the
        simulated fill bar of its closing leg (falling back to the current simulated bar).
        """
        before = {t.id for t in self._closed_transactions()}
        ok = super().refresh_transactions()
        sim_now = self._price.now()
        for txn in self._closed_transactions():
            if txn.id in before:
                continue  # already closed on an earlier bar — leave its simulated close_date.
            txn.close_date = self._closing_fill_date(txn) or sim_now
            if txn.open_date is None:
                entry = self._entry_order_for_transaction(txn)
                if entry is not None and entry.id is not None:
                    txn.open_date = self._fill_dates.get(entry.id)
            update_instance(txn)
        return ok

    def _closed_transactions(self) -> List[Transaction]:
        """All CLOSED transactions in the per-run trading DB (single-account backtest)."""
        from sqlmodel import select, Session
        from ba2_common.core.types import TransactionStatus

        with Session(get_db().bind) as session:
            return list(
                session.exec(
                    select(Transaction).where(Transaction.status == TransactionStatus.CLOSED)
                ).all()
            )

    def _closing_fill_date(self, transaction: Transaction) -> Optional[datetime]:
        """The simulated fill bar of the transaction's filled closing leg, if any."""
        executed = OrderStatus.get_executed_statuses()
        for o in self.get_orders():
            if (
                o.transaction_id == transaction.id
                and o.depends_on_order is not None
                and o.status in executed
                and o.id is not None
            ):
                dt = self._fill_dates.get(o.id)
                if dt is not None:
                    return dt
        return None

    def get_dividends(
        self,
        symbol: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[Dict]:
        """v1: no dividend simulation. Returns []."""
        return []

    def get_filled_trades(
        self,
        symbol: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[Dict]:
        """Filled-trade history derived from executed ``TradingOrder`` rows."""
        executed = OrderStatus.get_executed_statuses()
        trades: List[Dict] = []
        for o in self.get_orders():
            if o.status not in executed:
                continue
            qty = o.filled_qty if o.filled_qty else o.quantity
            if not qty:
                continue
            if symbol is not None and o.symbol != symbol:
                continue
            trades.append(self._order_to_trade(o, qty))
        return trades

    def get_round_trip_trades(self) -> List[Dict[str, Any]]:
        """Pair opening fills with their closing fills into round-trip trades with realised P&L.

        ``get_filled_trades`` returns one row per FILLED order (opens AND closers separately),
        which has no round-trip P&L — so trade-quality metrics (win_rate, profit_factor,
        expectancy, best/worst trade) are all zero. This method instead groups FILLED orders by
        their ``transaction_id`` and produces ONE row per transaction.

        Entries vs exits are classified by SIDE, not by ``depends_on_order``: the OPENING order
        is the EARLIEST-filled order in the transaction (you cannot close before you open) and
        its side is the ``opening_side``. Then:

          * ENTRIES = same-side fills (the open + any rebalance ADDs);
          * EXITS   = opposite-side fills — this covers BOTH plain market sells (FactorRanker
            rebalance/stop closers, ``depends_on_order IS NULL``) AND dependent TP/SL/OCO legs.
            Classifying by ``depends_on_order`` instead would mis-read a plain closing sell as
            an entry and drop the transaction into the ``open_at_end`` branch with garbage.
          * entry/exit price = quantity-weighted average ``open_price`` over each side; ``size``
            is the realised (exit) quantity; pnl = (exit_px - entry_px) * size * dir - commissions.
          * a transaction with NO exit fill is still OPEN at run end -> marked-to-market at the
            symbol's last available price (``exit_reason='open_at_end'``) so its unrealised P&L
            is counted (otherwise a run that ends mid-trade would understate performance).

        ``_exit_reason`` (called on the LATEST exit fill) returns ``"exit"`` for a plain market
        sell (no limit/stop), and ``take_profit``/``stop_loss`` for an OCO/TP/SL leg by the
        nearest price level. This is an APPROXIMATION for scaled add/reduce (one weighted-avg
        round-trip row per transaction) and EXACT for the dominant buy-once / sell-once case.

        Rows carry the field names ``results._trade_row`` maps (entry_time/exit_time/direction/
        entry_price/exit_price/size/pnl/pnl_pct/bars_held/exit_reason).
        """
        executed = OrderStatus.get_executed_statuses()
        commission = float(self._cfg["commission_per_trade"])

        def _fill_key(o):
            """Sort key for fill ordering.

            Order by simulated fill date; when a fill date is missing, fall back to ``o.id``
            (a monotonic insertion counter). The first tuple element separates rows that HAVE a
            fill date (0) from those that do not (1) so the two cases never compare a datetime
            against an id, while keeping ``id`` as the stable tiebreaker within each group.
            """
            fd = self._fill_dates.get(o.id) if o.id is not None else None
            oid = o.id or 0
            return (0, fd, oid) if fd is not None else (1, oid, oid)

        # Group FILLED orders (with a usable price) by transaction.
        by_txn: Dict[int, List[Any]] = {}
        for o in self.get_orders():
            if o.transaction_id is None:
                continue
            if o.status not in executed or not (o.filled_qty or o.quantity):
                continue
            if not o.open_price:
                continue
            by_txn.setdefault(o.transaction_id, []).append(o)

        trades: List[Dict[str, Any]] = []
        for txn_id, orders in by_txn.items():
            if not orders:
                continue
            # The opening order is the earliest-filled one; its side opens the position.
            orders_by_fill = sorted(orders, key=_fill_key)
            opening = orders_by_fill[0]
            opening_side = opening.side
            entries = [o for o in orders if o.side == opening_side]
            exits = [o for o in orders if o.side != opening_side]
            if not entries:
                continue

            def _wavg(group):
                """(quantity-weighted avg open_price, total qty) over a group of fills."""
                tot_qty = sum(abs(float(o.filled_qty or o.quantity or 0.0)) for o in group)
                if tot_qty <= 0:
                    return None, 0.0
                wsum = sum(
                    float(o.open_price) * abs(float(o.filled_qty or o.quantity or 0.0))
                    for o in group
                )
                return wsum / tot_qty, tot_qty

            entry_px, entry_qty = _wavg(entries)
            if entry_px is None or entry_qty <= 0:
                continue
            is_long = opening_side == OrderDirection.BUY
            direction = 1.0 if is_long else -1.0
            entry_dt = min(
                (self._fill_dates.get(o.id) for o in entries if self._fill_dates.get(o.id) is not None),
                default=None,
            )

            if exits:
                exit_px, exit_qty = _wavg(exits)
                size = exit_qty
                exits_by_fill = sorted(exits, key=_fill_key)
                last_exit_fill = exits_by_fill[-1]
                exit_dt = max(
                    (self._fill_dates.get(o.id) for o in exits if self._fill_dates.get(o.id) is not None),
                    default=None,
                )
                exit_reason = self._exit_reason(last_exit_fill, exit_px)
                comm = commission * 2.0
            else:
                # Still open at run end: mark-to-market at the last available price.
                size = entry_qty
                exit_px = self._price.close_at(opening.symbol)
                if exit_px is None:
                    exit_px = entry_px  # no closing price -> flat (counts as a near-zero trade)
                exit_dt = self._price.now()
                exit_reason = "open_at_end"
                comm = commission

            gross = (exit_px - entry_px) * size * direction
            pnl = gross - comm
            pnl_pct = ((exit_px / entry_px - 1.0) * 100.0 * direction) if entry_px else 0.0
            bars_held = self._bars_between(entry_dt, exit_dt)
            trades.append(
                {
                    "symbol": opening.symbol,
                    "entry_time": entry_dt,
                    "exit_time": exit_dt,
                    "direction": "buy" if is_long else "sell",
                    "entry_price": entry_px,
                    "exit_price": exit_px,
                    "size": size,
                    "pnl": pnl,
                    "pnl_pct": pnl_pct,
                    "bars_held": bars_held,
                    "exit_reason": exit_reason,
                }
            )
        # Deterministic order: by entry time then symbol.
        trades.sort(key=lambda t: (str(t["entry_time"]), t["symbol"]))
        return trades

    def _exit_reason(self, exit_order, fill_px: float) -> str:
        """Classify an OCO/TP/SL exit fill as take_profit / stop_loss by nearest price level."""
        tp = exit_order.limit_price
        sl = exit_order.stop_price
        if tp is not None and sl is not None:
            return "take_profit" if abs(fill_px - tp) <= abs(fill_px - sl) else "stop_loss"
        if tp is not None:
            return "take_profit"
        if sl is not None:
            return "stop_loss"
        return "exit"

    def _bars_between(self, start: Optional[datetime], end: Optional[datetime]) -> int:
        """Number of equity-curve bars between two simulated timestamps (>=0)."""
        if start is None or end is None:
            return 0
        n = 0
        for s in self._equity_snapshots:
            d = s["date"]
            if start <= d <= end:
                n += 1
        return max(n - 1, 0)

    def get_balance_history(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[Dict]:
        """The equity curve: the per-bar snapshots appended by ``snapshot_equity``."""
        snaps = list(self._equity_snapshots)
        if start_date is not None:
            snaps = [s for s in snaps if s["date"] >= start_date]
        if end_date is not None:
            snaps = [s for s in snaps if s["date"] <= end_date]
        return snaps

    # ======================================================================
    # The critical gotcha: defeat the inherited wall-clock price cache
    # ======================================================================
    def get_instrument_current_price(self, symbol_or_symbols, price_type: str = "bid"):
        """OVERRIDE: bypass the inherited _GLOBAL_PRICE_CACHE (wall-clock TTL).

        The virtual backtest clock moves far faster than wall time, so the inherited
        TTL cache would treat a price fetched on virtual day N as "fresh" on day N+5,
        leaking stale/look-ahead prices across bars. We delegate straight to the impl
        (the engine ALSO pops the per-account cache each bar as belt-and-braces).
        """
        return self._get_instrument_current_price_impl(symbol_or_symbols, price_type=price_type)

    # ======================================================================
    # Trading abstracts — baseline; expanded into the full engine in Task 3
    # ======================================================================
    def _next_broker_id(self) -> str:
        self._broker_seq += 1
        return f"BT-{self.id}-{self._broker_seq}"

    def _submit_order_impl(
        self,
        trading_order: TradingOrder,
        tp_price: Optional[float] = None,
        sl_price: Optional[float] = None,
        is_closing_order: bool = False,
    ) -> Any:
        """Called by the INHERITED ``submit_order`` after validation/persistence.

        Assign a synthetic broker id and mark the order working; the per-bar fill engine
        (``refresh_orders``) decides when/whether it fills. We do NOT reimplement
        ``submit_order`` (it is inherited and exercises the real validation path).

        Idempotency guard (mirrors AlpacaAccount): an order that already carries a
        broker_order_id was already "sent" — never re-stamp it.

        A WAITING_TRIGGER dependent leg keeps its WAITING_TRIGGER status (it must wait for
        its parent to reach the trigger status before becoming live); everything else
        becomes ACCEPTED (working / active per get_active_statuses()).
        """
        if trading_order.broker_order_id:
            return trading_order
        trading_order.broker_order_id = self._next_broker_id()
        if trading_order.status != OrderStatus.WAITING_TRIGGER:
            trading_order.status = OrderStatus.ACCEPTED
        update_instance(trading_order)
        return trading_order

    def cancel_order(self, order_id: str) -> Any:
        """Cancel a working order (reserved cash/position is notional-only in this sim)."""
        o = self.get_order(order_id)
        if o is None:
            return None
        o.status = OrderStatus.CANCELED
        update_instance(o)
        return o

    def modify_order(self, order_id: str) -> Any:
        """In-place pre-fill edit of a working order.

        The live ``modify_order`` signature is ``modify_order(self, order_id)`` (no
        trading_order param) — the caller mutates the order row, then calls this to
        "push" the change to the broker. In the sim there is no broker round-trip, so we
        simply re-persist the (non-terminal) order. A terminal order cannot be modified.
        """
        o = self.get_order(order_id)
        if o is None or o.status in OrderStatus.get_terminal_statuses():
            return None
        update_instance(o)
        return o

    def adjust_tp(self, transaction: Transaction, new_tp_price: float, source: str = "") -> bool:
        """Create/replace a TP leg for a transaction.

        TP for a LONG (BUY) transaction is a SELL_LIMIT above entry; for a SHORT (SELL)
        transaction it is a BUY_LIMIT below entry. The leg is created WAITING_TRIGGER on
        the entry order's FILL (mirrors AlpacaAccount). Returns False if the entry order
        cannot be found or the price is invalid.
        """
        if not new_tp_price or new_tp_price <= 0:
            return False
        entry = self._entry_order_for_transaction(transaction)
        if entry is None:
            return False
        is_long = entry.side == OrderDirection.BUY
        leg_type = OrderType.SELL_LIMIT if is_long else OrderType.BUY_LIMIT
        self._replace_leg(transaction, entry, leg="TP", order_type=leg_type,
                          limit_price=new_tp_price, stop_price=None, source=source)
        transaction.take_profit = new_tp_price
        update_instance(transaction)
        return True

    def adjust_sl(self, transaction: Transaction, new_sl_price: float, source: str = "") -> bool:
        """Create/replace an SL leg for a transaction.

        SL for a LONG (BUY) transaction is a SELL_STOP below entry; for a SHORT (SELL)
        transaction it is a BUY_STOP above entry. WAITING_TRIGGER on the entry's FILL.
        """
        if not new_sl_price or new_sl_price <= 0:
            return False
        entry = self._entry_order_for_transaction(transaction)
        if entry is None:
            return False
        is_long = entry.side == OrderDirection.BUY
        leg_type = OrderType.SELL_STOP if is_long else OrderType.BUY_STOP
        self._replace_leg(transaction, entry, leg="SL", order_type=leg_type,
                          limit_price=None, stop_price=new_sl_price, source=source)
        transaction.stop_loss = new_sl_price
        update_instance(transaction)
        return True

    def adjust_tp_sl(
        self,
        transaction: Transaction,
        new_tp_price: Optional[float] = None,
        new_sl_price: Optional[float] = None,
        source: str = "",
    ) -> bool:
        """Set a paired TP+SL as an OCO bracket (one-cancels-other).

        When BOTH prices are given we create a single ``OrderType.OCO`` leg carrying both
        ``limit_price`` (TP) and ``stop_price`` (SL); the fill engine fills it at whichever
        side the bar crosses first and ``refresh_transactions`` recognises the close via
        the ``OrderType.OCO`` / ``"OCO-"`` marker. When only one price is given we fall
        back to a single TP or SL leg.
        """
        if new_tp_price is not None and new_sl_price is not None:
            if new_tp_price <= 0 or new_sl_price <= 0:
                return False
            entry = self._entry_order_for_transaction(transaction)
            if entry is None:
                return False
            self._replace_leg(transaction, entry, leg="TPSL", order_type=OrderType.OCO,
                              limit_price=new_tp_price, stop_price=new_sl_price, source=source)
            transaction.take_profit = new_tp_price
            transaction.stop_loss = new_sl_price
            update_instance(transaction)
            return True

        ok = True
        if new_tp_price is not None:
            ok &= self.adjust_tp(transaction, new_tp_price, source=source)
        if new_sl_price is not None:
            ok &= self.adjust_sl(transaction, new_sl_price, source=source)
        return ok

    # ======================================================================
    # Fill helpers (baseline MARKET path; Task 3 adds LIMIT/STOP/OCO branches)
    # ======================================================================
    def _bar_for_fill(self, order, as_of: datetime) -> Optional[Dict[str, float]]:
        """The bar an order fills against, per the configured fill model."""
        if self._cfg["fill_model"] == "same_bar_close":
            return self._price.bar_at(order.symbol, as_of)
        return self._price.next_bar(order.symbol, as_of)  # default: next_bar_open

    def _slip(self, px: float, side_is_buy: bool) -> float:
        """Apply slippage in the worsening direction (buys up, sells down)."""
        bps = float(self._cfg["slippage_bps"]) / 10_000.0
        return px * (1.0 + bps) if side_is_buy else px * (1.0 - bps)

    def _evaluate_fill(self, order, as_of: datetime) -> Optional[float]:
        """Return the fill price for ``order`` against the chosen bar, or None if untriggered.

        Per-type rules (the bar's [low, high] range is the day's traded range):
          * MARKET            -> fills at the bar's open (or close for same_bar_close),
                                 worsened by slippage.
          * BUY_LIMIT         -> fills at the limit iff bar.low  <= limit (price traded down to it).
          * SELL_LIMIT        -> fills at the limit iff bar.high >= limit (price traded up to it).
          * BUY_STOP          -> triggers iff bar.high >= stop; fills at stop +slippage.
          * SELL_STOP         -> triggers iff bar.low  <= stop; fills at stop -slippage.
          * OCO (TP+SL leg)   -> evaluate TP (limit) and SL (stop) sides; fill the side the
                                 bar crosses (SL preferred when the bar straddles both, the
                                 conservative assumption that the stop hit first).
        """
        bar = self._bar_for_fill(order, as_of)
        if bar is None:
            return None
        ot = order.order_type

        if ot == OrderType.MARKET:
            ref = bar["close"] if self._cfg["fill_model"] == "same_bar_close" else bar["open"]
            return self._slip(ref, order.side == OrderDirection.BUY)

        if ot == OrderType.BUY_LIMIT:
            return order.limit_price if bar["low"] <= order.limit_price else None
        if ot == OrderType.SELL_LIMIT:
            return order.limit_price if bar["high"] >= order.limit_price else None

        if ot == OrderType.BUY_STOP:
            return self._slip(order.stop_price, True) if bar["high"] >= order.stop_price else None
        if ot == OrderType.SELL_STOP:
            return self._slip(order.stop_price, False) if bar["low"] <= order.stop_price else None

        if ot == OrderType.OCO:
            return self._evaluate_oco_fill(order, bar)

        return None

    def _evaluate_oco_fill(self, order, bar: Dict[str, float]) -> Optional[float]:
        """Fill price for an OCO leg (limit_price=TP, stop_price=SL) against ``bar``.

        The OCO closes the position, so its ``side`` is opposite the entry:
          * SELL OCO (closing a LONG):  TP = SELL_LIMIT @ limit (bar.high >= TP),
                                        SL = SELL_STOP  @ stop  (bar.low  <= SL).
          * BUY  OCO (closing a SHORT): TP = BUY_LIMIT  @ limit (bar.low  <= TP),
                                        SL = BUY_STOP   @ stop  (bar.high >= SL).
        When a single bar's range crosses BOTH legs we fill the STOP (loss) side — the
        conservative, no-look-ahead assumption (intrabar order is unknown).
        """
        tp = order.limit_price
        sl = order.stop_price
        is_sell = order.side == OrderDirection.SELL  # closing a long

        if is_sell:
            sl_hit = sl is not None and bar["low"] <= sl
            tp_hit = tp is not None and bar["high"] >= tp
            if sl_hit:
                return self._slip(sl, False)   # SELL_STOP fills at stop -slippage
            if tp_hit:
                return tp                       # SELL_LIMIT fills at limit (no slippage)
            return None
        else:
            sl_hit = sl is not None and bar["high"] >= sl
            tp_hit = tp is not None and bar["low"] <= tp
            if sl_hit:
                return self._slip(sl, True)    # BUY_STOP fills at stop +slippage
            if tp_hit:
                return tp                       # BUY_LIMIT fills at limit
            return None

    def _apply_fill(self, order, fill_px: float, as_of: datetime) -> None:
        """Apply a fill to cash + ledger and mark the order FILLED."""
        qty = float(order.quantity) if order.quantity is not None else 0.0
        signed = qty if order.side == OrderDirection.BUY else -qty
        commission = float(self._cfg["commission_per_trade"])
        # Buying spends cash (signed>0 -> cash decreases); selling adds cash.
        self._cash -= signed * fill_px
        self._cash -= commission
        self._update_position(order.symbol, signed, fill_px)
        order.filled_qty = qty
        order.open_price = fill_px
        order.status = OrderStatus.FILLED
        update_instance(order)
        # Record the SIMULATED fill bar (not wall-clock) so the trade history is deterministic.
        if order.id is not None:
            self._fill_dates[order.id] = as_of

    # ======================================================================
    # TP/SL/OCO leg helpers
    # ======================================================================
    def _entry_order_for_transaction(self, transaction: Transaction) -> Optional[TradingOrder]:
        """The market-entry order of a transaction: transaction_id matches + no parent.

        (Transaction has no entry_order_id column; the entry order is the one with
        ``depends_on_order IS NULL``. If several exist — e.g. scaled entries — the oldest
        is returned so legs depend on the original entry.)
        """
        from sqlmodel import select, Session

        with Session(get_db().bind) as session:
            rows = session.exec(
                select(TradingOrder).where(
                    TradingOrder.transaction_id == transaction.id,
                    TradingOrder.account_id == self.id,
                    TradingOrder.depends_on_order.is_(None),
                )
            ).all()
        if not rows:
            return None
        rows.sort(key=lambda o: (o.created_at or datetime.min.replace(tzinfo=timezone.utc), o.id or 0))
        return rows[0]

    def _existing_legs(self, transaction: Transaction) -> List[TradingOrder]:
        """All non-terminal dependent (TP/SL/OCO) legs for a transaction."""
        terminal = OrderStatus.get_terminal_statuses()
        legs: List[TradingOrder] = []
        for o in self.get_orders():
            if (
                o.transaction_id == transaction.id
                and o.depends_on_order is not None
                and o.status not in terminal
            ):
                legs.append(o)
        return legs

    def _replace_leg(
        self,
        transaction: Transaction,
        entry: TradingOrder,
        leg: str,
        order_type: OrderType,
        limit_price: Optional[float],
        stop_price: Optional[float],
        source: str,
    ) -> TradingOrder:
        """Cancel any existing protective leg(s) and create a fresh WAITING_TRIGGER leg.

        The new leg is the side that CLOSES the position (opposite the entry side), carries
        an ``OCO-`` comment marker + (for paired) ``OrderType.OCO`` so the inherited
        ``refresh_transactions`` recognises a TP/SL close, and depends on the entry order
        reaching FILLED before going live. Quantity is synced to the entry order's quantity.
        """
        # Cancel any existing non-terminal legs (single TP/SL replaced; OCO supersedes both).
        for old in self._existing_legs(transaction):
            old.status = OrderStatus.CANCELED
            update_instance(old)

        close_side = OrderDirection.SELL if entry.side == OrderDirection.BUY else OrderDirection.BUY
        ts = int(datetime.now(timezone.utc).timestamp())
        comment = f"{ts}-OCO-{leg}-[PARENT:{entry.id}/BROKER:{entry.broker_order_id}]"

        leg_order = TradingOrder(
            account_id=self.id,
            symbol=entry.symbol,
            quantity=entry.quantity,
            side=close_side,
            order_type=order_type,
            limit_price=limit_price,
            stop_price=stop_price,
            transaction_id=transaction.id,
            status=OrderStatus.WAITING_TRIGGER,
            depends_on_order=entry.id,
            depends_order_status_trigger=OrderStatus.FILLED,
            open_type=OrderOpenType.AUTOMATIC,
            broker_order_id=self._next_broker_id(),
            expert_recommendation_id=entry.expert_recommendation_id,
            comment=comment,
            created_at=datetime.now(timezone.utc),
        )
        add_instance(leg_order)
        return leg_order

    def _cancel_oco_sibling(self, filled_order) -> None:
        """When an OCO/TP/SL leg fills, cancel the sibling protective leg(s).

        A single ``OrderType.OCO`` leg has both TP+SL internally (no sibling). For the
        separate-TP + separate-SL case, the two legs share the same transaction and
        ``depends_on_order``; filling one cancels the other so the position closes once.
        """
        if filled_order.transaction_id is None or filled_order.depends_on_order is None:
            return
        terminal = OrderStatus.get_terminal_statuses()
        for o in self.get_orders():
            if (
                o.id != filled_order.id
                and o.transaction_id == filled_order.transaction_id
                and o.depends_on_order is not None
                and o.status not in terminal
                and o.status != OrderStatus.FILLED
            ):
                o.status = OrderStatus.CANCELED
                update_instance(o)

    def _order_to_trade(self, order, qty: float) -> Dict[str, Any]:
        """Map a filled ``TradingOrder`` row to the documented filled-trade dict shape.

        ``date`` is the SIMULATED fill bar (from ``_fill_dates``), NOT ``order.created_at``
        (which the DB stamps with wall-clock ``datetime.now()`` and would make the trade
        history non-deterministic run-to-run). Falls back to ``created_at`` only if a fill
        date was not recorded (e.g. an order that fills outside the engine loop in a unit
        test) so the field is never None for a filled order.
        """
        fill_date = self._fill_dates.get(order.id) if order.id is not None else None
        return {
            "symbol": order.symbol,
            "qty": abs(float(qty)),
            "side": order.side.value if order.side else None,
            "date": fill_date if fill_date is not None else order.created_at,
            "price": order.open_price,
        }
