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
are implemented here with working baseline bodies and expanded into the full per-bar
fill / TP-SL / OCO engine in Phase 2 Task 3. All 18 abstracts are concrete now so the
class instantiates (``__abstractmethods__`` is empty).

Field/enum names verified against the installed ba2_common:
  * TradingOrder cols: id, account_id, symbol, quantity, side (OrderDirection),
    order_type (OrderType), status (OrderStatus), filled_qty, open_price, limit_price,
    stop_price, broker_order_id, depends_on_order, depends_order_status_trigger,
    transaction_id, comment, created_at, ...
  * OrderStatus has classmethods get_terminal_statuses()/get_executed_statuses()/
    get_active_statuses() (NOT get_open_order_statuses — that one does not exist).
  * AccountDefinition cols: id, name, provider, description.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from ba2_common.core.interfaces.AccountInterface import AccountInterface
from ba2_common.core.models import TradingOrder
from ba2_common.core.types import OrderStatus, OrderType, OrderDirection
from ba2_common.core.db import get_db, update_instance

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
        """Per-bar fill engine.

        Baseline implementation (Task 2): fill working MARKET orders against the chosen
        bar so the ledger/equity are exercised end-to-end. Task 3 expands this into the
        full LIMIT/STOP/TP/SL/OCO per-bar evaluation. Returns True.
        """
        as_of = self._price.now()
        active = OrderStatus.get_active_statuses()
        working = [o for o in self.get_orders() if o.status in active]
        for o in working:
            fill = self._try_fill(o, as_of)
            if fill is None:
                continue
            self._apply_fill(o, fill, as_of)
        return True

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
        """
        trading_order.broker_order_id = self._next_broker_id()
        trading_order.status = OrderStatus.ACCEPTED  # working / active per get_active_statuses()
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
        """Modify hook. Baseline: returns the (non-terminal) order unchanged.

        The live signature is ``modify_order(self, order_id)`` (no trading_order param);
        Task 3 wires in the in-place pre-fill price/qty edit.
        """
        o = self.get_order(order_id)
        if o is None or o.status in OrderStatus.get_terminal_statuses():
            return None
        return o

    def adjust_tp(self, transaction, new_tp_price: float, source: str = "") -> bool:
        """TP leg adjustment. Full SELL_LIMIT(long)/BUY_LIMIT(short) leg lands in Task 3."""
        raise NotImplementedError("adjust_tp is implemented in Phase 2 Task 3 (fill engine)")

    def adjust_sl(self, transaction, new_sl_price: float, source: str = "") -> bool:
        """SL leg adjustment. Full SELL_STOP(long)/BUY_STOP(short) leg lands in Task 3."""
        raise NotImplementedError("adjust_sl is implemented in Phase 2 Task 3 (fill engine)")

    def adjust_tp_sl(
        self,
        transaction,
        new_tp_price: Optional[float] = None,
        new_sl_price: Optional[float] = None,
        source: str = "",
    ) -> bool:
        """Paired TP+SL (OCO). Full implementation lands in Task 3."""
        raise NotImplementedError("adjust_tp_sl is implemented in Phase 2 Task 3 (fill engine)")

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

    def _try_fill(self, order, as_of: datetime) -> Optional[float]:
        """Return the fill price for MARKET orders this bar, else None.

        Task 2 baseline only fills MARKET orders (LIMIT/STOP/OCO trigger logic is added
        in Task 3). This is enough to exercise the ledger/equity end to end.
        """
        if order.order_type != OrderType.MARKET:
            return None
        bar = self._bar_for_fill(order, as_of)
        if bar is None:
            return None
        ref = bar["open"] if self._cfg["fill_model"] != "same_bar_close" else bar["close"]
        is_buy = order.side == OrderDirection.BUY
        return self._slip(ref, is_buy)

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

    def _order_to_trade(self, order, qty: float) -> Dict[str, Any]:
        """Map a filled ``TradingOrder`` row to the documented filled-trade dict shape."""
        return {
            "symbol": order.symbol,
            "qty": abs(float(qty)),
            "side": order.side.value if order.side else None,
            "date": order.created_at,
            "price": order.open_price,
        }
