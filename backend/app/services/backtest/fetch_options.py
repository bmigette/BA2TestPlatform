"""Builds the offline options cache from Alpaca. CLI: `ba2-test fetch-options`.
Run with the editable venv (~/ba2-venvs/test) which has alpaca-py installed."""
from __future__ import annotations
import argparse
from datetime import date, timedelta
from typing import Any, Dict, List
from .options_cache import OptionsHistoryCache

def _g(obj, name):
    return getattr(obj, name, None)

def contract_to_chain_row(occ: str, underlying: str, opt_type: str, strike: float,
                          expiry: str, snap: Any) -> Dict[str, Any]:
    greeks = _g(snap, "greeks"); q = _g(snap, "latest_quote"); t = _g(snap, "latest_trade")
    return {"occ_symbol": occ, "option_type": opt_type, "strike": strike, "expiry": expiry,
            "bid": _g(q, "bid_price"), "ask": _g(q, "ask_price"), "last": _g(t, "price"),
            "iv": _g(snap, "implied_volatility"), "delta": _g(greeks, "delta"),
            "gamma": _g(greeks, "gamma"), "theta": _g(greeks, "theta"), "vega": _g(greeks, "vega"),
            "open_interest": _g(snap, "open_interest"), "volume": _g(t, "size")}

def bar_to_row(occ: str, d: str, bar: Any, underlying: str, opt_type: str, strike: float,
               expiry: str) -> Dict[str, Any]:
    return {"occ_symbol": occ, "date": d, "open": _g(bar, "open"), "high": _g(bar, "high"),
            "low": _g(bar, "low"), "close": _g(bar, "close"), "volume": _g(bar, "volume"),
            "underlying": underlying, "option_type": opt_type, "strike": strike, "expiry": expiry}

def _alpaca_keys():
    # The codebase configures Alpaca market-data creds as ALPACA_MARKET_API_KEY/_SECRET
    # (see .env / .env.example). Fall back to the generic ALPACA_API_KEY/_SECRET_KEY names.
    import os
    key = os.environ.get("ALPACA_MARKET_API_KEY") or os.environ.get("ALPACA_API_KEY")
    secret = os.environ.get("ALPACA_MARKET_API_SECRET") or os.environ.get("ALPACA_SECRET_KEY")
    if not key or not secret:
        raise RuntimeError(
            "Alpaca credentials not found. Set ALPACA_MARKET_API_KEY/ALPACA_MARKET_API_SECRET "
            "(or ALPACA_API_KEY/ALPACA_SECRET_KEY) in the environment / .env.")
    return key, secret

def build_cache(cache_db: str, underlyings: List[str], start: date, end: date,
                feed: str = "indicative") -> None:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import GetOptionContractsRequest
    from alpaca.data.historical.option import OptionHistoricalDataClient
    from alpaca.data.requests import OptionBarsRequest, OptionChainRequest
    from alpaca.data.timeframe import TimeFrame
    if start < date(2024, 2, 1):
        raise ValueError("Alpaca options history starts 2024-02-01; pick a later --start")
    key, secret = _alpaca_keys()
    tc = TradingClient(key, secret, paper=True)
    dc = OptionHistoricalDataClient(key, secret)
    cache = OptionsHistoryCache(cache_db)
    for u in underlyings:
        contracts = tc.get_option_contracts(GetOptionContractsRequest(
            underlying_symbols=[u], expiration_date_gte=start.isoformat(),
            expiration_date_lte=(end + timedelta(days=120)).isoformat(),
            limit=10000)).option_contracts
        chain = dc.get_option_chain(OptionChainRequest(underlying_symbol=u, feed=feed))
        rows = []
        for c in contracts:
            snap = chain.get(c.symbol) if isinstance(chain, dict) else None
            rows.append(contract_to_chain_row(c.symbol, u, c.type.value, float(c.strike_price),
                                              c.expiration_date.isoformat(), snap or object()))
        cache.write_chain_rows(u, start.isoformat(), rows)
        for c in contracts:
            resp = dc.get_option_bars(OptionBarsRequest(symbol_or_symbols=c.symbol,
                timeframe=TimeFrame.Day, start=start.isoformat(), end=end.isoformat()))
            bars = (resp.data or {}).get(c.symbol, [])
            cache.write_bar_rows([bar_to_row(c.symbol, b.timestamp.date().isoformat(), b, u,
                c.type.value, float(c.strike_price), c.expiration_date.isoformat()) for b in bars])

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="ba2-test fetch-options")
    ap.add_argument("--underlyings", required=True, help="comma list or @file")
    ap.add_argument("--start", required=True); ap.add_argument("--end", required=True)
    ap.add_argument("--cache-db", required=True); ap.add_argument("--feed", default="indicative")
    a = ap.parse_args(argv)
    unders = (open(a.underlyings[1:]).read().split() if a.underlyings.startswith("@")
              else [s.strip() for s in a.underlyings.split(",") if s.strip()])
    build_cache(a.cache_db, unders, date.fromisoformat(a.start), date.fromisoformat(a.end), a.feed)
    return 0
