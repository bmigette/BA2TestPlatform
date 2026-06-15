# backend/tests/backtest/test_fetch_options.py
from app.services.backtest.fetch_options import contract_to_chain_row, bar_to_row

class _Snap:
    def __init__(self):
        self.implied_volatility = 0.3
        class G: delta=0.42; gamma=0.02; theta=-0.04; vega=0.12
        self.greeks = G()
        class Q: bid_price=1.0; ask_price=1.2
        self.latest_quote = Q()
        class T: price=1.1; size=7
        self.latest_trade = T()
        self.open_interest = 500

def test_contract_to_chain_row():
    row = contract_to_chain_row("AAPL240315C00180000", "AAPL", "call", 180.0, "2024-03-15", _Snap())
    assert row["occ_symbol"] == "AAPL240315C00180000"
    assert row["iv"] == 0.3 and row["delta"] == 0.42 and row["bid"] == 1.0 and row["last"] == 1.1

def test_contract_to_chain_row_handles_missing_snapshot():
    row = contract_to_chain_row("AAPL240315C00180000", "AAPL", "call", 180.0, "2024-03-15", object())
    assert row["occ_symbol"] == "AAPL240315C00180000" and row["iv"] is None and row["delta"] is None

def test_bar_to_row():
    class B: open=2.0; high=2.5; low=1.9; close=2.3; volume=100
    row = bar_to_row("AAPL240315C00180000", "2024-03-05", B(), "AAPL", "call", 180.0, "2024-03-15")
    assert row["close"] == 2.3 and row["underlying"] == "AAPL"
