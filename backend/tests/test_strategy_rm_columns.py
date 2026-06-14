"""Phase 4 Task 2: the Strategy model exposes the classic-RM optimize columns.

Verifies the five classic-RM params each carry value/_optimize/_min/_max/_step columns
(the schema half of strategy_param_space._collect_rm) and that to_dict() surfaces the
camelCase keys the UI edits.
"""
from app.models.strategy import Strategy

_RM_PARAMS = (
    "risk_per_trade_pct",
    "per_instrument_cap_pct",
    "min_stop_pct",
    "atr_stop_mult",
    "max_concurrent_positions",
)


def test_strategy_has_rm_columns():
    for p in _RM_PARAMS:
        for suffix in ("", "_optimize", "_min", "_max", "_step"):
            assert hasattr(Strategy, f"rm_{p}{suffix}"), f"missing rm_{p}{suffix}"


def test_rm_columns_registered_on_table():
    from sqlalchemy import inspect

    cols = {c.name for c in inspect(Strategy).columns}
    for p in _RM_PARAMS:
        for suffix in ("", "_optimize", "_min", "_max", "_step"):
            assert f"rm_{p}{suffix}" in cols, f"rm_{p}{suffix} not a mapped column"


def test_max_concurrent_positions_is_integer():
    from sqlalchemy import Integer, Float

    col = Strategy.__table__.c["rm_max_concurrent_positions"]
    assert isinstance(col.type, Integer)
    # the four percent/multiplier params are Float
    for p in ("risk_per_trade_pct", "per_instrument_cap_pct", "min_stop_pct", "atr_stop_mult"):
        assert isinstance(Strategy.__table__.c[f"rm_{p}"].type, Float), p


def test_to_dict_exposes_rm_camelcase_keys():
    s = Strategy(
        name="rm-test",
        rm_risk_per_trade_pct=2.0,
        rm_per_instrument_cap_pct=15.0,
        rm_min_stop_pct=1.5,
        rm_atr_stop_mult=3.0,
        rm_max_concurrent_positions=7,
    )
    d = s.to_dict()
    for key in (
        "rmRiskPerTradePct",
        "rmRiskPerTradePctOptimize",
        "rmRiskPerTradePctMin",
        "rmRiskPerTradePctMax",
        "rmRiskPerTradePctStep",
        "rmPerInstrumentCapPct",
        "rmMinStopPct",
        "rmAtrStopMult",
        "rmMaxConcurrentPositions",
        "rmMaxConcurrentPositionsOptimize",
        "rmMaxConcurrentPositionsStep",
    ):
        assert key in d, f"to_dict missing {key}"
    assert d["rmRiskPerTradePct"] == 2.0
    assert d["rmMaxConcurrentPositions"] == 7
