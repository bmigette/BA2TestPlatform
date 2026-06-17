from app.services.strategy_param_space import collect_param_space, decode_params


class _Strat:  # minimal stand-in
    initial_tp_percent = 5.0
    initial_sl_percent = 5.0
    buy_entry_conditions = None
    sell_entry_conditions = None
    exit_conditions = []


def test_collect_screener_adds_namespaced_genes():
    space = collect_param_space(
        _Strat(), expert_cfg={"params": {}}, bypass=True,
        screener_cfg={
            "screener_market_cap_min": {"min": 1e9, "max": 5e9, "step": 1e9, "type": "float", "optimize": True},
            "screener_relative_volume_min": {"min": 1.0, "max": 2.0, "step": 0.1, "type": "float", "optimize": True},
        })
    assert "screener:screener_market_cap_min" in space
    assert "screener:screener_relative_volume_min" in space


def test_decode_screener_overrides():
    out = decode_params(_Strat(), {
        "tp": 6.0, "sl": 4.0,
        "screener:screener_market_cap_min": 2e9,
        "screener:screener_relative_volume_min": 1.4,
    })
    assert out["screener_overrides"] == {
        "screener_market_cap_min": 2e9, "screener_relative_volume_min": 1.4}
    assert out["tp"] == 6.0  # existing fields still present
