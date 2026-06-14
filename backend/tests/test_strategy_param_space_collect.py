import types
from app.services.strategy_param_space import collect_param_space, CLASSIC_RM_PARAMS


def _strategy(**kw):
    """Minimal Strategy-like object with the columns collect_param_space reads."""
    base = dict(
        initial_tp_optimize=False, initial_tp_min=None, initial_tp_max=None, initial_tp_step=None,
        initial_sl_optimize=False, initial_sl_min=None, initial_sl_max=None, initial_sl_step=None,
        buy_entry_conditions=None, sell_entry_conditions=None, entry_conditions=None,
        exit_conditions=[],
    )
    base.update(kw)
    return types.SimpleNamespace(**base)


def test_collect_tp_sl_only_when_optimize():
    s = _strategy(initial_tp_optimize=True, initial_tp_min=2.0, initial_tp_max=10.0,
                  initial_tp_step=0.5)
    space = collect_param_space(s)
    assert space["tp"] == {"type": "float", "min": 2.0, "max": 10.0, "step": 0.5}
    assert "sl" not in space


def test_collect_rm_namespaced():
    s = _strategy(initial_tp_optimize=True, initial_tp_min=1, initial_tp_max=2, initial_tp_step=0.5)
    rm = {"risk_per_trade_pct": {"optimize": True, "min": 0.5, "max": 3.0, "step": 0.25, "type": "float"},
          "max_concurrent_positions": {"optimize": True, "min": 1, "max": 10, "step": 1, "type": "int"}}
    space = collect_param_space(s, rm_cfg=rm)
    assert space["rm:risk_per_trade_pct"]["type"] == "float"
    assert space["rm:max_concurrent_positions"] == {"type": "int", "min": 1, "max": 10, "step": 1}


def test_collect_expert_namespaced():
    s = _strategy(initial_sl_optimize=True, initial_sl_min=1, initial_sl_max=5, initial_sl_step=0.5)
    expert = {"surprise_min_pct": {"optimize": True, "min": 1.0, "max": 20.0, "step": 1.0, "type": "float"},
              "max_days_since_report": {"optimize": False, "min": 1, "max": 30, "step": 1, "type": "int"}}
    space = collect_param_space(s, expert_cfg=expert)
    assert "model:surprise_min_pct" in space
    assert "model:max_days_since_report" not in space  # optimize=False


def test_collect_condition_value_and_confirmation():
    buy = {"operator": "AND", "conditions": [
        {"id": "c1", "field": "model:probability", "comparison": ">=", "value": 0.6,
         "optimize": True, "value_min": 0.5, "value_max": 0.9, "value_step": 0.05,
         "confirmation_bars_min": 1, "confirmation_bars_max": 5, "confirmation_bars_step": 1},
    ]}
    s = _strategy(buy_entry_conditions=buy,
                  initial_tp_optimize=True, initial_tp_min=1, initial_tp_max=2, initial_tp_step=0.5)
    space = collect_param_space(s)
    assert space["cond:c1:value"] == {"type": "float", "min": 0.5, "max": 0.9, "step": 0.05}
    assert space["cond:c1:confirmation_bars"] == {"type": "int", "min": 1, "max": 5, "step": 1}


def test_collect_exit_action_value():
    s = _strategy(exit_conditions=[
        {"id": "e1", "action": "adjust_sl", "action_value": 1.0, "action_value_optimize": True,
         "action_value_min": 0.5, "action_value_max": 3.0, "action_value_step": 0.5,
         "conditions": {}},
    ])
    space = collect_param_space(s)
    assert space["exit:e1:action_value"]["min"] == 0.5


def test_empty_space_raises():
    import pytest
    with pytest.raises(ValueError):
        collect_param_space(_strategy())
