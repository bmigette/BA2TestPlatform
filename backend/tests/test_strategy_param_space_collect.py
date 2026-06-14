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


def test_bypass_excludes_rm_tp_sl_cond_exit_keeps_only_model():
    """BYPASS expert (piece 1c): the param space drops rm:*/tp/sl/cond:*/exit:* and keeps
    ONLY the expert's own model:* params (FactorRanker rebalances via its own portfolio
    manager, so the classic RM / ruleset namespaces have no effect)."""
    buy = {"operator": "AND", "conditions": [
        {"id": "c1", "field": "model:probability", "comparison": ">=", "value": 0.6,
         "optimize": True, "value_min": 0.5, "value_max": 0.9, "value_step": 0.05},
    ]}
    s = _strategy(
        buy_entry_conditions=buy,
        initial_tp_optimize=True, initial_tp_min=2.0, initial_tp_max=10.0, initial_tp_step=0.5,
        initial_sl_optimize=True, initial_sl_min=1.0, initial_sl_max=5.0, initial_sl_step=0.5,
        exit_conditions=[
            {"id": "e1", "action": "adjust_sl", "action_value": 1.0,
             "action_value_optimize": True, "action_value_min": 0.5, "action_value_max": 3.0,
             "action_value_step": 0.5, "conditions": {}},
        ],
    )
    rm = {"risk_per_trade_pct": {"optimize": True, "min": 0.5, "max": 3.0, "step": 0.25,
                                 "type": "float"}}
    # FactorRanker's own params (factor weights / top_n / winsorize_pct).
    expert = {"top_n": {"optimize": True, "min": 5, "max": 30, "step": 5, "type": "int"},
              "winsorize_pct": {"optimize": True, "min": 0.0, "max": 0.1, "step": 0.01,
                                "type": "float"}}

    space = collect_param_space(s, expert_cfg=expert, rm_cfg=rm, bypass=True)

    # ONLY model:* survives.
    assert set(space) == {"model:top_n", "model:winsorize_pct"}
    assert all(k.startswith("model:") for k in space)
    # None of the excluded namespaces leak in.
    assert "tp" not in space and "sl" not in space
    assert not any(k.startswith("rm:") for k in space)
    assert not any(k.startswith("cond:") for k in space)
    assert not any(k.startswith("exit:") for k in space)


def test_bypass_vs_non_bypass_same_inputs_differ():
    """The SAME strategy/expert/RM inputs yield a strictly smaller space under bypass=True
    (rm:*/tp/sl present without bypass, gone with it)."""
    s = _strategy(initial_tp_optimize=True, initial_tp_min=2.0, initial_tp_max=10.0,
                  initial_tp_step=0.5)
    rm = {"risk_per_trade_pct": {"optimize": True, "min": 0.5, "max": 3.0, "step": 0.25,
                                 "type": "float"}}
    expert = {"top_n": {"optimize": True, "min": 5, "max": 30, "step": 5, "type": "int"}}

    classic = collect_param_space(s, expert_cfg=expert, rm_cfg=rm, bypass=False)
    bypass = collect_param_space(s, expert_cfg=expert, rm_cfg=rm, bypass=True)

    assert "tp" in classic and "rm:risk_per_trade_pct" in classic
    assert set(bypass) == {"model:top_n"}
    assert set(bypass) < set(classic)


def test_bypass_with_no_expert_params_raises():
    """A bypass expert with NO optimizable expert params has an empty space -> fail-early
    (the rm/tp/sl/cond it might have are excluded, so there is genuinely nothing to search)."""
    import pytest
    s = _strategy(initial_tp_optimize=True, initial_tp_min=2.0, initial_tp_max=10.0,
                  initial_tp_step=0.5)
    with pytest.raises(ValueError):
        collect_param_space(s, expert_cfg=None, bypass=True)
