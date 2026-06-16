from fastapi.testclient import TestClient
from app.main import app
client = TestClient(app)


def test_vocabulary_lists_flags_numerics_actions_refs():
    v = client.get("/api/ruleset/vocabulary").json()
    assert "bearish" in {f["value"] for f in v["flags"]}
    assert "profit_loss_percent" in {n["value"] for n in v["numerics"]}
    assert {"close", "sell", "adjust_take_profit", "adjust_stop_loss"}.issubset({a["value"] for a in v["actions"]})
    assert any(a["value"] == "buy_call" and a["is_option"] for a in v["actions"])
    assert any(a["value"] == "adjust_stop_loss" and a["needs_reference"] for a in v["actions"])
    assert "order_open_price" in v["reference_values"] and ">" in v["operators"]


def test_exit_presets_validate_against_model():
    from app.api.strategies import ExitCondition
    presets = client.get("/api/ruleset/exit-presets").json()["presets"]
    assert len(presets) >= 4
    for p in presets:
        ExitCondition(**p["rule"])   # must not raise
