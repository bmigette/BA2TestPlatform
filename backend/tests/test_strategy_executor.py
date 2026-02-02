"""Tests for StrategyExecutor service."""

import pytest
from app.services.strategy_executor import (
    StrategyExecutor,
    evaluate_condition,
    evaluate_comparison,
    ExitActionType
)


class TestEvaluateComparison:
    def test_greater_than(self):
        assert evaluate_comparison(0.7, ">", 0.5) is True
        assert evaluate_comparison(0.5, ">", 0.7) is False

    def test_greater_than_equal(self):
        assert evaluate_comparison(0.7, ">=", 0.7) is True
        assert evaluate_comparison(0.5, ">=", 0.7) is False

    def test_less_than(self):
        assert evaluate_comparison(0.3, "<", 0.5) is True
        assert evaluate_comparison(0.7, "<", 0.5) is False

    def test_equal(self):
        assert evaluate_comparison(1, "==", 1) is True
        assert evaluate_comparison(1, "==", 2) is False

    def test_not_equal(self):
        assert evaluate_comparison(1, "!=", 2) is True
        assert evaluate_comparison(1, "!=", 1) is False

    def test_between(self):
        assert evaluate_comparison(5, "between", [1, 10]) is True
        assert evaluate_comparison(15, "between", [1, 10]) is False


class TestEvaluateCondition:
    def test_simple_condition(self):
        condition = {"field": "price_up", "comparison": ">", "value": 0.6}
        context = {"price_up": 0.7}
        assert evaluate_condition(condition, context) is True

    def test_missing_field(self):
        condition = {"field": "nonexistent", "comparison": ">", "value": 0.6}
        context = {"price_up": 0.7}
        assert evaluate_condition(condition, context) is False

    def test_and_operator(self):
        condition = {
            "operator": "AND",
            "conditions": [
                {"field": "price_up", "comparison": ">", "value": 0.6},
                {"field": "hour", "comparison": ">=", "value": 9}
            ]
        }
        context = {"price_up": 0.7, "hour": 10}
        assert evaluate_condition(condition, context) is True

        context = {"price_up": 0.7, "hour": 8}
        assert evaluate_condition(condition, context) is False

    def test_or_operator(self):
        condition = {
            "operator": "OR",
            "conditions": [
                {"field": "price_up", "comparison": ">", "value": 0.8},
                {"field": "price_down", "comparison": ">", "value": 0.8}
            ]
        }
        context = {"price_up": 0.5, "price_down": 0.9}
        assert evaluate_condition(condition, context) is True


class TestStrategyExecutor:
    def test_check_entry(self):
        config = {
            "entry_conditions": {
                "operator": "AND",
                "conditions": [
                    {"field": "price_up_10pct", "comparison": ">", "value": 0.7}
                ]
            }
        }
        executor = StrategyExecutor(config)

        assert executor.check_entry({"price_up_10pct": 0.8}) is True
        assert executor.check_entry({"price_up_10pct": 0.5}) is False

    def test_check_exits(self):
        config = {
            "entry_conditions": {},
            "exit_conditions": [
                {
                    "conditions": {"field": "bars_in_trade", "comparison": ">", "value": 50},
                    "action": "close"
                },
                {
                    "conditions": {"field": "position_pnl_pct", "comparison": ">", "value": 5},
                    "action": "adjust_sl",
                    "action_value": 0
                }
            ]
        }
        executor = StrategyExecutor(config)

        # No exit triggered
        action = executor.check_exits({"bars_in_trade": 10, "position_pnl_pct": 1})
        assert action is None

        # Close triggered
        action = executor.check_exits({"bars_in_trade": 60, "position_pnl_pct": 1})
        assert action is not None
        assert action.action == ExitActionType.CLOSE

        # Adjust SL triggered (first matching rule)
        action = executor.check_exits({"bars_in_trade": 10, "position_pnl_pct": 6})
        assert action is not None
        assert action.action == ExitActionType.ADJUST_SL
        assert action.value == 0
