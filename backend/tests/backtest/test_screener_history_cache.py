"""Phase 3 / Task 5: the grouped/labeled screener-history cache.

Covers (gate item 4 anchor = ``test_cache_once_no_second_fetch``):
  * ``screen_config_hash`` is deterministic AND config-sensitive (a threshold change
    yields a distinct namespace) -> different criteria never mix, same criteria replay;
  * ``write`` then ``replay`` returns the survivors rank-ordered;
  * cache-once: the second ``screened_universe_for_bar`` for the same
    ``(scan_date, hash)`` replays from the table and runs ``StockScreener.screen()``
    ZERO additional times (no provider fetches on replay);
  * a config change writes into a DISTINCT hash namespace (no row mixing).

Run from the backend cwd (``app.*`` import root):
    ./venv/bin/python -m pytest tests/backtest/test_screener_history_cache.py -v
"""
from datetime import datetime, timezone

from app.services.screener_history_cache import (
    ScreenerHistoryCache,
    screen_config_hash,
    screened_universe_for_bar,
)
import app.services.screener_history_cache as M

SD = datetime(2020, 6, 30, tzinfo=timezone.utc)
BASE = {
    "screener_market_cap_min": 1_000_000_000,
    "screener_sort_metric": "market_cap",
    "universe_mode": "broad",
}


def test_hash_stable_and_config_sensitive():
    h1 = screen_config_hash({**BASE})
    h2 = screen_config_hash({**BASE})
    h3 = screen_config_hash({**BASE, "screener_market_cap_min": 2_000_000_000})
    assert h1 == h2  # deterministic
    assert h1 != h3  # a threshold change -> distinct namespace


def test_hash_only_depends_on_result_affecting_keys():
    # An irrelevant key (not in _HASH_KEYS) must NOT change the namespace.
    h1 = screen_config_hash({**BASE})
    h2 = screen_config_hash({**BASE, "some_unrelated_ui_key": "noise"})
    assert h1 == h2


def test_write_then_replay(tmp_path):
    cache = ScreenerHistoryCache(str(tmp_path / "sh.sqlite"))
    survivors = [
        {"symbol": "AAA", "market_cap": 5e9, "price": 50.0},
        {"symbol": "BBB", "market_cap": 3e9, "price": 40.0},
    ]
    n = cache.write(SD, "hashX", "FactorRanker", survivors, "broad")
    assert n == 2
    assert cache.has(SD, "hashX")
    rows = cache.replay(SD, "hashX")
    assert [r["symbol"] for r in rows] == ["AAA", "BBB"]  # rank-ordered (insertion order)
    # audit columns persisted
    assert rows[0]["group_label"] == "FactorRanker"
    assert rows[0]["universe_mode"] == "broad"
    assert rows[0]["price_as_of"] == 50.0
    assert rows[0]["market_cap_as_of"] == 5e9
    assert rows[0]["float_approx"] == 1  # default proxy


def test_write_is_idempotent(tmp_path):
    cache = ScreenerHistoryCache(str(tmp_path / "sh_idem.sqlite"))
    survivors = [{"symbol": "AAA", "market_cap": 5e9, "price": 50.0}]
    cache.write(SD, "hashX", "FactorRanker", survivors, "broad")
    cache.write(SD, "hashX", "FactorRanker", survivors, "broad")  # re-run, same key
    rows = cache.replay(SD, "hashX")
    assert len(rows) == 1  # UNIQUE(symbol, scan_date, hash) -> no duplicate


def test_distinct_hash_namespaces_do_not_mix(tmp_path):
    cache = ScreenerHistoryCache(str(tmp_path / "sh_ns.sqlite"))
    cache.write(SD, "hashA", "ExpertA", [{"symbol": "AAA", "market_cap": 1e9}], "broad")
    cache.write(SD, "hashB", "ExpertB", [{"symbol": "BBB", "market_cap": 2e9}], "broad")
    assert [r["symbol"] for r in cache.replay(SD, "hashA")] == ["AAA"]
    assert [r["symbol"] for r in cache.replay(SD, "hashB")] == ["BBB"]


def _full_settings(**overrides):
    """A complete post-coercion settings dict (all hash keys present)."""
    base = {
        "screener_provider": "fmp_historical",
        "screener_market_cap_min": 1_000_000_000,
        "screener_market_cap_max": 0,
        "screener_volume_min": 0,
        "screener_volume_max": 0,
        "screener_float_min": 0,
        "screener_float_max": 0,
        "screener_price_min": 0,
        "screener_price_max": 0,
        "screener_relative_volume_min": 0,
        "screener_price_drop_pct": 0,
        "screener_price_drop_days": 1,
        "screener_max_stocks": 10,
        "screener_sort_metric": "market_cap",
        "screener_weinstein_stage2_only": 0,
        "universe_mode": "broad",
    }
    base.update(overrides)
    return base


def test_cache_once_no_second_fetch(tmp_path, monkeypatch):
    """Gate item 4: first call runs screen() once; second call replays (screen() not re-run)."""
    cache = ScreenerHistoryCache(str(tmp_path / "sh2.sqlite"))
    calls = {"n": 0}

    class FakeSC:
        def __init__(self, settings, progress_callback=None, as_of=None):
            # Mirror StockScreener: expose the coerced settings the hash is built from.
            self._settings = _full_settings()

        def screen(self):
            calls["n"] += 1
            return {
                "results": [{"symbol": "AAA", "market_cap": 5e9, "price": 50.0}],
                "stats": {},
            }

    # screened_universe_for_bar resolves StockScreener via this module's global namespace.
    monkeypatch.setattr(M, "StockScreener", FakeSC)

    u1 = screened_universe_for_bar(BASE, SD, "FactorRanker", cache)
    u2 = screened_universe_for_bar(BASE, SD, "FactorRanker", cache)

    assert calls["n"] == 1  # second call replayed from cache -> screen() NOT re-run
    assert [r["symbol"] for r in u1] == [r["symbol"] for r in u2] == ["AAA"]
    # compute path annotated the cache-audit metadata
    assert u1[0]["sort_metric_value"] == 5e9
    assert u1[0]["market_cap_source"] == "shares_x_close"
    assert u1[0]["float_approx"] is True


def test_config_change_yields_distinct_namespace_and_recomputes(tmp_path, monkeypatch):
    """A settings change flips the hash -> a fresh namespace -> screen() runs again."""
    cache = ScreenerHistoryCache(str(tmp_path / "sh3.sqlite"))
    calls = {"n": 0}
    seen_settings = {"v": None}

    class FakeSC:
        def __init__(self, settings, progress_callback=None, as_of=None):
            self._settings = _full_settings(
                screener_market_cap_min=int(settings["screener_market_cap_min"])
            )
            seen_settings["v"] = self._settings

        def screen(self):
            calls["n"] += 1
            return {"results": [{"symbol": "AAA", "market_cap": 5e9}], "stats": {}}

    monkeypatch.setattr(M, "StockScreener", FakeSC)

    cfg1 = {**BASE, "screener_market_cap_min": 1_000_000_000}
    cfg2 = {**BASE, "screener_market_cap_min": 2_000_000_000}
    screened_universe_for_bar(cfg1, SD, "FactorRanker", cache)
    screened_universe_for_bar(cfg1, SD, "FactorRanker", cache)  # replay (no recompute)
    assert calls["n"] == 1
    screened_universe_for_bar(cfg2, SD, "FactorRanker", cache)  # new namespace -> recompute
    assert calls["n"] == 2
