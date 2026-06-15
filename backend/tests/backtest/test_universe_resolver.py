"""Task 8: resolve a screener-typed daily-backtest universe from the OFFLINE cache.

The resolver is READ-ONLY: it unions the cached survivor symbols across the scan dates
that fall in ``[start, end]`` (built ahead of time via ``ba2-test fetch-screener``). It
NEVER live-screens — if no scan date in the range is cached it raises ``ScreenerCacheMiss``
telling the user to build the cache first (fail-early, ``backend/CLAUDE.md``).

Run from the backend cwd:
    ./venv/bin/python -m pytest tests/backtest/test_universe_resolver.py -v
"""
from datetime import datetime, timezone

import pytest

from app.services.backtest.universe_resolver import (
    ScreenerCacheMiss,
    resolve_screener_universe,
)
from app.services.screener_history_cache import ScreenerHistoryCache, screen_config_hash

# A complete post-coercion settings dict (all hash keys present) so the resolver's
# computed config hash matches the hash rows are written under.
SETTINGS = {
    "screener_provider": "fmp",
    "screener_market_cap_min": 1_000_000_000,
    "screener_market_cap_max": 0,
    "screener_volume_min": 500_000,
    "screener_volume_max": 0,
    "screener_float_min": 10_000_000,
    "screener_float_max": 0,
    "screener_price_min": 20.0,
    "screener_price_max": 0,
    "screener_relative_volume_min": 1.05,
    "screener_price_drop_pct": 15.0,
    "screener_price_drop_days": 1,
    "screener_max_stocks": 10,
    "screener_sort_metric": "market_cap",
    "screener_weinstein_stage2_only": 0,
    "universe_mode": "broad",
}


def test_missing_cache_fails_fast(tmp_path):
    """No scan date in the range cached -> ScreenerCacheMiss (no live screen)."""
    with pytest.raises(ScreenerCacheMiss) as exc:
        resolve_screener_universe(
            screener_settings=SETTINGS,
            start="2024-01-02",
            end="2024-01-31",
            cache_db=str(tmp_path / "empty.db"),
            group="g",
        )
    # message points the user at the build command
    assert "fetch-screener" in str(exc.value)


def test_resolves_union_of_cached_survivors(tmp_path):
    """A pre-built tiny cache: resolver returns the sorted union across cached dates."""
    db = str(tmp_path / "sh.db")
    cache = ScreenerHistoryCache(db)
    cfg_hash = screen_config_hash(SETTINGS)

    # Two cached scan dates inside the requested range, plus one OUTSIDE it.
    cache.write(
        datetime(2024, 1, 5, tzinfo=timezone.utc), cfg_hash, "g",
        [{"symbol": "AAA", "market_cap": 5e9}, {"symbol": "BBB", "market_cap": 3e9}],
        "broad",
    )
    cache.write(
        datetime(2024, 1, 12, tzinfo=timezone.utc), cfg_hash, "g",
        [{"symbol": "BBB", "market_cap": 4e9}, {"symbol": "CCC", "market_cap": 2e9}],
        "broad",
    )
    # Outside the range -> must NOT contribute.
    cache.write(
        datetime(2024, 2, 9, tzinfo=timezone.utc), cfg_hash, "g",
        [{"symbol": "ZZZ", "market_cap": 9e9}],
        "broad",
    )
    # Different group -> must NOT contribute.
    cache.write(
        datetime(2024, 1, 5, tzinfo=timezone.utc), cfg_hash, "other",
        [{"symbol": "QQQ", "market_cap": 9e9}],
        "broad",
    )

    out = resolve_screener_universe(
        screener_settings=SETTINGS,
        start="2024-01-02",
        end="2024-01-31",
        cache_db=db,
        group="g",
    )
    assert out == ["AAA", "BBB", "CCC"]  # sorted union, deduped, range+group scoped


def test_datetime_inputs_accepted(tmp_path):
    """start/end may be datetimes (the handler passes parsed datetimes)."""
    db = str(tmp_path / "sh2.db")
    cache = ScreenerHistoryCache(db)
    cfg_hash = screen_config_hash(SETTINGS)
    cache.write(
        datetime(2024, 3, 4, tzinfo=timezone.utc), cfg_hash, "g",
        [{"symbol": "MMM", "market_cap": 5e9}],
        "broad",
    )
    out = resolve_screener_universe(
        screener_settings=SETTINGS,
        start=datetime(2024, 3, 1),
        end=datetime(2024, 3, 31),
        cache_db=db,
        group="g",
    )
    assert out == ["MMM"]
