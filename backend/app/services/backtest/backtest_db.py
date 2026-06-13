"""Per-run backtest *trading* DB lifecycle.

This DB is distinct from BOTH:
  (a) BA2TestPlatform's own results DB (the ``Backtest`` SQLAlchemy row lives there,
      via app.models.database.SessionLocal), and
  (b) the live BA2TradePlatform DB (~/Documents/ba2_trade_platform/db.sqlite).

The inherited ba2_common AccountInterface / refresh_transactions / submit_order /
TradeActionEvaluator / TradeRiskManagement logic reads & writes TradingOrder /
Transaction / AccountDefinition / ExpertInstance / ExpertRecommendation rows via
``ba2_common.core.db``. We point THAT db layer at a throwaway sqlite file (one file
per run) so a run is hermetic, reproducible, and never touches the live DB. The
schema is created by ba2_common's own ``init_db`` (SQLModel.metadata.create_all), so
it is byte-identical to the live trading schema and the inherited DB logic works
unchanged.

Confirmed against the installed ba2_common.core.db:
  * configure_db(db_file) -> points the (lazy) engine at db_file and resets it.
  * init_db() -> imports models + SQLModel.metadata.create_all(get_engine()).
  * get_engine() / get_db() / add_instance / get_instance.

AccountDefinition's real columns are {name, provider, description} (NOT the plan
draft's account_type/enabled, which do not exist on the model).
"""
from __future__ import annotations

import pathlib
import tempfile
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

from ba2_common.core import db as common_db


def backtest_db_root() -> pathlib.Path:
    """Directory holding per-run backtest sqlite files (created on demand)."""
    root = pathlib.Path(tempfile.gettempdir()) / "ba2_backtest_dbs"
    root.mkdir(parents=True, exist_ok=True)
    return root


def backtest_db_path(run_id: int | str) -> pathlib.Path:
    """Deterministic sqlite path for a run id (so re-runs reuse the same file name)."""
    return backtest_db_root() / f"run_{run_id}.sqlite"


@contextmanager
def backtest_trading_db(run_id: int | str) -> Iterator[str]:
    """Configure ba2_common.core.db at a fresh sqlite for this run, create the schema,
    and yield the file path.

    A FRESH file per run (the previous file for this run id, if any, is deleted) so
    the run is hermetic and reproducible. The sqlite FILE is intentionally LEFT on disk
    after the context exits (post-mortem debugging), but the global ba2_common DB engine
    is RESTORED to whatever it pointed at before entering — otherwise any code that runs
    after the context (other tests, or live-flavoured provider construction such as
    FMPOHLCVProvider reading FMP_API_KEY from the app-settings table) would silently
    read/write the backtest DB instead of the live one. We capture ``_db_file`` and
    re-``configure_db`` it on exit so the seam is properly hermetic.
    """
    prior_db_file = common_db._db_file  # noqa: SLF001 (intentional: save/restore the global)
    path = backtest_db_path(run_id)
    if path.exists():
        path.unlink()
    common_db.configure_db(str(path))  # Phase-0 DB seam: point the engine at this file
    common_db.init_db()                # SQLModel.metadata.create_all(get_engine())
    try:
        yield str(path)
    finally:
        # Keep the sqlite file (debugging) but restore the previous DB engine target so
        # the backtest DB never leaks into subsequent (live) code paths or other tests.
        common_db.configure_db(prior_db_file)


def seed_account_definition(
    account_id: int,
    settings: Optional[Dict[str, Any]] = None,
    *,
    name: Optional[str] = None,
    provider: str = "backtest",
    description: Optional[str] = None,
) -> int:
    """Insert an ``AccountDefinition`` row for the BacktestAccount into the backtest DB.

    The inherited ba2_common code loads ``AccountDefinition`` by id (e.g. when
    constructing/validating the account), so the row must exist before the engine
    drives the loop. Returns the row id (== ``account_id`` since we set the PK).

    ``settings`` is accepted for caller symmetry (the BacktestAccount's resolved
    config dict) but is NOT persisted onto AccountDefinition: AccountDefinition has no
    settings column; account settings live in the separate AccountSetting table and
    the BacktestAccount carries its config dict in-process. Passing it here is a no-op
    beyond documenting intent.
    """
    from ba2_common.core.models import AccountDefinition
    from ba2_common.core.db import add_instance

    row = AccountDefinition(
        id=int(account_id),
        name=name or f"backtest-{account_id}",
        provider=provider,
        description=description or f"Backtest simulated broker (run account {account_id})",
    )
    return add_instance(row)
