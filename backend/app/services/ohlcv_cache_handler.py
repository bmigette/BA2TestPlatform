"""
OHLCV Cache Fetch Handler

Background task handler for prefetching and caching OHLCV data
for multiple symbols and timeframes.
"""

import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Dict, Any

import pandas as pd

from app.services.task_queue import get_task_queue

logger = logging.getLogger(__name__)


def _parse_dates(payload: Dict[str, Any]):
    """Parse start/end dates from payload, defaulting to 15-year window."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=15 * 365)
    raw_start = payload.get('start_date')
    raw_end = payload.get('end_date')
    if raw_start:
        start_date = datetime.strptime(raw_start, '%Y-%m-%d')
    if raw_end:
        end_date = datetime.strptime(raw_end, '%Y-%m-%d')
    return start_date, end_date


def _fetch_symbol_timeframes(
    symbol: str,
    timeframes: list,
    provider,
    start_date: datetime,
    end_date: datetime,
    executor_workers: int,
    task_id: str,
    task_queue,
    progress_offset: float,
    progress_range: float,
    lock: threading.Lock,
    completed_units: list,
    total_units: int,
) -> Dict[str, Any]:
    """
    Fetch all timeframes for a single symbol, updating task progress.
    Returns per-timeframe result dict.
    """
    results = {}

    def fetch_timeframe(tf: str):
        try:
            cache_file = provider._get_cache_file(symbol, tf)
            if cache_file.exists():
                try:
                    cached = pd.read_csv(cache_file, usecols=['Date'])
                    if not cached.empty:
                        cached['Date'] = pd.to_datetime(cached['Date'])
                        c_min = cached['Date'].min().strftime('%Y-%m-%d')
                        c_max = cached['Date'].max().strftime('%Y-%m-%d')
                        req_start = start_date.strftime('%Y-%m-%d')
                        req_end = end_date.strftime('%Y-%m-%d')
                        if c_min <= req_start and c_max >= req_end:
                            cache_msg = f"Already cached ({c_min} to {c_max}), skipping"
                        else:
                            cache_msg = f"Extending {c_min}..{c_max} to {req_start}–{req_end}"
                except Exception:
                    cache_msg = "Cache unreadable, refetching"
            else:
                cache_msg = f"Fetching {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"

            with lock:
                pct = progress_offset + (completed_units[0] / total_units) * progress_range
                task_queue.update_progress(task_id, pct, f"{symbol}/{tf}: {cache_msg}")

            def make_progress_callback(timeframe: str):
                def callback(pct_inner: float, msg: str) -> None:
                    with lock:
                        overall = progress_offset + (completed_units[0] / total_units) * progress_range
                        task_queue.update_progress(task_id, overall, msg)
                return callback

            df = provider.extend_ohlcv_cache(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                interval=tf,
                progress_callback=make_progress_callback(tf),
                executor_workers=executor_workers,
            )
            rows = len(df) if df is not None else 0
            result = {'status': 'success', 'rows': rows}
            logger.info(f"Cached {symbol}/{tf}: {rows} rows")
        except Exception as e:
            result = {'status': 'error', 'error': str(e)}
            logger.error(f"Error caching {symbol}/{tf}: {e}")

        with lock:
            completed_units[0] += 1
            pct = progress_offset + (completed_units[0] / total_units) * progress_range
            msg = (
                f"Done {symbol}/{tf}: {result.get('rows', '?')} rows"
                if result.get('status') == 'success'
                else f"Failed {symbol}/{tf}"
            )
            task_queue.update_progress(task_id, pct, msg)

        return tf, result

    with ThreadPoolExecutor(max_workers=min(8, len(timeframes))) as executor:
        futures = {executor.submit(fetch_timeframe, tf): tf for tf in timeframes}
        for future in as_completed(futures):
            tf, result = future.result()
            results[tf] = result

    return results


def handle_ohlcv_cache_fetch(task_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Background task handler for OHLCV cache fetching (single symbol).

    Args:
        task_id: Task ID for progress tracking
        payload: Dict with keys:
            - provider: str
            - symbol: str
            - timeframes: list[str]
            - start_date: str ISO date (optional)
            - end_date: str ISO date (optional)
            - executor_workers: int (default 5) — gap-fill parallelism

    Returns:
        Summary dict with status and results per timeframe
    """
    from app.api.datasets import get_ohlcv_provider

    task_queue = get_task_queue()
    provider_name = payload.get('provider', 'yfinance')
    symbol = payload.get('symbol', '')
    timeframes = payload.get('timeframes', ['1d'])
    executor_workers = int(payload.get('executor_workers', 5))

    if not symbol:
        return {'status': 'failed', 'error': 'symbol is required'}

    start_date, end_date = _parse_dates(payload)
    provider = get_ohlcv_provider(provider_name)
    task_queue.update_progress(task_id, 0, f"Starting {symbol} ({len(timeframes)} timeframes)...")

    completed_units = [0]
    lock = threading.Lock()
    results = _fetch_symbol_timeframes(
        symbol=symbol,
        timeframes=timeframes,
        provider=provider,
        start_date=start_date,
        end_date=end_date,
        executor_workers=executor_workers,
        task_id=task_id,
        task_queue=task_queue,
        progress_offset=0.0,
        progress_range=100.0,
        lock=lock,
        completed_units=completed_units,
        total_units=len(timeframes),
    )

    task_queue.update_progress(task_id, 100, f"Completed {symbol}")
    return {
        'status': 'completed',
        'symbol': symbol,
        'provider': provider_name,
        'results': results,
    }


def handle_ohlcv_cache_batch(task_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Background task handler for batch OHLCV cache fetching (multiple symbols).

    Processes symbols in parallel using parallel_jobs workers, with each
    symbol's gap fills running with executor_workers workers.

    Args:
        task_id: Task ID for progress tracking
        payload: Dict with keys:
            - provider: str
            - symbols: list[str]
            - timeframes: list[str]
            - start_date: str ISO date (optional)
            - end_date: str ISO date (optional)
            - parallel_jobs: int (default 3) — symbol-level parallelism
            - executor_workers: int (default 5) — gap-fill parallelism per symbol

    Returns:
        Summary dict with status and results per symbol
    """
    from app.api.datasets import get_ohlcv_provider

    task_queue = get_task_queue()
    provider_name = payload.get('provider', 'yfinance')
    symbols = payload.get('symbols', [])
    timeframes = payload.get('timeframes', ['1d'])
    parallel_jobs = int(payload.get('parallel_jobs', 3))
    executor_workers = int(payload.get('executor_workers', 5))

    if not symbols:
        return {'status': 'failed', 'error': 'symbols list is required'}

    start_date, end_date = _parse_dates(payload)
    provider = get_ohlcv_provider(provider_name)

    total_units = len(symbols) * len(timeframes)
    completed_units = [0]
    lock = threading.Lock()
    all_results: Dict[str, Any] = {}

    task_queue.update_progress(
        task_id, 0,
        f"Starting {len(symbols)} symbols x {len(timeframes)} timeframes "
        f"({parallel_jobs} parallel jobs, {executor_workers} gap-fill workers)"
    )

    def process_symbol(symbol: str) -> tuple:
        results = _fetch_symbol_timeframes(
            symbol=symbol,
            timeframes=timeframes,
            provider=provider,
            start_date=start_date,
            end_date=end_date,
            executor_workers=executor_workers,
            task_id=task_id,
            task_queue=task_queue,
            progress_offset=0.0,
            progress_range=100.0,
            lock=lock,
            completed_units=completed_units,
            total_units=total_units,
        )
        return symbol, results

    with ThreadPoolExecutor(max_workers=parallel_jobs) as executor:
        futures = {executor.submit(process_symbol, sym): sym for sym in symbols}
        for future in as_completed(futures):
            sym, results = future.result()
            all_results[sym] = results

    task_queue.update_progress(task_id, 100, f"Completed {len(symbols)} symbols")
    return {
        'status': 'completed',
        'symbols': symbols,
        'provider': provider_name,
        'parallel_jobs': parallel_jobs,
        'executor_workers': executor_workers,
        'results': all_results,
    }
