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


def handle_ohlcv_cache_fetch(task_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Background task handler for OHLCV cache fetching.

    Fetches OHLCV data for a single symbol across multiple timeframes using
    extend-only semantics: already-cached date ranges are not re-fetched.

    Args:
        task_id: Task ID for progress tracking
        payload: Dict with keys:
            - provider: str (e.g., 'yfinance', 'fmp')
            - symbol: str (e.g., 'AAPL')
            - timeframes: list[str] (e.g., ['1d', '1h', '4h'])
            - start_date: str ISO date 'YYYY-MM-DD' (optional, default 15yr ago)
            - end_date: str ISO date 'YYYY-MM-DD' (optional, default today)

    Returns:
        Summary dict with status and results per timeframe
    """
    from app.api.datasets import get_ohlcv_provider

    task_queue = get_task_queue()
    provider_name = payload.get('provider', 'yfinance')
    symbol = payload.get('symbol', '')
    timeframes = payload.get('timeframes', ['1d'])

    if not symbol:
        return {'status': 'failed', 'error': 'symbol is required'}

    # Parse date range from payload or fall back to 15-year default
    end_date = datetime.now()
    start_date = end_date - timedelta(days=15 * 365)

    raw_start = payload.get('start_date')
    raw_end = payload.get('end_date')
    if raw_start:
        start_date = datetime.strptime(raw_start, '%Y-%m-%d')
    if raw_end:
        end_date = datetime.strptime(raw_end, '%Y-%m-%d')

    provider = get_ohlcv_provider(provider_name)
    task_queue.update_progress(task_id, 0, f"Starting {symbol} ({len(timeframes)} timeframes)...")

    results = {}
    total = len(timeframes)
    completed_count = [0]
    lock = threading.Lock()

    def fetch_timeframe(tf: str):
        try:
            # Check existing cache range and report status
            cache_file = provider._get_cache_file(symbol, tf)
            cache_msg = ""
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
                            cache_msg = f"Cache has {c_min} to {c_max}, extending to {req_start}–{req_end}"
                except Exception:
                    cache_msg = "Existing cache unreadable, refetching"
            else:
                cache_msg = f"No cache, fetching {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"

            with lock:
                task_queue.update_progress(
                    task_id,
                    (completed_count[0] / total) * 100,
                    f"{symbol}/{tf}: {cache_msg}"
                )

            def make_progress_callback(timeframe: str):
                def callback(pct: float, msg: str) -> None:
                    with lock:
                        task_queue.update_progress(task_id, (completed_count[0] / total) * 100, msg)
                return callback

            df = provider.extend_ohlcv_cache(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                interval=tf,
                progress_callback=make_progress_callback(tf),
            )
            rows = len(df) if df is not None else 0
            result = {'status': 'success', 'rows': rows}
            logger.info(f"Cached {symbol} {tf}: {rows} rows")
        except Exception as e:
            result = {'status': 'error', 'error': str(e)}
            logger.error(f"Error caching {symbol} {tf}: {e}")

        with lock:
            completed_count[0] += 1
            progress = (completed_count[0] / total) * 100
            msg = f"Done {symbol}/{tf}: {result.get('rows', '?')} rows" if result.get('status') == 'success' else f"Failed {symbol}/{tf}"
            task_queue.update_progress(task_id, progress, msg)

        return tf, result

    with ThreadPoolExecutor(max_workers=min(8, total)) as executor:
        futures = {executor.submit(fetch_timeframe, tf): tf for tf in timeframes}
        for future in as_completed(futures):
            tf, result = future.result()
            results[tf] = result

    task_queue.update_progress(task_id, 100, f"Completed {symbol}")

    return {
        'status': 'completed',
        'symbol': symbol,
        'provider': provider_name,
        'results': results
    }
