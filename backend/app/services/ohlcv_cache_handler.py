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
    results = {}
    total = len(timeframes)
    completed_count = [0]
    lock = threading.Lock()

    def fetch_timeframe(tf: str):
        try:
            df = provider.extend_ohlcv_cache(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                interval=tf
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
            task_queue.update_progress(task_id, progress, f"Fetched {symbol} {tf}")

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
