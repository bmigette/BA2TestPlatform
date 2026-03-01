"""
OHLCV Cache Fetch Handler

Background task handler for prefetching and caching OHLCV data
for multiple symbols and timeframes.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any

from app.services.task_queue import get_task_queue

logger = logging.getLogger(__name__)


def handle_ohlcv_cache_fetch(task_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Background task handler for OHLCV cache fetching.

    Fetches OHLCV data for a single symbol across multiple timeframes,
    forcing cache refresh.

    Args:
        task_id: Task ID for progress tracking
        payload: Dict with keys:
            - provider: str (e.g., 'yfinance', 'fmp')
            - symbol: str (e.g., 'AAPL')
            - timeframes: list[str] (e.g., ['1d', '1h', '4h'])

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

    provider = get_ohlcv_provider(provider_name)
    results = {}
    total = len(timeframes)

    # Default date range: 15 years back from today
    end_date = datetime.now()
    start_date = end_date - timedelta(days=15 * 365)

    for i, tf in enumerate(timeframes):
        progress = ((i) / total) * 100
        task_queue.update_progress(task_id, progress, f"Fetching {symbol} {tf}...")

        try:
            df = provider.get_ohlcv_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                interval=tf,
                use_cache=False  # Force fresh fetch
            )
            rows = len(df) if df is not None else 0
            results[tf] = {'status': 'success', 'rows': rows}
            logger.info(f"Cached {symbol} {tf}: {rows} rows")
        except Exception as e:
            results[tf] = {'status': 'error', 'error': str(e)}
            logger.error(f"Error caching {symbol} {tf}: {e}")

    task_queue.update_progress(task_id, 100, f"Completed {symbol}")

    return {
        'status': 'completed',
        'symbol': symbol,
        'provider': provider_name,
        'results': results
    }
