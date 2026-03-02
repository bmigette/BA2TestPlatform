"""
News Cache Fetch Handler

Background task handler for prefetching and caching news articles
for multiple symbols using the SentimentService pipeline.
"""

import logging
from datetime import datetime
from typing import Dict, Any

from app.services.task_queue import get_task_queue

logger = logging.getLogger(__name__)


def handle_news_cache_fetch(task_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Background task handler for news cache fetching.

    Fetches and caches news articles for a single symbol,
    optionally enriching with full article content.

    Args:
        task_id: Task ID for progress tracking
        payload: Dict with keys:
            - provider: str (e.g., 'fmp', 'finnhub')
            - symbol: str (e.g., 'AAPL')
            - start_date: str (ISO date, e.g., '2024-01-01')
            - end_date: str (ISO date, e.g., '2025-01-01')
            - enrich_content: bool (whether to fetch full article text)

    Returns:
        Summary dict with status and article count
    """
    from app.services.sentiment import SentimentService

    task_queue = get_task_queue()
    provider = payload.get('provider', 'fmp')
    symbol = payload.get('symbol', '')
    start_date_str = payload.get('start_date', '')
    end_date_str = payload.get('end_date', '')
    enrich_content = payload.get('enrich_content', True)

    if not symbol:
        return {'status': 'failed', 'error': 'symbol is required'}
    if not start_date_str or not end_date_str:
        return {'status': 'failed', 'error': 'start_date and end_date are required'}

    try:
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
    except ValueError as e:
        return {'status': 'failed', 'error': f'Invalid date format: {e}'}

    task_queue.update_progress(task_id, 10, f"Fetching news for {symbol}...")

    try:
        sentiment_service = SentimentService()
        articles = sentiment_service.fetch_news_for_ticker(
            ticker=symbol,
            start_date=start_date,
            end_date=end_date,
            provider=provider,
            enrich_content=enrich_content,
            use_cache=True
        )

        task_queue.update_progress(task_id, 90, f"Cached {len(articles)} articles for {symbol}")

        # Count enriched articles
        enriched_count = sum(1 for a in articles if a.get('content_fetched'))

        task_queue.update_progress(task_id, 100, f"Completed {symbol}: {len(articles)} articles")

        return {
            'status': 'completed',
            'symbol': symbol,
            'provider': provider,
            'article_count': len(articles),
            'enriched_count': enriched_count,
            'start_date': start_date_str,
            'end_date': end_date_str
        }

    except Exception as e:
        logger.error(f"Error caching news for {symbol}: {e}", exc_info=True)
        return {
            'status': 'failed',
            'symbol': symbol,
            'provider': provider,
            'error': str(e)
        }
