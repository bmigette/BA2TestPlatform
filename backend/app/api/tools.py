"""
Tools API endpoints

Provides endpoints for testing and debugging various providers.
"""

from fastapi import APIRouter, HTTPException, status, Query
from typing import Optional
from datetime import datetime, timedelta
import logging

from app.services.sentiment import SentimentService

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/news/fetch")
async def fetch_news(
    symbol: str = Query(..., description="Stock ticker symbol (e.g., AAPL)"),
    provider: str = Query("fmp", description="News provider (fmp, alpaca)"),
    days: int = Query(30, description="Number of days to look back"),
    limit: int = Query(50, description="Maximum number of articles")
):
    """
    Fetch news articles for a symbol without sentiment analysis.

    Args:
        symbol: Stock ticker symbol
        provider: News provider to use
        days: Days to look back
        limit: Maximum articles to return

    Returns:
        List of news articles
    """
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        logger.info(f"Fetching news for {symbol} from {provider}, last {days} days")

        sentiment_service = SentimentService()
        articles = sentiment_service.fetch_news_for_ticker(
            ticker=symbol,
            start_date=start_date,
            end_date=end_date,
            provider=provider,
            enrich_content=False
        )

        # Limit results
        articles = articles[:limit] if articles else []

        # Convert dates to strings for JSON serialization
        for article in articles:
            if isinstance(article.get('date'), datetime):
                article['date'] = article['date'].isoformat()
            if isinstance(article.get('published_at'), datetime):
                article['published_at'] = article['published_at'].isoformat()

        return {
            "symbol": symbol,
            "provider": provider,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "article_count": len(articles),
            "articles": articles
        }

    except Exception as e:
        logger.error(f"Error fetching news: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch news: {str(e)}"
        )


@router.post("/news/analyze-single")
async def analyze_single_article(
    title: str = Query(..., description="Article title"),
    content: str = Query("", description="Article content/summary")
):
    """
    Analyze sentiment for a single news article.

    Args:
        title: Article title
        content: Article content or summary

    Returns:
        Sentiment analysis result
    """
    try:
        logger.info(f"Analyzing sentiment for article: {title[:50]}...")

        sentiment_service = SentimentService()

        # Create a mock article for analysis
        article = {
            'title': title,
            'summary': content,
            'content': content,
            'date': datetime.now()
        }

        # Analyze the article
        analyzed = sentiment_service.analyze_news_articles([article])

        if analyzed:
            result = analyzed[0]
            return {
                "title": title,
                "sentiment": result.get('sentiment', 'neutral'),
                "sentiment_score": result.get('sentiment_score', 0.5),
                "confidence": result.get('confidence', 0.0),
                "model_used": "FinBERT"
            }
        else:
            return {
                "title": title,
                "sentiment": "neutral",
                "sentiment_score": 0.5,
                "confidence": 0.0,
                "error": "Analysis returned no results"
            }

    except Exception as e:
        logger.error(f"Error analyzing sentiment: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to analyze sentiment: {str(e)}"
        )


@router.get("/news/providers")
async def list_news_providers():
    """
    List available news providers and their status.

    Returns:
        List of providers with availability info
    """
    import os

    providers = [
        {
            "id": "fmp",
            "name": "Financial Modeling Prep",
            "description": "Company and market news from FMP API",
            "requires_api_key": True,
            "api_key_configured": bool(os.getenv("FMP_API_KEY")),
            "features": ["company_news", "market_news"]
        },
        {
            "id": "alpaca",
            "name": "Alpaca Markets",
            "description": "News from Alpaca trading platform",
            "requires_api_key": True,
            "api_key_configured": bool(os.getenv("ALPACA_API_KEY")),
            "features": ["company_news"]
        }
    ]

    return {
        "providers": providers,
        "default": "fmp"
    }
