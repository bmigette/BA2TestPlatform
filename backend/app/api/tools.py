"""
Tools API endpoints

Provides endpoints for testing and debugging various providers.
"""

from fastapi import APIRouter, HTTPException, status, Query
from fastapi.responses import JSONResponse
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from pathlib import Path
import logging
import json
import uuid

from app.services.sentiment import SentimentService

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/news/fetch")
async def fetch_news(
    symbol: Optional[str] = Query(None, description="Stock ticker symbol (e.g., AAPL). Leave empty for global news."),
    provider: str = Query("fmp", description="News provider (fmp, alpaca, alphavantage)"),
    news_type: str = Query("company", description="Type of news: 'company' (requires symbol) or 'global' (market/general news)"),
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD). If not provided, defaults to 30 days ago."),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD). If not provided, defaults to today."),
    days: Optional[int] = Query(None, description="Deprecated: Use start_date/end_date instead. Number of days to look back."),
    limit: int = Query(50, description="Maximum number of articles")
):
    """
    Fetch news articles for a symbol or global market news.

    Args:
        symbol: Stock ticker symbol (required for company news, optional for global)
        provider: News provider to use
        news_type: 'company' for ticker-specific news, 'global' for market news
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        days: Deprecated, use date range instead
        limit: Maximum articles to return

    Returns:
        List of news articles
    """
    try:
        # Validate inputs
        if news_type == "company" and not symbol:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Symbol is required for company news"
            )

        # Parse dates or use defaults
        if end_date:
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        else:
            end_dt = datetime.now()

        if start_date:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        elif days:
            # Legacy support for days parameter
            start_dt = end_dt - timedelta(days=days)
        else:
            # Default to last 30 days
            start_dt = end_dt - timedelta(days=30)

        logger.info(f"Fetching {news_type} news for {symbol or 'global'} from {provider}, {start_dt.date()} to {end_dt.date()}")

        sentiment_service = SentimentService()

        if news_type == "global":
            # Fetch global/market news
            articles = sentiment_service.fetch_global_news(
                start_date=start_dt,
                end_date=end_dt,
                provider=provider
            )
        else:
            # Fetch company-specific news
            articles = sentiment_service.fetch_news_for_ticker(
                ticker=symbol,
                start_date=start_dt,
                end_date=end_dt,
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
            "symbol": symbol or "global",
            "news_type": news_type,
            "provider": provider,
            "start_date": start_dt.isoformat(),
            "end_date": end_dt.isoformat(),
            "article_count": len(articles),
            "articles": articles
        }

    except HTTPException:
        raise
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
            "features": ["company_news", "market_news"],
            "has_sentiment": False
        },
        {
            "id": "alphavantage",
            "name": "Alpha Vantage",
            "description": "News with built-in sentiment analysis from Alpha Vantage API",
            "requires_api_key": True,
            "api_key_configured": bool(os.getenv("ALPHA_VANTAGE_API_KEY")),
            "features": ["company_news", "sentiment_analysis"],
            "has_sentiment": True
        },
        {
            "id": "finnhub",
            "name": "Finnhub",
            "description": "Company and market news from Finnhub API",
            "requires_api_key": True,
            "api_key_configured": bool(os.getenv("FINNHUB_API_KEY")),
            "features": ["company_news", "global_news"],
            "has_sentiment": False
        },
        {
            "id": "alpaca",
            "name": "Alpaca Markets",
            "description": "News from Alpaca trading platform",
            "requires_api_key": True,
            "api_key_configured": bool(os.getenv("ALPACA_API_KEY")),
            "features": ["company_news"],
            "has_sentiment": False
        },
        {
            "id": "localfiles",
            "name": "Local Files",
            "description": "Read from previously exported JSON files",
            "requires_api_key": False,
            "api_key_configured": True,
            "features": ["company_news", "cached_sentiment"],
            "has_sentiment": True
        }
    ]

    return {
        "providers": providers,
        "default": "fmp"
    }


# Directory for exported news files
NEWS_EXPORTS_DIR = Path("news_exports")


@router.post("/news/export")
async def export_news_to_json(
    symbol: Optional[str] = Query(None, description="Stock ticker symbol (required for company news)"),
    provider: str = Query(..., description="Provider used to fetch the news"),
    news_type: str = Query("company", description="Type of news: 'company' or 'global'"),
    articles: List[Dict[str, Any]] = None
):
    """
    Export news articles to a JSON file with standardized format.

    The exported format can be imported using the LocalFiles news provider.

    Args:
        symbol: Stock ticker symbol (required for company news, ignored for global)
        provider: Original provider name
        news_type: Type of news ('company' or 'global')
        articles: List of articles to export

    Returns:
        Export file path and metadata
    """
    if not articles:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No articles provided for export"
        )

    if news_type == "company" and not symbol:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Symbol is required for company news export"
        )

    try:
        # Ensure export directory exists
        NEWS_EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

        # Generate filename based on news type
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if news_type == "global":
            filename = f"global_{provider}_{timestamp}.json"
            symbol_value = "global"
        else:
            filename = f"{symbol}_{provider}_{timestamp}.json"
            symbol_value = symbol

        filepath = NEWS_EXPORTS_DIR / filename

        # Standardize article format for export
        export_data = {
            "version": "1.1",
            "export_date": datetime.now().isoformat(),
            "news_type": news_type,
            "symbol": symbol_value,
            "provider": provider,
            "article_count": len(articles),
            "articles": []
        }

        for article in articles:
            # Standardize date format
            date = article.get("date") or article.get("published_at") or ""
            if isinstance(date, datetime):
                date = date.isoformat()

            export_data["articles"].append({
                "title": article.get("title", ""),
                "summary": article.get("summary") or article.get("content", ""),
                "source": article.get("source", ""),
                "url": article.get("url", ""),
                "published_at": date,
                "sentiment": article.get("sentiment"),
                "sentiment_score": article.get("sentiment_score")
            })

        # Write to file
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Exported {len(articles)} articles to {filepath}")

        return {
            "success": True,
            "filename": filename,
            "filepath": str(filepath),
            "article_count": len(articles),
            "message": f"Exported {len(articles)} articles to {filename}"
        }

    except Exception as e:
        logger.error(f"Error exporting news: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to export news: {str(e)}"
        )


@router.get("/news/exports")
async def list_news_exports():
    """
    List all exported news files.

    Returns:
        List of export files with metadata
    """
    exports = []

    if NEWS_EXPORTS_DIR.exists():
        for filepath in NEWS_EXPORTS_DIR.glob("*.json"):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                exports.append({
                    "filename": filepath.name,
                    "filepath": str(filepath),
                    "symbol": data.get("symbol", ""),
                    "provider": data.get("provider", ""),
                    "article_count": data.get("article_count", 0),
                    "export_date": data.get("export_date", ""),
                    "size_kb": round(filepath.stat().st_size / 1024, 2)
                })
            except Exception as e:
                logger.warning(f"Error reading export file {filepath}: {e}")

    return {
        "exports": sorted(exports, key=lambda x: x["export_date"], reverse=True),
        "count": len(exports),
        "directory": str(NEWS_EXPORTS_DIR)
    }
