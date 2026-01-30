"""
Base classes for news providers.

Provides the MarketNewsInterface abstract base class that all news providers
should inherit from, along with utility functions for date handling and
content fetching with trafilatura.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, Any, Literal, Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import time
import requests

logger = logging.getLogger(__name__)

# Import trafilatura for content fetching (required dependency)
import trafilatura

# Headers for resolving Finnhub redirect URLs
BROWSER_HEADERS = {
    'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
    'accept-language': 'en-US,en;q=0.9',
    'sec-ch-ua': '"Not(A:Brand";v="8", "Chromium";v="131"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"Windows"',
    'sec-fetch-dest': 'document',
    'sec-fetch-mode': 'navigate',
    'sec-fetch-site': 'none',
    'sec-fetch-user': '?1',
    'upgrade-insecure-requests': '1',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36'
}


class MarketNewsInterface(ABC):
    """
    Abstract base class for market news providers.

    All news providers should inherit from this class and implement
    the required abstract methods.
    """

    def __init__(self):
        """Initialize the news provider."""
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        """Get the provider name (e.g., 'fmp', 'alphavantage')."""
        pass

    @abstractmethod
    def get_supported_features(self) -> list[str]:
        """Get list of supported features (e.g., ['company_news', 'global_news'])."""
        pass

    @abstractmethod
    def validate_config(self) -> bool:
        """Validate provider configuration (e.g., API keys)."""
        pass

    @abstractmethod
    def _format_as_dict(self, data: Any) -> Dict[str, Any]:
        """Format data as a structured dictionary."""
        pass

    @abstractmethod
    def _format_as_markdown(self, data: Any) -> str:
        """Format data as markdown."""
        pass

    @abstractmethod
    def get_company_news(
        self,
        symbol: str,
        end_date: datetime,
        start_date: Optional[datetime] = None,
        lookback_days: Optional[int] = None,
        limit: int = 50,
        format_type: Literal["dict", "markdown", "both"] = "markdown"
    ) -> Dict[str, Any] | str:
        """
        Get news articles for a specific company.

        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL', 'MSFT')
            end_date: End date (inclusive)
            start_date: Start date (use either this OR lookback_days, not both)
            lookback_days: Days to look back from end_date
            limit: Maximum number of articles to return
            format_type: Output format ('dict', 'markdown', or 'both')

        Returns:
            News data in requested format
        """
        pass

    @abstractmethod
    def get_global_news(
        self,
        end_date: datetime,
        start_date: Optional[datetime] = None,
        lookback_days: Optional[int] = None,
        limit: int = 50,
        format_type: Literal["dict", "markdown", "both"] = "markdown"
    ) -> Dict[str, Any] | str:
        """
        Get global/market news (not specific to any company).

        Args:
            end_date: End date (inclusive)
            start_date: Start date (use either this OR lookback_days, not both)
            lookback_days: Days to look back from end_date
            limit: Maximum number of articles to return
            format_type: Output format ('dict', 'markdown', or 'both')

        Returns:
            News data in requested format
        """
        pass

    # Content fetching utilities (shared by all providers)

    @staticmethod
    def fetch_url_content(url: str, timeout: int = 10) -> Optional[str]:
        """
        Fetch article content from URL using trafilatura.

        Args:
            url: Article URL to fetch
            timeout: Request timeout in seconds

        Returns:
            Extracted text content or None if failed
        """
        try:
            downloaded = trafilatura.fetch_url(url)
            if downloaded:
                text = trafilatura.extract(downloaded)
                return text
        except Exception as e:
            logger.debug(f"Failed to fetch content from {url}: {e}")

        return None

    @staticmethod
    def resolve_finnhub_redirect(url: str, max_retries: int = 3) -> Optional[str]:
        """
        Resolve Finnhub redirect URL to get the actual article URL.

        Finnhub returns URLs like https://finnhub.io/api/news?id=xxx that
        redirect to the actual article. This follows the redirect.

        Args:
            url: Finnhub redirect URL
            max_retries: Maximum number of retry attempts

        Returns:
            Resolved article URL or None if failed
        """
        delays = [3, 5, 15]  # Exponential backoff delays

        for attempt in range(max_retries):
            try:
                # Use GET with allow_redirects=True to follow the full redirect chain
                # HEAD requests often return incomplete Location headers (e.g., "/")
                response = requests.get(
                    url,
                    headers=BROWSER_HEADERS,
                    allow_redirects=True,
                    timeout=10
                )

                # Return final URL if it differs from the original
                if response.url != url and not response.url.endswith('/'):
                    return response.url

                return None

            except requests.RequestException as e:
                if attempt < max_retries - 1:
                    delay = delays[attempt] if attempt < len(delays) else delays[-1]
                    time.sleep(delay)
                else:
                    logger.warning(f"Failed to resolve Finnhub redirect after {max_retries} attempts: {url}")

        return None

    def resolve_finnhub_redirects(
        self,
        articles: List[Dict[str, Any]],
        max_workers: int = 8
    ) -> List[Dict[str, Any]]:
        """
        Resolve Finnhub redirect URLs in articles to actual article URLs.

        Args:
            articles: List of article dicts with 'url' key
            max_workers: Maximum parallel resolution threads

        Returns:
            Articles with resolved URLs
        """
        # Find articles with Finnhub redirect URLs
        finnhub_articles = []
        for i, article in enumerate(articles):
            url = article.get('url', '')
            if 'finnhub.io/api/' in url:
                finnhub_articles.append((i, url))

        if not finnhub_articles:
            return articles

        logger.info(f"Resolving {len(finnhub_articles)} Finnhub redirect URLs")

        # Resolve in parallel with limited workers
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(self.resolve_finnhub_redirect, url): idx
                for idx, url in finnhub_articles
            }

            resolved_count = 0
            processed_count = 0
            total_count = len(finnhub_articles)
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    resolved_url = future.result()
                    if resolved_url:
                        # Store resolved URL separately, keep original for cache indexing
                        articles[idx]['resolved_url'] = resolved_url
                        articles[idx]['finnhub_url_resolved'] = True
                        resolved_count += 1
                except Exception as e:
                    logger.debug(f"Error resolving Finnhub URL for article {idx}: {e}")

                processed_count += 1
                if processed_count % 100 == 0:
                    logger.info(f"Progress: {processed_count}/{total_count} URLs processed ({resolved_count} resolved)")

        logger.info(f"Resolved {resolved_count}/{len(finnhub_articles)} Finnhub redirect URLs")
        return articles

    def enrich_articles_with_content(
        self,
        articles: List[Dict[str, Any]],
        max_workers: int = 8,
        min_summary_length: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Enrich articles that have short/missing summaries by fetching URL content.
        Uses ThreadPoolExecutor for parallel fetching.

        Args:
            articles: List of article dicts with 'url' and 'summary' keys
            max_workers: Maximum parallel fetch threads
            min_summary_length: Minimum summary length before fetching is attempted

        Returns:
            Articles with enriched summaries
        """
        # First, resolve any Finnhub redirect URLs to actual article URLs
        articles = self.resolve_finnhub_redirects(articles, max_workers=max_workers)

        # Find articles needing enrichment
        needs_enrichment = []
        for i, article in enumerate(articles):
            summary = article.get('summary', '') or ''
            # Use resolved_url for content fetching if available (for Finnhub redirects)
            url = article.get('resolved_url') or article.get('url', '')
            if len(summary) < min_summary_length and url:
                # Skip unresolved Finnhub API URLs (they can't be scraped)
                if 'finnhub.io/api/' in url:
                    continue
                needs_enrichment.append((i, url))

        if not needs_enrichment:
            return articles

        logger.info(f"Enriching {len(needs_enrichment)} articles with URL content")

        # Fetch content in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(self.fetch_url_content, url): idx
                for idx, url in needs_enrichment
            }

            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    content = future.result()
                    if content and len(content) > len(articles[idx].get('summary', '') or ''):
                        # Truncate to reasonable length for ML processing
                        articles[idx]['summary'] = content[:2000]
                        articles[idx]['content_fetched'] = True
                except Exception as e:
                    logger.debug(f"Error enriching article {idx}: {e}")

        return articles


# Utility functions (replacing ba2_trade_platform.core.provider_utils)

def validate_date_range(
    start_date: datetime,
    end_date: datetime,
    max_days: int = 365
) -> tuple[datetime, datetime]:
    """
    Validate and normalize date range.

    Args:
        start_date: Start date
        end_date: End date
        max_days: Maximum allowed days between dates

    Returns:
        Tuple of (start_date, end_date)

    Raises:
        ValueError: If date range is invalid
    """
    if start_date > end_date:
        raise ValueError("start_date must be before end_date")

    delta = (end_date - start_date).days
    if delta > max_days:
        raise ValueError(f"Date range exceeds maximum of {max_days} days")

    return start_date, end_date


def validate_lookback_days(lookback_days: int, max_lookback: int = 365) -> int:
    """
    Validate lookback days.

    Args:
        lookback_days: Number of days to look back
        max_lookback: Maximum allowed lookback

    Returns:
        Validated lookback_days

    Raises:
        ValueError: If lookback_days is invalid
    """
    if lookback_days <= 0:
        raise ValueError("lookback_days must be positive")
    if lookback_days > max_lookback:
        raise ValueError(f"lookback_days exceeds maximum of {max_lookback}")
    return lookback_days


def calculate_date_range(
    end_date: datetime,
    lookback_days: int
) -> tuple[datetime, datetime]:
    """
    Calculate start date from end date and lookback days.

    Args:
        end_date: End date
        lookback_days: Number of days to look back

    Returns:
        Tuple of (start_date, end_date)
    """
    start_date = end_date - timedelta(days=lookback_days)
    return start_date, end_date
