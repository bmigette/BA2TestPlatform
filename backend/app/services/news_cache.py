"""
News Cache Service

Handles caching of news articles to avoid redundant fetching and sentiment analysis.
Articles are indexed in database and content is stored in files on disk.
"""

import hashlib
import json
import base64
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from sqlalchemy.orm import Session

from app.models.database import SessionLocal
from app.models.news_cache import NewsCache

logger = logging.getLogger(__name__)


class NewsCacheService:
    """
    Service for caching news articles.

    Architecture:
    - Database stores article metadata, URLs, and sentiment results
    - File system stores article content (base64 encoded JSON)
    - Files organized by provider: datasets/cache/news/{provider}/

    Usage:
        cache = NewsCacheService()

        # Check if article is cached
        cached = cache.get_cached_article(url, provider)
        if cached:
            return cached

        # Cache new article
        cache.cache_article(article, provider, ticker)
    """

    def __init__(self, cache_dir: str = "datasets/cache/news"):
        """
        Initialize NewsCacheService.

        Args:
            cache_dir: Base directory for news cache files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_url_hash(self, url: str) -> str:
        """Generate SHA256 hash of URL for indexing."""
        return hashlib.sha256(url.encode('utf-8')).hexdigest()

    def _get_provider_dir(self, provider: str) -> Path:
        """Get cache directory for a provider."""
        provider_dir = self.cache_dir / provider.lower()
        provider_dir.mkdir(parents=True, exist_ok=True)
        return provider_dir

    def _get_content_file_path(self, url_hash: str, provider: str) -> str:
        """Get relative path for content file."""
        return f"{provider.lower()}/{url_hash[:2]}/{url_hash}.json"

    def _save_content_file(self, content: str, file_path: str) -> bool:
        """
        Save article content to file as base64 encoded JSON.

        Args:
            content: Article content text
            file_path: Relative path from cache_dir

        Returns:
            True if saved successfully
        """
        try:
            full_path = self.cache_dir / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)

            # Create JSON structure and base64 encode
            data = {
                'content': content,
                'cached_at': datetime.now().isoformat()
            }
            json_str = json.dumps(data, ensure_ascii=False)
            encoded = base64.b64encode(json_str.encode('utf-8')).decode('ascii')

            with open(full_path, 'w') as f:
                f.write(encoded)

            return True

        except Exception as e:
            logger.error(f"Failed to save content file {file_path}: {e}")
            return False

    def _load_content_file(self, file_path: str) -> Optional[str]:
        """
        Load article content from base64 encoded JSON file.

        Args:
            file_path: Relative path from cache_dir

        Returns:
            Article content or None if not found
        """
        try:
            full_path = self.cache_dir / file_path
            if not full_path.exists():
                return None

            with open(full_path, 'r') as f:
                encoded = f.read()

            json_str = base64.b64decode(encoded.encode('ascii')).decode('utf-8')
            data = json.loads(json_str)
            return data.get('content', '')

        except Exception as e:
            logger.warning(f"Failed to load content file {file_path}: {e}")
            return None

    def get_cached_article(
        self,
        url: str,
        provider: str,
        db: Session = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached article by URL.

        Args:
            url: Article URL
            provider: News provider name
            db: Optional database session (creates new if not provided)

        Returns:
            Article dictionary or None if not cached
        """
        close_db = db is None
        if db is None:
            db = SessionLocal()

        try:
            url_hash = self._get_url_hash(url)
            cache_entry = db.query(NewsCache).filter(
                NewsCache.url_hash == url_hash
            ).first()

            if not cache_entry:
                return None

            # Load content from file
            content = None
            if cache_entry.content_file_path:
                content = self._load_content_file(cache_entry.content_file_path)

            return cache_entry.to_article_dict(content)

        finally:
            if close_db:
                db.close()

    def get_cached_articles_for_ticker(
        self,
        ticker: str,
        provider: str,
        start_date: datetime,
        end_date: datetime,
        db: Session = None
    ) -> List[Dict[str, Any]]:
        """
        Get all cached articles for a ticker in date range.

        Args:
            ticker: Stock ticker
            provider: News provider name
            start_date: Start of date range
            end_date: End of date range
            db: Optional database session

        Returns:
            List of article dictionaries
        """
        close_db = db is None
        if db is None:
            db = SessionLocal()

        try:
            cache_entries = db.query(NewsCache).filter(
                NewsCache.ticker == ticker,
                NewsCache.provider == provider.lower(),
                NewsCache.published_at >= start_date,
                NewsCache.published_at <= end_date
            ).all()

            articles = []
            for entry in cache_entries:
                content = None
                if entry.content_file_path:
                    content = self._load_content_file(entry.content_file_path)
                articles.append(entry.to_article_dict(content))

            return articles

        finally:
            if close_db:
                db.close()

    def cache_article(
        self,
        article: Dict[str, Any],
        provider: str,
        ticker: str = None,
        db: Session = None
    ) -> Optional[NewsCache]:
        """
        Cache an article.

        Args:
            article: Article dictionary with url, title, content, date, etc.
            provider: News provider name
            ticker: Optional ticker symbol
            db: Optional database session

        Returns:
            NewsCache entry or None if failed
        """
        url = article.get('url', '')
        if not url:
            logger.debug("Cannot cache article without URL")
            return None

        close_db = db is None
        if db is None:
            db = SessionLocal()

        try:
            url_hash = self._get_url_hash(url)

            # Check if already cached
            existing = db.query(NewsCache).filter(
                NewsCache.url_hash == url_hash
            ).first()

            if existing:
                # Update sentiment if newly analyzed
                if article.get('sentiment') and not existing.sentiment_label:
                    existing.sentiment_label = article.get('sentiment')
                    existing.sentiment_score = article.get('sentiment_score')
                    existing.positive_prob = article.get('positive_prob')
                    existing.neutral_prob = article.get('neutral_prob')
                    existing.negative_prob = article.get('negative_prob')
                    existing.analyzed_at = datetime.now()
                    db.commit()
                return existing

            # Save content to file
            content = article.get('content', '')
            content_file_path = None
            if content:
                content_file_path = self._get_content_file_path(url_hash, provider)
                self._save_content_file(content, content_file_path)

            # Parse published date
            pub_date = article.get('date')
            if isinstance(pub_date, str):
                try:
                    pub_date = datetime.fromisoformat(pub_date.replace('Z', '+00:00'))
                except ValueError:
                    pub_date = None

            # Create cache entry
            cache_entry = NewsCache(
                provider=provider.lower(),
                original_url=url,
                resolved_url=article.get('resolved_url'),
                url_hash=url_hash,
                ticker=ticker,
                title=article.get('title', '')[:500] if article.get('title') else None,
                source=article.get('source', '')[:200] if article.get('source') else None,
                published_at=pub_date,
                sentiment_label=article.get('sentiment'),
                sentiment_score=article.get('sentiment_score'),
                positive_prob=article.get('positive_prob'),
                neutral_prob=article.get('neutral_prob'),
                negative_prob=article.get('negative_prob'),
                content_file_path=content_file_path,
                content_fetched=1 if article.get('content_fetched') else 0,
                fetched_at=datetime.now(),
                analyzed_at=datetime.now() if article.get('sentiment') else None
            )

            db.add(cache_entry)
            db.commit()
            db.refresh(cache_entry)

            logger.debug(f"Cached article: {url[:80]}...")
            return cache_entry

        except Exception as e:
            db.rollback()
            logger.error(f"Failed to cache article: {e}")
            return None

        finally:
            if close_db:
                db.close()

    def cache_articles_batch(
        self,
        articles: List[Dict[str, Any]],
        provider: str,
        ticker: str = None
    ) -> Tuple[int, int]:
        """
        Cache multiple articles.

        Args:
            articles: List of article dictionaries
            provider: News provider name
            ticker: Optional ticker symbol

        Returns:
            Tuple of (cached_count, skipped_count)
        """
        db = SessionLocal()
        cached = 0
        skipped = 0

        try:
            for article in articles:
                result = self.cache_article(article, provider, ticker, db)
                if result:
                    cached += 1
                else:
                    skipped += 1

            logger.info(f"Cached {cached} articles, skipped {skipped}")
            return cached, skipped

        finally:
            db.close()

    def update_sentiment(
        self,
        url: str,
        sentiment_result: Dict[str, Any],
        db: Session = None
    ) -> bool:
        """
        Update sentiment for a cached article.

        Args:
            url: Article URL
            sentiment_result: Sentiment analysis result dict

        Returns:
            True if updated successfully
        """
        close_db = db is None
        if db is None:
            db = SessionLocal()

        try:
            url_hash = self._get_url_hash(url)
            cache_entry = db.query(NewsCache).filter(
                NewsCache.url_hash == url_hash
            ).first()

            if not cache_entry:
                return False

            cache_entry.sentiment_label = sentiment_result.get('label')
            cache_entry.sentiment_score = sentiment_result.get('score')
            cache_entry.positive_prob = sentiment_result.get('positive_prob')
            cache_entry.neutral_prob = sentiment_result.get('neutral_prob')
            cache_entry.negative_prob = sentiment_result.get('negative_prob')
            cache_entry.analyzed_at = datetime.now()

            db.commit()
            return True

        except Exception as e:
            db.rollback()
            logger.error(f"Failed to update sentiment: {e}")
            return False

        finally:
            if close_db:
                db.close()

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        db = SessionLocal()
        try:
            total = db.query(NewsCache).count()
            with_sentiment = db.query(NewsCache).filter(
                NewsCache.sentiment_label.isnot(None)
            ).count()
            with_content = db.query(NewsCache).filter(
                NewsCache.content_fetched == 1
            ).count()

            # Count by provider
            from sqlalchemy import func
            by_provider = dict(
                db.query(NewsCache.provider, func.count(NewsCache.id))
                .group_by(NewsCache.provider)
                .all()
            )

            return {
                'total_articles': total,
                'with_sentiment': with_sentiment,
                'with_content': with_content,
                'by_provider': by_provider
            }

        finally:
            db.close()

    def clear_cache(self, provider: str = None, ticker: str = None) -> int:
        """
        Clear cache entries.

        Args:
            provider: Optional provider to filter
            ticker: Optional ticker to filter

        Returns:
            Number of entries deleted
        """
        db = SessionLocal()
        try:
            query = db.query(NewsCache)

            if provider:
                query = query.filter(NewsCache.provider == provider.lower())
            if ticker:
                query = query.filter(NewsCache.ticker == ticker)

            count = query.delete()
            db.commit()

            logger.info(f"Cleared {count} cache entries")
            return count

        except Exception as e:
            db.rollback()
            logger.error(f"Failed to clear cache: {e}")
            return 0

        finally:
            db.close()
