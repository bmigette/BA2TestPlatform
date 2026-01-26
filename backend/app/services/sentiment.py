"""
Sentiment Analysis Service

Provides news sentiment analysis using Transformers library with
financial sentiment models. Creates aggregated sentiment features
for ML model training.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
import os

logger = logging.getLogger(__name__)

# Try to import transformers
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers library not available. Sentiment analysis will use fallback.")


class SentimentService:
    """
    Service for news sentiment analysis and feature creation.

    Uses Hugging Face Transformers with financial sentiment models like
    FinBERT or ProsusAI/finbert for analyzing news articles.

    Creates aggregated sentiment features like:
    - news_1d_positive_short: Count of positive short-term news in last day
    - news_1w_negative_long: Count of negative long-term news in last week
    """

    # Lookback periods for sentiment aggregation
    LOOKBACK_PERIODS = {
        '1d': 1,      # 1 day
        '1w': 7,      # 1 week
        '1m': 30,     # 1 month
        '6m': 180     # 6 months
    }

    # Sentiment categories
    SENTIMENT_CATEGORIES = ['positive', 'neutral', 'negative']

    # Impact timeframes
    IMPACT_TIMEFRAMES = ['short', 'medium', 'long']

    # Default financial sentiment model
    DEFAULT_MODEL = 'ProsusAI/finbert'

    def __init__(self, model_name: str = None):
        """
        Initialize SentimentService.

        Args:
            model_name: Hugging Face model name for sentiment analysis
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        self._pipeline = None
        self._initialized = False

    def _initialize_pipeline(self):
        """Lazy initialization of the sentiment pipeline."""
        if self._initialized:
            return

        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Transformers not available, using fallback sentiment")
            self._initialized = True
            return

        try:
            logger.info(f"Loading sentiment model: {self.model_name}")
            self._pipeline = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                tokenizer=self.model_name,
                truncation=True,
                max_length=512
            )
            self._initialized = True
            logger.info("Sentiment pipeline initialized successfully")
        except Exception as e:
            logger.error(f"Failed to load sentiment model: {e}")
            self._initialized = True  # Mark as initialized to avoid retry

    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of a text.

        Args:
            text: Text to analyze

        Returns:
            Dictionary with sentiment label, score, and probabilities
        """
        self._initialize_pipeline()

        if self._pipeline is None:
            # Fallback: simple keyword-based sentiment
            return self._fallback_sentiment(text)

        try:
            result = self._pipeline(text[:512])[0]  # Truncate to 512 chars

            # Map FinBERT labels to standard format
            label = result['label'].lower()
            score = result['score']

            return {
                'label': label,
                'score': score,
                'positive_prob': score if label == 'positive' else 0.0,
                'neutral_prob': score if label == 'neutral' else 0.0,
                'negative_prob': score if label == 'negative' else 0.0
            }

        except Exception as e:
            logger.error(f"Sentiment analysis error: {e}")
            return self._fallback_sentiment(text)

    def _fallback_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Simple keyword-based sentiment fallback.

        Args:
            text: Text to analyze

        Returns:
            Sentiment result dictionary
        """
        text_lower = text.lower()

        positive_words = ['up', 'rise', 'gain', 'bull', 'growth', 'profit', 'beat', 'surge', 'rally']
        negative_words = ['down', 'fall', 'loss', 'bear', 'decline', 'miss', 'crash', 'drop', 'plunge']

        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)

        if positive_count > negative_count:
            return {'label': 'positive', 'score': 0.6, 'positive_prob': 0.6, 'neutral_prob': 0.3, 'negative_prob': 0.1}
        elif negative_count > positive_count:
            return {'label': 'negative', 'score': 0.6, 'positive_prob': 0.1, 'neutral_prob': 0.3, 'negative_prob': 0.6}
        else:
            return {'label': 'neutral', 'score': 0.5, 'positive_prob': 0.25, 'neutral_prob': 0.5, 'negative_prob': 0.25}

    def analyze_news_articles(
        self,
        articles: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Analyze sentiment of multiple news articles.

        Args:
            articles: List of article dicts with 'title', 'content', 'date' keys

        Returns:
            List of articles with sentiment added
        """
        results = []

        for i, article in enumerate(articles):
            # Combine title and content for analysis
            title = article.get('title', '')
            content = article.get('content', '')[:500]
            text = f"{title} {content}"

            # Debug log: article content
            logger.debug(f"[Article {i+1}/{len(articles)}] Title: {title}")
            logger.debug(f"[Article {i+1}/{len(articles)}] Content preview: {content[:200]}...")

            sentiment = self.analyze_text(text)

            # Debug log: sentiment result
            logger.debug(
                f"[Article {i+1}/{len(articles)}] Sentiment: {sentiment['label']} "
                f"(score={sentiment['score']:.3f}, pos={sentiment['positive_prob']:.3f}, "
                f"neu={sentiment['neutral_prob']:.3f}, neg={sentiment['negative_prob']:.3f})"
            )

            impact = self._estimate_impact_timeframe(text)
            logger.debug(f"[Article {i+1}/{len(articles)}] Impact timeframe: {impact}")

            result = {
                **article,
                'sentiment': sentiment['label'],
                'sentiment_score': sentiment['score'],
                'positive_prob': sentiment['positive_prob'],
                'neutral_prob': sentiment['neutral_prob'],
                'negative_prob': sentiment['negative_prob'],
                'impact_timeframe': impact
            }
            results.append(result)

        logger.info(f"Analyzed sentiment for {len(results)} articles")
        return results

    def _estimate_impact_timeframe(self, text: str) -> str:
        """
        Estimate the impact timeframe of news based on content.

        Args:
            text: News text

        Returns:
            'short', 'medium', or 'long'
        """
        text_lower = text.lower()

        # Long-term indicators
        long_keywords = ['annual', 'yearly', 'decade', 'long-term', 'strategic', 'restructuring']
        if any(kw in text_lower for kw in long_keywords):
            return 'long'

        # Short-term indicators
        short_keywords = ['today', 'trading', 'intraday', 'immediate', 'breaking']
        if any(kw in text_lower for kw in short_keywords):
            return 'short'

        return 'medium'

    def create_sentiment_features(
        self,
        ohlc_df: pd.DataFrame,
        news_articles: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Create aggregated sentiment features for dataset.

        Creates features like:
        - news_1d_positive_short: Positive short-term news count in last day
        - news_1w_negative_long: Negative long-term news count in last week
        - etc.

        Args:
            ohlc_df: DataFrame with Date column
            news_articles: List of analyzed news articles with dates

        Returns:
            DataFrame with sentiment features added
        """
        result_df = ohlc_df.copy()
        result_df['Date'] = pd.to_datetime(result_df['Date'])

        # Analyze articles if not already analyzed
        if news_articles and 'sentiment' not in news_articles[0]:
            news_articles = self.analyze_news_articles(news_articles)

        # Convert to DataFrame for easier manipulation
        if news_articles:
            news_df = pd.DataFrame(news_articles)
            news_df['date'] = pd.to_datetime(news_df['date'])
        else:
            news_df = pd.DataFrame(columns=['date', 'sentiment', 'impact_timeframe'])

        # Create all sentiment feature columns
        feature_columns = []
        for period_name, period_days in self.LOOKBACK_PERIODS.items():
            for sentiment in self.SENTIMENT_CATEGORIES:
                for impact in self.IMPACT_TIMEFRAMES:
                    col_name = f'news_{period_name}_{sentiment}_{impact}'
                    feature_columns.append(col_name)

        # Initialize columns with zeros
        for col in feature_columns:
            result_df[col] = 0

        # Calculate features for each row
        for idx, row in result_df.iterrows():
            row_date = row['Date']

            for period_name, period_days in self.LOOKBACK_PERIODS.items():
                start_date = row_date - timedelta(days=period_days)

                # Filter news in this lookback period
                mask = (news_df['date'] >= start_date) & (news_df['date'] <= row_date)
                period_news = news_df[mask] if len(news_df) > 0 else pd.DataFrame()

                for sentiment in self.SENTIMENT_CATEGORIES:
                    for impact in self.IMPACT_TIMEFRAMES:
                        col_name = f'news_{period_name}_{sentiment}_{impact}'

                        if len(period_news) > 0:
                            count = len(period_news[
                                (period_news['sentiment'] == sentiment) &
                                (period_news['impact_timeframe'] == impact)
                            ])
                            result_df.at[idx, col_name] = count

        logger.info(f"Created {len(feature_columns)} sentiment features")
        return result_df

    def fetch_news_for_ticker(
        self,
        ticker: str,
        start_date: datetime,
        end_date: datetime,
        provider: str = "fmp",
        enrich_content: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Fetch news articles for a ticker in date range using real news providers.

        Args:
            ticker: Stock ticker symbol
            start_date: Start date
            end_date: End date
            provider: News provider to use ('fmp', 'alphavantage', 'finnhub', 'alpaca')
            enrich_content: Whether to fetch full article content for short summaries

        Returns:
            List of news articles with title, content, date, source

        Raises:
            ValueError: If provider is not available or unknown
            Exception: If news fetching fails (no fallback to mock data)
        """
        logger.info(f"Fetching news for {ticker} from {start_date} to {end_date} using {provider}")

        # Get the news provider - fail if not available
        news_provider = self._get_news_provider(provider)
        if news_provider is None:
            raise ValueError(f"News provider '{provider}' is not available or not configured")

        # Fetch news using the provider - no fallback on error
        result = news_provider.get_company_news(
            symbol=ticker,
            end_date=end_date,
            start_date=start_date,
            format_type="dict"
        )

        # Check for error response
        if isinstance(result, dict) and "error" in result:
            raise Exception(result["error"])

        raw_articles = result.get("articles", [])
        logger.info(f"Received {len(raw_articles)} raw articles from {provider}")

        # Debug log: raw articles
        for i, article in enumerate(raw_articles):
            logger.debug(f"[Raw {i+1}/{len(raw_articles)}] Title: {article.get('title', 'N/A')}")
            logger.debug(f"[Raw {i+1}/{len(raw_articles)}] URL: {article.get('url', 'N/A')}")
            summary = article.get('summary', article.get('snippet', ''))
            logger.debug(f"[Raw {i+1}/{len(raw_articles)}] Summary length: {len(summary)} chars")

        # Enrich articles with short summaries using trafilatura
        if enrich_content and hasattr(news_provider, 'enrich_articles_with_content'):
            logger.info("Enriching articles with URL content via trafilatura...")
            raw_articles = news_provider.enrich_articles_with_content(
                raw_articles,
                max_workers=5,
                min_summary_length=100
            )

        # Convert to standard format
        articles = []
        for article in raw_articles:
            pub_date = article.get("published_at", "")
            # Parse date string to datetime if needed
            if isinstance(pub_date, str) and pub_date:
                try:
                    pub_date = datetime.fromisoformat(pub_date.replace('Z', '+00:00'))
                except ValueError:
                    pub_date = start_date

            articles.append({
                'title': article.get('title', ''),
                'content': article.get('summary', article.get('snippet', '')),
                'date': pub_date,
                'source': article.get('source', provider.upper()),
                'url': article.get('url', ''),
                'content_fetched': article.get('content_fetched', False)
            })

        logger.info(f"Fetched {len(articles)} news articles for {ticker} from {provider}")
        return articles

    def _get_news_provider(self, provider: str):
        """
        Get news provider instance by name.

        Args:
            provider: Provider name ('fmp', 'alphavantage', 'finnhub', 'alpaca')

        Returns:
            News provider instance

        Raises:
            ValueError: If provider is unknown
            ImportError: If provider dependencies are not installed
            Exception: If provider initialization fails

        Note:
            GoogleNewsProvider has been removed (scraping unreliable).
            AINewsProvider requires ModelFactory dependency.
        """
        valid_providers = ['fmp', 'alphavantage', 'finnhub', 'alpaca']

        if provider not in valid_providers:
            raise ValueError(f"Unknown news provider: '{provider}'. Valid providers: {valid_providers}")

        logger.debug(f"Initializing news provider: {provider}")

        if provider == "fmp":
            from dataproviders.news import FMPNewsProvider
            if FMPNewsProvider is None:
                raise ImportError("FMPNewsProvider not available - check if fmpsdk is installed")
            return FMPNewsProvider()
        elif provider == "alphavantage":
            from dataproviders.news import AlphaVantageNewsProvider
            if AlphaVantageNewsProvider is None:
                raise ImportError("AlphaVantageNewsProvider not available - check dependencies")
            return AlphaVantageNewsProvider()
        elif provider == "finnhub":
            from dataproviders.news import FinnhubNewsProvider
            if FinnhubNewsProvider is None:
                raise ImportError("FinnhubNewsProvider not available - check if finnhub-python is installed")
            return FinnhubNewsProvider()
        elif provider == "alpaca":
            from dataproviders.news import AlpacaNewsProvider
            if AlpacaNewsProvider is None:
                raise ImportError("AlpacaNewsProvider not available - check if alpaca-py is installed")
            return AlpacaNewsProvider()

    @staticmethod
    def get_feature_descriptions() -> Dict[str, str]:
        """
        Get descriptions for all sentiment features.

        Returns:
            Dictionary mapping feature names to descriptions
        """
        descriptions = {}
        for period_name, period_days in SentimentService.LOOKBACK_PERIODS.items():
            for sentiment in SentimentService.SENTIMENT_CATEGORIES:
                for impact in SentimentService.IMPACT_TIMEFRAMES:
                    col_name = f'news_{period_name}_{sentiment}_{impact}'
                    descriptions[col_name] = (
                        f"Count of {sentiment} {impact}-term impact news articles "
                        f"in the last {period_name} ({period_days} days)"
                    )
        return descriptions
