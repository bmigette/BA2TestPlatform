import React, { useState, useEffect } from 'react';
import { Wrench, Newspaper, Search, Loader, CheckCircle, XCircle, AlertCircle, MessageSquare } from 'lucide-react';

interface NewsArticle {
  title: string;
  summary?: string;
  content?: string;
  source?: string;
  url?: string;
  date?: string;
  published_at?: string;
}

interface SentimentResult {
  sentiment: 'positive' | 'neutral' | 'negative';
  sentiment_score: number;
  confidence: number;
}

interface NewsProvider {
  id: string;
  name: string;
  description: string;
  api_key_configured: boolean;
}

const Tools: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'news'>('news');

  return (
    <div className="p-6">
      <div className="mb-6">
        <h1 className="text-3xl font-bold text-gray-900 dark:text-gray-100 flex items-center gap-2">
          <Wrench size={32} />
          Tools
        </h1>
        <p className="text-gray-600 dark:text-gray-400 mt-1">
          Testing and debugging utilities
        </p>
      </div>

      {/* Tabs */}
      <div className="border-b border-gray-200 dark:border-gray-700 mb-6">
        <nav className="flex space-x-4">
          <button
            onClick={() => setActiveTab('news')}
            className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
              activeTab === 'news'
                ? 'border-blue-500 text-blue-600 dark:text-blue-400'
                : 'border-transparent text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300'
            }`}
          >
            <div className="flex items-center gap-2">
              <Newspaper size={16} />
              News Providers
            </div>
          </button>
        </nav>
      </div>

      {/* Tab Content */}
      {activeTab === 'news' && <NewsProviderTester />}
    </div>
  );
};

const NewsProviderTester: React.FC = () => {
  const [symbol, setSymbol] = useState('AAPL');
  const [provider, setProvider] = useState('fmp');
  const [days, setDays] = useState(30);
  const [providers, setProviders] = useState<NewsProvider[]>([]);
  const [articles, setArticles] = useState<NewsArticle[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sentimentResults, setSentimentResults] = useState<Record<number, SentimentResult | null>>({});
  const [analyzingIndex, setAnalyzingIndex] = useState<number | null>(null);

  // Fetch available providers on mount
  useEffect(() => {
    const fetchProviders = async () => {
      try {
        const response = await fetch('http://localhost:8002/api/tools/news/providers');
        if (response.ok) {
          const data = await response.json();
          setProviders(data.providers || []);
        }
      } catch (err) {
        console.error('Failed to fetch providers:', err);
      }
    };
    fetchProviders();
  }, []);

  const fetchNews = async () => {
    setLoading(true);
    setError(null);
    setArticles([]);
    setSentimentResults({});

    try {
      const params = new URLSearchParams({
        symbol,
        provider,
        days: days.toString(),
        limit: '50'
      });

      const response = await fetch(`http://localhost:8002/api/tools/news/fetch?${params}`);

      if (response.ok) {
        const data = await response.json();
        setArticles(data.articles || []);
        if (data.articles?.length === 0) {
          setError('No articles found for this symbol and date range');
        }
      } else {
        const errorData = await response.json();
        setError(errorData.detail || 'Failed to fetch news');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Network error');
    } finally {
      setLoading(false);
    }
  };

  const analyzeSentiment = async (index: number, article: NewsArticle) => {
    setAnalyzingIndex(index);

    try {
      const params = new URLSearchParams({
        title: article.title,
        content: article.summary || article.content || ''
      });

      const response = await fetch(`http://localhost:8002/api/tools/news/analyze-single?${params}`, {
        method: 'POST'
      });

      if (response.ok) {
        const data = await response.json();
        setSentimentResults(prev => ({
          ...prev,
          [index]: {
            sentiment: data.sentiment,
            sentiment_score: data.sentiment_score,
            confidence: data.confidence
          }
        }));
      } else {
        setSentimentResults(prev => ({
          ...prev,
          [index]: null
        }));
      }
    } catch (err) {
      console.error('Sentiment analysis error:', err);
      setSentimentResults(prev => ({
        ...prev,
        [index]: null
      }));
    } finally {
      setAnalyzingIndex(null);
    }
  };

  const analyzeAllSentiments = async () => {
    for (let i = 0; i < articles.length; i++) {
      if (!sentimentResults[i]) {
        await analyzeSentiment(i, articles[i]);
      }
    }
  };

  const getSentimentColor = (sentiment: string) => {
    switch (sentiment) {
      case 'positive':
        return 'text-green-600 bg-green-100 dark:text-green-400 dark:bg-green-900/30';
      case 'negative':
        return 'text-red-600 bg-red-100 dark:text-red-400 dark:bg-red-900/30';
      default:
        return 'text-orange-600 bg-orange-100 dark:text-orange-400 dark:bg-orange-900/30';
    }
  };

  const getSentimentIcon = (sentiment: string) => {
    switch (sentiment) {
      case 'positive':
        return <CheckCircle size={16} />;
      case 'negative':
        return <XCircle size={16} />;
      default:
        return <AlertCircle size={16} />;
    }
  };

  return (
    <div className="space-y-6">
      {/* Search Form */}
      <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow">
        <h2 className="text-xl font-bold mb-4 text-gray-900 dark:text-gray-100">Test News Provider</h2>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
              Symbol
            </label>
            <input
              type="text"
              value={symbol}
              onChange={(e) => setSymbol(e.target.value.toUpperCase())}
              placeholder="AAPL"
              className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
              Provider
            </label>
            <select
              value={provider}
              onChange={(e) => setProvider(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
            >
              {providers.map((p) => (
                <option key={p.id} value={p.id} disabled={!p.api_key_configured}>
                  {p.name} {!p.api_key_configured && '(No API Key)'}
                </option>
              ))}
              {providers.length === 0 && (
                <>
                  <option value="fmp">Financial Modeling Prep</option>
                  <option value="alpaca">Alpaca</option>
                </>
              )}
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
              Days Back
            </label>
            <input
              type="number"
              value={days}
              onChange={(e) => setDays(parseInt(e.target.value) || 30)}
              min={1}
              max={365}
              className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
            />
          </div>

          <div className="flex items-end">
            <button
              onClick={fetchNews}
              disabled={loading || !symbol}
              className="w-full px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 flex items-center justify-center gap-2"
            >
              {loading ? (
                <>
                  <Loader size={16} className="animate-spin" />
                  Fetching...
                </>
              ) : (
                <>
                  <Search size={16} />
                  Fetch News
                </>
              )}
            </button>
          </div>
        </div>

        {/* Provider Status */}
        {providers.length > 0 && (
          <div className="flex flex-wrap gap-2 mt-2">
            {providers.map((p) => (
              <span
                key={p.id}
                className={`px-2 py-1 text-xs rounded-full ${
                  p.api_key_configured
                    ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400'
                    : 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400'
                }`}
              >
                {p.name}: {p.api_key_configured ? 'Configured' : 'Missing API Key'}
              </span>
            ))}
          </div>
        )}
      </div>

      {/* Error Display */}
      {error && (
        <div className="bg-red-50 dark:bg-red-900/30 border border-red-200 dark:border-red-800 rounded-lg p-4">
          <div className="flex items-center gap-2 text-red-700 dark:text-red-300">
            <AlertCircle size={20} />
            <span>{error}</span>
          </div>
        </div>
      )}

      {/* Results */}
      {articles.length > 0 && (
        <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-bold text-gray-900 dark:text-gray-100">
              Results ({articles.length} articles)
            </h2>
            <button
              onClick={analyzeAllSentiments}
              disabled={analyzingIndex !== null}
              className="px-4 py-2 bg-purple-600 text-white rounded-md hover:bg-purple-700 disabled:opacity-50 flex items-center gap-2"
            >
              <MessageSquare size={16} />
              Analyze All Sentiments
            </button>
          </div>

          <div className="space-y-4">
            {articles.map((article, index) => (
              <div
                key={index}
                className="border border-gray-200 dark:border-gray-700 rounded-lg p-4"
              >
                <div className="flex items-start justify-between gap-4">
                  <div className="flex-1">
                    <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-1">
                      {article.title}
                    </h3>
                    <div className="flex items-center gap-3 text-sm text-gray-500 dark:text-gray-400 mb-2">
                      {article.source && <span>{article.source}</span>}
                      {(article.date || article.published_at) && (
                        <span>
                          {new Date(article.date || article.published_at || '').toLocaleDateString()}
                        </span>
                      )}
                    </div>
                    {(article.summary || article.content) && (
                      <p className="text-sm text-gray-600 dark:text-gray-300 line-clamp-2">
                        {article.summary || article.content}
                      </p>
                    )}
                    {article.url && (
                      <a
                        href={article.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-sm text-blue-600 dark:text-blue-400 hover:underline mt-1 inline-block"
                      >
                        Read more
                      </a>
                    )}
                  </div>

                  <div className="flex flex-col items-end gap-2">
                    {sentimentResults[index] ? (
                      <div className={`px-3 py-1.5 rounded-full flex items-center gap-1.5 ${getSentimentColor(sentimentResults[index]!.sentiment)}`}>
                        {getSentimentIcon(sentimentResults[index]!.sentiment)}
                        <span className="text-sm font-medium capitalize">
                          {sentimentResults[index]!.sentiment}
                        </span>
                        <span className="text-xs opacity-75">
                          ({(sentimentResults[index]!.sentiment_score * 100).toFixed(0)}%)
                        </span>
                      </div>
                    ) : (
                      <button
                        onClick={() => analyzeSentiment(index, article)}
                        disabled={analyzingIndex === index}
                        className="px-3 py-1.5 bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-md hover:bg-gray-200 dark:hover:bg-gray-600 disabled:opacity-50 text-sm flex items-center gap-1"
                      >
                        {analyzingIndex === index ? (
                          <>
                            <Loader size={14} className="animate-spin" />
                            Analyzing...
                          </>
                        ) : (
                          <>
                            <MessageSquare size={14} />
                            Analyze
                          </>
                        )}
                      </button>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default Tools;
