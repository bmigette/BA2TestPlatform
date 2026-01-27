import React, { useState, useEffect } from 'react';
import { Wrench, Newspaper, Search, Loader, CheckCircle, XCircle, AlertCircle, MessageSquare, Download, DollarSign, TrendingUp } from 'lucide-react';

interface NewsArticle {
  title: string;
  summary?: string;
  content?: string;
  source?: string;
  url?: string;
  date?: string;
  published_at?: string;
  // Provider-specific sentiment fields (e.g., from Alpha Vantage)
  sentiment?: string;
  sentiment_score?: number;
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
  has_sentiment?: boolean;
}

const Tools: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'news' | 'fundamentals' | 'macro'>('news');

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
          <button
            onClick={() => setActiveTab('fundamentals')}
            className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
              activeTab === 'fundamentals'
                ? 'border-blue-500 text-blue-600 dark:text-blue-400'
                : 'border-transparent text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300'
            }`}
          >
            <div className="flex items-center gap-2">
              <DollarSign size={16} />
              Fundamentals
            </div>
          </button>
          <button
            onClick={() => setActiveTab('macro')}
            className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
              activeTab === 'macro'
                ? 'border-blue-500 text-blue-600 dark:text-blue-400'
                : 'border-transparent text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300'
            }`}
          >
            <div className="flex items-center gap-2">
              <TrendingUp size={16} />
              Macro Indicators
            </div>
          </button>
        </nav>
      </div>

      {/* Tab Content */}
      {activeTab === 'news' && <NewsProviderTester />}
      {activeTab === 'fundamentals' && <FundamentalsTester />}
      {activeTab === 'macro' && <MacroTester />}
    </div>
  );
};

const NewsProviderTester: React.FC = () => {
  const [symbol, setSymbol] = useState('AAPL');
  const [provider, setProvider] = useState('fmp');
  const [newsType, setNewsType] = useState<'company' | 'global'>('company');
  // Default to last 30 days
  const getDefaultDates = () => {
    const end = new Date();
    const start = new Date();
    start.setDate(start.getDate() - 30);
    return {
      start: start.toISOString().split('T')[0],
      end: end.toISOString().split('T')[0]
    };
  };
  const defaultDates = getDefaultDates();
  const [startDate, setStartDate] = useState(defaultDates.start);
  const [endDate, setEndDate] = useState(defaultDates.end);
  const [providers, setProviders] = useState<NewsProvider[]>([]);
  const [articles, setArticles] = useState<NewsArticle[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sentimentResults, setSentimentResults] = useState<Record<number, SentimentResult | null>>({});
  const [analyzingIndex, setAnalyzingIndex] = useState<number | null>(null);

  // Pagination state
  const [currentPage, setCurrentPage] = useState(1);
  const [itemsPerPage, setItemsPerPage] = useState(20);
  const totalPages = Math.ceil(articles.length / itemsPerPage);
  const paginatedArticles = articles.slice(
    (currentPage - 1) * itemsPerPage,
    currentPage * itemsPerPage
  );

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
    setCurrentPage(1); // Reset to first page

    try {
      const params = new URLSearchParams({
        provider,
        news_type: newsType,
        start_date: startDate,
        end_date: endDate,
        limit: '500' // Fetch more articles, paginate client-side
      });

      // Only add symbol for company news
      if (newsType === 'company' && symbol) {
        params.set('symbol', symbol);
      }

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

  const [exporting, setExporting] = useState(false);
  const [exportMessage, setExportMessage] = useState<string | null>(null);

  const exportToJson = async () => {
    setExporting(true);
    setExportMessage(null);

    try {
      // First fetch ALL news for the date range (no limit)
      const fetchParams = new URLSearchParams({
        provider,
        news_type: newsType,
        start_date: startDate,
        end_date: endDate,
        limit: '10000' // Fetch all articles for export
      });
      if (newsType === 'company' && symbol) {
        fetchParams.set('symbol', symbol);
      }

      setExportMessage('Fetching all articles...');
      const fetchResponse = await fetch(`http://localhost:8002/api/tools/news/fetch?${fetchParams}`);

      if (!fetchResponse.ok) {
        const errorData = await fetchResponse.json();
        throw new Error(errorData.detail || 'Failed to fetch articles for export');
      }

      const fetchData = await fetchResponse.json();
      const allArticles = fetchData.articles || [];

      if (allArticles.length === 0) {
        setExportMessage('No articles to export');
        setExporting(false);
        return;
      }

      setExportMessage(`Exporting ${allArticles.length} articles...`);

      // Build export URL with appropriate parameters
      const exportParams = new URLSearchParams({
        provider,
        news_type: newsType
      });
      if (newsType === 'company' && symbol) {
        exportParams.set('symbol', symbol);
      }

      const response = await fetch(
        `http://localhost:8002/api/tools/news/export?${exportParams}`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(allArticles)
        }
      );

      if (response.ok) {
        const data = await response.json();
        setExportMessage(`Exported ${allArticles.length} articles to ${data.filename}`);
      } else {
        const errorData = await response.json();
        setExportMessage(`Export failed: ${errorData.detail}`);
      }
    } catch (err) {
      setExportMessage(`Export error: ${err instanceof Error ? err.message : 'Unknown error'}`);
    } finally {
      setExporting(false);
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

        {/* News Type Toggle */}
        <div className="flex gap-2 mb-4">
          <button
            onClick={() => setNewsType('company')}
            className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
              newsType === 'company'
                ? 'bg-blue-600 text-white'
                : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
            }`}
          >
            Company News
          </button>
          <button
            onClick={() => setNewsType('global')}
            className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
              newsType === 'global'
                ? 'bg-blue-600 text-white'
                : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
            }`}
          >
            Global/Market News
          </button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-5 gap-4 mb-4">
          {newsType === 'company' && (
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
          )}

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
              Start Date
            </label>
            <input
              type="date"
              value={startDate}
              onChange={(e) => setStartDate(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
              End Date
            </label>
            <input
              type="date"
              value={endDate}
              onChange={(e) => setEndDate(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
            />
          </div>

          <div className="flex items-end">
            <button
              onClick={fetchNews}
              disabled={loading || (newsType === 'company' && !symbol)}
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
            <div className="flex items-center gap-3">
              {exportMessage && (
                <span className={`text-sm ${exportMessage.includes('failed') || exportMessage.includes('error') ? 'text-red-500' : 'text-green-500'}`}>
                  {exportMessage}
                </span>
              )}
              <button
                onClick={exportToJson}
                disabled={exporting || articles.length === 0}
                className="px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 disabled:opacity-50 flex items-center gap-2"
              >
                {exporting ? (
                  <>
                    <Loader size={16} className="animate-spin" />
                    Exporting...
                  </>
                ) : (
                  <>
                    <Download size={16} />
                    Export JSON
                  </>
                )}
              </button>
              <button
                onClick={analyzeAllSentiments}
                disabled={analyzingIndex !== null}
                className="px-4 py-2 bg-purple-600 text-white rounded-md hover:bg-purple-700 disabled:opacity-50 flex items-center gap-2"
              >
                <MessageSquare size={16} />
                Analyze All Sentiments
              </button>
            </div>
          </div>

          <div className="space-y-4">
            {paginatedArticles.map((article, pageIndex) => {
              // Calculate actual index in full articles array for sentiment tracking
              const actualIndex = (currentPage - 1) * itemsPerPage + pageIndex;
              return (
                <div
                  key={actualIndex}
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
                      {/* Show provider's built-in sentiment if available */}
                      {article.sentiment && (
                        <div className="px-3 py-1.5 rounded-full flex items-center gap-1.5 bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-300">
                          <span className="text-xs opacity-75">API:</span>
                          <span className="text-sm font-medium capitalize">
                            {article.sentiment}
                          </span>
                          {article.sentiment_score !== undefined && (
                            <span className="text-xs opacity-75">
                              ({(article.sentiment_score * 100).toFixed(0)}%)
                            </span>
                          )}
                        </div>
                      )}
                      {/* Show FinBERT analysis or analyze button */}
                      {sentimentResults[actualIndex] ? (
                        <div className={`px-3 py-1.5 rounded-full flex items-center gap-1.5 ${getSentimentColor(sentimentResults[actualIndex]!.sentiment)}`}>
                          {getSentimentIcon(sentimentResults[actualIndex]!.sentiment)}
                          <span className="text-sm font-medium capitalize">
                            {sentimentResults[actualIndex]!.sentiment}
                          </span>
                          <span className="text-xs opacity-75">
                            ({(sentimentResults[actualIndex]!.sentiment_score * 100).toFixed(0)}%)
                          </span>
                        </div>
                      ) : (
                        <button
                          onClick={() => analyzeSentiment(actualIndex, article)}
                          disabled={analyzingIndex === actualIndex}
                          className="px-3 py-1.5 bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-md hover:bg-gray-200 dark:hover:bg-gray-600 disabled:opacity-50 text-sm flex items-center gap-1"
                        >
                          {analyzingIndex === actualIndex ? (
                            <>
                              <Loader size={14} className="animate-spin" />
                              Analyzing...
                            </>
                          ) : (
                            <>
                              <MessageSquare size={14} />
                              FinBERT
                            </>
                          )}
                        </button>
                      )}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>

          {/* Pagination Controls */}
          {totalPages > 1 && (
            <div className="flex items-center justify-between mt-6 pt-4 border-t border-gray-200 dark:border-gray-700">
              <div className="flex items-center gap-2">
                <span className="text-sm text-gray-600 dark:text-gray-400">
                  Showing {(currentPage - 1) * itemsPerPage + 1}-{Math.min(currentPage * itemsPerPage, articles.length)} of {articles.length}
                </span>
                <select
                  value={itemsPerPage}
                  onChange={(e) => {
                    setItemsPerPage(Number(e.target.value));
                    setCurrentPage(1);
                  }}
                  className="ml-2 px-2 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-700 dark:text-gray-300"
                >
                  <option value={10}>10 per page</option>
                  <option value={20}>20 per page</option>
                  <option value={50}>50 per page</option>
                  <option value={100}>100 per page</option>
                </select>
              </div>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => setCurrentPage(1)}
                  disabled={currentPage === 1}
                  className="px-3 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded hover:bg-gray-100 dark:hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  First
                </button>
                <button
                  onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
                  disabled={currentPage === 1}
                  className="px-3 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded hover:bg-gray-100 dark:hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  Previous
                </button>
                <span className="px-3 py-1 text-sm text-gray-600 dark:text-gray-400">
                  Page {currentPage} of {totalPages}
                </span>
                <button
                  onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
                  disabled={currentPage === totalPages}
                  className="px-3 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded hover:bg-gray-100 dark:hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  Next
                </button>
                <button
                  onClick={() => setCurrentPage(totalPages)}
                  disabled={currentPage === totalPages}
                  className="px-3 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded hover:bg-gray-100 dark:hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  Last
                </button>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

// Fundamentals Tester Component
const FundamentalsTester: React.FC = () => {
  const [symbol, setSymbol] = useState('AAPL');
  const [provider, setProvider] = useState('yfinance');
  const [dataType, setDataType] = useState('balance_sheet');
  const [frequency, setFrequency] = useState('quarterly');
  const [lookbackPeriods, setLookbackPeriods] = useState(8);
  const [useCustomDates, setUseCustomDates] = useState(false);
  const [startDate, setStartDate] = useState(() => {
    const d = new Date();
    d.setFullYear(d.getFullYear() - 2);
    return d.toISOString().split('T')[0];
  });
  const [endDate, setEndDate] = useState(() => new Date().toISOString().split('T')[0]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [fundamentals, setFundamentals] = useState<Record<string, any> | null>(null);

  const availableProviders = [
    { id: 'yfinance', name: 'Yahoo Finance', description: 'Free, no API key required' },
    { id: 'fmp', name: 'Financial Modeling Prep', description: 'Requires FMP API key' },
    { id: 'alphavantage', name: 'Alpha Vantage', description: 'Requires Alpha Vantage API key' },
  ];

  const dataTypes = [
    { id: 'overview', name: 'Company Overview', description: 'P/E, EPS, Market Cap, etc.' },
    { id: 'balance_sheet', name: 'Balance Sheet', description: 'Assets, liabilities, equity' },
    { id: 'income_statement', name: 'Income Statement', description: 'Revenue, expenses, profit' },
    { id: 'cashflow_statement', name: 'Cash Flow Statement', description: 'Operating, investing, financing' },
    { id: 'past_earnings', name: 'Past Earnings', description: 'Historical EPS and surprises' },
    { id: 'earnings_estimates', name: 'Earnings Estimates', description: 'Future EPS forecasts' },
  ];

  const fetchFundamentals = async () => {
    setLoading(true);
    setError(null);
    setFundamentals(null);

    try {
      const params = new URLSearchParams({
        symbol,
        provider,
        data_type: dataType,
        frequency,
      });

      if (useCustomDates) {
        params.set('start_date', startDate);
        params.set('end_date', endDate);
      } else {
        params.set('lookback_periods', lookbackPeriods.toString());
        params.set('end_date', endDate);
      }

      const response = await fetch(`http://localhost:8002/api/tools/fundamentals/fetch?${params}`);

      if (response.ok) {
        const data = await response.json();
        setFundamentals(data);
      } else {
        const errorData = await response.json();
        setError(errorData.detail || 'Failed to fetch fundamentals');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Network error');
    } finally {
      setLoading(false);
    }
  };

  // Format large numbers for display
  const formatValue = (value: any): string => {
    if (value === null || value === undefined) return '-';
    if (typeof value === 'number') {
      if (Math.abs(value) >= 1e9) return `$${(value / 1e9).toFixed(2)}B`;
      if (Math.abs(value) >= 1e6) return `$${(value / 1e6).toFixed(2)}M`;
      if (Math.abs(value) >= 1e3) return `$${(value / 1e3).toFixed(2)}K`;
      return value.toLocaleString();
    }
    return String(value);
  };

  // Render periods data (balance sheet, income statement, cash flow)
  const renderPeriods = () => {
    if (!fundamentals?.periods || fundamentals.periods.length === 0) {
      return <p className="text-gray-500 dark:text-gray-400">No period data available</p>;
    }

    return (
      <div className="space-y-6">
        {fundamentals.periods.map((period: any, idx: number) => (
          <div key={idx} className="border border-gray-200 dark:border-gray-700 rounded-lg p-4">
            <h4 className="font-medium text-gray-900 dark:text-gray-100 mb-3">
              Period: {period.date}
            </h4>
            <div className="overflow-x-auto max-h-64 overflow-y-auto">
              <table className="min-w-full text-sm">
                <thead className="bg-gray-50 dark:bg-gray-700 sticky top-0">
                  <tr>
                    <th className="px-3 py-2 text-left font-medium text-gray-700 dark:text-gray-300">Item</th>
                    <th className="px-3 py-2 text-right font-medium text-gray-700 dark:text-gray-300">Value</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                  {Object.entries(period.items || {}).map(([key, value]) => (
                    <tr key={key} className="hover:bg-gray-50 dark:hover:bg-gray-700/50">
                      <td className="px-3 py-1 text-gray-900 dark:text-gray-100">{key}</td>
                      <td className="px-3 py-1 text-right text-gray-600 dark:text-gray-400 font-mono">
                        {formatValue(value)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        ))}
      </div>
    );
  };

  // Render earnings data
  const renderEarnings = () => {
    const earnings = fundamentals?.earnings || [];
    if (earnings.length === 0) {
      return <p className="text-gray-500 dark:text-gray-400">No earnings data available</p>;
    }

    return (
      <div className="overflow-x-auto">
        <table className="min-w-full text-sm">
          <thead className="bg-gray-50 dark:bg-gray-700">
            <tr>
              <th className="px-4 py-2 text-left font-medium text-gray-700 dark:text-gray-300">Date</th>
              <th className="px-4 py-2 text-right font-medium text-gray-700 dark:text-gray-300">Reported EPS</th>
              <th className="px-4 py-2 text-right font-medium text-gray-700 dark:text-gray-300">Estimated EPS</th>
              <th className="px-4 py-2 text-right font-medium text-gray-700 dark:text-gray-300">Surprise</th>
              <th className="px-4 py-2 text-right font-medium text-gray-700 dark:text-gray-300">Surprise %</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
            {earnings.map((earning: any, idx: number) => (
              <tr key={idx} className="hover:bg-gray-50 dark:hover:bg-gray-700/50">
                <td className="px-4 py-2 text-gray-900 dark:text-gray-100">{earning.fiscal_date_ending}</td>
                <td className="px-4 py-2 text-right font-mono text-gray-600 dark:text-gray-400">
                  ${earning.reported_eps?.toFixed(2) || '-'}
                </td>
                <td className="px-4 py-2 text-right font-mono text-gray-600 dark:text-gray-400">
                  ${earning.estimated_eps?.toFixed(2) || '-'}
                </td>
                <td className={`px-4 py-2 text-right font-mono ${earning.surprise > 0 ? 'text-green-600' : earning.surprise < 0 ? 'text-red-600' : 'text-gray-600'}`}>
                  {earning.surprise != null ? `$${earning.surprise.toFixed(2)}` : '-'}
                </td>
                <td className={`px-4 py-2 text-right font-mono ${earning.surprise_percent > 0 ? 'text-green-600' : earning.surprise_percent < 0 ? 'text-red-600' : 'text-gray-600'}`}>
                  {earning.surprise_percent != null ? `${earning.surprise_percent.toFixed(1)}%` : '-'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  };

  // Render estimates data
  const renderEstimates = () => {
    const estimates = fundamentals?.estimates || [];
    if (estimates.length === 0) {
      return <p className="text-gray-500 dark:text-gray-400">No estimates data available</p>;
    }

    return (
      <div className="overflow-x-auto">
        <table className="min-w-full text-sm">
          <thead className="bg-gray-50 dark:bg-gray-700">
            <tr>
              <th className="px-4 py-2 text-left font-medium text-gray-700 dark:text-gray-300">Date</th>
              <th className="px-4 py-2 text-right font-medium text-gray-700 dark:text-gray-300">Avg Estimate</th>
              <th className="px-4 py-2 text-right font-medium text-gray-700 dark:text-gray-300">High</th>
              <th className="px-4 py-2 text-right font-medium text-gray-700 dark:text-gray-300">Low</th>
              <th className="px-4 py-2 text-right font-medium text-gray-700 dark:text-gray-300"># Analysts</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
            {estimates.map((estimate: any, idx: number) => (
              <tr key={idx} className="hover:bg-gray-50 dark:hover:bg-gray-700/50">
                <td className="px-4 py-2 text-gray-900 dark:text-gray-100">{estimate.fiscal_date_ending}</td>
                <td className="px-4 py-2 text-right font-mono text-gray-600 dark:text-gray-400">
                  ${estimate.estimated_eps_avg?.toFixed(2) || '-'}
                </td>
                <td className="px-4 py-2 text-right font-mono text-gray-600 dark:text-gray-400">
                  ${estimate.estimated_eps_high?.toFixed(2) || '-'}
                </td>
                <td className="px-4 py-2 text-right font-mono text-gray-600 dark:text-gray-400">
                  ${estimate.estimated_eps_low?.toFixed(2) || '-'}
                </td>
                <td className="px-4 py-2 text-right font-mono text-gray-600 dark:text-gray-400">
                  {estimate.number_of_analysts || '-'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  };

  // Render overview data (current metrics)
  const renderOverview = () => {
    const current = fundamentals?.current || fundamentals?.metrics || {};
    if (Object.keys(current).length === 0) {
      return <p className="text-gray-500 dark:text-gray-400">No overview data available</p>;
    }

    return (
      <div className="overflow-x-auto">
        <table className="min-w-full text-sm">
          <thead className="bg-gray-50 dark:bg-gray-700">
            <tr>
              <th className="px-4 py-2 text-left font-medium text-gray-700 dark:text-gray-300">Metric</th>
              <th className="px-4 py-2 text-left font-medium text-gray-700 dark:text-gray-300">Value</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
            {Object.entries(current).map(([key, value]) => (
              <tr key={key} className="hover:bg-gray-50 dark:hover:bg-gray-700/50">
                <td className="px-4 py-2 text-gray-900 dark:text-gray-100 font-medium">
                  {key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                </td>
                <td className="px-4 py-2 text-gray-600 dark:text-gray-400 font-mono">
                  {formatValue(value)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  };

  // Render the appropriate data based on data type
  const renderData = () => {
    if (!fundamentals) return null;

    if (dataType === 'overview') {
      return renderOverview();
    } else if (dataType === 'past_earnings') {
      return renderEarnings();
    } else if (dataType === 'earnings_estimates') {
      return renderEstimates();
    } else {
      return renderPeriods();
    }
  };

  return (
    <div className="space-y-6">
      <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow">
        <h2 className="text-xl font-semibold mb-4 text-gray-900 dark:text-gray-100">
          Test Fundamentals Provider
        </h2>
        <p className="text-sm text-gray-500 dark:text-gray-400 mb-4">
          Fetch historical fundamental data (balance sheets, income statements, cash flow, earnings) for a ticker.
        </p>

        <div className="space-y-4">
          {/* Row 1: Symbol, Provider, Data Type */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Symbol
              </label>
              <input
                type="text"
                value={symbol}
                onChange={(e) => setSymbol(e.target.value.toUpperCase())}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                placeholder="AAPL"
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
                {availableProviders.map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.name}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Data Type
              </label>
              <select
                value={dataType}
                onChange={(e) => setDataType(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
              >
                {dataTypes.map((dt) => (
                  <option key={dt.id} value={dt.id}>
                    {dt.name}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Frequency
              </label>
              <select
                value={frequency}
                onChange={(e) => setFrequency(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                disabled={dataType === 'overview'}
              >
                <option value="quarterly">Quarterly</option>
                <option value="annual">Annual</option>
              </select>
            </div>
          </div>

          {/* Row 2: Date Range Options */}
          <div className="flex flex-wrap gap-4 items-end">
            <div className="flex items-center gap-2">
              <input
                type="checkbox"
                id="useCustomDates"
                checked={useCustomDates}
                onChange={(e) => setUseCustomDates(e.target.checked)}
                className="rounded border-gray-300 dark:border-gray-600"
              />
              <label htmlFor="useCustomDates" className="text-sm text-gray-700 dark:text-gray-300">
                Use custom date range
              </label>
            </div>

            {useCustomDates ? (
              <>
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Start Date
                  </label>
                  <input
                    type="date"
                    value={startDate}
                    onChange={(e) => setStartDate(e.target.value)}
                    className="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    End Date
                  </label>
                  <input
                    type="date"
                    value={endDate}
                    onChange={(e) => setEndDate(e.target.value)}
                    className="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                  />
                </div>
              </>
            ) : (
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Lookback Periods
                </label>
                <input
                  type="number"
                  value={lookbackPeriods}
                  onChange={(e) => setLookbackPeriods(parseInt(e.target.value) || 8)}
                  min={1}
                  max={20}
                  className="w-24 px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                />
              </div>
            )}

            <button
              onClick={fetchFundamentals}
              disabled={loading || !symbol}
              className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 flex items-center gap-2"
            >
              {loading ? (
                <>
                  <Loader size={16} className="animate-spin" />
                  Fetching...
                </>
              ) : (
                <>
                  <Search size={16} />
                  Fetch Data
                </>
              )}
            </button>
          </div>
        </div>

        {error && (
          <div className="mt-4 p-4 bg-red-50 dark:bg-red-900/20 text-red-600 dark:text-red-400 rounded-md flex items-center gap-2">
            <XCircle size={16} />
            {error}
          </div>
        )}
      </div>

      {fundamentals && (
        <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
              {dataTypes.find(dt => dt.id === dataType)?.name} for {fundamentals.symbol || symbol}
            </h3>
            <div className="flex items-center gap-2">
              <span className="px-2 py-1 text-xs bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 rounded">
                Provider: {fundamentals.provider || provider}
              </span>
              {fundamentals.frequency && (
                <span className="px-2 py-1 text-xs bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded">
                  {fundamentals.frequency}
                </span>
              )}
            </div>
          </div>

          {renderData()}
        </div>
      )}
    </div>
  );
};

// Macro Indicators Tester Component
const MacroTester: React.FC = () => {
  const [indicators, setIndicators] = useState<string[]>(['interest_rate', 'gdp', 'inflation', 'unemployment']);
  const [provider, setProvider] = useState('fred');
  const [startDate, setStartDate] = useState(() => {
    const d = new Date();
    d.setFullYear(d.getFullYear() - 1);
    return d.toISOString().split('T')[0];
  });
  const [endDate, setEndDate] = useState(() => new Date().toISOString().split('T')[0]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [macroData, setMacroData] = useState<Record<string, any> | null>(null);

  const availableProviders = [
    { id: 'fred', name: 'FRED', description: 'Federal Reserve Economic Data' },
  ];

  const availableIndicators = [
    { id: 'interest_rate', name: 'Federal Funds Rate' },
    { id: 'gdp', name: 'GDP' },
    { id: 'inflation', name: 'CPI (Inflation)' },
    { id: 'unemployment', name: 'Unemployment Rate' },
    { id: 'vix', name: 'VIX Volatility' },
    { id: 'yield_10y', name: '10-Year Treasury Yield' },
    { id: 'yield_2y', name: '2-Year Treasury Yield' },
  ];

  const toggleIndicator = (id: string) => {
    setIndicators(prev =>
      prev.includes(id) ? prev.filter(i => i !== id) : [...prev, id]
    );
  };

  const fetchMacroData = async () => {
    setLoading(true);
    setError(null);
    setMacroData(null);

    try {
      const params = new URLSearchParams({
        indicators: indicators.join(','),
        provider,
        start_date: startDate,
        end_date: endDate,
      });

      const response = await fetch(`http://localhost:8002/api/tools/macro/fetch?${params}`);

      if (response.ok) {
        const data = await response.json();
        setMacroData(data);
      } else {
        const errorData = await response.json();
        setError(errorData.detail || 'Failed to fetch macro data');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Network error');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow">
        <h2 className="text-xl font-semibold mb-4 text-gray-900 dark:text-gray-100">
          Test Macro Indicators Provider
        </h2>
        <p className="text-sm text-gray-500 dark:text-gray-400 mb-4">
          Fetch macroeconomic indicators from FRED (Federal Reserve Economic Data).
        </p>

        <div className="space-y-4">
          <div className="flex flex-wrap gap-4 items-end">
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Provider
              </label>
              <select
                value={provider}
                onChange={(e) => setProvider(e.target.value)}
                className="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
              >
                {availableProviders.map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.name}
                  </option>
                ))}
              </select>
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Indicators
            </label>
            <div className="flex flex-wrap gap-2">
              {availableIndicators.map(ind => (
                <label
                  key={ind.id}
                  className={`px-3 py-1.5 rounded-full cursor-pointer text-sm transition-colors ${
                    indicators.includes(ind.id)
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
                  }`}
                >
                  <input
                    type="checkbox"
                    checked={indicators.includes(ind.id)}
                    onChange={() => toggleIndicator(ind.id)}
                    className="sr-only"
                  />
                  {ind.name}
                </label>
              ))}
            </div>
          </div>

          <div className="flex flex-wrap gap-4 items-end">
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Start Date
              </label>
              <input
                type="date"
                value={startDate}
                onChange={(e) => setStartDate(e.target.value)}
                className="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                End Date
              </label>
              <input
                type="date"
                value={endDate}
                onChange={(e) => setEndDate(e.target.value)}
                className="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
              />
            </div>

            <button
              onClick={fetchMacroData}
              disabled={loading || indicators.length === 0}
              className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 flex items-center gap-2"
            >
              {loading ? (
                <>
                  <Loader size={16} className="animate-spin" />
                  Fetching...
                </>
              ) : (
                <>
                  <Search size={16} />
                  Fetch Macro Data
                </>
              )}
            </button>
          </div>
        </div>

        {error && (
          <div className="mt-4 p-4 bg-red-50 dark:bg-red-900/20 text-red-600 dark:text-red-400 rounded-md flex items-center gap-2">
            <XCircle size={16} />
            {error}
          </div>
        )}
      </div>

      {macroData && macroData.indicators && (
        <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
              Macro Data Results
            </h3>
            <span className="px-2 py-1 text-xs bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 rounded">
              Provider: {macroData.provider || 'fred'}
            </span>
          </div>
          <p className="text-sm text-gray-500 dark:text-gray-400 mb-4">
            {macroData.start_date} to {macroData.end_date}
          </p>

          <div className="space-y-6">
            {Object.entries(macroData.indicators).map(([indicator, data]: [string, any]) => (
              <div key={indicator} className="border border-gray-200 dark:border-gray-700 rounded-lg p-4">
                <h4 className="font-medium text-gray-900 dark:text-gray-100 mb-2">
                  {data.name || indicator}
                </h4>
                <p className="text-sm text-gray-500 dark:text-gray-400 mb-2">
                  {data.description} ({data.unit})
                </p>
                {data.data && data.data.length > 0 ? (
                  <div className="overflow-x-auto max-h-48 overflow-y-auto">
                    <table className="min-w-full text-sm">
                      <thead className="bg-gray-50 dark:bg-gray-700 sticky top-0">
                        <tr>
                          <th className="px-3 py-2 text-left font-medium text-gray-700 dark:text-gray-300">Date</th>
                          <th className="px-3 py-2 text-left font-medium text-gray-700 dark:text-gray-300">Value</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                        {data.data.slice(-20).map((row: any, idx: number) => (
                          <tr key={idx} className="hover:bg-gray-50 dark:hover:bg-gray-700/50">
                            <td className="px-3 py-1 text-gray-600 dark:text-gray-400">{row.date}</td>
                            <td className="px-3 py-1 text-gray-900 dark:text-gray-100 font-mono">{row.value?.toFixed(2) || '-'}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                    {data.data.length > 20 && (
                      <p className="text-xs text-gray-500 dark:text-gray-400 mt-2">
                        Showing last 20 of {data.data.length} data points
                      </p>
                    )}
                  </div>
                ) : (
                  <p className="text-gray-500 dark:text-gray-400">No data available</p>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default Tools;
