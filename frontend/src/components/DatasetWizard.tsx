import React, { useState } from 'react';
import { X, MessageSquare, TrendingUp, BarChart3, FileText, Settings, CheckCircle } from 'lucide-react';

interface DatasetWizardProps {
  isOpen: boolean;
  onClose: () => void;
  onComplete: () => void;
}

interface IndicatorConfig {
  name: string;
  type: string;
  period?: number;
  enabled: boolean;
}

interface SentimentConfig {
  enabled: boolean;
  newsSources: string[];
  lookbackPeriods: string[];
  sentimentCategories: string[];
  impactTimeframes: string[];
}

interface FundamentalsConfig {
  enabled: boolean;
  metrics: string[];
  macroIndicators: string[];
}

interface WizardData {
  ticker: string;
  timeframe: string;
  startDate: string;
  endDate: string;
  dataProvider: string;
  indicators: IndicatorConfig[];
  sentiment: SentimentConfig;
  fundamentals: FundamentalsConfig;
}

const DatasetWizard: React.FC<DatasetWizardProps> = ({ isOpen, onClose, onComplete }) => {
  const [currentStep, setCurrentStep] = useState(1);
  const [wizardData, setWizardData] = useState<WizardData>({
    ticker: '',
    timeframe: '1d',
    startDate: '',
    endDate: '',
    dataProvider: 'yfinance',
    indicators: [
      { name: 'SMA', type: 'sma', period: 20, enabled: false },
      { name: 'EMA', type: 'ema', period: 20, enabled: false },
      { name: 'RSI', type: 'rsi', period: 14, enabled: false },
      { name: 'MACD', type: 'macd', enabled: false },
      { name: 'Bollinger Bands', type: 'bbands', period: 20, enabled: false },
      { name: 'ATR', type: 'atr', period: 14, enabled: false },
      { name: 'Stochastic', type: 'stochastic', enabled: false },
    ],
    sentiment: {
      enabled: false,
      newsSources: ['google_news', 'fmp_news'],
      lookbackPeriods: ['1d', '1w', '1m', '6m'],
      sentimentCategories: ['positive', 'neutral', 'negative'],
      impactTimeframes: ['short', 'medium', 'long']
    },
    fundamentals: {
      enabled: false,
      metrics: ['fcf', 'pe', 'eps', 'revenue'],
      macroIndicators: ['interest_rate', 'gdp', 'inflation', 'unemployment']
    }
  });
  const [isCreating, setIsCreating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  if (!isOpen) return null;

  const validateTicker = (ticker: string): boolean => {
    // Ticker should be 1-5 uppercase letters, optionally with numbers
    const tickerRegex = /^[A-Z]{1,5}(\.[A-Z]{1,2})?$/;
    return tickerRegex.test(ticker);
  };

  const handleNext = () => {
    if (currentStep === 1) {
      if (!wizardData.ticker) {
        setError('Ticker is required');
        return;
      }
      if (!validateTicker(wizardData.ticker)) {
        setError('Invalid ticker format. Use 1-5 uppercase letters (e.g., AAPL, MSFT)');
        return;
      }
      // Validate date range
      if (wizardData.startDate && wizardData.endDate) {
        const start = new Date(wizardData.startDate);
        const end = new Date(wizardData.endDate);
        if (start >= end) {
          setError('Start date must be before end date');
          return;
        }
      }
    }

    if (currentStep === 2) {
      if (!wizardData.dataProvider) {
        setError('Please select a data provider');
        return;
      }
    }

    setError(null);
    setCurrentStep(currentStep + 1);
  };

  const handleBack = () => {
    setError(null);
    setCurrentStep(currentStep - 1);
  };

  const handleCreate = async () => {
    setIsCreating(true);
    setError(null);

    try {
      const response = await fetch('http://localhost:8002/api/datasets', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          ticker: wizardData.ticker,
          timeframe: wizardData.timeframe,
          start_date: wizardData.startDate || undefined,
          end_date: wizardData.endDate || undefined,
          technical_indicators: wizardData.indicators.filter(i => i.enabled).map(i => ({
            type: i.type,
            period: i.period
          })),
          sentiment_config: wizardData.sentiment.enabled ? wizardData.sentiment : null,
          fundamentals_config: wizardData.fundamentals.enabled ? wizardData.fundamentals : null
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to create dataset');
      }

      onComplete();
      onClose();

      // Reset wizard
      setCurrentStep(1);
      setWizardData({
        ticker: '',
        timeframe: '1d',
        startDate: '',
        endDate: '',
        dataProvider: 'yfinance',
        indicators: [
          { name: 'SMA', type: 'sma', period: 20, enabled: false },
          { name: 'EMA', type: 'ema', period: 20, enabled: false },
          { name: 'RSI', type: 'rsi', period: 14, enabled: false },
          { name: 'MACD', type: 'macd', enabled: false },
          { name: 'Bollinger Bands', type: 'bbands', period: 20, enabled: false },
          { name: 'ATR', type: 'atr', period: 14, enabled: false },
          { name: 'Stochastic', type: 'stochastic', enabled: false },
        ],
        sentiment: {
          enabled: false,
          newsSources: ['google_news', 'fmp_news'],
          lookbackPeriods: ['1d', '1w', '1m', '6m'],
          sentimentCategories: ['positive', 'neutral', 'negative'],
          impactTimeframes: ['short', 'medium', 'long']
        },
        fundamentals: {
          enabled: false,
          metrics: ['fcf', 'pe', 'eps', 'revenue'],
          macroIndicators: ['interest_rate', 'gdp', 'inflation', 'unemployment']
        }
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setIsCreating(false);
    }
  };

  const getTickerError = (): string | null => {
    if (!wizardData.ticker) return null;
    if (!validateTicker(wizardData.ticker)) {
      return 'Invalid ticker format';
    }
    return null;
  };

  const getDateError = (): string | null => {
    if (wizardData.startDate && wizardData.endDate) {
      const start = new Date(wizardData.startDate);
      const end = new Date(wizardData.endDate);
      if (start >= end) {
        return 'Start date must be before end date';
      }
    }
    return null;
  };

  const tickerError = getTickerError();
  const dateError = getDateError();

  const renderStep1 = () => (
    <div className="space-y-4">
      <div>
        <label className="block text-sm font-medium mb-2">
          Ticker Symbol <span className="text-red-500">*</span>
        </label>
        <input
          type="text"
          value={wizardData.ticker}
          onChange={(e) => setWizardData({ ...wizardData, ticker: e.target.value.toUpperCase() })}
          placeholder="e.g., AAPL, MSFT, GOOGL"
          className={`w-full px-3 py-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 ${
            tickerError
              ? 'border-red-500 focus:ring-red-500'
              : 'border-gray-300 dark:border-gray-600'
          }`}
        />
        {tickerError && (
          <p className="text-xs text-red-500 mt-1">{tickerError}</p>
        )}
      </div>

      <div>
        <label className="block text-sm font-medium mb-2">Timeframe</label>
        <select
          value={wizardData.timeframe}
          onChange={(e) => setWizardData({ ...wizardData, timeframe: e.target.value })}
          className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:border-gray-600"
        >
          <option value="1m">1 Minute</option>
          <option value="5m">5 Minutes</option>
          <option value="15m">15 Minutes</option>
          <option value="30m">30 Minutes</option>
          <option value="1h">1 Hour</option>
          <option value="4h">4 Hours</option>
          <option value="1d">1 Day</option>
          <option value="1w">1 Week</option>
          <option value="1mo">1 Month</option>
        </select>
      </div>

      <div>
        <label className="block text-sm font-medium mb-2">Start Date (optional)</label>
        <input
          type="date"
          value={wizardData.startDate}
          onChange={(e) => setWizardData({ ...wizardData, startDate: e.target.value })}
          className={`w-full px-3 py-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 ${
            dateError
              ? 'border-red-500 focus:ring-red-500'
              : 'border-gray-300 dark:border-gray-600'
          }`}
        />
        <p className="text-xs text-gray-500 mt-1">Leave empty for 1 year of data</p>
      </div>

      <div>
        <label className="block text-sm font-medium mb-2">End Date (optional)</label>
        <input
          type="date"
          value={wizardData.endDate}
          onChange={(e) => setWizardData({ ...wizardData, endDate: e.target.value })}
          className={`w-full px-3 py-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 ${
            dateError
              ? 'border-red-500 focus:ring-red-500'
              : 'border-gray-300 dark:border-gray-600'
          }`}
        />
        {dateError ? (
          <p className="text-xs text-red-500 mt-1">{dateError}</p>
        ) : (
          <p className="text-xs text-gray-500 mt-1">Leave empty for today</p>
        )}
      </div>
    </div>
  );

  const renderStep2 = () => (
    <div className="space-y-4">
      <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
        Select a data provider to fetch historical market data:
      </p>

      <div className="space-y-3 max-h-96 overflow-y-auto">
        {[
          { id: 'yfinance', name: 'Yahoo Finance', desc: 'Free, reliable market data. No API key required.', tags: ['Free', 'No API Key'], recommended: true },
          { id: 'alphavantage', name: 'Alpha Vantage', desc: 'Professional-grade financial data with fundamentals.', tags: ['API Key Required', '500/day'] },
          { id: 'fmp', name: 'Financial Modeling Prep', desc: 'Comprehensive financial data with earnings data.', tags: ['API Key Required', '250/day'] },
          { id: 'alpaca', name: 'Alpaca Markets', desc: 'Real-time and historical market data for trading.', tags: ['API Key Required'] },
        ].map(provider => (
          <label key={provider.id} className={`block p-4 border-2 rounded-lg cursor-pointer transition-colors ${
            wizardData.dataProvider === provider.id
              ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
              : 'border-gray-200 dark:border-gray-700 hover:border-blue-300'
          }`}>
            <input
              type="radio"
              name="dataProvider"
              value={provider.id}
              checked={wizardData.dataProvider === provider.id}
              onChange={(e) => setWizardData({ ...wizardData, dataProvider: e.target.value })}
              className="sr-only"
            />
            <div className="font-semibold text-lg mb-1">{provider.name}</div>
            <p className="text-sm text-gray-600 dark:text-gray-400">{provider.desc}</p>
            <div className="mt-2 flex items-center space-x-2 text-xs">
              {provider.tags.map(tag => (
                <span key={tag} className={`px-2 py-1 rounded ${tag.includes('Free') ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200' : 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200'}`}>
                  {tag}
                </span>
              ))}
              {provider.recommended && <span className="text-gray-500">Recommended</span>}
            </div>
          </label>
        ))}
      </div>
    </div>
  );

  const toggleIndicator = (index: number) => {
    const newIndicators = [...wizardData.indicators];
    newIndicators[index].enabled = !newIndicators[index].enabled;
    setWizardData({ ...wizardData, indicators: newIndicators });
  };

  const updateIndicatorPeriod = (index: number, period: number) => {
    const newIndicators = [...wizardData.indicators];
    newIndicators[index].period = period;
    setWizardData({ ...wizardData, indicators: newIndicators });
  };

  const renderStep3 = () => (
    <div className="space-y-4">
      <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
        Select technical indicators to calculate for your dataset:
      </p>

      <div className="space-y-3 max-h-80 overflow-y-auto">
        {wizardData.indicators.map((indicator, index) => (
          <div key={indicator.type} className="p-4 border border-gray-200 dark:border-gray-700 rounded-lg">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-3">
                <input
                  type="checkbox"
                  id={`indicator-${indicator.type}`}
                  checked={indicator.enabled}
                  onChange={() => toggleIndicator(index)}
                  className="w-4 h-4 text-blue-600 rounded focus:ring-blue-500"
                />
                <label htmlFor={`indicator-${indicator.type}`} className="font-medium cursor-pointer">
                  {indicator.name}
                </label>
              </div>
              {indicator.period && indicator.enabled && (
                <div className="flex items-center space-x-2">
                  <span className="text-sm text-gray-600 dark:text-gray-400">Period:</span>
                  <input
                    type="number"
                    min="1"
                    max="200"
                    value={indicator.period}
                    onChange={(e) => updateIndicatorPeriod(index, parseInt(e.target.value) || 1)}
                    className="w-20 px-2 py-1 border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-sm"
                  />
                </div>
              )}
            </div>
          </div>
        ))}
      </div>

      <div className="bg-blue-50 dark:bg-blue-900/20 p-3 rounded-md">
        <p className="text-sm text-blue-800 dark:text-blue-200">
          Selected: <strong>{wizardData.indicators.filter(i => i.enabled).length}</strong> indicators
        </p>
      </div>
    </div>
  );

  const renderStep4 = () => (
    <div className="space-y-4">
      <div className="flex items-center justify-between p-4 border border-gray-200 dark:border-gray-700 rounded-lg">
        <div className="flex items-center gap-3">
          <MessageSquare className="w-6 h-6 text-purple-500" />
          <div>
            <h3 className="font-semibold">Enable Sentiment Analysis</h3>
            <p className="text-sm text-gray-500">Analyze news sentiment for the ticker</p>
          </div>
        </div>
        <label className="relative inline-flex items-center cursor-pointer">
          <input
            type="checkbox"
            checked={wizardData.sentiment.enabled}
            onChange={(e) => setWizardData({
              ...wizardData,
              sentiment: { ...wizardData.sentiment, enabled: e.target.checked }
            })}
            className="sr-only peer"
          />
          <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 dark:peer-focus:ring-blue-800 rounded-full peer dark:bg-gray-700 peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all dark:border-gray-600 peer-checked:bg-blue-600"></div>
        </label>
      </div>

      {wizardData.sentiment.enabled && (
        <div className="space-y-4 p-4 bg-gray-50 dark:bg-gray-700/50 rounded-lg">
          <div>
            <label className="block text-sm font-medium mb-2">News Sources</label>
            <div className="flex flex-wrap gap-2">
              {['google_news', 'fmp_news', 'alpaca_news'].map(source => (
                <label key={source} className={`px-3 py-2 rounded-lg cursor-pointer border ${
                  wizardData.sentiment.newsSources.includes(source)
                    ? 'border-purple-500 bg-purple-50 dark:bg-purple-900/30'
                    : 'border-gray-300 dark:border-gray-600'
                }`}>
                  <input
                    type="checkbox"
                    checked={wizardData.sentiment.newsSources.includes(source)}
                    onChange={(e) => {
                      const newSources = e.target.checked
                        ? [...wizardData.sentiment.newsSources, source]
                        : wizardData.sentiment.newsSources.filter(s => s !== source);
                      setWizardData({
                        ...wizardData,
                        sentiment: { ...wizardData.sentiment, newsSources: newSources }
                      });
                    }}
                    className="sr-only"
                  />
                  <span className="text-sm capitalize">{source.replace('_', ' ')}</span>
                </label>
              ))}
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium mb-2">Lookback Periods</label>
            <div className="flex flex-wrap gap-2">
              {['1d', '1w', '1m', '6m'].map(period => (
                <label key={period} className={`px-3 py-2 rounded-lg cursor-pointer border ${
                  wizardData.sentiment.lookbackPeriods.includes(period)
                    ? 'border-purple-500 bg-purple-50 dark:bg-purple-900/30'
                    : 'border-gray-300 dark:border-gray-600'
                }`}>
                  <input
                    type="checkbox"
                    checked={wizardData.sentiment.lookbackPeriods.includes(period)}
                    onChange={(e) => {
                      const newPeriods = e.target.checked
                        ? [...wizardData.sentiment.lookbackPeriods, period]
                        : wizardData.sentiment.lookbackPeriods.filter(p => p !== period);
                      setWizardData({
                        ...wizardData,
                        sentiment: { ...wizardData.sentiment, lookbackPeriods: newPeriods }
                      });
                    }}
                    className="sr-only"
                  />
                  <span className="text-sm">{period}</span>
                </label>
              ))}
            </div>
          </div>

          <div className="bg-purple-50 dark:bg-purple-900/20 p-3 rounded-md">
            <p className="text-sm text-purple-800 dark:text-purple-200">
              Sentiment features like <code className="bg-purple-100 dark:bg-purple-800 px-1 rounded">news_1d_positive_short</code>,
              <code className="bg-purple-100 dark:bg-purple-800 px-1 rounded ml-1">news_1w_negative_long</code> will be generated.
            </p>
          </div>
        </div>
      )}
    </div>
  );

  const renderStep5 = () => (
    <div className="space-y-4">
      <div className="flex items-center justify-between p-4 border border-gray-200 dark:border-gray-700 rounded-lg">
        <div className="flex items-center gap-3">
          <BarChart3 className="w-6 h-6 text-green-500" />
          <div>
            <h3 className="font-semibold">Enable Fundamentals & Macro Data</h3>
            <p className="text-sm text-gray-500">Add company fundamentals and macroeconomic indicators</p>
          </div>
        </div>
        <label className="relative inline-flex items-center cursor-pointer">
          <input
            type="checkbox"
            checked={wizardData.fundamentals.enabled}
            onChange={(e) => setWizardData({
              ...wizardData,
              fundamentals: { ...wizardData.fundamentals, enabled: e.target.checked }
            })}
            className="sr-only peer"
          />
          <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 dark:peer-focus:ring-blue-800 rounded-full peer dark:bg-gray-700 peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all dark:border-gray-600 peer-checked:bg-blue-600"></div>
        </label>
      </div>

      {wizardData.fundamentals.enabled && (
        <div className="space-y-4 p-4 bg-gray-50 dark:bg-gray-700/50 rounded-lg">
          <div>
            <label className="block text-sm font-medium mb-2">Fundamental Metrics</label>
            <div className="grid grid-cols-2 gap-2">
              {[
                { id: 'fcf', label: 'Free Cash Flow (FCF)' },
                { id: 'pe', label: 'P/E Ratio' },
                { id: 'eps', label: 'Earnings Per Share (EPS)' },
                { id: 'revenue', label: 'Revenue' },
                { id: 'debt_equity', label: 'Debt/Equity Ratio' },
                { id: 'roe', label: 'Return on Equity (ROE)' }
              ].map(metric => (
                <label key={metric.id} className={`p-3 rounded-lg cursor-pointer border ${
                  wizardData.fundamentals.metrics.includes(metric.id)
                    ? 'border-green-500 bg-green-50 dark:bg-green-900/30'
                    : 'border-gray-300 dark:border-gray-600'
                }`}>
                  <input
                    type="checkbox"
                    checked={wizardData.fundamentals.metrics.includes(metric.id)}
                    onChange={(e) => {
                      const newMetrics = e.target.checked
                        ? [...wizardData.fundamentals.metrics, metric.id]
                        : wizardData.fundamentals.metrics.filter(m => m !== metric.id);
                      setWizardData({
                        ...wizardData,
                        fundamentals: { ...wizardData.fundamentals, metrics: newMetrics }
                      });
                    }}
                    className="sr-only"
                  />
                  <span className="text-sm">{metric.label}</span>
                </label>
              ))}
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium mb-2">Macro Economic Indicators</label>
            <div className="grid grid-cols-2 gap-2">
              {[
                { id: 'interest_rate', label: 'Interest Rates' },
                { id: 'gdp', label: 'GDP Growth' },
                { id: 'inflation', label: 'Inflation Rate (CPI)' },
                { id: 'unemployment', label: 'Unemployment Rate' }
              ].map(indicator => (
                <label key={indicator.id} className={`p-3 rounded-lg cursor-pointer border ${
                  wizardData.fundamentals.macroIndicators.includes(indicator.id)
                    ? 'border-green-500 bg-green-50 dark:bg-green-900/30'
                    : 'border-gray-300 dark:border-gray-600'
                }`}>
                  <input
                    type="checkbox"
                    checked={wizardData.fundamentals.macroIndicators.includes(indicator.id)}
                    onChange={(e) => {
                      const newIndicators = e.target.checked
                        ? [...wizardData.fundamentals.macroIndicators, indicator.id]
                        : wizardData.fundamentals.macroIndicators.filter(i => i !== indicator.id);
                      setWizardData({
                        ...wizardData,
                        fundamentals: { ...wizardData.fundamentals, macroIndicators: newIndicators }
                      });
                    }}
                    className="sr-only"
                  />
                  <span className="text-sm">{indicator.label}</span>
                </label>
              ))}
            </div>
          </div>

          <div className="bg-green-50 dark:bg-green-900/20 p-3 rounded-md">
            <p className="text-sm text-green-800 dark:text-green-200">
              Features like <code className="bg-green-100 dark:bg-green-800 px-1 rounded">days_to_last_fcf</code>,
              <code className="bg-green-100 dark:bg-green-800 px-1 rounded ml-1">last_eps_percent</code> will be generated.
            </p>
          </div>
        </div>
      )}
    </div>
  );

  const renderStep6 = () => (
    <div className="space-y-4">
      <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
        Review your dataset configuration:
      </p>
      <div className="bg-gray-50 dark:bg-gray-700 p-4 rounded-md space-y-3 max-h-96 overflow-y-auto">
        <div className="flex justify-between py-2 border-b border-gray-200 dark:border-gray-600">
          <span className="font-medium">Ticker:</span>
          <span className="font-mono">{wizardData.ticker}</span>
        </div>
        <div className="flex justify-between py-2 border-b border-gray-200 dark:border-gray-600">
          <span className="font-medium">Timeframe:</span>
          <span>{wizardData.timeframe}</span>
        </div>
        <div className="flex justify-between py-2 border-b border-gray-200 dark:border-gray-600">
          <span className="font-medium">Data Provider:</span>
          <span className="capitalize">{wizardData.dataProvider}</span>
        </div>
        <div className="flex justify-between py-2 border-b border-gray-200 dark:border-gray-600">
          <span className="font-medium">Date Range:</span>
          <span>{wizardData.startDate || '1 year ago'} - {wizardData.endDate || 'Today'}</span>
        </div>

        <div className="py-2 border-b border-gray-200 dark:border-gray-600">
          <div className="flex justify-between items-center">
            <span className="font-medium flex items-center gap-2">
              <TrendingUp className="w-4 h-4 text-blue-500" />
              Technical Indicators:
            </span>
            <span>{wizardData.indicators.filter(i => i.enabled).length} selected</span>
          </div>
          {wizardData.indicators.filter(i => i.enabled).length > 0 && (
            <div className="mt-2 text-sm text-gray-600 dark:text-gray-400 space-y-1">
              {wizardData.indicators.filter(i => i.enabled).map(indicator => (
                <div key={indicator.type} className="flex justify-between pl-6">
                  <span>{indicator.name}</span>
                  {indicator.period && <span>Period: {indicator.period}</span>}
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="py-2 border-b border-gray-200 dark:border-gray-600">
          <div className="flex justify-between items-center">
            <span className="font-medium flex items-center gap-2">
              <MessageSquare className="w-4 h-4 text-purple-500" />
              Sentiment Analysis:
            </span>
            <span className={wizardData.sentiment.enabled ? 'text-green-600' : 'text-gray-500'}>
              {wizardData.sentiment.enabled ? 'Enabled' : 'Disabled'}
            </span>
          </div>
          {wizardData.sentiment.enabled && (
            <div className="mt-2 text-sm text-gray-600 dark:text-gray-400 pl-6">
              <div>Sources: {wizardData.sentiment.newsSources.join(', ')}</div>
              <div>Periods: {wizardData.sentiment.lookbackPeriods.join(', ')}</div>
            </div>
          )}
        </div>

        <div className="py-2">
          <div className="flex justify-between items-center">
            <span className="font-medium flex items-center gap-2">
              <BarChart3 className="w-4 h-4 text-green-500" />
              Fundamentals & Macro:
            </span>
            <span className={wizardData.fundamentals.enabled ? 'text-green-600' : 'text-gray-500'}>
              {wizardData.fundamentals.enabled ? 'Enabled' : 'Disabled'}
            </span>
          </div>
          {wizardData.fundamentals.enabled && (
            <div className="mt-2 text-sm text-gray-600 dark:text-gray-400 pl-6">
              <div>Metrics: {wizardData.fundamentals.metrics.join(', ')}</div>
              <div>Macro: {wizardData.fundamentals.macroIndicators.join(', ')}</div>
            </div>
          )}
        </div>
      </div>

      <div className="bg-blue-50 dark:bg-blue-900/20 p-3 rounded-md">
        <p className="text-sm text-blue-800 dark:text-blue-200">
          Click "Create Dataset" to fetch data and build your dataset with all configured features.
        </p>
      </div>
    </div>
  );

  const steps = [
    { num: 1, label: 'Ticker', icon: FileText },
    { num: 2, label: 'Provider', icon: Settings },
    { num: 3, label: 'Indicators', icon: TrendingUp },
    { num: 4, label: 'Sentiment', icon: MessageSquare },
    { num: 5, label: 'Fundamentals', icon: BarChart3 },
    { num: 6, label: 'Review', icon: CheckCircle }
  ];

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-xl w-full max-w-2xl mx-4 max-h-[90vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-gray-200 dark:border-gray-700">
          <h2 className="text-2xl font-bold">Create New Dataset</h2>
          <button
            onClick={onClose}
            className="text-gray-500 hover:text-gray-700 dark:hover:text-gray-300"
          >
            <X size={24} />
          </button>
        </div>

        {/* Steps indicator */}
        <div className="px-6 pt-4 pb-2 overflow-x-auto">
          <div className="flex items-center justify-between min-w-max">
            {steps.map((step, index) => (
              <React.Fragment key={step.num}>
                <div className="flex flex-col items-center">
                  <div className={`w-8 h-8 rounded-full flex items-center justify-center ${
                    currentStep >= step.num ? 'bg-blue-500 text-white' : 'bg-gray-300 text-gray-600'
                  }`}>
                    <step.icon className="w-4 h-4" />
                  </div>
                  <span className={`text-xs mt-1 ${currentStep >= step.num ? 'font-medium' : 'text-gray-500'}`}>
                    {step.label}
                  </span>
                </div>
                {index < steps.length - 1 && (
                  <div className={`flex-1 h-0.5 mx-2 ${currentStep > step.num ? 'bg-blue-500' : 'bg-gray-300'}`}></div>
                )}
              </React.Fragment>
            ))}
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto px-6 py-4">
          {error && (
            <div className="mb-4 p-3 bg-red-100 dark:bg-red-900 text-red-700 dark:text-red-200 rounded-md">
              {error}
            </div>
          )}

          {currentStep === 1 && renderStep1()}
          {currentStep === 2 && renderStep2()}
          {currentStep === 3 && renderStep3()}
          {currentStep === 4 && renderStep4()}
          {currentStep === 5 && renderStep5()}
          {currentStep === 6 && renderStep6()}
        </div>

        {/* Footer */}
        <div className="flex justify-between p-6 border-t border-gray-200 dark:border-gray-700">
          <button
            onClick={currentStep === 1 ? onClose : handleBack}
            className="px-4 py-2 text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-md"
            disabled={isCreating}
          >
            {currentStep === 1 ? 'Cancel' : 'Back'}
          </button>
          <button
            onClick={currentStep === 6 ? handleCreate : handleNext}
            className="px-4 py-2 bg-blue-500 text-white rounded-md hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed"
            disabled={isCreating}
          >
            {isCreating ? 'Creating...' : currentStep === 6 ? 'Create Dataset' : 'Next'}
          </button>
        </div>
      </div>
    </div>
  );
};

export default DatasetWizard;
