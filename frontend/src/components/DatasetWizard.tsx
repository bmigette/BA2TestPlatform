import React, { useState, useEffect } from 'react';
import { X, MessageSquare, TrendingUp, BarChart3, FileText, Settings, CheckCircle, Plus, Trash2, Save, FolderOpen } from 'lucide-react';

type WizardMode = 'create' | 'duplicate' | 'edit';

interface InitialDataset {
  id: number;
  name: string;
  ticker: string;
  timeframe: string;
  start_date: string;
  end_date: string;
  normalization_buffer_pct?: number;
  technical_indicators?: any;
  generation_config?: any;
}

interface DatasetWizardProps {
  isOpen: boolean;
  onClose: () => void;
  onComplete: () => void;
  mode?: WizardMode;
  initialData?: InitialDataset | null;
}

// Updated indicator config with individual timeframe
interface IndicatorConfig {
  id: string;  // Unique ID for React keys
  name: string;
  type: string;
  timeframe: string;
  period?: number;
  fast?: number;
  slow?: number;
  signal?: number;
  std_dev?: number;
  k_period?: number;
  d_period?: number;
  smooth_k?: number;
}

interface IndicatorCollection {
  id: number;
  name: string;
  description: string;
  is_default: boolean;
  indicators: IndicatorConfig[];
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
  fundamentalsProvider: string;
  macroProvider: string;
}

interface WizardData {
  ticker: string;
  timeframe: string;
  startDate: string;
  endDate: string;
  dataProvider: string;
  normalizationBufferPct: number;
  indicators: IndicatorConfig[];
  sentiment: SentimentConfig;
  fundamentals: FundamentalsConfig;
}

// Timeframe ordering for validation
const TIMEFRAME_ORDER = ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w', '1mo'];

const TIMEFRAME_LABELS: Record<string, string> = {
  '1m': '1 Minute',
  '5m': '5 Minutes',
  '15m': '15 Minutes',
  '30m': '30 Minutes',
  '1h': '1 Hour',
  '4h': '4 Hours',
  '1d': '1 Day',
  '1w': '1 Week',
  '1mo': '1 Month'
};

// Available indicator types
const INDICATOR_TYPES = [
  { type: 'sma', name: 'SMA (Simple Moving Average)', hasPeriod: true, defaultPeriod: 20 },
  { type: 'ema', name: 'EMA (Exponential Moving Average)', hasPeriod: true, defaultPeriod: 20 },
  { type: 'rsi', name: 'RSI (Relative Strength Index)', hasPeriod: true, defaultPeriod: 14 },
  { type: 'macd', name: 'MACD', hasPeriod: false },
  { type: 'bbands', name: 'Bollinger Bands', hasPeriod: true, defaultPeriod: 20 },
  { type: 'atr', name: 'ATR (Average True Range)', hasPeriod: true, defaultPeriod: 14 },
  { type: 'stochastic', name: 'Stochastic Oscillator', hasPeriod: false },
];

const getDefaultWizardData = (): WizardData => ({
  ticker: '',
  timeframe: '1d',
  startDate: '',
  endDate: '',
  dataProvider: 'yfinance',
  normalizationBufferPct: 0.35,
  indicators: [],
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
    macroIndicators: ['interest_rate', 'gdp', 'inflation', 'unemployment'],
    fundamentalsProvider: 'yfinance',
    macroProvider: 'fred'
  }
});

const DatasetWizard: React.FC<DatasetWizardProps> = ({ isOpen, onClose, onComplete, mode = 'create', initialData = null }) => {
  const [currentStep, setCurrentStep] = useState(1);
  const [wizardData, setWizardData] = useState<WizardData>(getDefaultWizardData());

  // Initialize from initialData when mode changes
  useEffect(() => {
    if (isOpen && initialData && (mode === 'duplicate' || mode === 'edit')) {
      // Parse dates from ISO format
      const startDate = initialData.generation_config?.original_start_date ||
                       (initialData.start_date ? initialData.start_date.split('T')[0] : '');
      const endDate = initialData.generation_config?.original_end_date ||
                     (initialData.end_date ? initialData.end_date.split('T')[0] : '');

      // Parse indicators
      let indicators: IndicatorConfig[] = [];
      if (initialData.technical_indicators && Array.isArray(initialData.technical_indicators)) {
        indicators = initialData.technical_indicators.map((ind: any) => ({
          id: `ind_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
          ...ind
        }));
      }

      setWizardData({
        ticker: mode === 'duplicate' ? '' : initialData.ticker,  // Clear ticker for duplicate
        timeframe: initialData.timeframe,
        startDate,
        endDate,
        dataProvider: initialData.generation_config?.data_provider || 'yfinance',
        normalizationBufferPct: initialData.normalization_buffer_pct || 0.35,
        indicators,
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
          macroIndicators: ['interest_rate', 'gdp', 'inflation', 'unemployment'],
          fundamentalsProvider: 'yfinance',
          macroProvider: 'fred'
        }
      });
    } else if (isOpen && mode === 'create') {
      setWizardData(getDefaultWizardData());
    }
  }, [isOpen, mode, initialData]);
  const [isCreating, setIsCreating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Indicator add form state
  const [newIndicatorType, setNewIndicatorType] = useState('sma');
  const [newIndicatorTimeframe, setNewIndicatorTimeframe] = useState('1d');
  const [newIndicatorPeriod, setNewIndicatorPeriod] = useState(20);

  // Collections state
  const [collections, setCollections] = useState<IndicatorCollection[]>([]);
  const [selectedCollectionId, setSelectedCollectionId] = useState<number | null>(null);
  const [saveCollectionName, setSaveCollectionName] = useState('');
  const [showSaveDialog, setShowSaveDialog] = useState(false);

  // Fetch collections on mount
  useEffect(() => {
    if (isOpen) {
      fetchCollections();
    }
  }, [isOpen]);

  // Update indicator timeframe dropdown when dataset timeframe changes
  useEffect(() => {
    const validTimeframes = getAvailableTimeframes();
    if (!validTimeframes.includes(newIndicatorTimeframe)) {
      setNewIndicatorTimeframe(wizardData.timeframe);
    }
  }, [wizardData.timeframe]);

  const fetchCollections = async () => {
    try {
      const response = await fetch('http://localhost:8002/api/indicator-collections');
      if (response.ok) {
        const data = await response.json();
        setCollections(data.collections);
      }
    } catch (err) {
      console.error('Failed to fetch collections:', err);
    }
  };

  const getAvailableTimeframes = () => {
    const datasetIdx = TIMEFRAME_ORDER.indexOf(wizardData.timeframe);
    if (datasetIdx === -1) return TIMEFRAME_ORDER;
    return TIMEFRAME_ORDER.slice(datasetIdx);
  };

  const generateIndicatorId = () => {
    return `ind_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  };

  const getIndicatorDisplayName = (indicator: IndicatorConfig) => {
    const typeInfo = INDICATOR_TYPES.find(t => t.type === indicator.type);
    let name = typeInfo?.name.split(' ')[0] || indicator.type.toUpperCase();

    if (indicator.period) {
      name += ` ${indicator.period}`;
    }

    return `${name} @ ${indicator.timeframe.toUpperCase()}`;
  };

  const addIndicator = () => {
    const typeInfo = INDICATOR_TYPES.find(t => t.type === newIndicatorType);
    if (!typeInfo) return;

    const newIndicator: IndicatorConfig = {
      id: generateIndicatorId(),
      type: newIndicatorType,
      name: typeInfo.name,
      timeframe: newIndicatorTimeframe,
    };

    // Add type-specific parameters
    if (typeInfo.hasPeriod) {
      newIndicator.period = newIndicatorPeriod;
    }

    if (newIndicatorType === 'macd') {
      newIndicator.fast = 12;
      newIndicator.slow = 26;
      newIndicator.signal = 9;
    } else if (newIndicatorType === 'bbands') {
      newIndicator.std_dev = 2.0;
    } else if (newIndicatorType === 'stochastic') {
      newIndicator.k_period = 14;
      newIndicator.d_period = 3;
      newIndicator.smooth_k = 3;
    }

    setWizardData({
      ...wizardData,
      indicators: [...wizardData.indicators, newIndicator]
    });
  };

  const removeIndicator = (id: string) => {
    setWizardData({
      ...wizardData,
      indicators: wizardData.indicators.filter(i => i.id !== id)
    });
  };

  const loadCollection = (collectionId: number) => {
    const collection = collections.find(c => c.id === collectionId);
    if (!collection) return;

    // Filter indicators to only include those with valid timeframes
    const validTimeframes = getAvailableTimeframes();
    const validIndicators = collection.indicators
      .filter(ind => validTimeframes.includes(ind.timeframe))
      .map(ind => ({
        ...ind,
        id: generateIndicatorId()
      }));

    const invalidCount = collection.indicators.length - validIndicators.length;

    setWizardData({
      ...wizardData,
      indicators: validIndicators
    });

    setSelectedCollectionId(collectionId);

    if (invalidCount > 0) {
      setError(`${invalidCount} indicators were skipped because their timeframe is smaller than the dataset timeframe (${wizardData.timeframe})`);
    }
  };

  const saveCollection = async () => {
    if (!saveCollectionName.trim()) {
      setError('Please enter a collection name');
      return;
    }

    try {
      const response = await fetch('http://localhost:8002/api/indicator-collections', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: saveCollectionName,
          description: `Custom collection with ${wizardData.indicators.length} indicators`,
          indicators: wizardData.indicators.map(({ id, ...rest }) => rest)
        })
      });

      if (response.ok) {
        setShowSaveDialog(false);
        setSaveCollectionName('');
        fetchCollections();
        setError(null);
      } else {
        const data = await response.json();
        setError(data.detail || 'Failed to save collection');
      }
    } catch (err) {
      setError('Failed to save collection');
    }
  };

  if (!isOpen) return null;

  const validateTicker = (ticker: string): boolean => {
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
      // Convert indicators to API format
      const technicalIndicators = wizardData.indicators.map(({ id, name, ...rest }) => rest);

      let response: Response;

      if (mode === 'duplicate' && initialData) {
        // Duplicate: POST to /{id}/duplicate
        response = await fetch(`http://localhost:8002/api/datasets/${initialData.id}/duplicate`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            new_ticker: wizardData.ticker || undefined,
            new_name: undefined  // Let backend generate name
          }),
        });
      } else if (mode === 'edit' && initialData) {
        // Edit: PUT to /{id}
        response = await fetch(`http://localhost:8002/api/datasets/${initialData.id}`, {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            ticker: wizardData.ticker,
            timeframe: wizardData.timeframe,
            start_date: wizardData.startDate || undefined,
            end_date: wizardData.endDate || undefined,
            technical_indicators: technicalIndicators,
            normalization_buffer_pct: wizardData.normalizationBufferPct
          }),
        });
      } else {
        // Create: POST to /
        response = await fetch('http://localhost:8002/api/datasets', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            ticker: wizardData.ticker,
            timeframe: wizardData.timeframe,
            start_date: wizardData.startDate || undefined,
            end_date: wizardData.endDate || undefined,
            data_provider: wizardData.dataProvider,
            normalization_buffer_pct: wizardData.normalizationBufferPct,
            technical_indicators: technicalIndicators,
            sentiment_config: wizardData.sentiment.enabled ? wizardData.sentiment : null,
            fundamentals_config: wizardData.fundamentals.enabled ? wizardData.fundamentals : null
          }),
        });
      }

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `Failed to ${mode} dataset`);
      }

      onComplete();
      onClose();

      // Reset wizard
      setCurrentStep(1);
      setWizardData(getDefaultWizardData());
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setIsCreating(false);
    }
  };

  const getActionButtonText = () => {
    if (isCreating) {
      switch (mode) {
        case 'duplicate': return 'Duplicating...';
        case 'edit': return 'Updating...';
        default: return 'Creating...';
      }
    }
    switch (mode) {
      case 'duplicate': return 'Duplicate Dataset';
      case 'edit': return 'Update Dataset';
      default: return 'Create Dataset';
    }
  };

  const getModalTitle = () => {
    switch (mode) {
      case 'duplicate': return 'Duplicate Dataset';
      case 'edit': return 'Edit Dataset';
      default: return 'Create New Dataset';
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
        <label className="block text-sm font-medium mb-2 text-gray-700 dark:text-gray-200">
          Ticker Symbol <span className="text-red-500">*</span>
        </label>
        <input
          type="text"
          value={wizardData.ticker}
          onChange={(e) => setWizardData({ ...wizardData, ticker: e.target.value.toUpperCase() })}
          placeholder="e.g., AAPL, MSFT, GOOGL"
          className={`w-full px-3 py-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-gray-100 ${
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
        <label className="block text-sm font-medium mb-2 text-gray-700 dark:text-gray-200">Timeframe</label>
        <select
          value={wizardData.timeframe}
          onChange={(e) => setWizardData({ ...wizardData, timeframe: e.target.value })}
          className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:text-gray-100"
        >
          {TIMEFRAME_ORDER.map(tf => (
            <option key={tf} value={tf}>{TIMEFRAME_LABELS[tf]}</option>
          ))}
        </select>
      </div>

      <div>
        <label className="block text-sm font-medium mb-2 text-gray-700 dark:text-gray-200">Start Date (optional)</label>
        <input
          type="date"
          value={wizardData.startDate}
          onChange={(e) => setWizardData({ ...wizardData, startDate: e.target.value })}
          className={`w-full px-3 py-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-gray-100 ${
            dateError
              ? 'border-red-500 focus:ring-red-500'
              : 'border-gray-300 dark:border-gray-600'
          }`}
        />
        <p className="text-xs text-gray-400 dark:text-gray-400 mt-1">Leave empty for 1 year of data</p>
      </div>

      <div>
        <label className="block text-sm font-medium mb-2 text-gray-700 dark:text-gray-200">End Date (optional)</label>
        <input
          type="date"
          value={wizardData.endDate}
          onChange={(e) => setWizardData({ ...wizardData, endDate: e.target.value })}
          className={`w-full px-3 py-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-gray-100 ${
            dateError
              ? 'border-red-500 focus:ring-red-500'
              : 'border-gray-300 dark:border-gray-600'
          }`}
        />
        {dateError ? (
          <p className="text-xs text-red-500 mt-1">{dateError}</p>
        ) : (
          <p className="text-xs text-gray-400 dark:text-gray-400 mt-1">Leave empty for today</p>
        )}
      </div>
    </div>
  );

  const renderStep2 = () => (
    <div className="space-y-4">
      <p className="text-sm text-gray-600 dark:text-gray-300 mb-4">
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
            <div className="font-semibold text-lg mb-1 text-gray-900 dark:text-gray-100">{provider.name}</div>
            <p className="text-sm text-gray-600 dark:text-gray-300">{provider.desc}</p>
            <div className="mt-2 flex items-center space-x-2 text-xs">
              {provider.tags.map(tag => (
                <span key={tag} className={`px-2 py-1 rounded ${tag.includes('Free') ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200' : 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200'}`}>
                  {tag}
                </span>
              ))}
              {provider.recommended && <span className="text-gray-400 dark:text-gray-400">Recommended</span>}
            </div>
          </label>
        ))}
      </div>

      {/* Advanced Settings */}
      <div className="mt-6 pt-4 border-t border-gray-200 dark:border-gray-700">
        <details className="group">
          <summary className="cursor-pointer text-sm font-medium text-gray-700 dark:text-gray-300 hover:text-gray-900 dark:hover:text-gray-100">
            Advanced Settings
          </summary>
          <div className="mt-4 space-y-4">
            <div>
              <label className="block text-sm font-medium mb-2 text-gray-700 dark:text-gray-200">
                Normalization Buffer (%)
              </label>
              <p className="text-xs text-gray-500 dark:text-gray-400 mb-2">
                Extra headroom above/below observed min/max for live trading. Higher values handle more price growth.
              </p>
              <div className="flex items-center gap-4">
                <input
                  type="range"
                  min="10"
                  max="100"
                  step="5"
                  value={wizardData.normalizationBufferPct * 100}
                  onChange={(e) => setWizardData({ ...wizardData, normalizationBufferPct: parseInt(e.target.value) / 100 })}
                  className="flex-1"
                />
                <span className="w-16 text-center font-medium text-gray-900 dark:text-gray-100">
                  {(wizardData.normalizationBufferPct * 100).toFixed(0)}%
                </span>
              </div>
            </div>
          </div>
        </details>
      </div>
    </div>
  );

  const renderStep3 = () => {
    const availableTimeframes = getAvailableTimeframes();
    const selectedTypeInfo = INDICATOR_TYPES.find(t => t.type === newIndicatorType);

    return (
      <div className="space-y-4">
        <p className="text-sm text-gray-600 dark:text-gray-300 mb-2">
          Add technical indicators with individual timeframes. Indicator timeframe must be equal or greater than the dataset timeframe ({wizardData.timeframe}).
        </p>

        {/* Collection load/save controls */}
        <div className="flex items-center gap-2 pb-2 border-b border-gray-200 dark:border-gray-700">
          <div className="flex-1">
            <select
              value={selectedCollectionId || ''}
              onChange={(e) => {
                const id = parseInt(e.target.value);
                if (id) loadCollection(id);
              }}
              className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md text-sm dark:bg-gray-700 dark:text-gray-100"
            >
              <option value="">Load collection...</option>
              {collections.map(c => (
                <option key={c.id} value={c.id}>
                  {c.name} {c.is_default ? '(Default)' : ''}
                </option>
              ))}
            </select>
          </div>
          <button
            onClick={() => setShowSaveDialog(true)}
            disabled={wizardData.indicators.length === 0}
            className="flex items-center gap-1 px-3 py-2 text-sm border border-gray-300 dark:border-gray-600 rounded-md hover:bg-gray-100 dark:hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed text-gray-700 dark:text-gray-200"
          >
            <Save className="w-4 h-4" />
            Save
          </button>
        </div>

        {/* Save collection dialog */}
        {showSaveDialog && (
          <div className="p-3 bg-gray-50 dark:bg-gray-700/50 rounded-lg space-y-2">
            <input
              type="text"
              value={saveCollectionName}
              onChange={(e) => setSaveCollectionName(e.target.value)}
              placeholder="Collection name..."
              className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md text-sm dark:bg-gray-700 dark:text-gray-100"
            />
            <div className="flex gap-2">
              <button
                onClick={saveCollection}
                className="px-3 py-1 bg-blue-500 text-white text-sm rounded-md hover:bg-blue-600"
              >
                Save Collection
              </button>
              <button
                onClick={() => setShowSaveDialog(false)}
                className="px-3 py-1 text-gray-600 dark:text-gray-300 text-sm hover:bg-gray-200 dark:hover:bg-gray-600 rounded-md"
              >
                Cancel
              </button>
            </div>
          </div>
        )}

        {/* Add indicator form */}
        <div className="p-4 bg-gray-50 dark:bg-gray-700/50 rounded-lg space-y-3">
          <div className="font-medium text-sm text-gray-700 dark:text-gray-200">Add Indicator</div>
          <div className="grid grid-cols-3 gap-3">
            <div>
              <label className="block text-xs text-gray-500 dark:text-gray-400 mb-1">Type</label>
              <select
                value={newIndicatorType}
                onChange={(e) => {
                  setNewIndicatorType(e.target.value);
                  const typeInfo = INDICATOR_TYPES.find(t => t.type === e.target.value);
                  if (typeInfo?.defaultPeriod) {
                    setNewIndicatorPeriod(typeInfo.defaultPeriod);
                  }
                }}
                className="w-full px-2 py-1.5 border border-gray-300 dark:border-gray-600 rounded text-sm dark:bg-gray-700 dark:text-gray-100"
              >
                {INDICATOR_TYPES.map(type => (
                  <option key={type.type} value={type.type}>{type.name}</option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-xs text-gray-500 dark:text-gray-400 mb-1">Timeframe</label>
              <select
                value={newIndicatorTimeframe}
                onChange={(e) => setNewIndicatorTimeframe(e.target.value)}
                className="w-full px-2 py-1.5 border border-gray-300 dark:border-gray-600 rounded text-sm dark:bg-gray-700 dark:text-gray-100"
              >
                {availableTimeframes.map(tf => (
                  <option key={tf} value={tf}>{TIMEFRAME_LABELS[tf]}</option>
                ))}
              </select>
            </div>
            <div>
              {selectedTypeInfo?.hasPeriod && (
                <>
                  <label className="block text-xs text-gray-500 dark:text-gray-400 mb-1">Period</label>
                  <input
                    type="number"
                    min="1"
                    max="500"
                    value={newIndicatorPeriod}
                    onChange={(e) => setNewIndicatorPeriod(parseInt(e.target.value) || 1)}
                    className="w-full px-2 py-1.5 border border-gray-300 dark:border-gray-600 rounded text-sm dark:bg-gray-700 dark:text-gray-100"
                  />
                </>
              )}
            </div>
          </div>
          <button
            onClick={addIndicator}
            className="flex items-center gap-1 px-3 py-1.5 bg-blue-500 text-white text-sm rounded-md hover:bg-blue-600"
          >
            <Plus className="w-4 h-4" />
            Add Indicator
          </button>
        </div>

        {/* List of added indicators */}
        <div className="space-y-2 max-h-48 overflow-y-auto">
          {wizardData.indicators.length === 0 ? (
            <div className="text-center py-6 text-gray-400 dark:text-gray-500 text-sm">
              No indicators added yet. Use the form above to add indicators.
            </div>
          ) : (
            wizardData.indicators.map((indicator) => (
              <div
                key={indicator.id}
                className="flex items-center justify-between p-3 border border-gray-200 dark:border-gray-700 rounded-lg"
              >
                <div className="flex items-center gap-3">
                  <TrendingUp className="w-4 h-4 text-blue-500" />
                  <div>
                    <span className="font-medium text-gray-900 dark:text-gray-100">
                      {getIndicatorDisplayName(indicator)}
                    </span>
                  </div>
                </div>
                <button
                  onClick={() => removeIndicator(indicator.id)}
                  className="p-1 text-red-500 hover:bg-red-50 dark:hover:bg-red-900/20 rounded"
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              </div>
            ))
          )}
        </div>

        <div className="bg-blue-50 dark:bg-blue-900/20 p-3 rounded-md">
          <p className="text-sm text-blue-800 dark:text-blue-200">
            Added: <strong>{wizardData.indicators.length}</strong> indicators
          </p>
        </div>
      </div>
    );
  };

  const renderStep4 = () => (
    <div className="space-y-4">
      <div className="flex items-center justify-between p-4 border border-gray-200 dark:border-gray-700 rounded-lg">
        <div className="flex items-center gap-3">
          <MessageSquare className="w-6 h-6 text-purple-500" />
          <div>
            <h3 className="font-semibold text-gray-900 dark:text-gray-100">Enable Sentiment Analysis</h3>
            <p className="text-sm text-gray-500 dark:text-gray-400">Analyze news sentiment for the ticker</p>
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
            <label className="block text-sm font-medium mb-2 text-gray-700 dark:text-gray-200">News Sources</label>
            <div className="flex flex-wrap gap-2">
              {['google_news', 'fmp_news', 'alpaca_news'].map(source => (
                <label key={source} className={`px-3 py-2 rounded-lg cursor-pointer border text-sm ${
                  wizardData.sentiment.newsSources.includes(source)
                    ? 'border-purple-500 bg-purple-50 dark:bg-purple-900/30 text-purple-800 dark:text-purple-200'
                    : 'border-gray-300 dark:border-gray-600 text-gray-700 dark:text-gray-300'
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
                  <span className="capitalize">{source.replace('_', ' ')}</span>
                </label>
              ))}
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium mb-2 text-gray-700 dark:text-gray-200">Lookback Periods</label>
            <div className="flex flex-wrap gap-2">
              {['1d', '1w', '1m', '6m'].map(period => (
                <label key={period} className={`px-3 py-2 rounded-lg cursor-pointer border text-sm ${
                  wizardData.sentiment.lookbackPeriods.includes(period)
                    ? 'border-purple-500 bg-purple-50 dark:bg-purple-900/30 text-purple-800 dark:text-purple-200'
                    : 'border-gray-300 dark:border-gray-600 text-gray-700 dark:text-gray-300'
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
                  <span>{period}</span>
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
            <h3 className="font-semibold text-gray-900 dark:text-gray-100">Enable Fundamentals & Macro Data</h3>
            <p className="text-sm text-gray-500 dark:text-gray-400">Add company fundamentals and macroeconomic indicators</p>
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
            <label className="block text-sm font-medium mb-2 text-gray-700 dark:text-gray-200">Fundamental Metrics</label>
            <div className="grid grid-cols-2 gap-2">
              {[
                { id: 'fcf', label: 'Free Cash Flow (FCF)' },
                { id: 'pe', label: 'P/E Ratio' },
                { id: 'eps', label: 'Earnings Per Share (EPS)' },
                { id: 'revenue', label: 'Revenue' },
                { id: 'debt_equity', label: 'Debt/Equity Ratio' },
                { id: 'roe', label: 'Return on Equity (ROE)' }
              ].map(metric => (
                <label key={metric.id} className={`p-3 rounded-lg cursor-pointer border text-sm ${
                  wizardData.fundamentals.metrics.includes(metric.id)
                    ? 'border-green-500 bg-green-50 dark:bg-green-900/30 text-green-800 dark:text-green-200'
                    : 'border-gray-300 dark:border-gray-600 text-gray-700 dark:text-gray-300'
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
                  <span>{metric.label}</span>
                </label>
              ))}
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium mb-2 text-gray-700 dark:text-gray-200">Macro Economic Indicators</label>
            <div className="grid grid-cols-2 gap-2">
              {[
                { id: 'interest_rate', label: 'Interest Rates' },
                { id: 'gdp', label: 'GDP Growth' },
                { id: 'inflation', label: 'Inflation Rate (CPI)' },
                { id: 'unemployment', label: 'Unemployment Rate' }
              ].map(indicator => (
                <label key={indicator.id} className={`p-3 rounded-lg cursor-pointer border text-sm ${
                  wizardData.fundamentals.macroIndicators.includes(indicator.id)
                    ? 'border-green-500 bg-green-50 dark:bg-green-900/30 text-green-800 dark:text-green-200'
                    : 'border-gray-300 dark:border-gray-600 text-gray-700 dark:text-gray-300'
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
                  <span>{indicator.label}</span>
                </label>
              ))}
            </div>
          </div>

          {/* Data Provider Selection */}
          <div className="pt-4 border-t border-gray-200 dark:border-gray-600">
            <label className="block text-sm font-medium mb-2 text-gray-700 dark:text-gray-200">Data Providers</label>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-xs text-gray-500 dark:text-gray-400 mb-1">Fundamentals Provider</label>
                <select
                  value={wizardData.fundamentals.fundamentalsProvider}
                  onChange={(e) => setWizardData({
                    ...wizardData,
                    fundamentals: { ...wizardData.fundamentals, fundamentalsProvider: e.target.value }
                  })}
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md text-sm dark:bg-gray-700 dark:text-gray-100"
                >
                  <option value="yfinance">Yahoo Finance (Free)</option>
                  <option value="alphavantage">Alpha Vantage (API Key)</option>
                  <option value="fmp">Financial Modeling Prep (API Key)</option>
                </select>
              </div>
              <div>
                <label className="block text-xs text-gray-500 dark:text-gray-400 mb-1">Macro Data Provider</label>
                <select
                  value={wizardData.fundamentals.macroProvider}
                  onChange={(e) => setWizardData({
                    ...wizardData,
                    fundamentals: { ...wizardData.fundamentals, macroProvider: e.target.value }
                  })}
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md text-sm dark:bg-gray-700 dark:text-gray-100"
                >
                  <option value="fred">FRED (Federal Reserve)</option>
                  <option value="alphavantage">Alpha Vantage (API Key)</option>
                </select>
              </div>
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
      <p className="text-sm text-gray-600 dark:text-gray-300 mb-4">
        Review your dataset configuration:
      </p>
      <div className="bg-gray-50 dark:bg-gray-700 p-4 rounded-md space-y-3 max-h-96 overflow-y-auto">
        <div className="flex justify-between py-2 border-b border-gray-200 dark:border-gray-600">
          <span className="font-medium text-gray-700 dark:text-gray-200">Ticker:</span>
          <span className="font-mono text-gray-900 dark:text-gray-100">{wizardData.ticker}</span>
        </div>
        <div className="flex justify-between py-2 border-b border-gray-200 dark:border-gray-600">
          <span className="font-medium text-gray-700 dark:text-gray-200">Timeframe:</span>
          <span className="text-gray-900 dark:text-gray-100">{TIMEFRAME_LABELS[wizardData.timeframe]}</span>
        </div>
        <div className="flex justify-between py-2 border-b border-gray-200 dark:border-gray-600">
          <span className="font-medium text-gray-700 dark:text-gray-200">Data Provider:</span>
          <span className="capitalize text-gray-900 dark:text-gray-100">{wizardData.dataProvider}</span>
        </div>
        <div className="flex justify-between py-2 border-b border-gray-200 dark:border-gray-600">
          <span className="font-medium text-gray-700 dark:text-gray-200">Date Range:</span>
          <span className="text-gray-900 dark:text-gray-100">{wizardData.startDate || '1 year ago'} - {wizardData.endDate || 'Today'}</span>
        </div>

        <div className="py-2 border-b border-gray-200 dark:border-gray-600">
          <div className="flex justify-between items-center">
            <span className="font-medium flex items-center gap-2 text-gray-700 dark:text-gray-200">
              <TrendingUp className="w-4 h-4 text-blue-500" />
              Technical Indicators:
            </span>
            <span className="text-gray-900 dark:text-gray-100">{wizardData.indicators.length} selected</span>
          </div>
          {wizardData.indicators.length > 0 && (
            <div className="mt-2 text-sm text-gray-600 dark:text-gray-300 space-y-1">
              {wizardData.indicators.map(indicator => (
                <div key={indicator.id} className="pl-6">
                  {getIndicatorDisplayName(indicator)}
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="py-2 border-b border-gray-200 dark:border-gray-600">
          <div className="flex justify-between items-center">
            <span className="font-medium flex items-center gap-2 text-gray-700 dark:text-gray-200">
              <MessageSquare className="w-4 h-4 text-purple-500" />
              Sentiment Analysis:
            </span>
            <span className={wizardData.sentiment.enabled ? 'text-green-600 dark:text-green-400' : 'text-gray-500 dark:text-gray-400'}>
              {wizardData.sentiment.enabled ? 'Enabled' : 'Disabled'}
            </span>
          </div>
          {wizardData.sentiment.enabled && (
            <div className="mt-2 text-sm text-gray-600 dark:text-gray-300 pl-6">
              <div>Sources: {wizardData.sentiment.newsSources.join(', ')}</div>
              <div>Periods: {wizardData.sentiment.lookbackPeriods.join(', ')}</div>
            </div>
          )}
        </div>

        <div className="py-2">
          <div className="flex justify-between items-center">
            <span className="font-medium flex items-center gap-2 text-gray-700 dark:text-gray-200">
              <BarChart3 className="w-4 h-4 text-green-500" />
              Fundamentals & Macro:
            </span>
            <span className={wizardData.fundamentals.enabled ? 'text-green-600 dark:text-green-400' : 'text-gray-500 dark:text-gray-400'}>
              {wizardData.fundamentals.enabled ? 'Enabled' : 'Disabled'}
            </span>
          </div>
          {wizardData.fundamentals.enabled && (
            <div className="mt-2 text-sm text-gray-600 dark:text-gray-300 pl-6">
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
          <h2 className="text-2xl font-bold text-gray-900 dark:text-gray-100">{getModalTitle()}</h2>
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
                    currentStep >= step.num ? 'bg-blue-500 text-white' : 'bg-gray-300 dark:bg-gray-600 text-gray-600 dark:text-gray-400'
                  }`}>
                    <step.icon className="w-4 h-4" />
                  </div>
                  <span className={`text-xs mt-1 ${currentStep >= step.num ? 'font-medium text-gray-900 dark:text-gray-100' : 'text-gray-500 dark:text-gray-400'}`}>
                    {step.label}
                  </span>
                </div>
                {index < steps.length - 1 && (
                  <div className={`flex-1 h-0.5 mx-2 ${currentStep > step.num ? 'bg-blue-500' : 'bg-gray-300 dark:bg-gray-600'}`}></div>
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
            {currentStep === 6 ? getActionButtonText() : 'Next'}
          </button>
        </div>
      </div>
    </div>
  );
};

export default DatasetWizard;
