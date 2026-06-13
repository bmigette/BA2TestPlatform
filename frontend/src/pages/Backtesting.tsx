import React, { useEffect, useState, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Play,
  Loader2,
  AlertCircle,
  TrendingUp,
  TrendingDown,
  Target,
  Calendar,
  Settings,
  ChevronDown,
  ChevronUp,
  Download,
  Trash2,
  BarChart3,
  Activity,
  Brain,
  Clock,
  Award,
  ArrowDownRight,
  Filter,
  Save,
  X,
  Database,
  Layers
} from 'lucide-react';
import Tooltip from '../components/Tooltip';
import ConfirmDialog from '../components/ConfirmDialog';
import ConditionBuilder, {
  ExitConditionsBuilder,
  createEmptyGroup,
  isConditionGroup
} from '../components/ConditionBuilder';
// BacktestChart removed - price chart tab not used
import type {
  ConditionGroup,
  ConditionTree,
  ExitConditionSet,
  AvailableField
} from '../components/ConditionBuilder';
import {
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  ResponsiveContainer,
  AreaChart,
  Area,
  ReferenceLine,
} from 'recharts';

interface PredictionTarget {
  type: string;
  category?: string;
  direction?: string;
  horizon?: number;
  profitPct?: number;
  maxDd?: number;
  indicator?: string;
  indicatorType?: string;
  [key: string]: unknown;
}

interface Model {
  id: string;
  name: string;
  modelType: string;
  threshold?: number; // Classification threshold (default 0.5)
  predictionTargets?: PredictionTarget[];
  predictionHorizon?: number;
  performanceMetrics: {
    accuracy: number;
    sharpeRatio: number | null;
  };
}

interface Dataset {
  id: number;
  name: string;
  ticker: string;
  timeframe: string;
  startDate: string;
  endDate: string;
  rowsCount: number;
}

interface Strategy {
  id: number;
  name: string;
  description: string | null;
  requiredFields: string[];
  entryConditions?: ConditionTree;  // Deprecated, for backwards compatibility
  buyEntryConditions?: ConditionTree;
  sellEntryConditions?: ConditionTree;
  exitConditions: ExitConditionSet[];
  initialTpPercent: number;
  initialTpOptimize: boolean;
  initialTpMin: number | null;
  initialTpMax: number | null;
  initialTpStep: number | null;
  initialSlPercent: number;
  initialSlOptimize: boolean;
  initialSlMin: number | null;
  initialSlMax: number | null;
  initialSlStep: number | null;
  createdAt: string;
  updatedAt: string | null;
}

interface Trade {
  id: string | number;
  entryDate: string;
  exitDate: string;
  entryPrice: number;
  exitPrice: number;
  size: number;
  direction: 'long' | 'short';
  pnl: number;
  pnlPercent: number;
  duration: number;
  exitReason: string;
}

interface BacktestResults {
  equityCurve: Array<{ date: string; equity: number }>;
  drawdownCurve: Array<{ date: string; drawdown: number }>;
  trades: Trade[];
  priceData: Array<{ date: string; open: number; high: number; low: number; close: number; signal: number }>;
}

interface Backtest {
  id: number;
  name: string;
  // 'ml' = legacy model-driven backtesting.py run (modelId set);
  // 'daily_expert' = Phase-2 daily multi-asset expert engine (modelId null).
  engineType?: string;
  modelId: number | null;
  predictionDatasetId: number;
  executionDatasetId: number;
  strategyId: number | null;
  strategyParams: Record<string, unknown> | null;
  startDate: string;
  endDate: string;
  initialCapital: number;
  positionSizingType: string;
  positionSizingValue: number;
  commission: number;
  slippage: number;
  fitnessMetric: string | null;
  status: string;
  totalReturn: number | null;
  sharpeRatio: number | null;
  maxDrawdown: number | null;
  winRate: number | null;
  profitFactor: number | null;
  totalTrades: number | null;
  avgTradeDuration: number | null;
  bestTrade: number | null;
  worstTrade: number | null;
  results: BacktestResults | null;
  errorMessage: string | null;
  createdAt: string;
  completedAt: string | null;
}

const API_BASE = 'http://localhost:8000/api';

const Backtesting: React.FC = () => {
  const _navigate = useNavigate();
  void _navigate;

  // State
  const [models, setModels] = useState<Model[]>([]);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [strategies, setStrategies] = useState<Strategy[]>([]);
  const [backtests, setBacktests] = useState<Backtest[]>([]);

  // Form state
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [predictionDatasetId, setPredictionDatasetId] = useState<number | ''>('');
  const [executionDatasetId, setExecutionDatasetId] = useState<number | ''>('');
  const [backtestName, setBacktestName] = useState('');
  const [startDate, setStartDate] = useState('2025-01-01');
  const [endDate, setEndDate] = useState('2025-12-31');

  // Strategy configuration
  const [buyEntryConditions, setBuyEntryConditions] = useState<ConditionGroup>(createEmptyGroup('AND'));
  const [sellEntryConditions, setSellEntryConditions] = useState<ConditionGroup>(createEmptyGroup('AND'));
  const [exitConditions, setExitConditions] = useState<ExitConditionSet[]>([]);
  const [initialTpPercent, setInitialTpPercent] = useState(5.0);
  const [initialSlPercent, setInitialSlPercent] = useState(2.0);
  const [initialTpOptimize, setInitialTpOptimize] = useState(false);
  const [initialSlOptimize, setInitialSlOptimize] = useState(false);
  const [initialTpMin, setInitialTpMin] = useState(2.0);
  const [initialTpMax, setInitialTpMax] = useState(15.0);
  const [initialTpStep, setInitialTpStep] = useState(1.0);
  const [initialSlMin, setInitialSlMin] = useState(1.0);
  const [initialSlMax, setInitialSlMax] = useState(10.0);
  const [initialSlStep, setInitialSlStep] = useState(0.5);

  // Backtest settings
  const [initialCapital, setInitialCapital] = useState(10000);
  const [positionSizingType, setPositionSizingType] = useState('fixed');
  const [positionSizingValue, setPositionSizingValue] = useState(1000);
  const [commission, setCommission] = useState(0.1);
  const [slippage, setSlippage] = useState(0.05);

  // Available fields from model
  const [availableFields, setAvailableFields] = useState<AvailableField[]>([]);

  // UI state
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [showConditionModal, setShowConditionModal] = useState<'buy' | 'sell' | 'exit' | null>(null);
  const [loading, setLoading] = useState(true);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Results view
  const [selectedBacktest, setSelectedBacktest] = useState<Backtest | null>(null);
  const [activeTab, setActiveTab] = useState<'equity' | 'drawdown' | 'trades'>('equity');
  const [tradeFilter, setTradeFilter] = useState<'all' | 'profit' | 'loss'>('all');
  const [tradeSortField, setTradeSortField] = useState<'pnl' | 'date' | 'duration'>('date');
  const [tradeSortAsc, setTradeSortAsc] = useState(false);

  // Dialogs
  const [showSaveDialog, setShowSaveDialog] = useState(false);
  const [saveStrategyName, setSaveStrategyName] = useState('');
  const [saveStrategyDescription, setSaveStrategyDescription] = useState('');
  const [savingStrategy, setSavingStrategy] = useState(false);
  const [confirmDialog, setConfirmDialog] = useState<{
    isOpen: boolean;
    title: string;
    message: string;
    variant: 'danger' | 'warning' | 'info';
    onConfirm: () => void;
  }>({ isOpen: false, title: '', message: '', variant: 'warning', onConfirm: () => {} });

  // Fetch initial data
  useEffect(() => {
    fetchData();
  }, []);

  // Poll for backtest status updates when there are pending/running backtests
  useEffect(() => {
    const hasPendingOrRunning = backtests.some(bt => bt.status === 'pending' || bt.status === 'running');
    if (!hasPendingOrRunning) return;

    const pollInterval = setInterval(async () => {
      try {
        const res = await fetch(`${API_BASE}/backtests`);
        if (res.ok) {
          const data = await res.json();
          const updatedBacktests = data.backtests || [];
          setBacktests(updatedBacktests);

          // Update selected backtest if it was updated
          if (selectedBacktest) {
            const updated = updatedBacktests.find((bt: Backtest) => bt.id === selectedBacktest.id);
            if (updated && updated.status !== selectedBacktest.status) {
              // Fetch full details for the selected backtest
              const detailsRes = await fetch(`${API_BASE}/backtests/${selectedBacktest.id}`);
              if (detailsRes.ok) {
                const details = await detailsRes.json();
                setSelectedBacktest(details);
              }
            }
          }
        }
      } catch (err) {
        console.error('Failed to poll backtests:', err);
      }
    }, 2000); // Poll every 2 seconds

    return () => clearInterval(pollInterval);
  }, [backtests, selectedBacktest]);

  // Fetch prediction fields when model changes
  const fetchPredictionFields = useCallback(async (modelId: string) => {
    if (!modelId) {
      setAvailableFields([]);
      return;
    }

    try {
      const res = await fetch(`${API_BASE}/models/${modelId}/prediction-fields`);
      if (res.ok) {
        const data = await res.json();
        setAvailableFields(data.fields || []);
      }
    } catch (err) {
      console.error('Failed to fetch prediction fields:', err);
    }
  }, []);

  useEffect(() => {
    if (selectedModel) {
      fetchPredictionFields(selectedModel);
    }
  }, [selectedModel, fetchPredictionFields]);

  const fetchData = async () => {
    try {
      setLoading(true);

      // Fetch models, datasets, strategies, and backtests in parallel
      const [modelsRes, datasetsRes, strategiesRes, backtestsRes] = await Promise.all([
        fetch(`${API_BASE}/models`),
        fetch(`${API_BASE}/datasets`),
        fetch(`${API_BASE}/strategies`),
        fetch(`${API_BASE}/backtests`)
      ]);

      if (modelsRes.ok) {
        const data = await modelsRes.json();
        setModels(data.models || []);
      }

      if (datasetsRes.ok) {
        const data = await datasetsRes.json();
        // Transform snake_case to camelCase
        const transformedDatasets = (data.datasets || []).map((d: Record<string, unknown>) => ({
          id: d.id,
          name: d.name,
          ticker: d.ticker,
          timeframe: d.timeframe,
          startDate: d.start_date,
          endDate: d.end_date,
          rowsCount: d.rows_count
        }));
        setDatasets(transformedDatasets);
      }

      if (strategiesRes.ok) {
        const data = await strategiesRes.json();
        setStrategies(data.strategies || []);
      }

      if (backtestsRes.ok) {
        const data = await backtestsRes.json();
        setBacktests(data.backtests || []);
      }

      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load data');
    } finally {
      setLoading(false);
    }
  };

  const runBacktest = async () => {
    if (!selectedModel) {
      setError('Please select a model');
      return;
    }

    if (!predictionDatasetId) {
      setError('Please select a prediction dataset');
      return;
    }

    if (!executionDatasetId) {
      setError('Please select an execution dataset');
      return;
    }

    if (!backtestName.trim()) {
      setError('Please enter a backtest name');
      return;
    }

    // Validate conditions - check for empty fields
    const validateConditions = (tree: ConditionTree, path: string): string | null => {
      if (isConditionGroup(tree)) {
        for (let i = 0; i < tree.conditions.length; i++) {
          const error = validateConditions(tree.conditions[i], `${path}[${i}]`);
          if (error) return error;
        }
        return null;
      } else {
        // It's a ConditionNode
        if (!tree.field || tree.field.trim() === '') {
          return `Empty field in ${path}. Please select a field or remove the condition.`;
        }
        if (!tree.comparison || tree.comparison.trim() === '') {
          return `Empty comparison in ${path}. Please select a comparison operator.`;
        }
        return null;
      }
    };

    const buyError = validateConditions(buyEntryConditions, 'Buy Entry');
    if (buyError) {
      setError(buyError);
      return;
    }

    const sellError = validateConditions(sellEntryConditions, 'Sell Entry');
    if (sellError) {
      setError(sellError);
      return;
    }

    for (let i = 0; i < exitConditions.length; i++) {
      const exitError = validateConditions(exitConditions[i].conditions, `Exit Rule "${exitConditions[i].name}"`);
      if (exitError) {
        setError(exitError);
        return;
      }
    }

    try {
      setRunning(true);
      setError(null);

      // Get model numeric ID
      const model = models.find(m => m.id === selectedModel);
      if (!model) {
        throw new Error('Selected model not found');
      }

      // Build strategy params from current form state
      const strategyParams = {
        buyEntryConditions,
        sellEntryConditions,
        exitConditions: exitConditions.map(ec => ({
          id: ec.id,
          name: ec.name,
          conditions: ec.conditions,
          action: ec.action,
          actionValue: ec.actionValue,
          actionValueOptimize: ec.actionValueOptimize,
          actionValueMin: ec.actionValueMin,
          actionValueMax: ec.actionValueMax,
          actionValueStep: ec.actionValueStep
        })),
        initialTpPercent,
        initialTpOptimize,
        initialTpMin: initialTpOptimize ? initialTpMin : null,
        initialTpMax: initialTpOptimize ? initialTpMax : null,
        initialTpStep: initialTpOptimize ? initialTpStep : null,
        initialSlPercent,
        initialSlOptimize,
        initialSlMin: initialSlOptimize ? initialSlMin : null,
        initialSlMax: initialSlOptimize ? initialSlMax : null,
        initialSlStep: initialSlOptimize ? initialSlStep : null
      };

      const res = await fetch(`${API_BASE}/backtests`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: backtestName,
          model_id: selectedModel,  // String model ID like "mdl-abc123"
          prediction_dataset_id: predictionDatasetId,
          execution_dataset_id: executionDatasetId,
          strategy_params: strategyParams,
          start_date: startDate,
          end_date: endDate,
          initial_capital: initialCapital,
          position_sizing_type: positionSizingType,
          position_sizing_value: positionSizingValue,
          commission,
          slippage
        })
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || 'Failed to run backtest');
      }

      const backtest = await res.json();

      // Fetch full details with results
      const detailsRes = await fetch(`${API_BASE}/backtests/${backtest.id}`);
      if (detailsRes.ok) {
        const details = await detailsRes.json();
        setSelectedBacktest(details);
        setBacktests(prev => [details, ...prev]);
      }

      // Reset form
      setBacktestName('');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to run backtest');
    } finally {
      setRunning(false);
    }
  };

  const viewBacktest = async (id: number) => {
    try {
      const res = await fetch(`${API_BASE}/backtests/${id}`);
      if (res.ok) {
        const data = await res.json();
        setSelectedBacktest(data);
      }
    } catch (err) {
      setError('Failed to load backtest details');
    }
  };

  const deleteBacktest = (id: number) => {
    setConfirmDialog({
      isOpen: true,
      title: 'Delete Backtest',
      message: 'Are you sure you want to delete this backtest?',
      variant: 'danger',
      onConfirm: async () => {
        try {
          await fetch(`${API_BASE}/backtests/${id}`, { method: 'DELETE' });
          setBacktests(prev => prev.filter(b => b.id !== id));
          if (selectedBacktest?.id === id) {
            setSelectedBacktest(null);
          }
        } catch (err) {
          setError('Failed to delete backtest');
        }
      },
    });
  };

  const exportBacktest = async (id: number) => {
    try {
      const res = await fetch(`${API_BASE}/backtests/${id}/export?format=csv`, { method: 'POST' });
      if (res.ok) {
        const data = await res.json();
        alert(`Exported to ${data.path}`);
      }
    } catch (err) {
      setError('Failed to export backtest');
    }
  };

  const getFilteredTrades = () => {
    if (!selectedBacktest?.results?.trades) return [];

    let trades = [...selectedBacktest.results.trades];

    // Filter
    if (tradeFilter === 'profit') {
      trades = trades.filter(t => t.pnl > 0);
    } else if (tradeFilter === 'loss') {
      trades = trades.filter(t => t.pnl < 0);
    }

    // Sort
    trades.sort((a, b) => {
      let cmp = 0;
      if (tradeSortField === 'pnl') {
        cmp = a.pnl - b.pnl;
      } else if (tradeSortField === 'date') {
        cmp = a.entryDate.localeCompare(b.entryDate);
      } else if (tradeSortField === 'duration') {
        cmp = a.duration - b.duration;
      }
      return tradeSortAsc ? cmp : -cmp;
    });

    return trades;
  };

  const saveStrategy = async () => {
    if (!saveStrategyName.trim()) {
      setError('Please enter a strategy name');
      return;
    }

    try {
      setSavingStrategy(true);
      setError(null);

      const res = await fetch(`${API_BASE}/strategies`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: saveStrategyName,
          description: saveStrategyDescription || null,
          buy_entry_conditions: buyEntryConditions,
          sell_entry_conditions: sellEntryConditions,
          exit_conditions: exitConditions.map(ec => ({
            id: ec.id,
            name: ec.name,
            conditions: ec.conditions,
            action: ec.action,
            action_value: ec.actionValue,
            action_value_optimize: ec.actionValueOptimize,
            action_value_min: ec.actionValueMin,
            action_value_max: ec.actionValueMax,
            action_value_step: ec.actionValueStep
          })),
          initial_tp_percent: initialTpPercent,
          initial_tp_optimize: initialTpOptimize,
          initial_tp_min: initialTpOptimize ? initialTpMin : null,
          initial_tp_max: initialTpOptimize ? initialTpMax : null,
          initial_tp_step: initialTpOptimize ? initialTpStep : null,
          initial_sl_percent: initialSlPercent,
          initial_sl_optimize: initialSlOptimize,
          initial_sl_min: initialSlOptimize ? initialSlMin : null,
          initial_sl_max: initialSlOptimize ? initialSlMax : null,
          initial_sl_step: initialSlOptimize ? initialSlStep : null
        })
      });

      if (!res.ok) {
        throw new Error('Failed to save strategy');
      }

      const saved = await res.json();
      setStrategies(prev => [saved, ...prev]);
      setShowSaveDialog(false);
      setSaveStrategyName('');
      setSaveStrategyDescription('');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save strategy');
    } finally {
      setSavingStrategy(false);
    }
  };

  const loadStrategy = (strategy: Strategy) => {
    // Load buy entry conditions - ensure it's a valid group
    if (strategy.buyEntryConditions && isConditionGroup(strategy.buyEntryConditions)) {
      setBuyEntryConditions(strategy.buyEntryConditions);
    } else if (strategy.entryConditions && isConditionGroup(strategy.entryConditions)) {
      // Backwards compatibility: load old entryConditions as buyEntryConditions
      setBuyEntryConditions(strategy.entryConditions);
    } else {
      setBuyEntryConditions(createEmptyGroup('AND'));
    }

    // Load sell entry conditions
    if (strategy.sellEntryConditions && isConditionGroup(strategy.sellEntryConditions)) {
      setSellEntryConditions(strategy.sellEntryConditions);
    } else {
      setSellEntryConditions(createEmptyGroup('AND'));
    }

    // Load exit conditions
    setExitConditions(strategy.exitConditions || []);

    // Load TP/SL settings
    setInitialTpPercent(strategy.initialTpPercent ?? 5.0);
    setInitialTpOptimize(strategy.initialTpOptimize ?? false);
    setInitialTpMin(strategy.initialTpMin ?? 2.0);
    setInitialTpMax(strategy.initialTpMax ?? 15.0);
    setInitialTpStep(strategy.initialTpStep ?? 1.0);
    setInitialSlPercent(strategy.initialSlPercent ?? 2.0);
    setInitialSlOptimize(strategy.initialSlOptimize ?? false);
    setInitialSlMin(strategy.initialSlMin ?? 1.0);
    setInitialSlMax(strategy.initialSlMax ?? 10.0);
    setInitialSlStep(strategy.initialSlStep ?? 0.5);
  };

  if (loading) {
    return (
      <div className="p-6 flex items-center justify-center min-h-96">
        <Loader2 className="w-8 h-8 animate-spin text-blue-500" />
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold flex items-center gap-2 text-gray-900 dark:text-gray-100">
          <BarChart3 className="w-8 h-8 text-blue-500" />
          Backtesting
        </h1>
      </div>

      {error && (
        <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4">
          <div className="flex items-center gap-2 text-red-600 dark:text-red-400">
            <AlertCircle className="w-5 h-5" />
            <span>{error}</span>
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        {/* Configuration Panel */}
        <div className="xl:col-span-1 space-y-4">
          {/* New Backtest Form */}
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-4">
            <h2 className="text-lg font-semibold mb-4 flex items-center gap-2 text-gray-900 dark:text-gray-100">
              <Play className="w-5 h-5 text-green-500" />
              New Backtest
            </h2>

            <div className="space-y-4">
              {/* Backtest Name */}
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Backtest Name
                </label>
                <input
                  type="text"
                  value={backtestName}
                  onChange={e => setBacktestName(e.target.value)}
                  placeholder="e.g., LSTM_AAPL_Conservative"
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                />
              </div>

              {/* Model Selection */}
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  <Brain className="w-4 h-4 inline mr-1" />
                  Select Model
                </label>
                <select
                  value={selectedModel}
                  onChange={e => setSelectedModel(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                >
                  <option value="">-- Select a model --</option>
                  {models.map(model => (
                    <option key={model.id} value={model.id}>
                      {model.name} ({model.modelType})
                    </option>
                  ))}
                </select>
              </div>

              {/* Dual Dataset Selection */}
              <div className="space-y-3 border-t border-gray-200 dark:border-gray-700 pt-4">
                <h3 className="text-sm font-semibold flex items-center gap-2 text-gray-900 dark:text-gray-100">
                  <Database className="w-4 h-4 text-blue-500" />
                  Dataset Selection
                </h3>

                <div>
                  <label className="block text-xs text-gray-500 dark:text-gray-400 mb-1">
                    Prediction Dataset (for model signals)
                  </label>
                  <select
                    value={predictionDatasetId}
                    onChange={e => setPredictionDatasetId(e.target.value ? parseInt(e.target.value) : '')}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-sm"
                  >
                    <option value="">-- Select prediction dataset --</option>
                    {datasets.map(ds => (
                      <option key={ds.id} value={ds.id}>
                        {ds.name} ({ds.ticker} {ds.timeframe})
                      </option>
                    ))}
                  </select>
                </div>

                <div>
                  <label className="block text-xs text-gray-500 dark:text-gray-400 mb-1">
                    Execution Dataset (for price simulation)
                  </label>
                  <select
                    value={executionDatasetId}
                    onChange={e => setExecutionDatasetId(e.target.value ? parseInt(e.target.value) : '')}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-sm"
                  >
                    <option value="">-- Select execution dataset --</option>
                    {datasets.map(ds => (
                      <option key={ds.id} value={ds.id}>
                        {ds.name} ({ds.ticker} {ds.timeframe})
                      </option>
                    ))}
                  </select>
                </div>

                <button
                  type="button"
                  onClick={() => setExecutionDatasetId(predictionDatasetId)}
                  disabled={!predictionDatasetId}
                  className="text-xs text-blue-600 dark:text-blue-400 hover:underline disabled:opacity-50"
                >
                  Use same dataset for both
                </button>
              </div>

              {/* Date Range */}
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    <Calendar className="w-4 h-4 inline mr-1" />
                    Start Date
                  </label>
                  <input
                    type="date"
                    value={startDate}
                    onChange={e => setStartDate(e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    End Date
                  </label>
                  <input
                    type="date"
                    value={endDate}
                    onChange={e => setEndDate(e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  />
                </div>
              </div>

              {/* Strategy Selection */}
              <div className="border-t border-gray-200 dark:border-gray-700 pt-4">
                <h3 className="text-sm font-semibold mb-3 flex items-center gap-2 text-gray-900 dark:text-gray-100">
                  <Layers className="w-4 h-4 text-purple-500" />
                  Strategy
                </h3>

                {/* Load Strategy Dropdown */}
                <div className="flex items-center gap-2 mb-3">
                  <div className="relative flex-1">
                    <select
                      value=""
                      onChange={e => {
                        const stratId = parseInt(e.target.value);
                        const strat = strategies.find(s => s.id === stratId);
                        if (strat) loadStrategy(strat);
                      }}
                      className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-sm"
                    >
                      <option value="">Load from saved strategy...</option>
                      {strategies.map(strat => (
                        <option key={strat.id} value={strat.id}>
                          {strat.name}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>
              </div>

              {/* Entry/Exit Condition Buttons */}
              <div className="space-y-2 border border-gray-200 dark:border-gray-700 rounded-lg p-3">
                  <h4 className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-2">
                    Strategy Conditions
                  </h4>
                  <button
                    onClick={() => setShowConditionModal('buy')}
                    className="flex items-center gap-2 text-sm text-gray-700 dark:text-gray-300 w-full p-2 bg-green-50 dark:bg-green-900/20 hover:bg-green-100 dark:hover:bg-green-900/30 rounded-lg border border-green-200 dark:border-green-800"
                  >
                    <TrendingUp className="w-4 h-4 text-green-600" />
                    <span className="flex-1 text-left">Buy Entry Conditions</span>
                    <span className="text-xs text-gray-500">{buyEntryConditions.conditions.length} condition{buyEntryConditions.conditions.length !== 1 ? 's' : ''}</span>
                    <ChevronDown className="w-4 h-4 text-gray-400" />
                  </button>
                  <button
                    onClick={() => setShowConditionModal('sell')}
                    className="flex items-center gap-2 text-sm text-gray-700 dark:text-gray-300 w-full p-2 bg-red-50 dark:bg-red-900/20 hover:bg-red-100 dark:hover:bg-red-900/30 rounded-lg border border-red-200 dark:border-red-800"
                  >
                    <TrendingDown className="w-4 h-4 text-red-600" />
                    <span className="flex-1 text-left">Sell Entry Conditions</span>
                    <span className="text-xs text-gray-500">{sellEntryConditions.conditions.length} condition{sellEntryConditions.conditions.length !== 1 ? 's' : ''}</span>
                    <ChevronDown className="w-4 h-4 text-gray-400" />
                  </button>
                  <button
                    onClick={() => setShowConditionModal('exit')}
                    className="flex items-center gap-2 text-sm text-gray-700 dark:text-gray-300 w-full p-2 bg-gray-50 dark:bg-gray-700/50 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg border border-gray-200 dark:border-gray-600"
                  >
                    <Target className="w-4 h-4 text-orange-500" />
                    <span className="flex-1 text-left">Exit Conditions</span>
                    <span className="text-xs text-gray-500">{exitConditions.length} rule{exitConditions.length !== 1 ? 's' : ''}</span>
                    <ChevronDown className="w-4 h-4 text-gray-400" />
                  </button>
                </div>

              {/* Initial TP/SL */}
              <div className="border border-gray-200 dark:border-gray-700 rounded-lg p-3">
                <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Initial Take Profit / Stop Loss
                  </h4>
                    <div className="grid grid-cols-2 gap-3">
                      <div>
                        <label className="block text-xs text-gray-500 dark:text-gray-400 mb-1">Take Profit %</label>
                        <div className="flex items-center gap-2">
                          <input
                            type="number"
                            step="0.5"
                            value={initialTpPercent}
                            onChange={e => setInitialTpPercent(parseFloat(e.target.value))}
                            className="flex-1 px-2 py-1.5 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                          />
                          <label className="flex items-center gap-1 text-xs">
                            <input
                              type="checkbox"
                              checked={initialTpOptimize}
                              onChange={e => setInitialTpOptimize(e.target.checked)}
                              className="rounded"
                            />
                            Opt
                          </label>
                        </div>
                        {initialTpOptimize && (
                          <div className="flex items-center gap-1 mt-1">
                            <input
                              type="number"
                              step="0.5"
                              value={initialTpMin}
                              onChange={e => setInitialTpMin(parseFloat(e.target.value))}
                              className="w-14 px-1 py-0.5 text-xs border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700"
                              placeholder="Min"
                            />
                            <span className="text-xs text-gray-500">-</span>
                            <input
                              type="number"
                              step="0.5"
                              value={initialTpMax}
                              onChange={e => setInitialTpMax(parseFloat(e.target.value))}
                              className="w-14 px-1 py-0.5 text-xs border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700"
                              placeholder="Max"
                            />
                            <input
                              type="number"
                              step="0.1"
                              value={initialTpStep}
                              onChange={e => setInitialTpStep(parseFloat(e.target.value))}
                              className="w-12 px-1 py-0.5 text-xs border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700"
                              placeholder="Step"
                            />
                          </div>
                        )}
                      </div>
                      <div>
                        <label className="block text-xs text-gray-500 dark:text-gray-400 mb-1">Stop Loss %</label>
                        <div className="flex items-center gap-2">
                          <input
                            type="number"
                            step="0.5"
                            value={initialSlPercent}
                            onChange={e => setInitialSlPercent(parseFloat(e.target.value))}
                            className="flex-1 px-2 py-1.5 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                          />
                          <label className="flex items-center gap-1 text-xs">
                            <input
                              type="checkbox"
                              checked={initialSlOptimize}
                              onChange={e => setInitialSlOptimize(e.target.checked)}
                              className="rounded"
                            />
                            Opt
                          </label>
                        </div>
                        {initialSlOptimize && (
                          <div className="flex items-center gap-1 mt-1">
                            <input
                              type="number"
                              step="0.5"
                              value={initialSlMin}
                              onChange={e => setInitialSlMin(parseFloat(e.target.value))}
                              className="w-14 px-1 py-0.5 text-xs border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700"
                              placeholder="Min"
                            />
                            <span className="text-xs text-gray-500">-</span>
                            <input
                              type="number"
                              step="0.5"
                              value={initialSlMax}
                              onChange={e => setInitialSlMax(parseFloat(e.target.value))}
                              className="w-14 px-1 py-0.5 text-xs border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700"
                              placeholder="Max"
                            />
                            <input
                              type="number"
                              step="0.1"
                              value={initialSlStep}
                              onChange={e => setInitialSlStep(parseFloat(e.target.value))}
                              className="w-12 px-1 py-0.5 text-xs border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700"
                              placeholder="Step"
                            />
                          </div>
                        )}
                      </div>
                    </div>
                </div>

              {/* Save Strategy Button */}
              <div className="flex items-center gap-2 border-t border-gray-200 dark:border-gray-700 pt-4">
                <Tooltip content="Save current strategy configuration for later use">
                  <button
                    onClick={() => setShowSaveDialog(true)}
                    className="flex items-center gap-1.5 px-3 py-1.5 text-sm text-gray-700 dark:text-gray-300 border border-gray-300 dark:border-gray-600 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
                  >
                    <Save className="w-4 h-4" />
                    Save Strategy
                  </button>
                </Tooltip>
              </div>

              {/* Advanced Options Toggle */}
              <button
                onClick={() => setShowAdvanced(!showAdvanced)}
                className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 hover:text-gray-800 dark:hover:text-gray-200"
              >
                <Settings className="w-4 h-4" />
                Backtest Settings
                {showAdvanced ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
              </button>

              {/* Advanced Options */}
              {showAdvanced && (
                <div className="border border-gray-200 dark:border-gray-700 rounded-lg p-3 space-y-3">
                  <div>
                    <label className="block text-xs text-gray-500 dark:text-gray-400 mb-1">Initial Capital</label>
                    <input
                      type="number"
                      min="1000"
                      step="1000"
                      value={initialCapital}
                      onChange={e => setInitialCapital(parseFloat(e.target.value))}
                      className="w-full px-2 py-1.5 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                    />
                  </div>

                  <div>
                    <label className="block text-xs text-gray-500 dark:text-gray-400 mb-1">Position Sizing</label>
                    <select
                      value={positionSizingType}
                      onChange={e => setPositionSizingType(e.target.value)}
                      className="w-full px-2 py-1.5 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                    >
                      <option value="fixed">Fixed Amount</option>
                      <option value="percent">Percent of Capital</option>
                    </select>
                  </div>

                  <div>
                    <label className="block text-xs text-gray-500 dark:text-gray-400 mb-1">
                      {positionSizingType === 'percent' ? 'Position %' : 'Position Size ($)'}
                    </label>
                    <input
                      type="number"
                      min="0"
                      step={positionSizingType === 'percent' ? '1' : '100'}
                      value={positionSizingValue}
                      onChange={e => setPositionSizingValue(parseFloat(e.target.value))}
                      className="w-full px-2 py-1.5 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                    />
                  </div>

                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <label className="block text-xs text-gray-500 dark:text-gray-400 mb-1">Commission %</label>
                      <input
                        type="number"
                        min="0"
                        step="0.01"
                        value={commission}
                        onChange={e => setCommission(parseFloat(e.target.value))}
                        className="w-full px-2 py-1.5 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                      />
                    </div>
                    <div>
                      <label className="block text-xs text-gray-500 dark:text-gray-400 mb-1">Slippage %</label>
                      <input
                        type="number"
                        min="0"
                        step="0.01"
                        value={slippage}
                        onChange={e => setSlippage(parseFloat(e.target.value))}
                        className="w-full px-2 py-1.5 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                      />
                    </div>
                  </div>
                </div>
              )}

              {/* Run Button */}
              <button
                onClick={runBacktest}
                disabled={running || !selectedModel || !predictionDatasetId || !executionDatasetId}
                className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-green-500 text-white font-medium rounded-lg hover:bg-green-600 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {running ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin" />
                    Running Backtest...
                  </>
                ) : (
                  <>
                    <Play className="w-5 h-5" />
                    Run Backtest
                  </>
                )}
              </button>
            </div>
          </div>

          {/* Previous Backtests */}
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-4">
            <h3 className="text-sm font-semibold mb-3 flex items-center gap-2 text-gray-900 dark:text-gray-100">
              <Clock className="w-4 h-4 text-gray-500" />
              Previous Backtests
            </h3>
            <div className="space-y-2 max-h-80 overflow-y-auto">
              {backtests.length === 0 ? (
                <p className="text-sm text-gray-500 dark:text-gray-400 text-center py-4">No backtests yet</p>
              ) : (
                backtests.map(bt => (
                  <div
                    key={bt.id}
                    className={`p-3 border rounded-lg cursor-pointer transition-colors ${
                      selectedBacktest?.id === bt.id
                        ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                        : 'border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600'
                    }`}
                    onClick={() => viewBacktest(bt.id)}
                  >
                    <div className="flex items-center justify-between mb-1">
                      <span className="font-medium text-sm truncate text-gray-900 dark:text-gray-100">{bt.name}</span>
                      <div className="flex items-center gap-1">
                        <button
                          onClick={e => { e.stopPropagation(); exportBacktest(bt.id); }}
                          className="p-1 hover:bg-gray-200 dark:hover:bg-gray-700 rounded"
                          title="Export"
                        >
                          <Download className="w-3.5 h-3.5 text-gray-500" />
                        </button>
                        <button
                          onClick={e => { e.stopPropagation(); deleteBacktest(bt.id); }}
                          className="p-1 hover:bg-red-100 dark:hover:bg-red-900/20 rounded"
                          title="Delete"
                        >
                          <Trash2 className="w-3.5 h-3.5 text-red-500" />
                        </button>
                      </div>
                    </div>
                    <p className={`text-xs mb-2 ${bt.status === 'failed' ? 'text-red-500' : 'text-gray-500 dark:text-gray-400'}`}>
                      {bt.status === 'pending' ? 'Pending...' :
                       bt.status === 'running' ? 'Running...' :
                       bt.status === 'failed' ? 'Failed' :
                       bt.engineType === 'daily_expert' ? 'Daily expert (multi-asset)' :
                       `Model #${bt.modelId}`}
                    </p>
                    {bt.status === 'failed' && bt.errorMessage && (
                      <p className="text-xs text-red-400 mb-2 truncate" title={bt.errorMessage}>
                        {bt.errorMessage}
                      </p>
                    )}
                    {bt.status === 'completed' && (
                      <div className="flex items-center gap-3 text-xs">
                        <span className={`font-medium ${(bt.totalReturn || 0) >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                          {(bt.totalReturn || 0) >= 0 ? '+' : ''}{bt.totalReturn?.toFixed(1)}%
                        </span>
                        <span className="text-gray-500">Sharpe: {bt.sharpeRatio?.toFixed(2)}</span>
                        <span className="text-gray-500">{bt.totalTrades} trades</span>
                      </div>
                    )}
                  </div>
                ))
              )}
            </div>
          </div>
        </div>

        {/* Results Panel */}
        <div className="xl:col-span-2 space-y-4">
          {selectedBacktest ? (
            <>
              {/* Header: name + engine-type badge (daily expert = multi-asset; ml = model-driven) */}
              <div className="flex items-center justify-between flex-wrap gap-2">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 truncate">
                  {selectedBacktest.name}
                </h3>
                {selectedBacktest.engineType === 'daily_expert' ? (
                  <span className="px-2 py-0.5 text-xs font-medium rounded-full bg-purple-100 text-purple-700 dark:bg-purple-900/30 dark:text-purple-300">
                    Daily expert &middot; multi-asset
                  </span>
                ) : (
                  <span className="px-2 py-0.5 text-xs font-medium rounded-full bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-300">
                    ML strategy{selectedBacktest.modelId != null ? ` · Model #${selectedBacktest.modelId}` : ''}
                  </span>
                )}
              </div>
              {/* Metrics Summary */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm text-gray-500 dark:text-gray-400">Total Return</p>
                      <p className={`text-2xl font-bold ${(selectedBacktest.totalReturn || 0) >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                        {(selectedBacktest.totalReturn || 0) >= 0 ? '+' : ''}{selectedBacktest.totalReturn?.toFixed(1)}%
                      </p>
                    </div>
                    {(selectedBacktest.totalReturn || 0) >= 0 ? (
                      <TrendingUp className="w-8 h-8 text-green-500" />
                    ) : (
                      <TrendingDown className="w-8 h-8 text-red-500" />
                    )}
                  </div>
                </div>

                <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm text-gray-500 dark:text-gray-400">Sharpe Ratio</p>
                      <p className="text-2xl font-bold text-blue-600">{selectedBacktest.sharpeRatio?.toFixed(2)}</p>
                    </div>
                    <Activity className="w-8 h-8 text-blue-500" />
                  </div>
                </div>

                <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm text-gray-500 dark:text-gray-400">Max Drawdown</p>
                      <p className="text-2xl font-bold text-red-600">-{selectedBacktest.maxDrawdown?.toFixed(1)}%</p>
                    </div>
                    <ArrowDownRight className="w-8 h-8 text-red-500" />
                  </div>
                </div>

                <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm text-gray-500 dark:text-gray-400">Win Rate</p>
                      <p className="text-2xl font-bold text-purple-600">{selectedBacktest.winRate?.toFixed(1)}%</p>
                    </div>
                    <Award className="w-8 h-8 text-purple-500" />
                  </div>
                </div>
              </div>

              {/* Additional Metrics */}
              <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
                <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-3 text-center">
                  <p className="text-xs text-gray-500 dark:text-gray-400">Profit Factor</p>
                  <p className="text-lg font-bold text-gray-900 dark:text-gray-100">{selectedBacktest.profitFactor?.toFixed(2)}</p>
                </div>
                <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-3 text-center">
                  <p className="text-xs text-gray-500 dark:text-gray-400">Total Trades</p>
                  <p className="text-lg font-bold text-gray-900 dark:text-gray-100">{selectedBacktest.totalTrades}</p>
                </div>
                <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-3 text-center">
                  <p className="text-xs text-gray-500 dark:text-gray-400">Avg Duration</p>
                  <p className="text-lg font-bold text-gray-900 dark:text-gray-100">{selectedBacktest.avgTradeDuration?.toFixed(1)} bars</p>
                </div>
                <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-3 text-center">
                  <p className="text-xs text-gray-500 dark:text-gray-400">Best Trade</p>
                  <p className="text-lg font-bold text-green-600">+{selectedBacktest.bestTrade?.toFixed(1)}%</p>
                </div>
                <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-3 text-center">
                  <p className="text-xs text-gray-500 dark:text-gray-400">Worst Trade</p>
                  <p className="text-lg font-bold text-red-600">{selectedBacktest.worstTrade?.toFixed(1)}%</p>
                </div>
              </div>

              {/* Chart Tabs */}
              <div className="bg-white dark:bg-gray-800 rounded-lg shadow">
                <div className="border-b border-gray-200 dark:border-gray-700">
                  <nav className="flex">
                    {[
                      { id: 'equity', label: 'Equity Curve', icon: TrendingUp },
                      { id: 'drawdown', label: 'Drawdown', icon: TrendingDown },
                      { id: 'trades', label: 'Trade List', icon: Activity }
                    ].map(tab => (
                      <button
                        key={tab.id}
                        onClick={() => setActiveTab(tab.id as 'equity' | 'drawdown' | 'trades')}
                        className={`flex items-center gap-2 px-4 py-3 border-b-2 transition-colors text-sm ${
                          activeTab === tab.id
                            ? 'border-blue-500 text-blue-600'
                            : 'border-transparent text-gray-500 hover:text-gray-700'
                        }`}
                      >
                        <tab.icon className="w-4 h-4" />
                        {tab.label}
                      </button>
                    ))}
                  </nav>
                </div>

                <div className="p-4">
                  {activeTab === 'equity' && selectedBacktest.results?.equityCurve && (
                    <div className="h-80">
                      <ResponsiveContainer width="100%" height="100%">
                        <AreaChart data={selectedBacktest.results.equityCurve}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                          <XAxis
                            dataKey="date"
                            tickFormatter={(d: string) => {
                              const date = new Date(d);
                              return `${(date.getMonth() + 1).toString().padStart(2, '0')}/${date.getDate().toString().padStart(2, '0')}`;
                            }}
                            tick={{ fontSize: 11 }}
                            interval="preserveStartEnd"
                          />
                          <YAxis
                            domain={['auto', 'auto']}
                            tickFormatter={(v: number) => `$${(v / 1000).toFixed(1)}k`}
                            width={65}
                            tick={{ fontSize: 11 }}
                          />
                          <RechartsTooltip
                            formatter={(value) => [`$${(value as number)?.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 }) ?? '0'}`, 'Equity']}
                            labelFormatter={(label) => {
                              const date = new Date(String(label));
                              return `${date.toLocaleDateString()} ${date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`;
                            }}
                          />
                          <Area type="monotone" dataKey="equity" stroke="#22c55e" fill="#22c55e" fillOpacity={0.2} />
                          <ReferenceLine y={selectedBacktest.initialCapital || 10000} stroke="#888" strokeDasharray="3 3" label={{ value: 'Initial', position: 'right', fontSize: 11 }} />
                        </AreaChart>
                      </ResponsiveContainer>
                    </div>
                  )}

                  {activeTab === 'drawdown' && selectedBacktest.results?.drawdownCurve && (
                    <div className="h-80">
                      <ResponsiveContainer width="100%" height="100%">
                        <AreaChart data={selectedBacktest.results.drawdownCurve}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                          <XAxis
                            dataKey="date"
                            tickFormatter={(d: string) => {
                              const date = new Date(d);
                              return `${(date.getMonth() + 1).toString().padStart(2, '0')}/${date.getDate().toString().padStart(2, '0')}`;
                            }}
                            tick={{ fontSize: 11 }}
                            interval="preserveStartEnd"
                          />
                          <YAxis
                            domain={[0, 'auto']}
                            tickFormatter={(v: number) => `${v.toFixed(1)}%`}
                            width={50}
                            tick={{ fontSize: 11 }}
                            reversed
                          />
                          <RechartsTooltip
                            formatter={(value) => [`${((value as number) ?? 0).toFixed(2)}%`, 'Drawdown']}
                            labelFormatter={(label) => {
                              const date = new Date(String(label));
                              return `${date.toLocaleDateString()} ${date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`;
                            }}
                          />
                          <Area type="monotone" dataKey="drawdown" stroke="#ef4444" fill="#ef4444" fillOpacity={0.3} />
                        </AreaChart>
                      </ResponsiveContainer>
                    </div>
                  )}

                  {activeTab === 'trades' && selectedBacktest.results?.trades && (
                    <div>
                      {/* Trade Filters */}
                      <div className="flex items-center gap-4 mb-4">
                        <div className="flex items-center gap-2">
                          <Filter className="w-4 h-4 text-gray-500" />
                          <select
                            value={tradeFilter}
                            onChange={e => setTradeFilter(e.target.value as 'all' | 'profit' | 'loss')}
                            className="text-sm border border-gray-300 dark:border-gray-600 rounded px-2 py-1 bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                          >
                            <option value="all">All Trades</option>
                            <option value="profit">Profitable</option>
                            <option value="loss">Losing</option>
                          </select>
                        </div>
                        <div className="flex items-center gap-2">
                          <span className="text-sm text-gray-500 dark:text-gray-400">Sort:</span>
                          <select
                            value={tradeSortField}
                            onChange={e => setTradeSortField(e.target.value as 'pnl' | 'date' | 'duration')}
                            className="text-sm border border-gray-300 dark:border-gray-600 rounded px-2 py-1 bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                          >
                            <option value="date">Date</option>
                            <option value="pnl">P&L</option>
                            <option value="duration">Duration</option>
                          </select>
                          <button
                            onClick={() => setTradeSortAsc(!tradeSortAsc)}
                            className="p-1 hover:bg-gray-100 dark:hover:bg-gray-700 rounded"
                          >
                            {tradeSortAsc ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                          </button>
                        </div>
                      </div>

                      {/* Trade Table */}
                      <div className="overflow-x-auto max-h-64 overflow-y-auto">
                        <table className="w-full text-sm">
                          <thead className="bg-gray-50 dark:bg-gray-700/50 sticky top-0">
                            <tr>
                              <th className="px-3 py-2 text-left text-gray-700 dark:text-gray-300">Entry</th>
                              <th className="px-3 py-2 text-left text-gray-700 dark:text-gray-300">Exit</th>
                              <th className="px-3 py-2 text-right text-gray-700 dark:text-gray-300">Entry $</th>
                              <th className="px-3 py-2 text-right text-gray-700 dark:text-gray-300">Exit $</th>
                              <th className="px-3 py-2 text-center text-gray-700 dark:text-gray-300">Dir</th>
                              <th className="px-3 py-2 text-right text-gray-700 dark:text-gray-300">Size</th>
                              <th className="px-3 py-2 text-right text-gray-700 dark:text-gray-300">P&L</th>
                              <th className="px-3 py-2 text-right text-gray-700 dark:text-gray-300">P&L %</th>
                              <th className="px-3 py-2 text-center text-gray-700 dark:text-gray-300">Duration</th>
                              <th className="px-3 py-2 text-center text-gray-700 dark:text-gray-300">Reason</th>
                            </tr>
                          </thead>
                          <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                            {getFilteredTrades().map(trade => (
                              <tr key={trade.id} className="hover:bg-gray-50 dark:hover:bg-gray-700/30">
                                <td className="px-3 py-2 text-gray-900 dark:text-gray-100">{trade.entryDate}</td>
                                <td className="px-3 py-2 text-gray-900 dark:text-gray-100">{trade.exitDate}</td>
                                <td className="px-3 py-2 text-right text-gray-900 dark:text-gray-100">${trade.entryPrice.toFixed(2)}</td>
                                <td className="px-3 py-2 text-right text-gray-900 dark:text-gray-100">${trade.exitPrice.toFixed(2)}</td>
                                <td className="px-3 py-2 text-center">
                                  <span className={`px-2 py-0.5 rounded text-xs font-medium ${
                                    trade.direction === 'long'
                                      ? 'bg-green-100 text-green-700 dark:bg-green-900 dark:text-green-300'
                                      : 'bg-red-100 text-red-700 dark:bg-red-900 dark:text-red-300'
                                  }`}>
                                    {trade.direction}
                                  </span>
                                </td>
                                <td className="px-3 py-2 text-right text-gray-900 dark:text-gray-100">{trade.size}</td>
                                <td className={`px-3 py-2 text-right font-medium ${trade.pnl >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                                  {trade.pnl >= 0 ? '+' : ''}${trade.pnl.toFixed(2)}
                                </td>
                                <td className={`px-3 py-2 text-right font-medium ${trade.pnl >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                                  {trade.pnl >= 0 ? '+' : ''}{trade.pnlPercent.toFixed(2)}%
                                </td>
                                <td className="px-3 py-2 text-center text-gray-900 dark:text-gray-100">{trade.duration} bars</td>
                                <td className="px-3 py-2 text-center">
                                  <span className="px-2 py-0.5 rounded bg-gray-100 dark:bg-gray-700 text-xs text-gray-700 dark:text-gray-300">
                                    {trade.exitReason}
                                  </span>
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </>
          ) : (
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-8 text-center">
              <BarChart3 className="w-16 h-16 text-gray-300 dark:text-gray-600 mx-auto mb-4" />
              <h3 className="text-lg font-medium text-gray-600 dark:text-gray-400 mb-2">No Backtest Selected</h3>
              <p className="text-sm text-gray-500 dark:text-gray-400">
                Configure and run a new backtest, or select a previous backtest to view results.
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Save Strategy Dialog */}
      {showSaveDialog && (
        <div className="fixed inset-0 z-50 overflow-y-auto">
          <div
            className="fixed inset-0 bg-black bg-opacity-50 transition-opacity"
            onClick={() => setShowSaveDialog(false)}
          />
          <div className="flex min-h-full items-center justify-center p-4">
            <div className="relative bg-white dark:bg-gray-800 rounded-lg shadow-xl max-w-md w-full p-6">
              <button
                onClick={() => setShowSaveDialog(false)}
                className="absolute top-4 right-4 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
              >
                <X className="w-5 h-5" />
              </button>

              <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4 flex items-center gap-2">
                <Save className="w-5 h-5 text-blue-500" />
                Save Strategy
              </h3>

              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Strategy Name *
                  </label>
                  <input
                    type="text"
                    value={saveStrategyName}
                    onChange={e => setSaveStrategyName(e.target.value)}
                    placeholder="e.g., Conservative Trend Follower"
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    autoFocus
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Description (optional)
                  </label>
                  <textarea
                    value={saveStrategyDescription}
                    onChange={e => setSaveStrategyDescription(e.target.value)}
                    placeholder="Describe the strategy approach..."
                    rows={3}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 resize-none"
                  />
                </div>

                <div className="bg-gray-50 dark:bg-gray-700/50 rounded-lg p-3 text-sm">
                  <p className="font-medium text-gray-700 dark:text-gray-300 mb-2">Current Configuration:</p>
                  <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-gray-600 dark:text-gray-400">
                    <span>Buy conditions: {buyEntryConditions.conditions.length}</span>
                    <span>Sell conditions: {sellEntryConditions.conditions.length}</span>
                    <span>Exit rules: {exitConditions.length}</span>
                    <span>TP: {initialTpPercent}% / SL: {initialSlPercent}%</span>
                  </div>
                </div>
              </div>

              <div className="flex justify-end gap-3 mt-6">
                <button
                  onClick={() => setShowSaveDialog(false)}
                  className="px-4 py-2 bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-600"
                >
                  Cancel
                </button>
                <button
                  onClick={saveStrategy}
                  disabled={savingStrategy || !saveStrategyName.trim()}
                  className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
                >
                  {savingStrategy ? (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin" />
                      Saving...
                    </>
                  ) : (
                    <>
                      <Save className="w-4 h-4" />
                      Save
                    </>
                  )}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Condition Builder Modal */}
      {showConditionModal && (
        <div className="fixed inset-0 z-50 overflow-y-auto">
          <div
            className="fixed inset-0 bg-black bg-opacity-50 transition-opacity"
            onClick={() => setShowConditionModal(null)}
          />
          <div className="flex min-h-full items-center justify-center p-4">
            <div className="relative bg-white dark:bg-gray-800 rounded-lg shadow-xl max-w-3xl w-full max-h-[80vh] flex flex-col">
              <div className="flex items-center justify-between p-4 border-b border-gray-200 dark:border-gray-700">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 flex items-center gap-2">
                  {showConditionModal === 'buy' && (
                    <>
                      <TrendingUp className="w-5 h-5 text-green-500" />
                      Buy Entry Conditions
                    </>
                  )}
                  {showConditionModal === 'sell' && (
                    <>
                      <TrendingDown className="w-5 h-5 text-red-500" />
                      Sell Entry Conditions
                    </>
                  )}
                  {showConditionModal === 'exit' && (
                    <>
                      <Target className="w-5 h-5 text-orange-500" />
                      Exit Conditions
                    </>
                  )}
                </h3>
                <button
                  onClick={() => setShowConditionModal(null)}
                  className="p-1 hover:bg-gray-100 dark:hover:bg-gray-700 rounded text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>

              <div className="p-4 overflow-y-auto flex-1">
                {/* Model Info: Targets and Threshold */}
                {selectedModel && (() => {
                  const model = models.find(m => m.id === selectedModel);
                  const targets = model?.predictionTargets || [];
                  const threshold = model?.threshold ?? 0.5;
                  const horizon = model?.predictionHorizon;

                  return (
                    <div className="mb-4 space-y-3">
                      {/* Prediction Targets Info */}
                      {targets.length > 0 && (
                        <div className="p-3 bg-purple-50 dark:bg-purple-900/20 rounded-lg border border-purple-200 dark:border-purple-800">
                          <p className="text-xs font-semibold text-purple-700 dark:text-purple-300 mb-2 flex items-center gap-1">
                            <Target className="w-3 h-3" />
                            Prediction Targets
                            {horizon !== undefined && (
                              <span className="ml-2 font-normal text-purple-600 dark:text-purple-400">
                                (Horizon: {horizon} bar{horizon !== 1 ? 's' : ''})
                              </span>
                            )}
                          </p>
                          <div className="space-y-2">
                            {targets.map((target, idx) => {
                              const targetType = target.type || 'unknown';
                              const category = target.category || 'binary_classification';

                              // Build readable description
                              let description = '';
                              if (targetType === 'directional') {
                                description = `${target.direction === 'up' ? 'Price Up' : 'Price Down'} prediction`;
                              } else if (targetType === 'price_based') {
                                description = `Profit ${target.profitPct}%${target.maxDd ? `, Max DD ${target.maxDd}%` : ''}`;
                              } else if (targetType === 'trend_reversal') {
                                description = `${target.indicator || 'Indicator'} ${target.indicatorType || 'reversal'}`;
                              } else {
                                // Show raw properties for other types
                                const props = Object.entries(target)
                                  .filter(([k]) => !['type', 'category', 'enabled', 'color'].includes(k))
                                  .map(([k, v]) => `${k}: ${v}`)
                                  .join(', ');
                                description = props || targetType;
                              }

                              return (
                                <div key={idx} className="flex items-center gap-2 text-xs">
                                  <span className={`px-1.5 py-0.5 rounded text-xs font-medium ${
                                    category === 'binary_classification' ? 'bg-blue-100 text-blue-700 dark:bg-blue-900/50 dark:text-blue-300' :
                                    category === 'multiclass_classification' ? 'bg-orange-100 text-orange-700 dark:bg-orange-900/50 dark:text-orange-300' :
                                    'bg-gray-100 text-gray-700 dark:bg-gray-700 dark:text-gray-300'
                                  }`}>
                                    {targetType.replace(/_/g, ' ')}
                                  </span>
                                  <span className="text-purple-600 dark:text-purple-400">
                                    {description}
                                  </span>
                                </div>
                              );
                            })}
                          </div>
                        </div>
                      )}

                      {/* Model Threshold Info */}
                      <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
                        <p className="text-xs text-blue-700 dark:text-blue-300">
                          <strong>Model Threshold:</strong> {threshold.toFixed(2)}
                        </p>
                        <p className="text-xs text-blue-600 dark:text-blue-400 mt-1">
                          <strong>Prediction</strong> fields use this threshold (Prediction = 1 when Probability ≥ {(threshold * 100).toFixed(0)}%).
                          Use <strong>Probability</strong> fields for custom thresholds.
                        </p>
                      </div>
                    </div>
                  );
                })()}

                {showConditionModal === 'buy' && (
                  <div>
                    <p className="text-sm text-gray-500 dark:text-gray-400 mb-4">
                      Define conditions for opening a long (buy) position. All conditions in a group must be met.
                    </p>
                    <ConditionBuilder
                      value={buyEntryConditions}
                      onChange={(val) => {
                        if (isConditionGroup(val)) {
                          setBuyEntryConditions(val);
                        }
                      }}
                      availableFields={availableFields}
                      showOptimization={true}
                    />
                  </div>
                )}

                {showConditionModal === 'sell' && (
                  <div>
                    <p className="text-sm text-gray-500 dark:text-gray-400 mb-4">
                      Define conditions for opening a short (sell) position. All conditions in a group must be met.
                    </p>
                    <ConditionBuilder
                      value={sellEntryConditions}
                      onChange={(val) => {
                        if (isConditionGroup(val)) {
                          setSellEntryConditions(val);
                        }
                      }}
                      availableFields={availableFields}
                      showOptimization={true}
                    />
                  </div>
                )}

                {showConditionModal === 'exit' && (
                  <div>
                    <p className="text-sm text-gray-500 dark:text-gray-400 mb-4">
                      Define conditions for closing positions. Each rule can trigger a close or adjust TP/SL.
                    </p>
                    <ExitConditionsBuilder
                      value={exitConditions}
                      onChange={setExitConditions}
                      availableFields={availableFields}
                      showOptimization={true}
                    />
                  </div>
                )}
              </div>

              <div className="flex justify-end p-4 border-t border-gray-200 dark:border-gray-700">
                <button
                  onClick={() => setShowConditionModal(null)}
                  className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600"
                >
                  Done
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Confirm Dialog */}
      <ConfirmDialog
        isOpen={confirmDialog.isOpen}
        onClose={() => setConfirmDialog(prev => ({ ...prev, isOpen: false }))}
        onConfirm={confirmDialog.onConfirm}
        title={confirmDialog.title}
        message={confirmDialog.message}
        variant={confirmDialog.variant}
        confirmText="Delete"
      />
    </div>
  );
};

export default Backtesting;
