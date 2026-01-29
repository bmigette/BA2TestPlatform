import React, { useEffect, useState } from 'react';
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
  FolderOpen,
  X
} from 'lucide-react';
import Tooltip from '../components/Tooltip';
import {
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  Legend,
  ResponsiveContainer,
  AreaChart,
  Area,
  ComposedChart,
  Bar,
  ReferenceLine
} from 'recharts';

interface Model {
  id: string;
  name: string;
  modelType: string;
  performanceMetrics: {
    accuracy: number;
    sharpeRatio: number | null;
  };
}

interface StrategyConfig {
  entryThreshold: number;
  exitThreshold: number;
  stopLossPercent: number;
  takeProfitPercent: number;
  trailingStop: boolean;
  trailingStopPercent: number;
  positionSizing: string;
  positionSize: number;
  maxPositions: number;
  commission: number;
  slippage: number;
}

interface AdvancedOptions {
  useMarginTrading: boolean;
  leverage: number;
  requireConfirmation: boolean;
  confirmationBars: number;
  cooldownBars: number;
  allowShorts: boolean;
  hedging: boolean;
}

interface Trade {
  id: string;
  entryDate: string;
  exitDate: string;
  entryPrice: number;
  exitPrice: number;
  size: number;
  direction: string;
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
  id: string;
  name: string;
  modelId: string;
  modelName: string;
  startDate: string;
  endDate: string;
  status: string;
  strategyConfig: StrategyConfig;
  advancedOptions: AdvancedOptions | null;
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
  createdAt: string;
  completedAt: string | null;
}

interface SavedStrategy {
  id: string;
  name: string;
  description: string | null;
  strategyConfig: StrategyConfig;
  advancedOptions: AdvancedOptions | null;
  createdAt: string;
  updatedAt: string;
}

const API_BASE = 'http://localhost:8000/api';

const defaultStrategyConfig: StrategyConfig = {
  entryThreshold: 0.6,
  exitThreshold: 0.4,
  stopLossPercent: 5.0,
  takeProfitPercent: 10.0,
  trailingStop: false,
  trailingStopPercent: 2.0,
  positionSizing: 'fixed',
  positionSize: 1000,
  maxPositions: 1,
  commission: 0.1,
  slippage: 0.05
};

const defaultAdvancedOptions: AdvancedOptions = {
  useMarginTrading: false,
  leverage: 1.0,
  requireConfirmation: false,
  confirmationBars: 2,
  cooldownBars: 0,
  allowShorts: false,
  hedging: false
};

const Backtesting: React.FC = () => {
  // useNavigate available if needed for routing
  const _navigate = useNavigate();
  void _navigate; // Suppress unused warning

  // State
  const [models, setModels] = useState<Model[]>([]);
  const [backtests, setBacktests] = useState<Backtest[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [backtestName, setBacktestName] = useState('');
  const [startDate, setStartDate] = useState('2025-01-01');
  const [endDate, setEndDate] = useState('2025-12-31');
  const [strategyConfig, setStrategyConfig] = useState<StrategyConfig>(defaultStrategyConfig);
  const [advancedOptions, setAdvancedOptions] = useState<AdvancedOptions>(defaultAdvancedOptions);
  const [showAdvanced, setShowAdvanced] = useState(false);

  const [loading, setLoading] = useState(true);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Results view
  const [selectedBacktest, setSelectedBacktest] = useState<Backtest | null>(null);
  const [activeTab, setActiveTab] = useState<'equity' | 'drawdown' | 'price' | 'trades'>('equity');
  const [tradeFilter, setTradeFilter] = useState<'all' | 'profit' | 'loss'>('all');
  const [tradeSortField, setTradeSortField] = useState<'pnl' | 'date' | 'duration'>('date');
  const [tradeSortAsc, setTradeSortAsc] = useState(false);

  // Saved strategies
  const [savedStrategies, setSavedStrategies] = useState<SavedStrategy[]>([]);
  const [showSaveDialog, setShowSaveDialog] = useState(false);
  const [saveStrategyName, setSaveStrategyName] = useState('');
  const [saveStrategyDescription, setSaveStrategyDescription] = useState('');
  const [savingStrategy, setSavingStrategy] = useState(false);
  const [showLoadDropdown, setShowLoadDropdown] = useState(false);

  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    try {
      setLoading(true);

      // Fetch models
      const modelsRes = await fetch(`${API_BASE}/models`);
      if (modelsRes.ok) {
        const data = await modelsRes.json();
        setModels(data.models || []);
      }

      // Fetch backtests
      const backtestsRes = await fetch(`${API_BASE}/backtests`);
      if (backtestsRes.ok) {
        const data = await backtestsRes.json();
        setBacktests(data.backtests || []);
      }

      // Fetch saved strategies
      const strategiesRes = await fetch(`${API_BASE}/backtests/strategies/saved`);
      if (strategiesRes.ok) {
        const data = await strategiesRes.json();
        setSavedStrategies(data.strategies || []);
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

    if (!backtestName.trim()) {
      setError('Please enter a backtest name');
      return;
    }

    try {
      setRunning(true);
      setError(null);

      const res = await fetch(`${API_BASE}/backtests`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: backtestName,
          modelId: selectedModel,
          startDate,
          endDate,
          strategyConfig,
          advancedOptions: showAdvanced ? advancedOptions : null
        })
      });

      if (!res.ok) {
        throw new Error('Failed to run backtest');
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

  const viewBacktest = async (id: string) => {
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

  const deleteBacktest = async (id: string) => {
    if (!confirm('Delete this backtest?')) return;

    try {
      await fetch(`${API_BASE}/backtests/${id}`, { method: 'DELETE' });
      setBacktests(prev => prev.filter(b => b.id !== id));
      if (selectedBacktest?.id === id) {
        setSelectedBacktest(null);
      }
    } catch (err) {
      setError('Failed to delete backtest');
    }
  };

  const exportBacktest = async (id: string) => {
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

      const res = await fetch(`${API_BASE}/backtests/strategies/save`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: saveStrategyName,
          description: saveStrategyDescription || null,
          strategyConfig,
          advancedOptions: showAdvanced ? advancedOptions : null
        })
      });

      if (!res.ok) {
        throw new Error('Failed to save strategy');
      }

      const saved = await res.json();
      setSavedStrategies(prev => [saved, ...prev]);
      setShowSaveDialog(false);
      setSaveStrategyName('');
      setSaveStrategyDescription('');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save strategy');
    } finally {
      setSavingStrategy(false);
    }
  };

  const loadStrategy = (strategy: SavedStrategy) => {
    setStrategyConfig(strategy.strategyConfig);
    if (strategy.advancedOptions) {
      setAdvancedOptions(strategy.advancedOptions);
      setShowAdvanced(true);
    } else {
      setAdvancedOptions(defaultAdvancedOptions);
      setShowAdvanced(false);
    }
    setShowLoadDropdown(false);
  };

  const deleteStrategy = async (strategyId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (!confirm('Delete this saved strategy?')) return;

    try {
      await fetch(`${API_BASE}/backtests/strategies/${strategyId}`, { method: 'DELETE' });
      setSavedStrategies(prev => prev.filter(s => s.id !== strategyId));
    } catch (err) {
      setError('Failed to delete strategy');
    }
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

              {/* Strategy Parameters */}
              <div className="border-t border-gray-200 dark:border-gray-700 pt-4">
                <h3 className="text-sm font-semibold mb-3 flex items-center gap-2 text-gray-900 dark:text-gray-100">
                  <Target className="w-4 h-4 text-blue-500" />
                  Strategy Parameters
                </h3>

                <div className="space-y-3">
                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <label className="block text-xs text-gray-500 dark:text-gray-400 mb-1">Entry Threshold</label>
                      <input
                        type="number"
                        min="0"
                        max="1"
                        step="0.05"
                        value={strategyConfig.entryThreshold}
                        onChange={e => setStrategyConfig({ ...strategyConfig, entryThreshold: parseFloat(e.target.value) })}
                        className="w-full px-2 py-1.5 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                      />
                    </div>
                    <div>
                      <label className="block text-xs text-gray-500 dark:text-gray-400 mb-1">Exit Threshold</label>
                      <input
                        type="number"
                        min="0"
                        max="1"
                        step="0.05"
                        value={strategyConfig.exitThreshold}
                        onChange={e => setStrategyConfig({ ...strategyConfig, exitThreshold: parseFloat(e.target.value) })}
                        className="w-full px-2 py-1.5 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                      />
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <label className="block text-xs text-gray-500 dark:text-gray-400 mb-1">Stop Loss %</label>
                      <input
                        type="number"
                        min="0"
                        step="0.5"
                        value={strategyConfig.stopLossPercent}
                        onChange={e => setStrategyConfig({ ...strategyConfig, stopLossPercent: parseFloat(e.target.value) })}
                        className="w-full px-2 py-1.5 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                      />
                    </div>
                    <div>
                      <label className="block text-xs text-gray-500 dark:text-gray-400 mb-1">Take Profit %</label>
                      <input
                        type="number"
                        min="0"
                        step="0.5"
                        value={strategyConfig.takeProfitPercent}
                        onChange={e => setStrategyConfig({ ...strategyConfig, takeProfitPercent: parseFloat(e.target.value) })}
                        className="w-full px-2 py-1.5 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                      />
                    </div>
                  </div>

                  <div className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      id="trailingStop"
                      checked={strategyConfig.trailingStop}
                      onChange={e => setStrategyConfig({ ...strategyConfig, trailingStop: e.target.checked })}
                      className="rounded"
                    />
                    <label htmlFor="trailingStop" className="text-sm text-gray-600 dark:text-gray-400">
                      Trailing Stop
                    </label>
                    {strategyConfig.trailingStop && (
                      <input
                        type="number"
                        min="0"
                        step="0.5"
                        value={strategyConfig.trailingStopPercent}
                        onChange={e => setStrategyConfig({ ...strategyConfig, trailingStopPercent: parseFloat(e.target.value) })}
                        className="w-20 px-2 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                      />
                    )}
                  </div>

                  <div>
                    <label className="block text-xs text-gray-500 dark:text-gray-400 mb-1">Position Sizing</label>
                    <select
                      value={strategyConfig.positionSizing}
                      onChange={e => setStrategyConfig({ ...strategyConfig, positionSizing: e.target.value })}
                      className="w-full px-2 py-1.5 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                    >
                      <option value="fixed">Fixed Amount</option>
                      <option value="percent">Percent of Capital</option>
                      <option value="kelly">Kelly Criterion</option>
                    </select>
                  </div>

                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <label className="block text-xs text-gray-500 dark:text-gray-400 mb-1">
                        {strategyConfig.positionSizing === 'percent' ? 'Position %' : 'Position Size'}
                      </label>
                      <input
                        type="number"
                        min="0"
                        step={strategyConfig.positionSizing === 'percent' ? '1' : '100'}
                        value={strategyConfig.positionSize}
                        onChange={e => setStrategyConfig({ ...strategyConfig, positionSize: parseFloat(e.target.value) })}
                        className="w-full px-2 py-1.5 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                      />
                    </div>
                    <div>
                      <label className="block text-xs text-gray-500 dark:text-gray-400 mb-1">Max Positions</label>
                      <input
                        type="number"
                        min="1"
                        max="10"
                        value={strategyConfig.maxPositions}
                        onChange={e => setStrategyConfig({ ...strategyConfig, maxPositions: parseInt(e.target.value) })}
                        className="w-full px-2 py-1.5 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                      />
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <label className="block text-xs text-gray-500 dark:text-gray-400 mb-1">Commission %</label>
                      <input
                        type="number"
                        min="0"
                        step="0.01"
                        value={strategyConfig.commission}
                        onChange={e => setStrategyConfig({ ...strategyConfig, commission: parseFloat(e.target.value) })}
                        className="w-full px-2 py-1.5 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                      />
                    </div>
                    <div>
                      <label className="block text-xs text-gray-500 dark:text-gray-400 mb-1">Slippage %</label>
                      <input
                        type="number"
                        min="0"
                        step="0.01"
                        value={strategyConfig.slippage}
                        onChange={e => setStrategyConfig({ ...strategyConfig, slippage: parseFloat(e.target.value) })}
                        className="w-full px-2 py-1.5 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                      />
                    </div>
                  </div>
                </div>
              </div>

              {/* Save/Load Strategy Buttons */}
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

                <div className="relative">
                  <Tooltip content="Load a previously saved strategy configuration">
                    <button
                      onClick={() => setShowLoadDropdown(!showLoadDropdown)}
                      className="flex items-center gap-1.5 px-3 py-1.5 text-sm text-gray-700 dark:text-gray-300 border border-gray-300 dark:border-gray-600 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
                    >
                      <FolderOpen className="w-4 h-4" />
                      Load Strategy
                      <ChevronDown className="w-3 h-3" />
                    </button>
                  </Tooltip>

                  {showLoadDropdown && (
                    <div className="absolute z-10 left-0 mt-1 w-64 bg-white dark:bg-gray-800 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700 py-1 max-h-60 overflow-y-auto">
                      {savedStrategies.length === 0 ? (
                        <p className="text-sm text-gray-500 dark:text-gray-400 dark:text-gray-400 text-center py-3">No saved strategies</p>
                      ) : (
                        savedStrategies.map(strat => (
                          <div
                            key={strat.id}
                            onClick={() => loadStrategy(strat)}
                            className="px-3 py-2 hover:bg-gray-100 dark:hover:bg-gray-700 cursor-pointer flex items-center justify-between"
                          >
                            <div>
                              <p className="text-sm font-medium truncate">{strat.name}</p>
                              {strat.description && (
                                <p className="text-xs text-gray-500 dark:text-gray-400 truncate">{strat.description}</p>
                              )}
                            </div>
                            <button
                              onClick={(e) => deleteStrategy(strat.id, e)}
                              className="p-1 hover:bg-red-100 dark:hover:bg-red-900/30 rounded text-red-500"
                            >
                              <Trash2 className="w-3.5 h-3.5" />
                            </button>
                          </div>
                        ))
                      )}
                    </div>
                  )}
                </div>
              </div>

              {/* Advanced Options Toggle */}
              <button
                onClick={() => setShowAdvanced(!showAdvanced)}
                className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 hover:text-gray-800 dark:hover:text-gray-200"
              >
                <Settings className="w-4 h-4" />
                Advanced Options
                {showAdvanced ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
              </button>

              {/* Advanced Options */}
              {showAdvanced && (
                <div className="border border-gray-200 dark:border-gray-700 rounded-lg p-3 space-y-3">
                  <div className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      id="marginTrading"
                      checked={advancedOptions.useMarginTrading}
                      onChange={e => setAdvancedOptions({ ...advancedOptions, useMarginTrading: e.target.checked })}
                      className="rounded"
                    />
                    <label htmlFor="marginTrading" className="text-sm text-gray-600 dark:text-gray-400">
                      Margin Trading
                    </label>
                    {advancedOptions.useMarginTrading && (
                      <input
                        type="number"
                        min="1"
                        max="10"
                        step="0.5"
                        value={advancedOptions.leverage}
                        onChange={e => setAdvancedOptions({ ...advancedOptions, leverage: parseFloat(e.target.value) })}
                        className="w-20 px-2 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                        placeholder="Leverage"
                      />
                    )}
                  </div>

                  <div className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      id="requireConfirmation"
                      checked={advancedOptions.requireConfirmation}
                      onChange={e => setAdvancedOptions({ ...advancedOptions, requireConfirmation: e.target.checked })}
                      className="rounded"
                    />
                    <label htmlFor="requireConfirmation" className="text-sm text-gray-600 dark:text-gray-400">
                      Require Signal Confirmation
                    </label>
                    {advancedOptions.requireConfirmation && (
                      <input
                        type="number"
                        min="1"
                        max="10"
                        value={advancedOptions.confirmationBars}
                        onChange={e => setAdvancedOptions({ ...advancedOptions, confirmationBars: parseInt(e.target.value) })}
                        className="w-16 px-2 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                        placeholder="Bars"
                      />
                    )}
                  </div>

                  <div>
                    <label className="block text-xs text-gray-500 dark:text-gray-400 mb-1">Cooldown Bars (after exit)</label>
                    <input
                      type="number"
                      min="0"
                      max="20"
                      value={advancedOptions.cooldownBars}
                      onChange={e => setAdvancedOptions({ ...advancedOptions, cooldownBars: parseInt(e.target.value) })}
                      className="w-full px-2 py-1.5 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                    />
                  </div>

                  <div className="flex items-center gap-4">
                    <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
                      <input
                        type="checkbox"
                        checked={advancedOptions.allowShorts}
                        onChange={e => setAdvancedOptions({ ...advancedOptions, allowShorts: e.target.checked })}
                        className="rounded"
                      />
                      Allow Shorts
                    </label>
                    <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
                      <input
                        type="checkbox"
                        checked={advancedOptions.hedging}
                        onChange={e => setAdvancedOptions({ ...advancedOptions, hedging: e.target.checked })}
                        className="rounded"
                      />
                      Hedging
                    </label>
                  </div>
                </div>
              )}

              {/* Run Button */}
              <button
                onClick={runBacktest}
                disabled={running || !selectedModel}
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
                <p className="text-sm text-gray-500 dark:text-gray-400 dark:text-gray-400 text-center py-4">No backtests yet</p>
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
                      <span className="font-medium text-sm truncate">{bt.name}</span>
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
                    <p className="text-xs text-gray-500 dark:text-gray-400 mb-2">{bt.modelName}</p>
                    <div className="flex items-center gap-3 text-xs">
                      <span className={`font-medium ${(bt.totalReturn || 0) >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                        {(bt.totalReturn || 0) >= 0 ? '+' : ''}{bt.totalReturn?.toFixed(1)}%
                      </span>
                      <span className="text-gray-500">Sharpe: {bt.sharpeRatio?.toFixed(2)}</span>
                      <span className="text-gray-500">{bt.totalTrades} trades</span>
                    </div>
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
                      { id: 'price', label: 'Price Chart', icon: BarChart3 },
                      { id: 'trades', label: 'Trade List', icon: Activity }
                    ].map(tab => (
                      <button
                        key={tab.id}
                        onClick={() => setActiveTab(tab.id as any)}
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
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="date" tickFormatter={d => d.slice(5)} />
                          <YAxis domain={['dataMin - 500', 'dataMax + 500']} tickFormatter={v => `$${v.toLocaleString()}`} />
                          <RechartsTooltip
                            formatter={(value) => [`$${(value as number)?.toLocaleString() ?? '0'}`, 'Equity']}
                            labelFormatter={label => `Date: ${label}`}
                          />
                          <Area type="monotone" dataKey="equity" stroke="#22c55e" fill="#22c55e" fillOpacity={0.2} />
                          <ReferenceLine y={10000} stroke="#888" strokeDasharray="3 3" label="Initial" />
                        </AreaChart>
                      </ResponsiveContainer>
                    </div>
                  )}

                  {activeTab === 'drawdown' && selectedBacktest.results?.drawdownCurve && (
                    <div className="h-80">
                      <ResponsiveContainer width="100%" height="100%">
                        <AreaChart data={selectedBacktest.results.drawdownCurve}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="date" tickFormatter={d => d.slice(5)} />
                          <YAxis domain={[0, 'dataMax + 2']} tickFormatter={v => `${v}%`} reversed />
                          <RechartsTooltip
                            formatter={(value) => [`${((value as number) ?? 0).toFixed(2)}%`, 'Drawdown']}
                            labelFormatter={label => `Date: ${label}`}
                          />
                          <Area type="monotone" dataKey="drawdown" stroke="#ef4444" fill="#ef4444" fillOpacity={0.3} />
                        </AreaChart>
                      </ResponsiveContainer>
                    </div>
                  )}

                  {activeTab === 'price' && selectedBacktest.results?.priceData && (
                    <div className="h-80">
                      <ResponsiveContainer width="100%" height="100%">
                        <ComposedChart data={selectedBacktest.results.priceData.filter((_, i) => i % 5 === 0)}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="date" tickFormatter={d => d.slice(5)} />
                          <YAxis yAxisId="price" domain={['dataMin - 5', 'dataMax + 5']} />
                          <YAxis yAxisId="signal" orientation="right" domain={[-1, 1]} />
                          <RechartsTooltip />
                          <Legend />
                          <Line yAxisId="price" type="monotone" dataKey="close" stroke="#3b82f6" name="Price" dot={false} />
                          <Bar yAxisId="signal" dataKey="signal" name="Signal" fill="#8884d8" opacity={0.3} />
                          <ReferenceLine yAxisId="signal" y={0.6} stroke="#22c55e" strokeDasharray="3 3" label="Buy" />
                          <ReferenceLine yAxisId="signal" y={-0.6} stroke="#ef4444" strokeDasharray="3 3" label="Sell" />
                        </ComposedChart>
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
                            onChange={e => setTradeFilter(e.target.value as any)}
                            className="text-sm border border-gray-300 dark:border-gray-600 rounded px-2 py-1 bg-white dark:bg-gray-700"
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
                            onChange={e => setTradeSortField(e.target.value as any)}
                            className="text-sm border border-gray-300 dark:border-gray-600 rounded px-2 py-1 bg-white dark:bg-gray-700"
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
                              <th className="px-3 py-2 text-left">Entry</th>
                              <th className="px-3 py-2 text-left">Exit</th>
                              <th className="px-3 py-2 text-right">Entry $</th>
                              <th className="px-3 py-2 text-right">Exit $</th>
                              <th className="px-3 py-2 text-center">Dir</th>
                              <th className="px-3 py-2 text-right">Size</th>
                              <th className="px-3 py-2 text-right">P&L</th>
                              <th className="px-3 py-2 text-right">P&L %</th>
                              <th className="px-3 py-2 text-center">Duration</th>
                              <th className="px-3 py-2 text-center">Reason</th>
                            </tr>
                          </thead>
                          <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                            {getFilteredTrades().map(trade => (
                              <tr key={trade.id} className="hover:bg-gray-50 dark:hover:bg-gray-700/30">
                                <td className="px-3 py-2">{trade.entryDate}</td>
                                <td className="px-3 py-2">{trade.exitDate}</td>
                                <td className="px-3 py-2 text-right">${trade.entryPrice.toFixed(2)}</td>
                                <td className="px-3 py-2 text-right">${trade.exitPrice.toFixed(2)}</td>
                                <td className="px-3 py-2 text-center">
                                  <span className={`px-2 py-0.5 rounded text-xs font-medium ${
                                    trade.direction === 'long'
                                      ? 'bg-green-100 text-green-700 dark:bg-green-900 dark:text-green-300'
                                      : 'bg-red-100 text-red-700 dark:bg-red-900 dark:text-red-300'
                                  }`}>
                                    {trade.direction}
                                  </span>
                                </td>
                                <td className="px-3 py-2 text-right">{trade.size}</td>
                                <td className={`px-3 py-2 text-right font-medium ${trade.pnl >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                                  {trade.pnl >= 0 ? '+' : ''}${trade.pnl.toFixed(2)}
                                </td>
                                <td className={`px-3 py-2 text-right font-medium ${trade.pnl >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                                  {trade.pnl >= 0 ? '+' : ''}{trade.pnlPercent.toFixed(2)}%
                                </td>
                                <td className="px-3 py-2 text-center">{trade.duration} bars</td>
                                <td className="px-3 py-2 text-center">
                                  <span className="px-2 py-0.5 rounded bg-gray-100 dark:bg-gray-700 text-xs">
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
                    <span>Entry: {strategyConfig.entryThreshold}</span>
                    <span>Exit: {strategyConfig.exitThreshold}</span>
                    <span>Stop Loss: {strategyConfig.stopLossPercent}%</span>
                    <span>Take Profit: {strategyConfig.takeProfitPercent}%</span>
                    <span>Position: {strategyConfig.positionSizing}</span>
                    <span>Trailing: {strategyConfig.trailingStop ? 'Yes' : 'No'}</span>
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
    </div>
  );
};

export default Backtesting;
