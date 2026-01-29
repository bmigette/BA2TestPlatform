import React, { useState, useCallback } from 'react';
import {
  Target,
  TrendingUp,
  TrendingDown,
  BarChart3,
  Activity,
  Plus,
  X,
  Play,
  Save,
  FolderOpen,
  Eye,
  EyeOff,
  AlertCircle,
} from 'lucide-react';
import type {
  TargetConfig,
  PriceBasedTarget,
  DirectionalTarget,
  TripleBarrierTarget,
  TrendReversalTarget,
  VolatilityTarget,
  CalculatedTarget,
  IndicatorParams,
} from '../types/targets';
import { TARGET_COLORS } from '../types/targets';

type TabType = 'price_based' | 'directional' | 'triple_barrier' | 'trend_reversal' | 'volatility';

interface PredictionTargetsPanelProps {
  datasetId: number;
  onTargetsCalculated: (targets: CalculatedTarget[]) => void;
  onSaveSet: (targets: TargetConfig[]) => void;
  onLoadSet: () => void;
}

const PredictionTargetsPanel: React.FC<PredictionTargetsPanelProps> = ({
  datasetId,
  onTargetsCalculated,
  onSaveSet,
  onLoadSet,
}) => {
  const [activeTab, setActiveTab] = useState<TabType>('price_based');
  const [targets, setTargets] = useState<CalculatedTarget[]>([]);
  const [isCalculating, setIsCalculating] = useState(false);
  const [showRemoveConfirm, setShowRemoveConfirm] = useState<number | null>(null);

  // Form states for each tab
  const [priceBasedForm, setPriceBasedForm] = useState({
    direction: 'up' as 'up' | 'down',
    profitPct: 10,
    maxDrawdownPct: 5,
    timeBars: 14,
  });

  const [directionalForm, setDirectionalForm] = useState({
    direction: 'up' as 'up' | 'down',
    horizon: 5,
  });

  const [tripleBarrierForm, setTripleBarrierForm] = useState({
    profitPct: 3,
    stopPct: 2,
    maxBars: 10,
  });

  const [trendReversalForm, setTrendReversalForm] = useState({
    indicator: 'rsi' as 'rsi' | 'macd' | 'sar' | 'zigzag',
    direction: 'bullish' as 'bullish' | 'bearish',
    threshold: 30,
    // RSI params
    rsiPeriod: 14,
    // MACD params
    macdFast: 12,
    macdSlow: 26,
    macdSignal: 9,
    // SAR params
    sarAfStart: 0.02,
    sarAfMax: 0.2,
    // ZigZag params
    zigzagDeviation: 5.0,
  });

  const [volatilityForm, setVolatilityForm] = useState({
    horizon: 5,
    method: 'std' as 'std' | 'range' | 'atr',
  });

  // Color assignment for targets
  const getNextColor = useCallback((type: string, direction?: string): string => {
    const colorMap: Record<string, string> = {
      'price_based_up': TARGET_COLORS.priceBased.up,
      'price_based_down': TARGET_COLORS.priceBased.down,
      'directional_up': TARGET_COLORS.directional.up,
      'directional_down': TARGET_COLORS.directional.down,
      'triple_barrier': TARGET_COLORS.tripleBarrier.profit,
      'trend_reversal_bullish': TARGET_COLORS.trendReversal.bullish,
      'trend_reversal_bearish': TARGET_COLORS.trendReversal.bearish,
      'volatility': TARGET_COLORS.volatility,
    };
    const key = direction ? `${type}_${direction}` : type;
    return colorMap[key] || '#6B7280';
  }, []);

  // Add target from form
  const addTarget = useCallback(() => {
    let config: TargetConfig;
    let color: string;

    switch (activeTab) {
      case 'price_based':
        config = {
          type: 'price_based',
          category: 'binary_classification',
          direction: priceBasedForm.direction,
          profitPct: priceBasedForm.profitPct,
          maxDrawdownPct: priceBasedForm.maxDrawdownPct,
          timeBars: priceBasedForm.timeBars,
        } as PriceBasedTarget;
        color = getNextColor('price_based', priceBasedForm.direction);
        break;

      case 'directional':
        config = {
          type: 'directional',
          category: 'binary_classification',
          direction: directionalForm.direction,
          horizon: directionalForm.horizon,
        } as DirectionalTarget;
        color = getNextColor('directional', directionalForm.direction);
        break;

      case 'triple_barrier':
        config = {
          type: 'triple_barrier',
          category: 'multiclass_classification',
          profitPct: tripleBarrierForm.profitPct,
          stopPct: tripleBarrierForm.stopPct,
          maxBars: tripleBarrierForm.maxBars,
        } as TripleBarrierTarget;
        color = getNextColor('triple_barrier');
        break;

      case 'trend_reversal':
        let indicatorParams: IndicatorParams;
        if (trendReversalForm.indicator === 'rsi') {
          indicatorParams = { period: trendReversalForm.rsiPeriod };
        } else if (trendReversalForm.indicator === 'macd') {
          indicatorParams = {
            fast: trendReversalForm.macdFast,
            slow: trendReversalForm.macdSlow,
            signal: trendReversalForm.macdSignal,
          };
        } else if (trendReversalForm.indicator === 'sar') {
          indicatorParams = {
            afStart: trendReversalForm.sarAfStart,
            afMax: trendReversalForm.sarAfMax,
          };
        } else {
          indicatorParams = { deviationPct: trendReversalForm.zigzagDeviation };
        }
        config = {
          type: 'trend_reversal',
          category: 'binary_classification',
          indicator: trendReversalForm.indicator,
          indicatorParams,
          threshold: trendReversalForm.threshold,
          direction: trendReversalForm.direction,
        } as TrendReversalTarget;
        color = getNextColor('trend_reversal', trendReversalForm.direction);
        break;

      case 'volatility':
        config = {
          type: 'volatility',
          category: 'regression',
          horizon: volatilityForm.horizon,
          method: volatilityForm.method,
        } as VolatilityTarget;
        color = getNextColor('volatility');
        break;

      default:
        return;
    }

    // Add to targets list (will be calculated on preview)
    const newTarget: CalculatedTarget = {
      config,
      columnName: '',
      data: [],
      stats: { totalRows: 0, validRows: 0 },
      color,
      visible: true,
    };

    setTargets([...targets, newTarget]);
  }, [activeTab, priceBasedForm, directionalForm, tripleBarrierForm, trendReversalForm, volatilityForm, targets, getNextColor]);

  // Remove target with confirmation
  const removeTarget = useCallback((index: number) => {
    setTargets(targets.filter((_, i) => i !== index));
    setShowRemoveConfirm(null);
  }, [targets]);

  // Toggle target visibility
  const toggleVisibility = useCallback((index: number) => {
    setTargets(targets.map((t, i) =>
      i === index ? { ...t, visible: !t.visible } : t
    ));
  }, [targets]);

  // Calculate all targets
  const calculateTargets = useCallback(async () => {
    if (targets.length === 0) return;

    setIsCalculating(true);
    try {
      const response = await fetch(`http://localhost:8002/api/datasets/${datasetId}/calculate-targets`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          targets: targets.map(t => t.config),
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to calculate targets');
      }

      const data = await response.json();

      // Update targets with calculated data
      const updatedTargets = targets.map((target, index) => {
        const result = data.targets[index];
        if (result) {
          return {
            ...target,
            columnName: result.columnName,
            data: result.data,
            stats: result.stats,
          };
        }
        return target;
      });

      setTargets(updatedTargets);
      onTargetsCalculated(updatedTargets);
    } catch (error) {
      console.error('Error calculating targets:', error);
    } finally {
      setIsCalculating(false);
    }
  }, [datasetId, targets, onTargetsCalculated]);

  // Get label for a target
  const getTargetLabel = (config: TargetConfig): string => {
    switch (config.type) {
      case 'price_based': {
        const c = config as PriceBasedTarget;
        return `Price ${c.direction} ${c.profitPct}%/${c.maxDrawdownPct}%DD/${c.timeBars}b`;
      }
      case 'directional': {
        const c = config as DirectionalTarget;
        return `Directional ${c.direction} ${c.horizon}b`;
      }
      case 'triple_barrier': {
        const c = config as TripleBarrierTarget;
        return `Barrier ${c.profitPct}%P/${c.stopPct}%S/${c.maxBars}b`;
      }
      case 'trend_reversal': {
        const c = config as TrendReversalTarget;
        return `${c.indicator.toUpperCase()} ${c.direction}`;
      }
      case 'volatility': {
        const c = config as VolatilityTarget;
        return `Volatility ${c.method} ${c.horizon}b`;
      }
      default:
        return 'Unknown';
    }
  };

  // Get stats display for a target
  const getStatsDisplay = (target: CalculatedTarget): string => {
    const { stats, config } = target;
    if (stats.validRows === 0) return 'Not calculated';

    if (config.category === 'binary_classification') {
      return `${stats.positiveCount || 0} hits (${stats.positivePct || 0}%)`;
    } else if (config.category === 'multiclass_classification') {
      return `P:${stats.profitHitCount || 0} S:${stats.stopHitCount || 0} T:${stats.timeoutCount || 0}`;
    } else if (config.category === 'regression') {
      return `μ=${stats.mean?.toFixed(2) || 0} σ=${stats.std?.toFixed(2) || 0}`;
    }
    return '';
  };

  const tabs: { id: TabType; label: string; icon: React.ReactNode }[] = [
    { id: 'price_based', label: 'Price-Based', icon: <Target size={16} /> },
    { id: 'directional', label: 'Directional', icon: <TrendingUp size={16} /> },
    { id: 'triple_barrier', label: 'Triple-Barrier', icon: <BarChart3 size={16} /> },
    { id: 'trend_reversal', label: 'Trend Reversal', icon: <TrendingDown size={16} /> },
    { id: 'volatility', label: 'Volatility', icon: <Activity size={16} /> },
  ];

  return (
    <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-bold flex items-center gap-2">
          <Target size={20} className="text-purple-500" />
          Prediction Targets
        </h2>
        <div className="flex gap-2">
          <button
            onClick={() => onSaveSet(targets.map(t => t.config))}
            disabled={targets.length === 0}
            className="px-3 py-1.5 text-sm bg-gray-100 dark:bg-gray-700 rounded hover:bg-gray-200 dark:hover:bg-gray-600 disabled:opacity-50 flex items-center gap-1"
          >
            <Save size={14} />
            Save Set
          </button>
          <button
            onClick={onLoadSet}
            className="px-3 py-1.5 text-sm bg-gray-100 dark:bg-gray-700 rounded hover:bg-gray-200 dark:hover:bg-gray-600 flex items-center gap-1"
          >
            <FolderOpen size={14} />
            Load Set
          </button>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex border-b border-gray-200 dark:border-gray-700 mb-4">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`px-4 py-2 text-sm font-medium flex items-center gap-1.5 border-b-2 transition-colors ${
              activeTab === tab.id
                ? 'border-purple-500 text-purple-600 dark:text-purple-400'
                : 'border-transparent text-gray-500 hover:text-gray-700 dark:hover:text-gray-300'
            }`}
          >
            {tab.icon}
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      <div className="p-4 bg-gray-50 dark:bg-gray-700 rounded-lg mb-4">
        {activeTab === 'price_based' && (
          <div className="flex flex-wrap items-end gap-4">
            <div>
              <label className="block text-xs font-medium text-gray-600 dark:text-gray-400 mb-1">Direction</label>
              <select
                value={priceBasedForm.direction}
                onChange={(e) => setPriceBasedForm({ ...priceBasedForm, direction: e.target.value as 'up' | 'down' })}
                className="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md text-sm bg-white dark:bg-gray-800"
              >
                <option value="up">Up (Buy)</option>
                <option value="down">Down (Sell)</option>
              </select>
            </div>
            <div>
              <label className="block text-xs font-medium text-gray-600 dark:text-gray-400 mb-1">Profit Target (%)</label>
              <input
                type="number"
                min="0.1"
                step="0.1"
                value={priceBasedForm.profitPct}
                onChange={(e) => setPriceBasedForm({ ...priceBasedForm, profitPct: parseFloat(e.target.value) || 0 })}
                className="w-24 px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md text-sm bg-white dark:bg-gray-800"
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-gray-600 dark:text-gray-400 mb-1">Max Drawdown (%)</label>
              <input
                type="number"
                min="0.1"
                step="0.1"
                value={priceBasedForm.maxDrawdownPct}
                onChange={(e) => setPriceBasedForm({ ...priceBasedForm, maxDrawdownPct: parseFloat(e.target.value) || 0 })}
                className="w-24 px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md text-sm bg-white dark:bg-gray-800"
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-gray-600 dark:text-gray-400 mb-1">Time Window (bars)</label>
              <input
                type="number"
                min="1"
                value={priceBasedForm.timeBars}
                onChange={(e) => setPriceBasedForm({ ...priceBasedForm, timeBars: parseInt(e.target.value) || 1 })}
                className="w-24 px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md text-sm bg-white dark:bg-gray-800"
              />
            </div>
          </div>
        )}

        {activeTab === 'directional' && (
          <div className="flex flex-wrap items-end gap-4">
            <div>
              <label className="block text-xs font-medium text-gray-600 dark:text-gray-400 mb-1">Direction</label>
              <select
                value={directionalForm.direction}
                onChange={(e) => setDirectionalForm({ ...directionalForm, direction: e.target.value as 'up' | 'down' })}
                className="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md text-sm bg-white dark:bg-gray-800"
              >
                <option value="up">Up (Higher)</option>
                <option value="down">Down (Lower)</option>
              </select>
            </div>
            <div>
              <label className="block text-xs font-medium text-gray-600 dark:text-gray-400 mb-1">Horizon (bars)</label>
              <input
                type="number"
                min="1"
                value={directionalForm.horizon}
                onChange={(e) => setDirectionalForm({ ...directionalForm, horizon: parseInt(e.target.value) || 1 })}
                className="w-24 px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md text-sm bg-white dark:bg-gray-800"
              />
            </div>
            <p className="text-xs text-gray-500 dark:text-gray-400">
              Predicts if price will be {directionalForm.direction === 'up' ? 'higher' : 'lower'} in {directionalForm.horizon} bars
            </p>
          </div>
        )}

        {activeTab === 'triple_barrier' && (
          <div className="flex flex-wrap items-end gap-4">
            <div>
              <label className="block text-xs font-medium text-gray-600 dark:text-gray-400 mb-1">Profit Target (%)</label>
              <input
                type="number"
                min="0.1"
                step="0.1"
                value={tripleBarrierForm.profitPct}
                onChange={(e) => setTripleBarrierForm({ ...tripleBarrierForm, profitPct: parseFloat(e.target.value) || 0 })}
                className="w-24 px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md text-sm bg-white dark:bg-gray-800"
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-gray-600 dark:text-gray-400 mb-1">Stop Loss (%)</label>
              <input
                type="number"
                min="0.1"
                step="0.1"
                value={tripleBarrierForm.stopPct}
                onChange={(e) => setTripleBarrierForm({ ...tripleBarrierForm, stopPct: parseFloat(e.target.value) || 0 })}
                className="w-24 px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md text-sm bg-white dark:bg-gray-800"
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-gray-600 dark:text-gray-400 mb-1">Max Bars (timeout)</label>
              <input
                type="number"
                min="1"
                value={tripleBarrierForm.maxBars}
                onChange={(e) => setTripleBarrierForm({ ...tripleBarrierForm, maxBars: parseInt(e.target.value) || 1 })}
                className="w-24 px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md text-sm bg-white dark:bg-gray-800"
              />
            </div>
            <p className="text-xs text-gray-500 dark:text-gray-400">
              3 classes: Profit hit, Stop hit, or Timeout
            </p>
          </div>
        )}

        {activeTab === 'trend_reversal' && (
          <div className="space-y-4">
            <div className="flex flex-wrap items-end gap-4">
              <div>
                <label className="block text-xs font-medium text-gray-600 dark:text-gray-400 mb-1">Indicator</label>
                <select
                  value={trendReversalForm.indicator}
                  onChange={(e) => setTrendReversalForm({ ...trendReversalForm, indicator: e.target.value as 'rsi' | 'macd' | 'sar' | 'zigzag' })}
                  className="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md text-sm bg-white dark:bg-gray-800"
                >
                  <option value="rsi">RSI</option>
                  <option value="macd">MACD</option>
                  <option value="sar">Parabolic SAR</option>
                  <option value="zigzag">ZigZag</option>
                </select>
              </div>
              <div>
                <label className="block text-xs font-medium text-gray-600 dark:text-gray-400 mb-1">Direction</label>
                <select
                  value={trendReversalForm.direction}
                  onChange={(e) => setTrendReversalForm({ ...trendReversalForm, direction: e.target.value as 'bullish' | 'bearish' })}
                  className="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md text-sm bg-white dark:bg-gray-800"
                >
                  <option value="bullish">Bullish (Buy)</option>
                  <option value="bearish">Bearish (Sell)</option>
                </select>
              </div>
              {trendReversalForm.indicator === 'rsi' && (
                <>
                  <div>
                    <label className="block text-xs font-medium text-gray-600 dark:text-gray-400 mb-1">Threshold</label>
                    <input
                      type="number"
                      min="1"
                      max="99"
                      value={trendReversalForm.threshold}
                      onChange={(e) => setTrendReversalForm({ ...trendReversalForm, threshold: parseInt(e.target.value) || 30 })}
                      className="w-20 px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md text-sm bg-white dark:bg-gray-800"
                    />
                  </div>
                  <div>
                    <label className="block text-xs font-medium text-gray-600 dark:text-gray-400 mb-1">Period</label>
                    <input
                      type="number"
                      min="2"
                      value={trendReversalForm.rsiPeriod}
                      onChange={(e) => setTrendReversalForm({ ...trendReversalForm, rsiPeriod: parseInt(e.target.value) || 14 })}
                      className="w-20 px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md text-sm bg-white dark:bg-gray-800"
                    />
                  </div>
                </>
              )}
              {trendReversalForm.indicator === 'macd' && (
                <>
                  <div>
                    <label className="block text-xs font-medium text-gray-600 dark:text-gray-400 mb-1">Fast</label>
                    <input
                      type="number"
                      min="1"
                      value={trendReversalForm.macdFast}
                      onChange={(e) => setTrendReversalForm({ ...trendReversalForm, macdFast: parseInt(e.target.value) || 12 })}
                      className="w-16 px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md text-sm bg-white dark:bg-gray-800"
                    />
                  </div>
                  <div>
                    <label className="block text-xs font-medium text-gray-600 dark:text-gray-400 mb-1">Slow</label>
                    <input
                      type="number"
                      min="1"
                      value={trendReversalForm.macdSlow}
                      onChange={(e) => setTrendReversalForm({ ...trendReversalForm, macdSlow: parseInt(e.target.value) || 26 })}
                      className="w-16 px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md text-sm bg-white dark:bg-gray-800"
                    />
                  </div>
                  <div>
                    <label className="block text-xs font-medium text-gray-600 dark:text-gray-400 mb-1">Signal</label>
                    <input
                      type="number"
                      min="1"
                      value={trendReversalForm.macdSignal}
                      onChange={(e) => setTrendReversalForm({ ...trendReversalForm, macdSignal: parseInt(e.target.value) || 9 })}
                      className="w-16 px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md text-sm bg-white dark:bg-gray-800"
                    />
                  </div>
                </>
              )}
              {trendReversalForm.indicator === 'sar' && (
                <>
                  <div>
                    <label className="block text-xs font-medium text-gray-600 dark:text-gray-400 mb-1">AF Start</label>
                    <input
                      type="number"
                      min="0.01"
                      step="0.01"
                      value={trendReversalForm.sarAfStart}
                      onChange={(e) => setTrendReversalForm({ ...trendReversalForm, sarAfStart: parseFloat(e.target.value) || 0.02 })}
                      className="w-20 px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md text-sm bg-white dark:bg-gray-800"
                    />
                  </div>
                  <div>
                    <label className="block text-xs font-medium text-gray-600 dark:text-gray-400 mb-1">AF Max</label>
                    <input
                      type="number"
                      min="0.01"
                      step="0.01"
                      value={trendReversalForm.sarAfMax}
                      onChange={(e) => setTrendReversalForm({ ...trendReversalForm, sarAfMax: parseFloat(e.target.value) || 0.2 })}
                      className="w-20 px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md text-sm bg-white dark:bg-gray-800"
                    />
                  </div>
                </>
              )}
              {trendReversalForm.indicator === 'zigzag' && (
                <div>
                  <label className="block text-xs font-medium text-gray-600 dark:text-gray-400 mb-1">Deviation (%)</label>
                  <input
                    type="number"
                    min="0.1"
                    step="0.1"
                    value={trendReversalForm.zigzagDeviation}
                    onChange={(e) => setTrendReversalForm({ ...trendReversalForm, zigzagDeviation: parseFloat(e.target.value) || 5 })}
                    className="w-20 px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md text-sm bg-white dark:bg-gray-800"
                  />
                </div>
              )}
            </div>
          </div>
        )}

        {activeTab === 'volatility' && (
          <div className="flex flex-wrap items-end gap-4">
            <div>
              <label className="block text-xs font-medium text-gray-600 dark:text-gray-400 mb-1">Method</label>
              <select
                value={volatilityForm.method}
                onChange={(e) => setVolatilityForm({ ...volatilityForm, method: e.target.value as 'std' | 'range' | 'atr' })}
                className="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md text-sm bg-white dark:bg-gray-800"
              >
                <option value="std">Std Deviation</option>
                <option value="range">High-Low Range</option>
                <option value="atr">ATR</option>
              </select>
            </div>
            <div>
              <label className="block text-xs font-medium text-gray-600 dark:text-gray-400 mb-1">Horizon (bars)</label>
              <input
                type="number"
                min="1"
                value={volatilityForm.horizon}
                onChange={(e) => setVolatilityForm({ ...volatilityForm, horizon: parseInt(e.target.value) || 1 })}
                className="w-24 px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md text-sm bg-white dark:bg-gray-800"
              />
            </div>
            <p className="text-xs text-gray-500 dark:text-gray-400">
              Regression target: predicts volatility over next {volatilityForm.horizon} bars
            </p>
          </div>
        )}

        <button
          onClick={addTarget}
          className="mt-4 px-4 py-2 bg-purple-600 text-white rounded-md hover:bg-purple-700 flex items-center gap-1"
        >
          <Plus size={16} />
          Add Target
        </button>
      </div>

      {/* Active Targets List */}
      {targets.length > 0 && (
        <div className="mb-4">
          <h4 className="text-sm font-medium mb-2">Active Targets:</h4>
          <div className="space-y-2">
            {targets.map((target, index) => (
              <div
                key={index}
                className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-700 rounded-lg"
              >
                <div className="flex items-center gap-3">
                  <button
                    onClick={() => toggleVisibility(index)}
                    className="text-gray-500 hover:text-gray-700 dark:hover:text-gray-300"
                  >
                    {target.visible ? <Eye size={16} /> : <EyeOff size={16} />}
                  </button>
                  <div
                    className="w-4 h-4 rounded-full"
                    style={{ backgroundColor: target.color }}
                  />
                  <span className="text-sm font-medium">{getTargetLabel(target.config)}</span>
                  <span className="text-xs text-gray-500 dark:text-gray-400">
                    {getStatsDisplay(target)}
                  </span>
                </div>
                <div className="relative">
                  {showRemoveConfirm === index ? (
                    <div className="flex items-center gap-2">
                      <span className="text-xs text-gray-500">Remove?</span>
                      <button
                        onClick={() => removeTarget(index)}
                        className="px-2 py-1 text-xs bg-red-500 text-white rounded hover:bg-red-600"
                      >
                        Yes
                      </button>
                      <button
                        onClick={() => setShowRemoveConfirm(null)}
                        className="px-2 py-1 text-xs bg-gray-300 dark:bg-gray-600 rounded hover:bg-gray-400"
                      >
                        No
                      </button>
                    </div>
                  ) : (
                    <button
                      onClick={() => setShowRemoveConfirm(index)}
                      className="text-gray-400 hover:text-red-500"
                    >
                      <X size={16} />
                    </button>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Action Buttons */}
      <div className="flex gap-3">
        <button
          onClick={calculateTargets}
          disabled={targets.length === 0 || isCalculating}
          className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 flex items-center gap-2"
        >
          <Play size={16} />
          {isCalculating ? 'Calculating...' : 'Preview on Chart'}
        </button>
      </div>

      {/* Warning for class imbalance */}
      {targets.some(t => t.stats.validRows > 0 && t.config.category === 'binary_classification' && (t.stats.positivePct || 0) < 10) && (
        <div className="mt-4 p-3 bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 rounded-lg flex items-start gap-2">
          <AlertCircle className="w-5 h-5 text-amber-500 flex-shrink-0 mt-0.5" />
          <div className="text-sm">
            <p className="font-medium text-amber-800 dark:text-amber-200">Class Imbalance Detected</p>
            <p className="text-amber-700 dark:text-amber-300 mt-1">
              Some targets have less than 10% positive samples. Use <strong>F1-score</strong> as
              the fitness metric and <strong>Focal Loss</strong> during training.
            </p>
          </div>
        </div>
      )}
    </div>
  );
};

export default PredictionTargetsPanel;
