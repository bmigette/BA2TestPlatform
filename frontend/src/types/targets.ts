/**
 * Type definitions for prediction targets system
 */

// Target categories for metric selection
export type TargetCategory = 'binary_classification' | 'multiclass_classification' | 'regression';

// Base target configuration
export interface BaseTargetConfig {
  type: string;
  category: TargetCategory;
  enabled?: boolean;
  color?: string;
}

// Price-based target (existing)
export interface PriceBasedTarget extends BaseTargetConfig {
  type: 'price_based';
  category: 'binary_classification';
  direction: 'up' | 'down';
  profitPct: number;
  maxDrawdownPct: number;
  timeBars: number;
}

// Directional movement target
export interface DirectionalTarget extends BaseTargetConfig {
  type: 'directional';
  category: 'binary_classification';
  direction: 'up' | 'down';
  horizon: number; // bars ahead
}

// Triple-barrier target
export interface TripleBarrierTarget extends BaseTargetConfig {
  type: 'triple_barrier';
  category: 'multiclass_classification';
  profitPct: number;
  stopPct: number;
  maxBars: number;
}

// Trend reversal target
export interface TrendReversalTarget extends BaseTargetConfig {
  type: 'trend_reversal';
  category: 'binary_classification';
  indicator: 'rsi' | 'macd' | 'sar' | 'zigzag';
  indicatorParams: IndicatorParams;
  threshold: number;
  direction: 'bullish' | 'bearish';
}

// Volatility target
export interface VolatilityTarget extends BaseTargetConfig {
  type: 'volatility';
  category: 'regression';
  horizon: number; // bars ahead
  method: 'std' | 'range' | 'atr';
}

// Union type for all target configs
export type TargetConfig =
  | PriceBasedTarget
  | DirectionalTarget
  | TripleBarrierTarget
  | TrendReversalTarget
  | VolatilityTarget;

// Indicator parameters
export interface RSIParams {
  period: number;
}

export interface MACDParams {
  fast: number;
  slow: number;
  signal: number;
}

export interface SARParams {
  afStart: number;
  afMax: number;
}

export interface ZigZagParams {
  deviationPct: number;
}

export type IndicatorParams = RSIParams | MACDParams | SARParams | ZigZagParams;

// Indicator configuration for API
export interface IndicatorConfig {
  type: 'rsi' | 'macd' | 'sar' | 'zigzag';
  period?: number;
  fast?: number;
  slow?: number;
  signal?: number;
  af_start?: number;
  af_max?: number;
  deviation_pct?: number;
}

// Calculated target with data and stats
export interface CalculatedTarget {
  config: TargetConfig;
  columnName: string;
  data: TargetDataPoint[];
  stats: TargetStats;
  color: string;
  visible: boolean;
}

export interface TargetDataPoint {
  date: string;
  value: number | null;
}

export interface TargetStats {
  totalRows: number;
  validRows: number;
  // For classification
  positiveCount?: number;
  negativeCount?: number;
  positivePct?: number;
  negativePct?: number;
  // For multiclass (triple barrier)
  profitHitCount?: number;
  stopHitCount?: number;
  timeoutCount?: number;
  // For regression
  mean?: number;
  std?: number;
  min?: number;
  max?: number;
}

// Saved target set
export interface TargetSet {
  id: number;
  name: string;
  description?: string;
  targets: TargetConfig[];
  createdAt: string;
  updatedAt: string;
}

// API request/response types
export interface CalculateIndicatorsRequest {
  indicators: IndicatorConfig[];
}

export interface CalculateIndicatorsResponse {
  data: Record<string, number | null>[];
}

export interface CalculateTargetsRequest {
  targets: TargetConfig[];
}

export interface CalculateTargetsResponse {
  targets: CalculatedTarget[];
}

// Chart marker types
export type MarkerShape = 'arrowUp' | 'arrowDown' | 'circle' | 'square';
export type MarkerPosition = 'aboveBar' | 'belowBar' | 'inBar';

export interface ChartMarker {
  time: number;
  position: MarkerPosition;
  shape: MarkerShape;
  color: string;
  text?: string;
  size?: number;
}

// Color palette for targets
export const TARGET_COLORS = {
  priceBased: {
    up: '#10B981',    // green
    down: '#EF4444',  // red
  },
  directional: {
    up: '#3B82F6',    // blue
    down: '#8B5CF6',  // purple
  },
  tripleBarrier: {
    profit: '#10B981',  // green
    stop: '#EF4444',    // red
    timeout: '#F59E0B', // yellow/amber
  },
  trendReversal: {
    bullish: '#A855F7', // purple
    bearish: '#EC4899', // pink
  },
  volatility: '#6366F1', // indigo
} as const;

// Default indicator parameters
export const DEFAULT_INDICATOR_PARAMS = {
  rsi: { period: 14 },
  macd: { fast: 12, slow: 26, signal: 9 },
  sar: { afStart: 0.02, afMax: 0.2 },
  zigzag: { deviationPct: 5.0 },
} as const;

// Metric options by category
export const METRICS_BY_CATEGORY = {
  binary_classification: [
    { value: 'accuracy', label: 'Accuracy' },
    { value: 'f1', label: 'F1 Score' },
    { value: 'precision', label: 'Precision' },
    { value: 'recall', label: 'Recall' },
    { value: 'auc_roc', label: 'AUC-ROC' },
  ],
  multiclass_classification: [
    { value: 'accuracy', label: 'Accuracy' },
    { value: 'macro_f1', label: 'Macro F1' },
    { value: 'weighted_f1', label: 'Weighted F1' },
  ],
  regression: [
    { value: 'mse', label: 'MSE' },
    { value: 'rmse', label: 'RMSE' },
    { value: 'mae', label: 'MAE' },
    { value: 'r2', label: 'R²' },
    { value: 'mape', label: 'MAPE' },
  ],
} as const;
