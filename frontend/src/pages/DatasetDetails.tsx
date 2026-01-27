import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { ArrowLeft, Calendar, TrendingUp, Database, ZoomIn, ZoomOut, Maximize2, Download, Eye, EyeOff, MessageSquare, Target, Plus, X, Play, Save, RefreshCw, AlertCircle, CheckCircle, Loader, Settings, ChevronDown, ChevronUp } from 'lucide-react';
import {
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ComposedChart,
  Bar,
  Brush,
  Line,
  ReferenceDot,
} from 'recharts';

interface Dataset {
  id: number;
  name: string;
  ticker: string;
  timeframe: string;
  start_date: string;
  end_date: string;
  rows_count: number;
  status: 'pending' | 'building' | 'ready' | 'error';
  error_message: string | null;
  technical_indicators: any;
  fundamentals_config: any;
  sentiment_config: any;
  file_path: string;
  created_at: string;
}

interface OHLCData {
  Date: string;
  Open: number;
  High: number;
  Low: number;
  Close: number;
  Volume: number;
  // Technical indicators
  SMA_20?: number;
  SMA_50?: number;
  EMA_12?: number;
  EMA_26?: number;
  MACD?: number;
  MACD_signal?: number;
  RSI?: number;
  BB_upper?: number;
  BB_lower?: number;
  BB_middle?: number;
}

interface SentimentMarker {
  date: string;
  sentiment: 'positive' | 'neutral' | 'negative';
  score: number;
  headline: string;
  source?: string;
}

interface NewsFrequency {
  date: string;
  count: number;
  positiveCount: number;
  negativeCount: number;
  neutralCount: number;
  dominantSentiment: 'positive' | 'neutral' | 'negative';
}

interface PredictionTarget {
  profitPct: number;
  maxDd: number;
  days: number;
}

interface PredictionPreview {
  target_columns: string[];
  statistics: Record<string, {
    positive_count: number;
    negative_count: number;
    positive_pct: number;
    negative_pct: number;
    total_valid: number;
  }>;
  target_data: any[];  // All rows with Date + target columns
  total_rows: number;
}

// Custom diamond shape for prediction target markers
const DiamondShape = (props: any) => {
  const { cx, cy, fill, stroke, strokeWidth } = props;
  const size = 6;
  return (
    <polygon
      points={`${cx},${cy - size} ${cx + size},${cy} ${cx},${cy + size} ${cx - size},${cy}`}
      fill={fill}
      stroke={stroke}
      strokeWidth={strokeWidth}
    />
  );
};

interface IndicatorVisibility {
  sma20: boolean;
  sma50: boolean;
  ema12: boolean;
  ema26: boolean;
  bollingerBands: boolean;
  volume: boolean;
  showMacd: boolean;
  showRsi: boolean;
  showSentiment: boolean;
  showTargets: boolean;
}

interface ColumnInfo {
  name: string;
  dtype: string;
  category: string;
}

interface DatasetColumns {
  dataset_id: number;
  total_columns: number;
  category_counts: Record<string, number>;
  columns: Record<string, ColumnInfo[]>;
  all_columns: string[];
}

// Colors for dynamic indicators
const INDICATOR_COLORS = [
  '#3B82F6', // blue
  '#F97316', // orange
  '#10B981', // green
  '#8B5CF6', // purple
  '#EC4899', // pink
  '#06B6D4', // cyan
  '#F59E0B', // amber
  '#6366F1', // indigo
  '#84CC16', // lime
  '#EF4444', // red
];

// Custom Candlestick component for Recharts
const Candlestick = (props: any) => {
  const { x, y, width, height, low: _low, high: _high, openClose } = props;
  void _low; void _high; // Available for wick calculations if needed
  const isGrowing = openClose[1] > openClose[0];
  const color = isGrowing ? '#10B981' : '#EF4444'; // Green for bullish, red for bearish
  const ratio = Math.abs(height / (openClose[0] - openClose[1]));

  return (
    <g stroke={color} fill="none" strokeWidth="2">
      {/* High-Low wick line */}
      <path
        d={`
          M ${x + width / 2}, ${y}
          L ${x + width / 2}, ${y + height}
        `}
      />
      {/* Open-Close body rectangle */}
      <rect
        x={x + 1}
        y={isGrowing ? y + height - ratio * (openClose[1] - openClose[0]) : y}
        width={Math.max(width - 2, 1)}
        height={Math.abs(ratio * (openClose[1] - openClose[0]))}
        fill={color}
        fillOpacity={0.8}
      />
    </g>
  );
};

const DatasetDetails: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [dataset, setDataset] = useState<Dataset | null>(null);
  const [chartData, setChartData] = useState<OHLCData[]>([]);
  const [sentimentMarkers, setSentimentMarkers] = useState<SentimentMarker[]>([]);
  const [sentimentLoading, setSentimentLoading] = useState(false);
  const [sentimentIsMock, setSentimentIsMock] = useState(false);
  const [sentimentError, setSentimentError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [initialZoomSet, setInitialZoomSet] = useState(false);
  const [isRegenerating, setIsRegenerating] = useState(false);

  // Prediction targets state
  const [predictionTargets, setPredictionTargets] = useState<PredictionTarget[]>([]);
  const [newTarget, setNewTarget] = useState<PredictionTarget>({ profitPct: 10, maxDd: 5, days: 14 });
  const [predictionPreview, setPredictionPreview] = useState<PredictionPreview | null>(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [generateLoading, setGenerateLoading] = useState(false);
  const [generatedFiles, setGeneratedFiles] = useState<{ training: string; normalization: string } | null>(null);
  const [zoomDomain, setZoomDomain] = useState<{ startIndex: number; endIndex: number } | null>(null);
  const [indicators, setIndicators] = useState<IndicatorVisibility>({
    sma20: true,
    sma50: false,
    ema12: false,
    ema26: false,
    bollingerBands: false,
    volume: true,
    showMacd: false,
    showRsi: false,
    showSentiment: true,
    showTargets: true,
  });

  // Dynamic indicators from dataset columns
  const [datasetColumns, setDatasetColumns] = useState<DatasetColumns | null>(null);
  const [columnsLoading, setColumnsLoading] = useState(false);
  const [showIndicatorPopup, setShowIndicatorPopup] = useState(false);
  const [enabledIndicators, setEnabledIndicators] = useState<Set<string>>(new Set());
  const [expandedCategories, setExpandedCategories] = useState<Set<string>>(new Set(['technical']));

  // Fetch real sentiment markers from API
  const fetchSentimentMarkers = async (datasetId: number) => {
    setSentimentLoading(true);
    setSentimentError(null);
    try {
      const response = await fetch(`http://localhost:8002/api/datasets/${datasetId}/sentiment?provider=fmp`);
      if (response.ok) {
        const data = await response.json();
        setSentimentMarkers(data.markers || []);
        setSentimentIsMock(data.is_mock || false);
        setSentimentError(null);
      } else {
        // Try to extract error message from response
        let errorMsg = 'Failed to fetch sentiment markers';
        try {
          const errorData = await response.json();
          errorMsg = errorData.detail || errorMsg;
        } catch {
          // Ignore JSON parse errors
        }
        console.error('Failed to fetch sentiment markers:', errorMsg);
        setSentimentMarkers([]);
        setSentimentError(errorMsg);
      }
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'Network error';
      console.error('Error fetching sentiment:', errorMsg);
      setSentimentMarkers([]);
      setSentimentError(errorMsg);
    } finally {
      setSentimentLoading(false);
    }
  };

  // Fetch dataset columns for indicator selection
  const fetchDatasetColumns = async (datasetId: number) => {
    setColumnsLoading(true);
    try {
      const response = await fetch(`http://localhost:8002/api/datasets/${datasetId}/columns`);
      if (response.ok) {
        const data = await response.json();
        setDatasetColumns(data);
      } else {
        console.error('Failed to fetch dataset columns');
      }
    } catch (err) {
      console.error('Error fetching columns:', err);
    } finally {
      setColumnsLoading(false);
    }
  };

  // Toggle indicator visibility
  const toggleDynamicIndicator = (columnName: string) => {
    setEnabledIndicators(prev => {
      const next = new Set(prev);
      if (next.has(columnName)) {
        next.delete(columnName);
      } else {
        next.add(columnName);
      }
      return next;
    });
  };

  // Toggle category expansion in popup
  const toggleCategory = (category: string) => {
    setExpandedCategories(prev => {
      const next = new Set(prev);
      if (next.has(category)) {
        next.delete(category);
      } else {
        next.add(category);
      }
      return next;
    });
  };

  // Get color for an indicator based on its index
  const getIndicatorColor = (columnName: string): string => {
    const index = Array.from(enabledIndicators).indexOf(columnName);
    return INDICATOR_COLORS[index % INDICATOR_COLORS.length];
  };

  // Handle dataset regeneration
  const handleRegenerate = async () => {
    if (!dataset) return;

    setIsRegenerating(true);
    setError(null);

    try {
      const response = await fetch(`http://localhost:8002/api/datasets/${dataset.id}/regenerate`, {
        method: 'POST',
      });

      if (response.ok) {
        const updatedDataset = await response.json();
        setDataset(updatedDataset);

        // Refetch chart data if successful
        if (updatedDataset.status === 'ready') {
          const csvResponse = await fetch(`http://localhost:8002/api/datasets/${dataset.id}/preview`);
          if (csvResponse.ok) {
            const csvData = await csvResponse.json();
            const rawData = csvData.data || [];
            const enrichedData = addIndicatorsToData(rawData);
            setChartData(enrichedData);
          }
        }
      } else {
        const errorData = await response.json();
        setError(errorData.detail || 'Failed to regenerate dataset');
        // Refetch dataset to get updated status
        const refreshResponse = await fetch(`http://localhost:8002/api/datasets/${dataset.id}`);
        if (refreshResponse.ok) {
          setDataset(await refreshResponse.json());
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Network error');
    } finally {
      setIsRegenerating(false);
    }
  };

  // Prediction target functions
  const addPredictionTarget = () => {
    if (newTarget.profitPct > 0 && newTarget.maxDd > 0 && newTarget.days > 0) {
      setPredictionTargets([...predictionTargets, { ...newTarget }]);
      setNewTarget({ profitPct: 10, maxDd: 5, days: 14 });
    }
  };

  const removePredictionTarget = (index: number) => {
    setPredictionTargets(predictionTargets.filter((_, i) => i !== index));
  };

  const previewPredictionTargets = async () => {
    if (!dataset || predictionTargets.length === 0) return;

    setPreviewLoading(true);
    setPredictionPreview(null);

    try {
      // Convert targets to API format (both up and down directions)
      const apiTargets = predictionTargets.flatMap(t => [
        { profit_pct: t.profitPct, max_dd: t.maxDd, days: t.days, direction: 'up' },
        { profit_pct: t.profitPct, max_dd: t.maxDd, days: t.days, direction: 'down' }
      ]);

      const response = await fetch(
        `http://localhost:8002/api/ml/datasets/${dataset.id}/preview-targets`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(apiTargets)
        }
      );

      if (response.ok) {
        const data = await response.json();
        setPredictionPreview(data);
      } else {
        console.error('Preview failed');
      }
    } catch (err) {
      console.error('Preview error:', err);
    } finally {
      setPreviewLoading(false);
    }
  };

  const generateTrainingData = async () => {
    if (!dataset || predictionTargets.length === 0) return;

    setGenerateLoading(true);
    setGeneratedFiles(null);

    try {
      const apiTargets = predictionTargets.flatMap(t => [
        { profit_pct: t.profitPct, max_dd: t.maxDd, days: t.days, direction: 'up' },
        { profit_pct: t.profitPct, max_dd: t.maxDd, days: t.days, direction: 'down' }
      ]);

      const response = await fetch(
        `http://localhost:8002/api/ml/datasets/${dataset.id}/generate-training-data`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ targets: apiTargets, normalize: true })
        }
      );

      if (response.ok) {
        const data = await response.json();
        setGeneratedFiles({
          training: data.training_file,
          normalization: data.normalization_file
        });
      } else {
        console.error('Generate failed');
      }
    } catch (err) {
      console.error('Generate error:', err);
    } finally {
      setGenerateLoading(false);
    }
  };

  // Calculate simple moving average (utility function for future use)
  const calculateSMA = (data: OHLCData[], period: number): OHLCData[] => {
    return data.map((d, i) => {
      if (i < period - 1) return d;
      const sum = data.slice(i - period + 1, i + 1).reduce((acc, curr) => acc + curr.Close, 0);
      return {
        ...d,
        [`SMA_${period}`]: sum / period,
      };
    });
  };
  void calculateSMA; // Reserved for dynamic period calculations

  // Add indicators to chart data
  const addIndicatorsToData = (data: OHLCData[]): OHLCData[] => {
    if (data.length === 0) return data;

    let enrichedData = [...data];

    // Calculate SMA 20
    enrichedData = enrichedData.map((d, i) => {
      if (i < 19) return d;
      const sum = enrichedData.slice(i - 19, i + 1).reduce((acc, curr) => acc + curr.Close, 0);
      return { ...d, SMA_20: sum / 20 };
    });

    // Calculate SMA 50
    enrichedData = enrichedData.map((d, i) => {
      if (i < 49) return { ...d };
      const sum = enrichedData.slice(i - 49, i + 1).reduce((acc, curr) => acc + curr.Close, 0);
      return { ...d, SMA_50: sum / 50 };
    });

    // Calculate Bollinger Bands (20-period, 2 std dev)
    enrichedData = enrichedData.map((d, i) => {
      if (i < 19) return { ...d };
      const slice = enrichedData.slice(i - 19, i + 1);
      const mean = slice.reduce((acc, curr) => acc + curr.Close, 0) / 20;
      const variance = slice.reduce((acc, curr) => acc + Math.pow(curr.Close - mean, 2), 0) / 20;
      const stdDev = Math.sqrt(variance);
      return {
        ...d,
        BB_middle: mean,
        BB_upper: mean + 2 * stdDev,
        BB_lower: mean - 2 * stdDev,
      };
    });

    return enrichedData;
  };

  const toggleIndicator = (key: keyof IndicatorVisibility) => {
    setIndicators(prev => ({ ...prev, [key]: !prev[key] }));
  };

  useEffect(() => {
    const fetchDataset = async () => {
      setIsLoading(true);
      setError(null);
      try {
        // Fetch dataset metadata
        const response = await fetch(`http://localhost:8002/api/datasets/${id}`);
        if (!response.ok) {
          throw new Error('Failed to fetch dataset details');
        }
        const data = await response.json();
        setDataset(data);

        // Fetch dataset CSV data for charting
        // Note: Preview endpoint requires backend restart to be available
        try {
          const csvResponse = await fetch(`http://localhost:8002/api/datasets/${id}/preview`);
          if (csvResponse.ok) {
            const csvData = await csvResponse.json();
            const rawData = csvData.data || [];
            const enrichedData = addIndicatorsToData(rawData);
            setChartData(enrichedData);
          }
        } catch (err) {
          console.log('Preview endpoint not available yet');
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : 'An error occurred');
      } finally {
        setIsLoading(false);
      }
    };

    if (id) {
      fetchDataset();
    }
  }, [id]);

  // Fetch sentiment when dataset loads and sentiment toggle is on
  useEffect(() => {
    if (dataset && indicators.showSentiment) {
      fetchSentimentMarkers(dataset.id);
    }
  }, [dataset?.id, indicators.showSentiment]);

  // Fetch dataset columns for indicator popup
  useEffect(() => {
    if (dataset) {
      fetchDatasetColumns(dataset.id);
    }
  }, [dataset?.id]);

  // Set initial zoom based on data length (show last 100 bars by default)
  useEffect(() => {
    if (chartData.length > 0 && !initialZoomSet) {
      const dataLength = chartData.length;
      const maxInitialBars = 100;

      if (dataLength > maxInitialBars) {
        // Start from the end (most recent data)
        setZoomDomain({
          startIndex: dataLength - maxInitialBars,
          endIndex: dataLength - 1
        });
      }
      setInitialZoomSet(true);
    }
  }, [chartData.length, initialZoomSet]);

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString();
  };

  const formatNumber = (num: number) => {
    return new Intl.NumberFormat('en-US').format(num);
  };

  // Prepare data for candlestick chart
  const prepareChartData = () => {
    return chartData.map((d) => ({
      ...d,
      openClose: [d.Open, d.Close],
      highLow: [d.High, d.Low],
    }));
  };

  const candlestickData = prepareChartData();

  // Aggregate news by date for frequency-based visualization
  const newsFrequencyByDate = useMemo<NewsFrequency[]>(() => {
    if (!sentimentMarkers || sentimentMarkers.length === 0) return [];

    // Group markers by date
    const byDate = new Map<string, SentimentMarker[]>();
    sentimentMarkers.forEach(marker => {
      const existing = byDate.get(marker.date) || [];
      existing.push(marker);
      byDate.set(marker.date, existing);
    });

    // Convert to NewsFrequency array
    const frequencies: NewsFrequency[] = [];
    byDate.forEach((markers, date) => {
      const positiveCount = markers.filter(m => m.sentiment === 'positive').length;
      const negativeCount = markers.filter(m => m.sentiment === 'negative').length;
      const neutralCount = markers.filter(m => m.sentiment === 'neutral').length;

      let dominantSentiment: 'positive' | 'neutral' | 'negative' = 'neutral';
      if (positiveCount > negativeCount && positiveCount > neutralCount) {
        dominantSentiment = 'positive';
      } else if (negativeCount > positiveCount && negativeCount > neutralCount) {
        dominantSentiment = 'negative';
      }

      frequencies.push({
        date,
        count: markers.length,
        positiveCount,
        negativeCount,
        neutralCount,
        dominantSentiment
      });
    });

    return frequencies;
  }, [sentimentMarkers]);

  // Calculate circle size based on news count
  const getNewsCircleRadius = (count: number): number => {
    const minRadius = 4;
    const maxRadius = 16;
    // Use log scale for better visualization
    const maxCount = Math.max(...newsFrequencyByDate.map(n => n.count), 1);
    if (maxCount <= 1) return minRadius;
    const scale = (Math.log(count + 1) / Math.log(maxCount + 1));
    return minRadius + (maxRadius - minRadius) * scale;
  };

  // Zoom control functions
  const handleZoomIn = () => {
    const currentStart = zoomDomain?.startIndex ?? 0;
    const currentEnd = zoomDomain?.endIndex ?? candlestickData.length - 1;
    const range = currentEnd - currentStart;
    const newRange = Math.max(Math.floor(range * 0.7), 10); // Zoom in by 30%, min 10 points
    const center = Math.floor((currentStart + currentEnd) / 2);
    const newStart = Math.max(0, center - Math.floor(newRange / 2));
    const newEnd = Math.min(candlestickData.length - 1, newStart + newRange);
    setZoomDomain({ startIndex: newStart, endIndex: newEnd });
    setBrushKey(prev => prev + 1);
  };

  const handleZoomOut = () => {
    const currentStart = zoomDomain?.startIndex ?? 0;
    const currentEnd = zoomDomain?.endIndex ?? candlestickData.length - 1;
    const range = currentEnd - currentStart;
    const newRange = Math.min(Math.floor(range * 1.5), candlestickData.length);
    const center = Math.floor((currentStart + currentEnd) / 2);
    const newStart = Math.max(0, center - Math.floor(newRange / 2));
    const newEnd = Math.min(candlestickData.length - 1, newStart + newRange);

    if (newStart === 0 && newEnd === candlestickData.length - 1) {
      setZoomDomain(null);
    } else {
      setZoomDomain({ startIndex: newStart, endIndex: newEnd });
    }
    setBrushKey(prev => prev + 1);
  };

  const handleResetZoom = () => {
    setZoomDomain(null);
    setBrushKey(prev => prev + 1); // Force brush to reset
  };

  // Brush key to force re-creation when zoom buttons are used
  const [brushKey, setBrushKey] = useState(0);

  // Debounce timer ref
  const brushDebounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const handleBrushChange = useCallback((domain: any) => {
    if (domain && domain.startIndex !== undefined && domain.endIndex !== undefined) {
      // Debounce the state update to prevent rapid re-renders
      if (brushDebounceRef.current) {
        clearTimeout(brushDebounceRef.current);
      }
      brushDebounceRef.current = setTimeout(() => {
        setZoomDomain({ startIndex: domain.startIndex, endIndex: domain.endIndex });
      }, 50);
    }
  }, []);

  const handleExport = async () => {
    try {
      const response = await fetch(`http://localhost:8002/api/datasets/${id}/export`);
      if (!response.ok) {
        throw new Error('Failed to export dataset');
      }

      // Create a blob from the response
      const blob = await response.blob();

      // Create a temporary download link
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${dataset?.name || 'dataset'}.csv`;
      document.body.appendChild(a);
      a.click();

      // Cleanup
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (err) {
      console.error('Export error:', err);
      alert('Failed to export dataset');
    }
  };

  const handleExportParquet = async () => {
    try {
      const response = await fetch(`http://localhost:8002/api/datasets/${id}/export/parquet`);
      if (!response.ok) {
        throw new Error('Failed to export dataset to Parquet');
      }

      // Create a blob from the response
      const blob = await response.blob();

      // Create a temporary download link
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${dataset?.name || 'dataset'}.parquet`;
      document.body.appendChild(a);
      a.click();

      // Cleanup
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (err) {
      console.error('Export Parquet error:', err);
      alert('Failed to export dataset to Parquet');
    }
  };

  if (isLoading) {
    return (
      <div className="p-6">
        <div className="text-center py-12">
          <p className="text-gray-600 dark:text-gray-400">Loading dataset...</p>
        </div>
      </div>
    );
  }

  if (error || !dataset) {
    return (
      <div className="p-6">
        <div className="mb-4 p-4 bg-red-100 dark:bg-red-900 text-red-700 dark:text-red-200 rounded-md">
          {error || 'Dataset not found'}
        </div>
        <button
          onClick={() => navigate('/datasets')}
          className="px-4 py-2 bg-gray-500 text-white rounded-md hover:bg-gray-600 flex items-center space-x-2"
        >
          <ArrowLeft size={16} />
          <span>Back to Datasets</span>
        </button>
      </div>
    );
  }

  return (
    <div className="p-6">
      {/* Header */}
      <div className="mb-6">
        <div className="flex items-center justify-between mb-4">
          <button
            onClick={() => navigate('/datasets')}
            className="px-4 py-2 bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-md hover:bg-gray-200 dark:hover:bg-gray-600 flex items-center space-x-2"
          >
            <ArrowLeft size={16} />
            <span>Back to Datasets</span>
          </button>
          <div className="flex space-x-2">
            <button
              onClick={handleRegenerate}
              disabled={isRegenerating}
              className="px-4 py-2 bg-orange-600 text-white rounded-md hover:bg-orange-700 disabled:opacity-50 flex items-center space-x-2"
            >
              <RefreshCw size={16} className={isRegenerating ? 'animate-spin' : ''} />
              <span>{isRegenerating ? 'Regenerating...' : 'Regenerate'}</span>
            </button>
            <button
              onClick={handleExport}
              disabled={dataset.status !== 'ready'}
              className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 flex items-center space-x-2"
            >
              <Download size={16} />
              <span>Export CSV</span>
            </button>
            <button
              onClick={handleExportParquet}
              disabled={dataset.status !== 'ready'}
              className="px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 disabled:opacity-50 flex items-center space-x-2"
            >
              <Download size={16} />
              <span>Export Parquet</span>
            </button>
          </div>
        </div>
        <div className="flex items-center gap-3 mb-2">
          <h1 className="text-3xl font-bold text-gray-900 dark:text-gray-100">{dataset.name}</h1>
          {/* Status Badge */}
          {dataset.status === 'ready' && (
            <span className="px-2.5 py-1 text-xs font-medium bg-green-100 text-green-700 dark:bg-green-900/50 dark:text-green-300 rounded-full flex items-center gap-1">
              <CheckCircle size={12} />
              Ready
            </span>
          )}
          {dataset.status === 'building' && (
            <span className="px-2.5 py-1 text-xs font-medium bg-blue-100 text-blue-700 dark:bg-blue-900/50 dark:text-blue-300 rounded-full flex items-center gap-1">
              <Loader size={12} className="animate-spin" />
              Building
            </span>
          )}
          {dataset.status === 'pending' && (
            <span className="px-2.5 py-1 text-xs font-medium bg-gray-100 text-gray-700 dark:bg-gray-700 dark:text-gray-300 rounded-full flex items-center gap-1">
              <Loader size={12} />
              Pending
            </span>
          )}
          {dataset.status === 'error' && (
            <span className="px-2.5 py-1 text-xs font-medium bg-red-100 text-red-700 dark:bg-red-900/50 dark:text-red-300 rounded-full flex items-center gap-1">
              <AlertCircle size={12} />
              Error
            </span>
          )}
        </div>
        <p className="text-gray-600 dark:text-gray-400">
          {dataset.ticker} • {dataset.timeframe}
        </p>
        {/* Error Message Banner */}
        {dataset.status === 'error' && dataset.error_message && (
          <div className="mt-3 p-3 bg-red-50 dark:bg-red-900/30 border border-red-200 dark:border-red-800 rounded-lg">
            <div className="flex items-start gap-2">
              <AlertCircle size={18} className="text-red-500 flex-shrink-0 mt-0.5" />
              <div>
                <p className="text-sm font-medium text-red-700 dark:text-red-300">Dataset Generation Failed</p>
                <p className="text-sm text-red-600 dark:text-red-400 mt-1">{dataset.error_message}</p>
                <button
                  onClick={handleRegenerate}
                  disabled={isRegenerating}
                  className="mt-2 px-3 py-1.5 text-sm bg-red-600 text-white rounded hover:bg-red-700 disabled:opacity-50 flex items-center gap-1"
                >
                  <RefreshCw size={14} className={isRegenerating ? 'animate-spin' : ''} />
                  {isRegenerating ? 'Retrying...' : 'Retry Generation'}
                </button>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
        <div className="bg-white dark:bg-gray-800 p-4 rounded-lg shadow">
          <div className="flex items-center space-x-2 mb-2">
            <Database size={20} className="text-blue-500" />
            <h3 className="text-sm font-medium text-gray-600 dark:text-gray-400">
              Data Points
            </h3>
          </div>
          <p className="text-2xl font-bold text-gray-900 dark:text-gray-100">
            {formatNumber(dataset.rows_count)}
          </p>
        </div>

        <div className="bg-white dark:bg-gray-800 p-4 rounded-lg shadow">
          <div className="flex items-center space-x-2 mb-2">
            <Calendar size={20} className="text-green-500" />
            <h3 className="text-sm font-medium text-gray-600 dark:text-gray-400">
              Start Date
            </h3>
          </div>
          <p className="text-2xl font-bold text-gray-900 dark:text-gray-100">
            {formatDate(dataset.start_date)}
          </p>
        </div>

        <div className="bg-white dark:bg-gray-800 p-4 rounded-lg shadow">
          <div className="flex items-center space-x-2 mb-2">
            <Calendar size={20} className="text-orange-500" />
            <h3 className="text-sm font-medium text-gray-600 dark:text-gray-400">
              End Date
            </h3>
          </div>
          <p className="text-2xl font-bold text-gray-900 dark:text-gray-100">
            {formatDate(dataset.end_date)}
          </p>
        </div>

        <div className="bg-white dark:bg-gray-800 p-4 rounded-lg shadow">
          <div className="flex items-center space-x-2 mb-2">
            <TrendingUp size={20} className="text-purple-500" />
            <h3 className="text-sm font-medium text-gray-600 dark:text-gray-400">
              Timeframe
            </h3>
          </div>
          <p className="text-2xl font-bold text-gray-900 dark:text-gray-100">
            {dataset.timeframe}
          </p>
        </div>
      </div>

      {/* Price Chart - Candlestick */}
      <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow mb-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-bold">Price Chart (Candlestick)</h2>
          <div className="flex items-center space-x-2">
            <button
              onClick={handleZoomIn}
              className="p-2 bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 rounded-md transition-colors"
              title="Zoom In"
            >
              <ZoomIn size={18} className="text-gray-700 dark:text-gray-300" />
            </button>
            <button
              onClick={handleZoomOut}
              className="p-2 bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 rounded-md transition-colors"
              title="Zoom Out"
            >
              <ZoomOut size={18} className="text-gray-700 dark:text-gray-300" />
            </button>
            <button
              onClick={handleResetZoom}
              className="p-2 bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 rounded-md transition-colors"
              title="Reset Zoom"
            >
              <Maximize2 size={18} className="text-gray-700 dark:text-gray-300" />
            </button>
          </div>
        </div>

        {/* Indicator Toggle Controls */}
        <div className="flex flex-wrap gap-3 mb-4 p-3 bg-gray-50 dark:bg-gray-700/50 rounded-lg">
          <span className="text-sm font-medium text-gray-600 dark:text-gray-400 self-center">Overlays:</span>
          <button
            onClick={() => toggleIndicator('sma20')}
            className={`px-3 py-1.5 text-sm rounded-md flex items-center gap-1.5 transition-colors ${
              indicators.sma20
                ? 'bg-blue-100 text-blue-700 dark:bg-blue-900/50 dark:text-blue-300'
                : 'bg-gray-100 text-gray-500 dark:bg-gray-600 dark:text-gray-400'
            }`}
          >
            {indicators.sma20 ? <Eye size={14} /> : <EyeOff size={14} />}
            SMA 20
          </button>
          <button
            onClick={() => toggleIndicator('sma50')}
            className={`px-3 py-1.5 text-sm rounded-md flex items-center gap-1.5 transition-colors ${
              indicators.sma50
                ? 'bg-orange-100 text-orange-700 dark:bg-orange-900/50 dark:text-orange-300'
                : 'bg-gray-100 text-gray-500 dark:bg-gray-600 dark:text-gray-400'
            }`}
          >
            {indicators.sma50 ? <Eye size={14} /> : <EyeOff size={14} />}
            SMA 50
          </button>
          <button
            onClick={() => toggleIndicator('bollingerBands')}
            className={`px-3 py-1.5 text-sm rounded-md flex items-center gap-1.5 transition-colors ${
              indicators.bollingerBands
                ? 'bg-purple-100 text-purple-700 dark:bg-purple-900/50 dark:text-purple-300'
                : 'bg-gray-100 text-gray-500 dark:bg-gray-600 dark:text-gray-400'
            }`}
          >
            {indicators.bollingerBands ? <Eye size={14} /> : <EyeOff size={14} />}
            Bollinger Bands
          </button>
          <button
            onClick={() => toggleIndicator('volume')}
            className={`px-3 py-1.5 text-sm rounded-md flex items-center gap-1.5 transition-colors ${
              indicators.volume
                ? 'bg-violet-100 text-violet-700 dark:bg-violet-900/50 dark:text-violet-300'
                : 'bg-gray-100 text-gray-500 dark:bg-gray-600 dark:text-gray-400'
            }`}
          >
            {indicators.volume ? <Eye size={14} /> : <EyeOff size={14} />}
            Volume
          </button>
          <button
            onClick={() => setShowIndicatorPopup(true)}
            className="px-3 py-1.5 text-sm rounded-md flex items-center gap-1.5 transition-colors bg-indigo-100 text-indigo-700 dark:bg-indigo-900/50 dark:text-indigo-300 hover:bg-indigo-200 dark:hover:bg-indigo-800/50"
          >
            <Settings size={14} />
            More Indicators
            {enabledIndicators.size > 0 && (
              <span className="ml-1 px-1.5 py-0.5 text-xs bg-indigo-500 text-white rounded-full">
                {enabledIndicators.size}
              </span>
            )}
          </button>
          <div className="border-l border-gray-300 dark:border-gray-600 mx-2"></div>
          <button
            onClick={() => toggleIndicator('showSentiment')}
            className={`px-3 py-1.5 text-sm rounded-md flex items-center gap-1.5 transition-colors ${
              indicators.showSentiment
                ? 'bg-green-100 text-green-700 dark:bg-green-900/50 dark:text-green-300'
                : 'bg-gray-100 text-gray-500 dark:bg-gray-600 dark:text-gray-400'
            }`}
          >
            <MessageSquare size={14} />
            News Sentiment
            {sentimentLoading && <span className="ml-1 animate-spin">...</span>}
          </button>
          <button
            onClick={() => toggleIndicator('showTargets')}
            className={`px-3 py-1.5 text-sm rounded-md flex items-center gap-1.5 transition-colors ${
              indicators.showTargets
                ? 'bg-purple-100 text-purple-700 dark:bg-purple-900/50 dark:text-purple-300'
                : 'bg-gray-100 text-gray-500 dark:bg-gray-600 dark:text-gray-400'
            }`}
          >
            <Target size={14} />
            Prediction Targets
          </button>
          {/* Target Legend */}
          {indicators.showTargets && predictionPreview && (
            <div className="flex items-center gap-3 ml-2 text-xs">
              <div className="flex items-center gap-1">
                <div className="w-2.5 h-2.5 rotate-45 bg-green-500 border border-green-700"></div>
                <span className="text-gray-500 dark:text-gray-400">Up Target</span>
              </div>
              <div className="flex items-center gap-1">
                <div className="w-2.5 h-2.5 rotate-45 bg-red-500 border border-red-700"></div>
                <span className="text-gray-500 dark:text-gray-400">Down Target</span>
              </div>
            </div>
          )}
          {/* Sentiment Legend */}
          {indicators.showSentiment && (
            <div className="flex items-center gap-3 ml-3 text-xs">
              {sentimentError ? (
                <>
                  <span className="text-red-500">Error: {sentimentError}</span>
                  <button
                    onClick={() => dataset && fetchSentimentMarkers(dataset.id)}
                    disabled={sentimentLoading}
                    className="px-2 py-0.5 bg-red-100 text-red-700 dark:bg-red-900/50 dark:text-red-300 rounded hover:bg-red-200 dark:hover:bg-red-800/50 disabled:opacity-50"
                  >
                    {sentimentLoading ? 'Retrying...' : 'Retry'}
                  </button>
                </>
              ) : (
                <>
                  <div className="flex items-center gap-1">
                    <div className="w-3 h-3 rounded-full bg-green-500"></div>
                    <span className="text-gray-500 dark:text-gray-400">Positive</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <div className="w-3 h-3 rounded-full bg-orange-500"></div>
                    <span className="text-gray-500 dark:text-gray-400">Neutral</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <div className="w-3 h-3 rounded-full bg-red-500"></div>
                    <span className="text-gray-500 dark:text-gray-400">Negative</span>
                  </div>
                  <span className="text-gray-400 dark:text-gray-500 mx-1">|</span>
                  <span className="text-gray-500 dark:text-gray-400">Size = frequency</span>
                  {sentimentIsMock && (
                    <span className="text-orange-500 text-xs">(Mock Data)</span>
                  )}
                  {!sentimentIsMock && sentimentMarkers.length > 0 && (
                    <span className="text-gray-400 text-xs">({sentimentMarkers.length} articles, {newsFrequencyByDate.length} days)</span>
                  )}
                </>
              )}
            </div>
          )}
        </div>
        {chartData.length > 0 ? (
          <ResponsiveContainer width="100%" height={500}>
            <ComposedChart data={candlestickData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis
                dataKey="Date"
                stroke="#9CA3AF"
                tick={{ fill: '#9CA3AF', fontSize: 12 }}
                tickFormatter={(value) => {
                  const date = new Date(value);
                  return date.toLocaleDateString('en-US', {
                    month: 'short',
                    day: 'numeric',
                  });
                }}
                interval="preserveStartEnd"
                minTickGap={50}
              />
              <YAxis
                yAxisId="price"
                stroke="#9CA3AF"
                tick={{ fill: '#9CA3AF' }}
                domain={['auto', 'auto']}
                label={{ value: 'Price ($)', angle: -90, position: 'insideLeft', fill: '#9CA3AF' }}
              />
              <YAxis
                yAxisId="volume"
                orientation="right"
                stroke="#9CA3AF"
                tick={{ fill: '#9CA3AF' }}
                domain={[0, 'auto']}
                label={{ value: 'Volume', angle: 90, position: 'insideRight', fill: '#9CA3AF' }}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#1F2937',
                  border: '1px solid #374151',
                  borderRadius: '8px',
                  color: '#F3F4F6',
                }}
                labelStyle={{ color: '#F3F4F6' }}
                content={({ active, payload }) => {
                  if (active && payload && payload.length) {
                    const data = payload[0].payload;
                    const isGrowing = data.Close > data.Open;
                    return (
                      <div className="bg-gray-800 border border-gray-700 rounded-lg p-3">
                        <p className="text-gray-300 mb-2">{new Date(data.Date).toLocaleDateString()}</p>
                        <p className="text-sm text-gray-300">Open: <span className="font-semibold">${data.Open.toFixed(2)}</span></p>
                        <p className="text-sm text-gray-300">High: <span className="font-semibold text-green-400">${data.High.toFixed(2)}</span></p>
                        <p className="text-sm text-gray-300">Low: <span className="font-semibold text-red-400">${data.Low.toFixed(2)}</span></p>
                        <p className="text-sm text-gray-300">Close: <span className={`font-semibold ${isGrowing ? 'text-green-400' : 'text-red-400'}`}>${data.Close.toFixed(2)}</span></p>
                        <p className="text-sm text-gray-300 mt-1">Volume: <span className="font-semibold">{formatNumber(data.Volume)}</span></p>
                        <p className={`text-sm mt-1 ${isGrowing ? 'text-green-400' : 'text-red-400'}`}>
                          {isGrowing ? '↑' : '↓'} {Math.abs(((data.Close - data.Open) / data.Open) * 100).toFixed(2)}%
                        </p>
                      </div>
                    );
                  }
                  return null;
                }}
              />
              <Legend wrapperStyle={{ color: '#9CA3AF' }} />
              <Bar
                yAxisId="price"
                dataKey="highLow"
                fill="#8884d8"
                shape={<Candlestick />}
                label={false}
                name="OHLC"
              />
              {indicators.volume && (
                <Bar
                  yAxisId="volume"
                  dataKey="Volume"
                  fill="#8B5CF6"
                  opacity={0.3}
                  name="Volume"
                />
              )}
              {/* Technical Indicator Overlays */}
              {indicators.sma20 && (
                <Line
                  yAxisId="price"
                  type="monotone"
                  dataKey="SMA_20"
                  stroke="#3B82F6"
                  strokeWidth={2}
                  dot={false}
                  name="SMA 20"
                />
              )}
              {indicators.sma50 && (
                <Line
                  yAxisId="price"
                  type="monotone"
                  dataKey="SMA_50"
                  stroke="#F97316"
                  strokeWidth={2}
                  dot={false}
                  name="SMA 50"
                />
              )}
              {indicators.bollingerBands && (
                <>
                  <Line
                    yAxisId="price"
                    type="monotone"
                    dataKey="BB_upper"
                    stroke="#A855F7"
                    strokeWidth={1}
                    strokeDasharray="5 5"
                    dot={false}
                    name="BB Upper"
                  />
                  <Line
                    yAxisId="price"
                    type="monotone"
                    dataKey="BB_middle"
                    stroke="#A855F7"
                    strokeWidth={1}
                    dot={false}
                    name="BB Middle"
                  />
                  <Line
                    yAxisId="price"
                    type="monotone"
                    dataKey="BB_lower"
                    stroke="#A855F7"
                    strokeWidth={1}
                    strokeDasharray="5 5"
                    dot={false}
                    name="BB Lower"
                  />
                </>
              )}
              {/* Dynamic Indicators from Dataset */}
              {Array.from(enabledIndicators).map((columnName) => (
                <Line
                  key={`dynamic-${columnName}`}
                  yAxisId="price"
                  type="monotone"
                  dataKey={columnName}
                  stroke={getIndicatorColor(columnName)}
                  strokeWidth={1.5}
                  dot={false}
                  name={columnName}
                  connectNulls
                />
              ))}
              {/* News Frequency Circles - size based on article count */}
              {indicators.showSentiment && newsFrequencyByDate.map((newsFreq, idx) => {
                const dataPoint = candlestickData.find(d => d.Date === newsFreq.date);
                if (!dataPoint) return null;

                const color = newsFreq.dominantSentiment === 'positive' ? '#10B981' :
                              newsFreq.dominantSentiment === 'negative' ? '#EF4444' : '#F59E0B';
                const radius = getNewsCircleRadius(newsFreq.count);

                return (
                  <ReferenceDot
                    key={`news-freq-${idx}`}
                    x={newsFreq.date}
                    y={dataPoint.High * 1.02}
                    yAxisId="price"
                    r={radius}
                    fill={color}
                    fillOpacity={0.6}
                    stroke={color}
                    strokeWidth={2}
                  />
                );
              })}
              {/* Prediction Target Markers */}
              {indicators.showTargets && predictionPreview && predictionPreview.target_data &&
                predictionPreview.target_data.map((sample, idx) => {
                  const dataPoint = candlestickData.find(d => d.Date === sample.Date);
                  if (!dataPoint) return null;

                  // Check for up targets (green) - price_up_* columns with value 1
                  const upTargetCols = predictionPreview.target_columns.filter(col => col.includes('_up_'));
                  const hasUpTarget = upTargetCols.some(col => sample[col] === 1);

                  // Check for down targets (red) - price_down_* columns with value 1
                  const downTargetCols = predictionPreview.target_columns.filter(col => col.includes('_down_'));
                  const hasDownTarget = downTargetCols.some(col => sample[col] === 1);

                  const markers = [];
                  if (hasUpTarget) {
                    markers.push(
                      <ReferenceDot
                        key={`target-up-${idx}`}
                        x={sample.Date}
                        y={dataPoint.Low * 0.98}
                        yAxisId="price"
                        r={6}
                        fill="#10B981"
                        stroke="#065F46"
                        strokeWidth={1.5}
                        shape={(props) => <DiamondShape {...props} fill="#10B981" stroke="#065F46" strokeWidth={1.5} />}
                      />
                    );
                  }
                  if (hasDownTarget) {
                    markers.push(
                      <ReferenceDot
                        key={`target-down-${idx}`}
                        x={sample.Date}
                        y={dataPoint.High * 1.02}
                        yAxisId="price"
                        r={6}
                        fill="#EF4444"
                        stroke="#991B1B"
                        strokeWidth={1.5}
                        shape={(props) => <DiamondShape {...props} fill="#EF4444" stroke="#991B1B" strokeWidth={1.5} />}
                      />
                    );
                  }
                  return markers;
                }).flat().filter(Boolean)
              }
              <Brush
                key={brushKey}
                dataKey="Date"
                height={30}
                stroke="#8B5CF6"
                fill="#1F2937"
                tickFormatter={(value) => {
                  const date = new Date(value);
                  return date.toLocaleDateString('en-US', {
                    month: 'short',
                    day: 'numeric',
                  });
                }}
                onChange={handleBrushChange}
                startIndex={zoomDomain?.startIndex}
                endIndex={zoomDomain?.endIndex}
              />
            </ComposedChart>
          </ResponsiveContainer>
        ) : (
          <div className="text-center py-12">
            <p className="text-gray-600 dark:text-gray-400">
              No chart data available
            </p>
          </div>
        )}
      </div>

      {/* Non-Chart Data Table (Fundamental, Sentiment, Macro, and other non-chart columns) */}
      {datasetColumns && chartData.length > 0 && (
        (() => {
          // Get columns that are NOT price or technical (include all other categories)
          const chartCategories = ['price', 'technical'];  // These are displayed on the chart
          const nonChartColumns = Object.entries(datasetColumns.columns)
            .filter(([category]) => !chartCategories.includes(category))
            .flatMap(([_, cols]) => cols.map(c => c.name))
            .filter(col => chartData[0] && col in chartData[0]);

          // Get visible data based on zoom
          const startIdx = zoomDomain?.startIndex ?? 0;
          const endIdx = zoomDomain?.endIndex ?? chartData.length - 1;
          const visibleData = chartData.slice(startIdx, endIdx + 1);

          // Limit displayed rows
          const displayData = visibleData.slice(-50); // Last 50 rows

          return (
            <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow mb-6">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-xl font-bold text-gray-900 dark:text-gray-100">
                  Non-Chart Data
                </h2>
                {nonChartColumns.length > 0 && (
                  <span className="text-sm text-gray-500 dark:text-gray-400">
                    Showing {displayData.length} of {visibleData.length} rows ({nonChartColumns.length} columns)
                  </span>
                )}
              </div>
              <p className="text-sm text-gray-500 dark:text-gray-400 mb-4">
                Fundamental, sentiment, and macro data that are not displayed on the chart.
              </p>

              {nonChartColumns.length === 0 ? (
                <div className="text-center py-8 text-gray-500 dark:text-gray-400">
                  <p>No fundamental, sentiment, or macro data in this dataset.</p>
                  <p className="text-sm mt-2">
                    Enable sentiment sources or add fundamental data when creating a dataset to see data here.
                  </p>
                </div>
              ) : (
                <div className="overflow-x-auto max-h-96 overflow-y-auto border border-gray-200 dark:border-gray-700 rounded-lg">
                  <table className="text-sm" style={{ minWidth: 'max-content' }}>
                    <thead className="bg-gray-50 dark:bg-gray-700 sticky top-0">
                      <tr>
                        <th className="px-3 py-2 text-left font-medium text-gray-700 dark:text-gray-300">
                          Date
                        </th>
                        {nonChartColumns.map(col => (
                          <th key={col} className="px-3 py-2 text-left font-medium text-gray-700 dark:text-gray-300 whitespace-nowrap">
                            {col}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                      {displayData.map((row, idx) => (
                        <tr key={idx} className="hover:bg-gray-50 dark:hover:bg-gray-700/50">
                          <td className="px-3 py-2 text-gray-600 dark:text-gray-400 whitespace-nowrap font-mono text-xs">
                            {new Date(row.Date).toLocaleString()}
                          </td>
                          {nonChartColumns.map(col => {
                            const value = (row as any)[col];
                            const formatted = value === null || value === undefined
                              ? '-'
                              : typeof value === 'number'
                                ? Number.isInteger(value) ? value : value.toFixed(4)
                                : String(value);
                            return (
                              <td key={col} className="px-3 py-2 text-gray-900 dark:text-gray-100 whitespace-nowrap font-mono text-xs">
                                {formatted}
                              </td>
                            );
                          })}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          );
        })()
      )}

      {/* Dataset Information */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow">
          <h2 className="text-xl font-bold mb-4">Dataset Information</h2>
          <dl className="space-y-3">
            <div>
              <dt className="text-sm font-medium text-gray-600 dark:text-gray-400">
                Dataset ID
              </dt>
              <dd className="text-sm text-gray-900 dark:text-gray-100">
                {dataset.id}
              </dd>
            </div>
            <div>
              <dt className="text-sm font-medium text-gray-600 dark:text-gray-400">
                File Path
              </dt>
              <dd className="text-sm text-gray-900 dark:text-gray-100 font-mono break-all">
                {dataset.file_path}
              </dd>
            </div>
            <div>
              <dt className="text-sm font-medium text-gray-600 dark:text-gray-400">
                Created At
              </dt>
              <dd className="text-sm text-gray-900 dark:text-gray-100">
                {new Date(dataset.created_at).toLocaleString()}
              </dd>
            </div>
            <div>
              <dt className="text-sm font-medium text-gray-600 dark:text-gray-400 flex items-center gap-1">
                <MessageSquare size={14} />
                News Articles
              </dt>
              <dd className="text-sm text-gray-900 dark:text-gray-100">
                {sentimentLoading ? (
                  <span className="text-gray-500">Loading...</span>
                ) : sentimentError ? (
                  <span className="text-red-500">Error loading news</span>
                ) : sentimentMarkers.length > 0 ? (
                  <div>
                    <span className="font-semibold">{sentimentMarkers.length}</span> articles found
                    <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                      {(() => {
                        const dates = sentimentMarkers
                          .map(m => m.date)
                          .filter(d => d)
                          .sort();
                        if (dates.length === 0) return null;
                        const first = new Date(dates[0]).toLocaleDateString();
                        const last = new Date(dates[dates.length - 1]).toLocaleDateString();
                        return `${first} — ${last}`;
                      })()}
                    </div>
                  </div>
                ) : (
                  <span className="text-gray-500">No news articles found</span>
                )}
              </dd>
            </div>
          </dl>
        </div>

        <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow">
          <h2 className="text-xl font-bold mb-4">Configuration</h2>
          <dl className="space-y-3">
            <div>
              <dt className="text-sm font-medium text-gray-600 dark:text-gray-400">
                Technical Indicators
              </dt>
              <dd className="text-sm text-gray-900 dark:text-gray-100">
                {dataset.technical_indicators
                  ? JSON.stringify(dataset.technical_indicators)
                  : 'Not configured'}
              </dd>
            </div>
            <div>
              <dt className="text-sm font-medium text-gray-600 dark:text-gray-400">
                Fundamentals
              </dt>
              <dd className="text-sm text-gray-900 dark:text-gray-100">
                {dataset.fundamentals_config
                  ? JSON.stringify(dataset.fundamentals_config)
                  : 'Not configured'}
              </dd>
            </div>
            <div>
              <dt className="text-sm font-medium text-gray-600 dark:text-gray-400">
                Sentiment Analysis
              </dt>
              <dd className="text-sm text-gray-900 dark:text-gray-100">
                {dataset.sentiment_config
                  ? JSON.stringify(dataset.sentiment_config)
                  : 'Not configured'}
              </dd>
            </div>
          </dl>
        </div>
      </div>

      {/* Prediction Targets Preview Panel */}
      <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow mt-6">
        <h2 className="text-xl font-bold mb-4 flex items-center gap-2">
          <Target size={20} className="text-purple-500" />
          Prediction Targets Preview
        </h2>
        <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
          Define prediction targets to generate training labels. Each target creates binary columns
          indicating whether the price moved by the specified percentage within the time window.
        </p>

        {/* Target configuration form */}
        <div className="flex flex-wrap items-end gap-4 mb-4 p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
          <div>
            <label className="block text-xs font-medium text-gray-600 dark:text-gray-400 mb-1">
              Profit Target (%)
            </label>
            <input
              type="number"
              min="1"
              max="100"
              value={newTarget.profitPct}
              onChange={(e) => setNewTarget({ ...newTarget, profitPct: parseFloat(e.target.value) || 0 })}
              className="w-24 px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md text-sm bg-white dark:bg-gray-800"
            />
          </div>
          <div>
            <label className="block text-xs font-medium text-gray-600 dark:text-gray-400 mb-1">
              Max Drawdown (%)
            </label>
            <input
              type="number"
              min="1"
              max="50"
              value={newTarget.maxDd}
              onChange={(e) => setNewTarget({ ...newTarget, maxDd: parseFloat(e.target.value) || 0 })}
              className="w-24 px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md text-sm bg-white dark:bg-gray-800"
            />
          </div>
          <div>
            <label className="block text-xs font-medium text-gray-600 dark:text-gray-400 mb-1">
              Time Window (days)
            </label>
            <input
              type="number"
              min="1"
              max="365"
              value={newTarget.days}
              onChange={(e) => setNewTarget({ ...newTarget, days: parseInt(e.target.value) || 0 })}
              className="w-24 px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md text-sm bg-white dark:bg-gray-800"
            />
          </div>
          <button
            onClick={addPredictionTarget}
            className="px-4 py-2 bg-purple-600 text-white rounded-md hover:bg-purple-700 flex items-center gap-1"
          >
            <Plus size={16} />
            Add Target
          </button>
        </div>

        {/* Added targets list */}
        {predictionTargets.length > 0 && (
          <div className="mb-4">
            <h4 className="text-sm font-medium mb-2">Configured Targets:</h4>
            <div className="flex flex-wrap gap-2">
              {predictionTargets.map((t, i) => (
                <span
                  key={i}
                  className="px-3 py-1.5 bg-purple-100 dark:bg-purple-900/50 text-purple-700 dark:text-purple-300 rounded-full text-sm flex items-center gap-2"
                >
                  {t.profitPct}% profit / {t.maxDd}% max DD / {t.days} days
                  <button
                    onClick={() => removePredictionTarget(i)}
                    className="hover:text-red-500"
                  >
                    <X size={14} />
                  </button>
                </span>
              ))}
            </div>
          </div>
        )}

        {/* Action buttons */}
        <div className="flex gap-3 mb-4">
          <button
            onClick={previewPredictionTargets}
            disabled={predictionTargets.length === 0 || previewLoading}
            className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 flex items-center gap-2"
          >
            <Play size={16} />
            {previewLoading ? 'Calculating...' : 'Preview Targets'}
          </button>
          <button
            onClick={generateTrainingData}
            disabled={predictionTargets.length === 0 || generateLoading}
            className="px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 disabled:opacity-50 flex items-center gap-2"
          >
            <Save size={16} />
            {generateLoading ? 'Generating...' : 'Generate Training Data'}
          </button>
        </div>

        {/* Generated files info */}
        {generatedFiles && (
          <div className="mb-4 p-3 bg-green-50 dark:bg-green-900/30 rounded-lg">
            <h4 className="text-sm font-medium text-green-700 dark:text-green-300 mb-2">
              Training Data Generated
            </h4>
            <div className="text-xs text-gray-600 dark:text-gray-400 space-y-1">
              <div><strong>Training file:</strong> {generatedFiles.training}</div>
              {generatedFiles.normalization && (
                <div><strong>Normalization params:</strong> {generatedFiles.normalization}</div>
              )}
            </div>
          </div>
        )}

        {/* Preview statistics */}
        {predictionPreview && (
          <div className="mt-4">
            <h4 className="text-sm font-medium mb-2">Target Statistics:</h4>
            <p className="text-xs text-gray-500 dark:text-gray-400 mb-3">
              Number of data points where each target condition was detected
            </p>
            {/* Class imbalance warning */}
            {Object.values(predictionPreview.statistics).some(s => s.positive_pct < 10) && (
              <div className="mb-3 p-3 bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 rounded-lg flex items-start gap-2">
                <AlertCircle className="w-5 h-5 text-amber-500 flex-shrink-0 mt-0.5" />
                <div className="text-sm">
                  <p className="font-medium text-amber-800 dark:text-amber-200">Class Imbalance Detected</p>
                  <p className="text-amber-700 dark:text-amber-300 mt-1">
                    Some targets have less than 10% positive samples. During training, use <strong>F1-score</strong> as
                    the fitness metric and <strong>Focal Loss</strong> to prevent the model from always predicting "no target".
                  </p>
                </div>
              </div>
            )}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
              {Object.entries(predictionPreview.statistics).map(([col, stats]) => (
                <div key={col} className="p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">
                  <div className="font-mono text-xs text-gray-600 dark:text-gray-400 mb-2 break-all">
                    {col}
                  </div>
                  <div className="flex items-center justify-between text-sm mb-2">
                    <div className="flex items-center gap-1.5">
                      <div className="w-3 h-3 rounded-full bg-green-500"></div>
                      <span className="text-gray-600 dark:text-gray-300">Detected:</span>
                      <span className="font-semibold text-green-600 dark:text-green-400">
                        {stats.positive_count}
                      </span>
                      <span className="text-gray-500 text-xs">({stats.positive_pct}%)</span>
                    </div>
                  </div>
                  <div className="flex items-center justify-between text-sm">
                    <div className="flex items-center gap-1.5">
                      <div className="w-3 h-3 rounded-full bg-gray-400"></div>
                      <span className="text-gray-600 dark:text-gray-300">Not detected:</span>
                      <span className="font-semibold text-gray-600 dark:text-gray-400">
                        {stats.negative_count}
                      </span>
                      <span className="text-gray-500 text-xs">({stats.negative_pct}%)</span>
                    </div>
                  </div>
                  {/* Visual ratio bar */}
                  <div className="mt-3 flex items-center gap-1">
                    <div
                      className="h-1.5 bg-green-500 rounded-l"
                      style={{ width: `${stats.positive_pct}%`, minWidth: stats.positive_count > 0 ? '4px' : '0' }}
                    />
                    <div
                      className="h-1.5 bg-gray-300 dark:bg-gray-500 rounded-r flex-1"
                    />
                  </div>
                </div>
              ))}
            </div>
            <div className="mt-3 text-xs text-gray-500">
              Total rows: {predictionPreview.total_rows} | Valid samples: {Object.values(predictionPreview.statistics)[0]?.total_valid || 0}
            </div>
          </div>
        )}
      </div>

      {/* Indicator Selection Popup */}
      {showIndicatorPopup && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-xl w-full max-w-2xl max-h-[80vh] flex flex-col">
            {/* Header */}
            <div className="flex items-center justify-between p-4 border-b border-gray-200 dark:border-gray-700">
              <div>
                <h3 className="text-lg font-bold text-gray-900 dark:text-gray-100">
                  Dataset Indicators
                </h3>
                <p className="text-sm text-gray-500 dark:text-gray-400">
                  Select indicators to display on the chart (loaded from dataset)
                </p>
              </div>
              <button
                onClick={() => setShowIndicatorPopup(false)}
                className="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-full"
              >
                <X size={20} className="text-gray-500" />
              </button>
            </div>

            {/* Content */}
            <div className="flex-1 overflow-y-auto p-4">
              {columnsLoading ? (
                <div className="flex items-center justify-center py-8">
                  <Loader size={24} className="animate-spin text-gray-500" />
                  <span className="ml-2 text-gray-500">Loading columns...</span>
                </div>
              ) : datasetColumns ? (
                <div className="space-y-4">
                  {/* Category sections */}
                  {['technical', 'fundamental', 'sentiment', 'macro', 'other'].map(category => {
                    const columns = datasetColumns.columns[category] || [];
                    if (columns.length === 0) return null;

                    const isExpanded = expandedCategories.has(category);
                    const categoryLabel = category.charAt(0).toUpperCase() + category.slice(1);
                    const enabledCount = columns.filter(c => enabledIndicators.has(c.name)).length;

                    return (
                      <div key={category} className="border border-gray-200 dark:border-gray-700 rounded-lg overflow-hidden">
                        <button
                          onClick={() => toggleCategory(category)}
                          className="w-full flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-700/50 hover:bg-gray-100 dark:hover:bg-gray-700"
                        >
                          <div className="flex items-center gap-2">
                            {isExpanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                            <span className="font-medium text-gray-900 dark:text-gray-100">
                              {categoryLabel}
                            </span>
                            <span className="text-sm text-gray-500">
                              ({columns.length} columns)
                            </span>
                            {enabledCount > 0 && (
                              <span className="px-2 py-0.5 text-xs bg-indigo-100 text-indigo-700 dark:bg-indigo-900/50 dark:text-indigo-300 rounded-full">
                                {enabledCount} selected
                              </span>
                            )}
                          </div>
                        </button>
                        {isExpanded && (
                          <div className="p-3 grid grid-cols-2 md:grid-cols-3 gap-2">
                            {columns.map(col => {
                              // Skip non-numeric columns
                              if (!col.dtype.includes('float') && !col.dtype.includes('int')) {
                                return null;
                              }
                              const isEnabled = enabledIndicators.has(col.name);
                              const color = isEnabled ? getIndicatorColor(col.name) : undefined;

                              return (
                                <button
                                  key={col.name}
                                  onClick={() => toggleDynamicIndicator(col.name)}
                                  className={`px-3 py-2 text-sm rounded-md flex items-center gap-2 transition-colors text-left ${
                                    isEnabled
                                      ? 'bg-indigo-100 dark:bg-indigo-900/50 text-indigo-700 dark:text-indigo-300'
                                      : 'bg-gray-100 dark:bg-gray-600 text-gray-600 dark:text-gray-400 hover:bg-gray-200 dark:hover:bg-gray-500'
                                  }`}
                                >
                                  {isEnabled && (
                                    <div
                                      className="w-3 h-3 rounded-full flex-shrink-0"
                                      style={{ backgroundColor: color }}
                                    />
                                  )}
                                  <span className="truncate">{col.name}</span>
                                </button>
                              );
                            })}
                          </div>
                        )}
                      </div>
                    );
                  })}

                  {/* Currently selected indicators */}
                  {enabledIndicators.size > 0 && (
                    <div className="mt-4 p-3 bg-indigo-50 dark:bg-indigo-900/20 rounded-lg">
                      <h4 className="text-sm font-medium text-indigo-700 dark:text-indigo-300 mb-2">
                        Selected Indicators ({enabledIndicators.size})
                      </h4>
                      <div className="flex flex-wrap gap-2">
                        {Array.from(enabledIndicators).map(name => (
                          <span
                            key={name}
                            className="px-2 py-1 text-xs rounded-full flex items-center gap-1.5"
                            style={{
                              backgroundColor: `${getIndicatorColor(name)}20`,
                              color: getIndicatorColor(name),
                              border: `1px solid ${getIndicatorColor(name)}`
                            }}
                          >
                            <div
                              className="w-2 h-2 rounded-full"
                              style={{ backgroundColor: getIndicatorColor(name) }}
                            />
                            {name}
                            <button
                              onClick={() => toggleDynamicIndicator(name)}
                              className="ml-1 hover:opacity-70"
                            >
                              <X size={12} />
                            </button>
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                <div className="text-center py-8 text-gray-500">
                  No column data available
                </div>
              )}
            </div>

            {/* Footer */}
            <div className="flex items-center justify-between p-4 border-t border-gray-200 dark:border-gray-700">
              <button
                onClick={() => setEnabledIndicators(new Set())}
                className="px-4 py-2 text-sm text-gray-600 dark:text-gray-400 hover:text-gray-800 dark:hover:text-gray-200"
              >
                Clear All
              </button>
              <button
                onClick={() => setShowIndicatorPopup(false)}
                className="px-4 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700"
              >
                Done
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default DatasetDetails;
