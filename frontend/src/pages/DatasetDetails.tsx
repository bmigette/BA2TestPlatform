import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { ArrowLeft, Calendar, TrendingUp, Database, ZoomIn, ZoomOut, Maximize2, Download, Eye, EyeOff, MessageSquare } from 'lucide-react';
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
}

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
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
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
  });

  // Generate mock sentiment markers based on chart data
  const generateSentimentMarkers = (data: OHLCData[]): SentimentMarker[] => {
    if (data.length === 0) return [];

    const markers: SentimentMarker[] = [];
    const sampleDates = data.filter((_, i) => i % 15 === 0).slice(0, 10);

    const headlines = [
      { text: 'Strong earnings beat expectations', sentiment: 'positive' as const },
      { text: 'Analyst upgrade to Buy rating', sentiment: 'positive' as const },
      { text: 'Market volatility concerns', sentiment: 'negative' as const },
      { text: 'Mixed quarterly results', sentiment: 'neutral' as const },
      { text: 'New product launch announced', sentiment: 'positive' as const },
      { text: 'Supply chain disruptions', sentiment: 'negative' as const },
      { text: 'Dividend increase announced', sentiment: 'positive' as const },
      { text: 'Regulatory concerns emerge', sentiment: 'negative' as const },
      { text: 'Industry outlook stable', sentiment: 'neutral' as const },
      { text: 'Revenue growth exceeds forecast', sentiment: 'positive' as const },
    ];

    sampleDates.forEach((d, i) => {
      const h = headlines[i % headlines.length];
      markers.push({
        date: d.Date,
        sentiment: h.sentiment,
        score: h.sentiment === 'positive' ? 0.7 + Math.random() * 0.3 :
               h.sentiment === 'negative' ? -(0.7 + Math.random() * 0.3) :
               -0.2 + Math.random() * 0.4,
        headline: h.text,
        source: ['Reuters', 'Bloomberg', 'WSJ', 'CNBC'][Math.floor(Math.random() * 4)],
      });
    });

    return markers;
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
            setSentimentMarkers(generateSentimentMarkers(enrichedData));
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
  };

  const handleResetZoom = () => {
    setZoomDomain(null);
  };

  const handleBrushChange = (domain: any) => {
    if (domain && domain.startIndex !== undefined && domain.endIndex !== undefined) {
      setZoomDomain({ startIndex: domain.startIndex, endIndex: domain.endIndex });
    }
  };

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
              onClick={handleExport}
              className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 flex items-center space-x-2"
            >
              <Download size={16} />
              <span>Export CSV</span>
            </button>
            <button
              onClick={handleExportParquet}
              className="px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 flex items-center space-x-2"
            >
              <Download size={16} />
              <span>Export Parquet</span>
            </button>
          </div>
        </div>
        <h1 className="text-3xl font-bold mb-2 text-gray-900 dark:text-gray-100">{dataset.name}</h1>
        <p className="text-gray-600 dark:text-gray-400">
          {dataset.ticker} • {dataset.timeframe}
        </p>
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
          </button>
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
              {/* Sentiment Markers */}
              {indicators.showSentiment && sentimentMarkers.map((marker, idx) => {
                const dataPoint = candlestickData.find(d => d.Date === marker.date);
                if (!dataPoint) return null;
                const color = marker.sentiment === 'positive' ? '#10B981' :
                              marker.sentiment === 'negative' ? '#EF4444' : '#F59E0B';
                return (
                  <ReferenceDot
                    key={`sentiment-${idx}`}
                    x={marker.date}
                    y={dataPoint.High * 1.02}
                    yAxisId="price"
                    r={6}
                    fill={color}
                    stroke="#1F2937"
                    strokeWidth={2}
                  />
                );
              })}
              <Brush
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
    </div>
  );
};

export default DatasetDetails;
