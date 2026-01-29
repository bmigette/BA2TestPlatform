import React, { useEffect, useRef, useCallback, useState } from 'react';
import {
  createChart,
  ColorType,
  CandlestickSeries,
  LineSeries,
  HistogramSeries,
  createSeriesMarkers,
} from 'lightweight-charts';
import type {
  IChartApi,
  ISeriesApi,
  CandlestickData,
  LineData,
  HistogramData,
  Time,
  SeriesMarker,
} from 'lightweight-charts';
import type { CalculatedTarget } from '../types/targets';

export interface OHLCData {
  Date: string;
  Open: number;
  High: number;
  Low: number;
  Close: number;
  Volume?: number;
  [key: string]: any;
}

export interface NewsFrequency {
  date: string;
  count: number;
  dominantSentiment: 'positive' | 'negative' | 'neutral';
}

export interface TrendPoint {
  date: string;
  trend: string;
}

export interface PredictionTarget {
  Date: string;
  [key: string]: any;
}

export interface IndicatorData {
  rsi?: { date: string; value: number | null }[];
  macd?: { date: string; macd: number | null; signal: number | null; histogram: number | null }[];
  sar?: { date: string; value: number | null }[];
  zigzag?: { date: string; value: number | null }[];
}

export interface TradingChartProps {
  data: OHLCData[];
  indicators: {
    showSMA20: boolean;
    showSMA50: boolean;
    showBollinger: boolean;
    showVolume: boolean;
    showSentiment: boolean;
    showTargets: boolean;
    showTrends: boolean;
  };
  newsFrequencyByDate?: NewsFrequency[];
  trendData?: TrendPoint[];
  predictionPreview?: {
    target_columns: string[];
    target_data: PredictionTarget[];
  } | null;
  calculatedTargets?: CalculatedTarget[];
  indicatorData?: IndicatorData;
  height?: number;
}

const TradingChart: React.FC<TradingChartProps> = ({
  data,
  indicators,
  newsFrequencyByDate = [],
  trendData = [],
  calculatedTargets = [],
  indicatorData,
  height = 500,
}) => {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candlestickSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Convert date string to Time format
  const toTime = useCallback((dateStr: string): Time => {
    const date = new Date(dateStr);
    return Math.floor(date.getTime() / 1000) as Time;
  }, []);

  // Calculate SMA
  const calculateSMA = useCallback((sourceData: OHLCData[], period: number): LineData[] => {
    const result: LineData[] = [];
    for (let i = period - 1; i < sourceData.length; i++) {
      let sum = 0;
      for (let j = 0; j < period; j++) {
        sum += sourceData[i - j].Close;
      }
      result.push({
        time: toTime(sourceData[i].Date),
        value: sum / period,
      });
    }
    return result;
  }, [toTime]);

  // Calculate Bollinger Bands
  const calculateBollinger = useCallback((sourceData: OHLCData[], period: number = 20, stdDev: number = 2): {
    upper: LineData[];
    middle: LineData[];
    lower: LineData[];
  } => {
    const upper: LineData[] = [];
    const middle: LineData[] = [];
    const lower: LineData[] = [];

    for (let i = period - 1; i < sourceData.length; i++) {
      let sum = 0;
      for (let j = 0; j < period; j++) {
        sum += sourceData[i - j].Close;
      }
      const sma = sum / period;

      let squaredSum = 0;
      for (let j = 0; j < period; j++) {
        squaredSum += Math.pow(sourceData[i - j].Close - sma, 2);
      }
      const std = Math.sqrt(squaredSum / period);

      const time = toTime(sourceData[i].Date);
      middle.push({ time, value: sma });
      upper.push({ time, value: sma + stdDev * std });
      lower.push({ time, value: sma - stdDev * std });
    }

    return { upper, middle, lower };
  }, [toTime]);

  // Initialize and update chart
  useEffect(() => {
    if (!chartContainerRef.current || data.length === 0) return;

    try {
      setError(null);

      // Clean up previous chart
      if (chartRef.current) {
        chartRef.current.remove();
        chartRef.current = null;
      }

      const chart = createChart(chartContainerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: '#1F2937' },
        textColor: '#9CA3AF',
      },
      grid: {
        vertLines: { color: '#374151' },
        horzLines: { color: '#374151' },
      },
      width: chartContainerRef.current.clientWidth,
      height: height,
      crosshair: {
        mode: 1,
      },
      rightPriceScale: {
        borderColor: '#374151',
      },
      timeScale: {
        borderColor: '#374151',
        timeVisible: true,
        secondsVisible: false,
      },
    });

    chartRef.current = chart;

    // Create candlestick series using v5 API
    const candlestickSeries = chart.addSeries(CandlestickSeries, {
      upColor: '#10B981',
      downColor: '#EF4444',
      borderUpColor: '#10B981',
      borderDownColor: '#EF4444',
      wickUpColor: '#10B981',
      wickDownColor: '#EF4444',
      lastValueVisible: false,
      priceLineVisible: false,
    });
    candlestickSeriesRef.current = candlestickSeries as unknown as ISeriesApi<'Candlestick'>;

    // Set candlestick data
    const candlestickData: CandlestickData[] = data.map((d) => ({
      time: toTime(d.Date),
      open: d.Open,
      high: d.High,
      low: d.Low,
      close: d.Close,
    }));
    candlestickSeries.setData(candlestickData);

    // Add markers for calculated targets
    const markers: SeriesMarker<Time>[] = [];
    calculatedTargets.forEach((target) => {
      if (!target.visible || !target.data || target.data.length === 0) return;

      target.data.forEach((point) => {
        if (point.value === null || point.value === undefined) return;

        const time = toTime(point.date);
        const config = target.config;

        // Determine marker based on target type and value
        if (config.category === 'binary_classification') {
          if (point.value === 1) {
            // Positive hit - arrow above
            const direction = (config as { direction?: string }).direction;
            const isUp = direction === 'up' || direction === 'bullish';
            markers.push({
              time,
              position: isUp ? 'aboveBar' : 'belowBar',
              color: target.color,
              shape: isUp ? 'arrowUp' : 'arrowDown',
              text: '',
            });
          }
        } else if (config.category === 'multiclass_classification') {
          // Triple barrier: 0=stop, 1=profit, 2=timeout
          if (point.value === 1) {
            markers.push({
              time,
              position: 'aboveBar',
              color: '#10B981', // green
              shape: 'arrowUp',
              text: '',
            });
          } else if (point.value === 0) {
            markers.push({
              time,
              position: 'belowBar',
              color: '#EF4444', // red
              shape: 'arrowDown',
              text: '',
            });
          } else if (point.value === 2) {
            markers.push({
              time,
              position: 'inBar',
              color: '#F59E0B', // yellow
              shape: 'circle',
              text: '',
            });
          }
        }
        // Regression targets don't show markers
      });
    });

    // Add trend markers if showTrends is enabled
    if (indicators.showTrends && trendData.length > 0) {
      trendData.forEach((point, index) => {
        // Only show markers at trend changes
        const prevTrend = index > 0 ? trendData[index - 1].trend : null;
        if (point.trend !== prevTrend) {
          const time = toTime(point.date);
          if (point.trend === 'uptrend') {
            markers.push({
              time,
              position: 'belowBar',
              color: '#22C55E',
              shape: 'arrowUp',
              text: 'Up',
            });
          } else if (point.trend === 'downtrend') {
            markers.push({
              time,
              position: 'aboveBar',
              color: '#EF4444',
              shape: 'arrowDown',
              text: 'Down',
            });
          } else if (point.trend === 'sideways') {
            markers.push({
              time,
              position: 'inBar',
              color: '#F59E0B',
              shape: 'circle',
              text: 'Side',
            });
          }
        }
      });
    }

    // Sort markers by time and set them on the candlestick series
    if (markers.length > 0) {
      markers.sort((a, b) => (a.time as number) - (b.time as number));
      createSeriesMarkers(candlestickSeries, markers);
    }

    // News sentiment histogram
    if (indicators.showSentiment && newsFrequencyByDate.length > 0) {
      const sentimentSeries = chart.addSeries(HistogramSeries, {
        priceFormat: { type: 'volume' },
        priceScaleId: 'sentiment',
        lastValueVisible: false,
        priceLineVisible: false,
      });
      chart.priceScale('sentiment').applyOptions({
        scaleMargins: { top: 0.9, bottom: 0 },
      });

      const sentimentData: HistogramData[] = newsFrequencyByDate.map((d) => {
        const color = d.dominantSentiment === 'positive' ? '#22C55E80' :
                      d.dominantSentiment === 'negative' ? '#EF444480' :
                      '#6B728080';
        return {
          time: toTime(d.date),
          value: d.count,
          color,
        };
      });
      sentimentSeries.setData(sentimentData);
    }

    // Volume series
    if (indicators.showVolume) {
      const volumeSeries = chart.addSeries(HistogramSeries, {
        color: '#6366F1',
        priceFormat: { type: 'volume' },
        priceScaleId: 'volume',
        lastValueVisible: false,
        priceLineVisible: false,
      });
      chart.priceScale('volume').applyOptions({
        scaleMargins: { top: 0.8, bottom: 0 },
      });
      const volumeData: HistogramData[] = data.map((d) => ({
        time: toTime(d.Date),
        value: d.Volume || 0,
        color: d.Close >= d.Open ? '#10B98140' : '#EF444440',
      }));
      volumeSeries.setData(volumeData);
    }

    // SMA 20
    if (indicators.showSMA20) {
      const sma20Series = chart.addSeries(LineSeries, {
        color: '#3B82F6',
        lineWidth: 1,
        lastValueVisible: false,
        priceLineVisible: false,
      });
      sma20Series.setData(calculateSMA(data, 20));
    }

    // SMA 50
    if (indicators.showSMA50) {
      const sma50Series = chart.addSeries(LineSeries, {
        color: '#F97316',
        lineWidth: 1,
        lastValueVisible: false,
        priceLineVisible: false,
      });
      sma50Series.setData(calculateSMA(data, 50));
    }

    // Bollinger Bands
    if (indicators.showBollinger) {
      const bollinger = calculateBollinger(data);

      const bollingerUpper = chart.addSeries(LineSeries, {
        color: '#8B5CF6',
        lineWidth: 1,
        lineStyle: 2,
        lastValueVisible: false,
        priceLineVisible: false,
      });
      bollingerUpper.setData(bollinger.upper);

      const bollingerMiddle = chart.addSeries(LineSeries, {
        color: '#8B5CF6',
        lineWidth: 1,
        lastValueVisible: false,
        priceLineVisible: false,
      });
      bollingerMiddle.setData(bollinger.middle);

      const bollingerLower = chart.addSeries(LineSeries, {
        color: '#8B5CF6',
        lineWidth: 1,
        lineStyle: 2,
        lastValueVisible: false,
        priceLineVisible: false,
      });
      bollingerLower.setData(bollinger.lower);
    }

    // SAR overlay (on price chart) - use transparent line with visible markers
    if (indicatorData?.sar && indicatorData.sar.length > 0) {
      const sarSeries = chart.addSeries(LineSeries, {
        color: '#F59E0B00', // Transparent line
        lineWidth: 1,
        pointMarkersVisible: true,
        pointMarkersRadius: 3,
        lastValueVisible: false,
        priceLineVisible: false,
      });
      const sarData: LineData[] = indicatorData.sar
        .filter(d => d.value !== null)
        .map(d => ({
          time: toTime(d.date),
          value: d.value as number,
        }));
      sarSeries.setData(sarData);
    }

    // ZigZag overlay (on price chart)
    if (indicatorData?.zigzag && indicatorData.zigzag.length > 0) {
      const zigzagSeries = chart.addSeries(LineSeries, {
        color: '#EC4899',
        lineWidth: 2,
        lastValueVisible: false,
        priceLineVisible: false,
      });
      const zigzagData: LineData[] = indicatorData.zigzag
        .filter(d => d.value !== null)
        .map(d => ({
          time: toTime(d.date),
          value: d.value as number,
        }));
      zigzagSeries.setData(zigzagData);
    }

    // RSI pane (separate scale)
    if (indicatorData?.rsi && indicatorData.rsi.length > 0) {
      const rsiSeries = chart.addSeries(LineSeries, {
        color: '#8B5CF6',
        lineWidth: 1,
        priceScaleId: 'rsi',
        lastValueVisible: false,
        priceLineVisible: false,
      });
      chart.priceScale('rsi').applyOptions({
        scaleMargins: { top: 0.85, bottom: 0 },
        borderVisible: false,
      });
      const rsiData: LineData[] = indicatorData.rsi
        .filter(d => d.value !== null)
        .map(d => ({
          time: toTime(d.date),
          value: d.value as number,
        }));
      rsiSeries.setData(rsiData);

      // Add overbought/oversold lines
      const rsiUpperSeries = chart.addSeries(LineSeries, {
        color: '#EF444480',
        lineWidth: 1,
        lineStyle: 2,
        priceScaleId: 'rsi',
        lastValueVisible: false,
        priceLineVisible: false,
      });
      const rsiLowerSeries = chart.addSeries(LineSeries, {
        color: '#10B98180',
        lineWidth: 1,
        lineStyle: 2,
        priceScaleId: 'rsi',
        lastValueVisible: false,
        priceLineVisible: false,
      });
      // Create constant lines at 70 and 30
      const rsiDates = indicatorData.rsi.filter(d => d.value !== null);
      if (rsiDates.length >= 2) {
        rsiUpperSeries.setData([
          { time: toTime(rsiDates[0].date), value: 70 },
          { time: toTime(rsiDates[rsiDates.length - 1].date), value: 70 },
        ]);
        rsiLowerSeries.setData([
          { time: toTime(rsiDates[0].date), value: 30 },
          { time: toTime(rsiDates[rsiDates.length - 1].date), value: 30 },
        ]);
      }
    }

    // MACD pane (separate scale)
    if (indicatorData?.macd && indicatorData.macd.length > 0) {
      // MACD histogram
      const macdHistSeries = chart.addSeries(HistogramSeries, {
        color: '#6366F1',
        priceScaleId: 'macd',
        lastValueVisible: false,
        priceLineVisible: false,
      });
      chart.priceScale('macd').applyOptions({
        scaleMargins: { top: 0.92, bottom: 0 },
        borderVisible: false,
      });
      const histData: HistogramData[] = indicatorData.macd
        .filter(d => d.histogram !== null)
        .map(d => ({
          time: toTime(d.date),
          value: d.histogram as number,
          color: (d.histogram as number) >= 0 ? '#10B98180' : '#EF444480',
        }));
      macdHistSeries.setData(histData);

      // MACD line
      const macdLineSeries = chart.addSeries(LineSeries, {
        color: '#3B82F6',
        lineWidth: 1,
        priceScaleId: 'macd',
        lastValueVisible: false,
        priceLineVisible: false,
      });
      const macdLineData: LineData[] = indicatorData.macd
        .filter(d => d.macd !== null)
        .map(d => ({
          time: toTime(d.date),
          value: d.macd as number,
        }));
      macdLineSeries.setData(macdLineData);

      // Signal line
      const signalLineSeries = chart.addSeries(LineSeries, {
        color: '#F97316',
        lineWidth: 1,
        lineStyle: 2,
        priceScaleId: 'macd',
        lastValueVisible: false,
        priceLineVisible: false,
      });
      const signalLineData: LineData[] = indicatorData.macd
        .filter(d => d.signal !== null)
        .map(d => ({
          time: toTime(d.date),
          value: d.signal as number,
        }));
      signalLineSeries.setData(signalLineData);
    }

    // Fit content
    chart.timeScale().fitContent();

    // Handle resize
    const handleResize = () => {
      if (chartContainerRef.current && chartRef.current) {
        chartRef.current.applyOptions({ width: chartContainerRef.current.clientWidth });
      }
    };
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      if (chartRef.current) {
        chartRef.current.remove();
        chartRef.current = null;
      }
    };
    } catch (err) {
      console.error('TradingChart error:', err);
      setError(err instanceof Error ? err.message : String(err));
    }
  }, [data, indicators, height, toTime, calculateSMA, calculateBollinger, newsFrequencyByDate, trendData, calculatedTargets, indicatorData]);

  if (error) {
    return (
      <div
        className="bg-red-900/20 border border-red-500 rounded p-4 text-red-400"
        style={{ width: '100%', height: `${height}px` }}
      >
        <p className="font-bold">Chart Error:</p>
        <p className="font-mono text-sm">{error}</p>
        <p className="mt-2 text-gray-400">Data points: {data.length}</p>
      </div>
    );
  }

  return (
    <div
      ref={chartContainerRef}
      style={{ width: '100%', height: `${height}px` }}
    />
  );
};

export default TradingChart;
