"""
Technical Indicator Service

Provides calculations for technical indicators used in prediction targets
and chart visualization.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class IndicatorService:
    """
    Calculate technical indicators for datasets.

    All methods expect a DataFrame with OHLC columns (Open, High, Low, Close)
    and return calculated indicator values as Series or Dict of Series.
    """

    def calculate_rsi(
        self,
        df: pd.DataFrame,
        period: int = 14
    ) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).

        RSI measures the speed and magnitude of recent price changes
        to evaluate overbought or oversold conditions.

        Args:
            df: DataFrame with 'Close' column
            period: RSI period (default 14)

        Returns:
            Series with RSI values (0-100 scale)
        """
        if 'Close' not in df.columns:
            raise ValueError("DataFrame must have 'Close' column")

        close = df['Close'].values
        n = len(close)

        # Calculate price changes
        delta = np.diff(close, prepend=close[0])

        # Separate gains and losses
        gains = np.where(delta > 0, delta, 0)
        losses = np.where(delta < 0, -delta, 0)

        # Calculate smoothed averages using Wilder's method
        avg_gain = np.zeros(n)
        avg_loss = np.zeros(n)

        # Initial SMA for first period
        if n >= period:
            avg_gain[period - 1] = np.mean(gains[1:period + 1])
            avg_loss[period - 1] = np.mean(losses[1:period + 1])

            # Subsequent values use exponential smoothing
            for i in range(period, n):
                avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gains[i]) / period
                avg_loss[i] = (avg_loss[i - 1] * (period - 1) + losses[i]) / period

        # Calculate RS and RSI
        rsi = np.full(n, np.nan)
        for i in range(period - 1, n):
            if avg_loss[i] == 0:
                rsi[i] = 100.0
            else:
                rs = avg_gain[i] / avg_loss[i]
                rsi[i] = 100.0 - (100.0 / (1.0 + rs))

        return pd.Series(rsi, index=df.index, name=f'rsi_{period}')

    def calculate_macd(
        self,
        df: pd.DataFrame,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Dict[str, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence).

        MACD shows the relationship between two EMAs of price,
        with a signal line for trade signals.

        Args:
            df: DataFrame with 'Close' column
            fast: Fast EMA period (default 12)
            slow: Slow EMA period (default 26)
            signal: Signal line EMA period (default 9)

        Returns:
            Dict with 'macd', 'signal', 'histogram' Series
        """
        if 'Close' not in df.columns:
            raise ValueError("DataFrame must have 'Close' column")

        close = df['Close']

        # Calculate EMAs
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()

        # MACD line = Fast EMA - Slow EMA
        macd_line = ema_fast - ema_slow

        # Signal line = EMA of MACD line
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()

        # Histogram = MACD - Signal
        histogram = macd_line - signal_line

        return {
            'macd': pd.Series(macd_line.values, index=df.index, name=f'macd_{fast}_{slow}_{signal}'),
            'signal': pd.Series(signal_line.values, index=df.index, name=f'macd_signal_{fast}_{slow}_{signal}'),
            'histogram': pd.Series(histogram.values, index=df.index, name=f'macd_hist_{fast}_{slow}_{signal}')
        }

    def calculate_sar(
        self,
        df: pd.DataFrame,
        af_start: float = 0.02,
        af_max: float = 0.2
    ) -> pd.Series:
        """
        Calculate Parabolic SAR (Stop and Reverse).

        SAR provides potential entry and exit points, appearing as
        dots above or below price.

        Args:
            df: DataFrame with 'High' and 'Low' columns
            af_start: Starting acceleration factor (default 0.02)
            af_max: Maximum acceleration factor (default 0.2)

        Returns:
            Series with SAR values (price scale)
        """
        if 'High' not in df.columns or 'Low' not in df.columns:
            raise ValueError("DataFrame must have 'High' and 'Low' columns")

        high = df['High'].values
        low = df['Low'].values
        n = len(high)

        if n < 2:
            return pd.Series(np.full(n, np.nan), index=df.index, name='sar')

        sar = np.zeros(n)
        af = af_start
        is_uptrend = True

        # Initialize
        ep = high[0]  # Extreme point
        sar[0] = low[0]

        for i in range(1, n):
            # Calculate SAR for current bar
            sar[i] = sar[i - 1] + af * (ep - sar[i - 1])

            if is_uptrend:
                # In uptrend, SAR cannot be above prior two lows
                sar[i] = min(sar[i], low[i - 1])
                if i >= 2:
                    sar[i] = min(sar[i], low[i - 2])

                # Check for reversal
                if low[i] < sar[i]:
                    is_uptrend = False
                    sar[i] = ep
                    ep = low[i]
                    af = af_start
                else:
                    # Update extreme point and AF
                    if high[i] > ep:
                        ep = high[i]
                        af = min(af + af_start, af_max)
            else:
                # In downtrend, SAR cannot be below prior two highs
                sar[i] = max(sar[i], high[i - 1])
                if i >= 2:
                    sar[i] = max(sar[i], high[i - 2])

                # Check for reversal
                if high[i] > sar[i]:
                    is_uptrend = True
                    sar[i] = ep
                    ep = high[i]
                    af = af_start
                else:
                    # Update extreme point and AF
                    if low[i] < ep:
                        ep = low[i]
                        af = min(af + af_start, af_max)

        return pd.Series(sar, index=df.index, name=f'sar_{af_start}_{af_max}')

    def calculate_zigzag(
        self,
        df: pd.DataFrame,
        deviation_pct: float = 5.0
    ) -> pd.Series:
        """
        Calculate ZigZag indicator.

        ZigZag connects significant swing highs and lows, filtering
        out smaller price movements below the deviation threshold.

        Args:
            df: DataFrame with 'High' and 'Low' columns
            deviation_pct: Minimum percentage move to form new pivot (default 5.0)

        Returns:
            Series with ZigZag values (NaN between pivots, price at pivots)
        """
        if 'High' not in df.columns or 'Low' not in df.columns:
            raise ValueError("DataFrame must have 'High' and 'Low' columns")

        high = df['High'].values
        low = df['Low'].values
        n = len(high)

        if n < 2:
            return pd.Series(np.full(n, np.nan), index=df.index, name='zigzag')

        zigzag = np.full(n, np.nan)
        pivots = []  # List of (index, price, type) where type is 'high' or 'low'

        deviation = deviation_pct / 100.0

        # Find initial direction
        first_high_idx = 0
        first_low_idx = 0

        # Start with first bar
        last_pivot_type = None
        last_pivot_idx = 0
        last_pivot_price = (high[0] + low[0]) / 2

        # Determine initial trend by looking at first significant move
        for i in range(1, min(n, 20)):
            high_change = (high[i] - low[0]) / low[0]
            low_change = (high[0] - low[i]) / high[0]

            if high_change >= deviation:
                last_pivot_type = 'low'
                last_pivot_idx = 0
                last_pivot_price = low[0]
                pivots.append((0, low[0], 'low'))
                break
            elif low_change >= deviation:
                last_pivot_type = 'high'
                last_pivot_idx = 0
                last_pivot_price = high[0]
                pivots.append((0, high[0], 'high'))
                break

        if last_pivot_type is None:
            # No significant move found, use first bar high
            last_pivot_type = 'high'
            last_pivot_idx = 0
            last_pivot_price = high[0]
            pivots.append((0, high[0], 'high'))

        # Scan for pivots
        for i in range(1, n):
            if last_pivot_type == 'high':
                # Looking for a low pivot
                if low[i] < last_pivot_price * (1 - deviation):
                    # Found significant low
                    pivots.append((i, low[i], 'low'))
                    last_pivot_type = 'low'
                    last_pivot_idx = i
                    last_pivot_price = low[i]
                elif high[i] > last_pivot_price:
                    # Extend the high pivot
                    pivots[-1] = (i, high[i], 'high')
                    last_pivot_idx = i
                    last_pivot_price = high[i]
            else:
                # Looking for a high pivot
                if high[i] > last_pivot_price * (1 + deviation):
                    # Found significant high
                    pivots.append((i, high[i], 'high'))
                    last_pivot_type = 'high'
                    last_pivot_idx = i
                    last_pivot_price = high[i]
                elif low[i] < last_pivot_price:
                    # Extend the low pivot
                    pivots[-1] = (i, low[i], 'low')
                    last_pivot_idx = i
                    last_pivot_price = low[i]

        # Fill zigzag array at pivot points
        for idx, price, _ in pivots:
            zigzag[idx] = price

        # Interpolate between pivots for visualization
        zigzag_interp = np.copy(zigzag)
        for i in range(len(pivots) - 1):
            start_idx, start_price, _ = pivots[i]
            end_idx, end_price, _ = pivots[i + 1]

            if end_idx > start_idx:
                for j in range(start_idx, end_idx + 1):
                    t = (j - start_idx) / (end_idx - start_idx)
                    zigzag_interp[j] = start_price + t * (end_price - start_price)

        return pd.Series(zigzag_interp, index=df.index, name=f'zigzag_{deviation_pct}')

    def calculate_indicators(
        self,
        df: pd.DataFrame,
        indicators: list
    ) -> Dict[str, Any]:
        """
        Calculate multiple indicators at once.

        Args:
            df: DataFrame with OHLC columns
            indicators: List of indicator configs, e.g.:
                [
                    {"type": "rsi", "period": 14},
                    {"type": "macd", "fast": 12, "slow": 26, "signal": 9}
                ]

        Returns:
            Dict with indicator names as keys and Series/Dict as values
        """
        results = {}

        for ind in indicators:
            ind_type = ind.get('type', '').lower()

            try:
                if ind_type == 'rsi':
                    period = ind.get('period', 14)
                    results[f'rsi_{period}'] = self.calculate_rsi(df, period)

                elif ind_type == 'macd':
                    fast = ind.get('fast', 12)
                    slow = ind.get('slow', 26)
                    signal = ind.get('signal', 9)
                    macd_data = self.calculate_macd(df, fast, slow, signal)
                    results[f'macd_{fast}_{slow}_{signal}'] = macd_data['macd']
                    results[f'macd_signal_{fast}_{slow}_{signal}'] = macd_data['signal']
                    results[f'macd_hist_{fast}_{slow}_{signal}'] = macd_data['histogram']

                elif ind_type == 'sar':
                    af_start = ind.get('af_start', 0.02)
                    af_max = ind.get('af_max', 0.2)
                    results[f'sar_{af_start}_{af_max}'] = self.calculate_sar(df, af_start, af_max)

                elif ind_type == 'zigzag':
                    deviation_pct = ind.get('deviation_pct', 5.0)
                    results[f'zigzag_{deviation_pct}'] = self.calculate_zigzag(df, deviation_pct)

                else:
                    logger.warning(f"Unknown indicator type: {ind_type}")

            except Exception as e:
                logger.error(f"Error calculating {ind_type}: {e}")
                raise

        return results
