"""
Technical Indicators Module

Calculates technical indicators for financial time series data.
Uses pandas and pandas_ta for indicator calculations.
"""

import pandas as pd
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """
    Technical indicators calculator for financial datasets.
    Provides common indicators like SMA, EMA, RSI, MACD, etc.
    """

    @staticmethod
    def calculate_sma(df: pd.DataFrame, column: str = 'Close', period: int = 20) -> pd.Series:
        """
        Calculate Simple Moving Average (SMA).

        Args:
            df: DataFrame with OHLC data
            column: Column name to calculate SMA on (default: 'Close')
            period: Period for moving average (default: 20)

        Returns:
            Series with SMA values
        """
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")

        sma = df[column].rolling(window=period, min_periods=period).mean()
        logger.debug(f"Calculated SMA({period}) on {column}")
        return sma

    @staticmethod
    def calculate_ema(df: pd.DataFrame, column: str = 'Close', period: int = 20) -> pd.Series:
        """
        Calculate Exponential Moving Average (EMA).

        Args:
            df: DataFrame with OHLC data
            column: Column name to calculate EMA on (default: 'Close')
            period: Period for exponential average (default: 20)

        Returns:
            Series with EMA values
        """
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")

        ema = df[column].ewm(span=period, adjust=False).mean()
        logger.debug(f"Calculated EMA({period}) on {column}")
        return ema

    @staticmethod
    def calculate_rsi(df: pd.DataFrame, column: str = 'Close', period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).

        Args:
            df: DataFrame with OHLC data
            column: Column name to calculate RSI on (default: 'Close')
            period: Period for RSI calculation (default: 14)

        Returns:
            Series with RSI values (0-100)
        """
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")

        # Calculate price changes
        delta = df[column].diff()

        # Separate gains and losses
        gains = delta.where(delta > 0, 0.0)
        losses = -delta.where(delta < 0, 0.0)

        # Calculate average gains and losses
        avg_gains = gains.rolling(window=period, min_periods=period).mean()
        avg_losses = losses.rolling(window=period, min_periods=period).mean()

        # Calculate RS and RSI
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))

        logger.debug(f"Calculated RSI({period}) on {column}")
        return rsi

    @staticmethod
    def calculate_macd(
        df: pd.DataFrame,
        column: str = 'Close',
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9
    ) -> Dict[str, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence).

        Args:
            df: DataFrame with OHLC data
            column: Column name to calculate MACD on (default: 'Close')
            fast_period: Fast EMA period (default: 12)
            slow_period: Slow EMA period (default: 26)
            signal_period: Signal line EMA period (default: 9)

        Returns:
            Dictionary with 'macd', 'signal', and 'histogram' Series
        """
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")

        # Calculate fast and slow EMAs
        fast_ema = df[column].ewm(span=fast_period, adjust=False).mean()
        slow_ema = df[column].ewm(span=slow_period, adjust=False).mean()

        # Calculate MACD line
        macd_line = fast_ema - slow_ema

        # Calculate signal line
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()

        # Calculate histogram
        histogram = macd_line - signal_line

        logger.debug(f"Calculated MACD({fast_period},{slow_period},{signal_period}) on {column}")

        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }

    @staticmethod
    def calculate_bollinger_bands(
        df: pd.DataFrame,
        column: str = 'Close',
        period: int = 20,
        std_dev: float = 2.0
    ) -> Dict[str, pd.Series]:
        """
        Calculate Bollinger Bands.

        Args:
            df: DataFrame with OHLC data
            column: Column name to calculate bands on (default: 'Close')
            period: Period for moving average (default: 20)
            std_dev: Number of standard deviations (default: 2.0)

        Returns:
            Dictionary with 'upper', 'middle', and 'lower' Series
        """
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")

        # Calculate middle band (SMA)
        middle_band = df[column].rolling(window=period, min_periods=period).mean()

        # Calculate standard deviation
        std = df[column].rolling(window=period, min_periods=period).std()

        # Calculate upper and lower bands
        upper_band = middle_band + (std * std_dev)
        lower_band = middle_band - (std * std_dev)

        logger.debug(f"Calculated Bollinger Bands({period},{std_dev}) on {column}")

        return {
            'upper': upper_band,
            'middle': middle_band,
            'lower': lower_band
        }

    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range (ATR).

        Args:
            df: DataFrame with OHLC data (must have High, Low, Close)
            period: Period for ATR calculation (default: 14)

        Returns:
            Series with ATR values
        """
        required_columns = ['High', 'Low', 'Close']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame. ATR requires High, Low, Close.")

        # Calculate true range components
        high_low = df['High'] - df['Low']
        high_close = (df['High'] - df['Close'].shift()).abs()
        low_close = (df['Low'] - df['Close'].shift()).abs()

        # True range is the maximum of the three
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        # ATR is the moving average of true range
        atr = true_range.rolling(window=period, min_periods=period).mean()

        logger.debug(f"Calculated ATR({period})")
        return atr

    @staticmethod
    def calculate_stochastic(
        df: pd.DataFrame,
        k_period: int = 14,
        d_period: int = 3,
        smooth_k: int = 3
    ) -> Dict[str, pd.Series]:
        """
        Calculate Stochastic Oscillator (%K and %D).

        Args:
            df: DataFrame with OHLC data (must have High, Low, Close)
            k_period: Period for %K calculation (default: 14)
            d_period: Period for %D (SMA of %K) (default: 3)
            smooth_k: Smoothing period for %K (default: 3)

        Returns:
            Dictionary with 'k' (%K) and 'd' (%D) Series (values 0-100)
        """
        required_columns = ['High', 'Low', 'Close']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame. Stochastic requires High, Low, Close.")

        # Calculate rolling high and low
        low_min = df['Low'].rolling(window=k_period, min_periods=k_period).min()
        high_max = df['High'].rolling(window=k_period, min_periods=k_period).max()

        # Calculate raw %K (Fast Stochastic)
        raw_k = 100 * (df['Close'] - low_min) / (high_max - low_min)

        # Smooth %K if smooth_k > 1 (Slow Stochastic)
        if smooth_k > 1:
            k = raw_k.rolling(window=smooth_k, min_periods=smooth_k).mean()
        else:
            k = raw_k

        # Calculate %D (SMA of %K)
        d = k.rolling(window=d_period, min_periods=d_period).mean()

        logger.debug(f"Calculated Stochastic({k_period},{d_period},{smooth_k})")

        return {
            'k': k,
            'd': d
        }

    @staticmethod
    def add_indicators_to_dataframe(
        df: pd.DataFrame,
        indicators_config: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Add multiple technical indicators to a DataFrame.

        Args:
            df: DataFrame with OHLC data
            indicators_config: Dictionary with indicator configurations
                Example: {
                    'sma_20': {'type': 'sma', 'period': 20},
                    'ema_50': {'type': 'ema', 'period': 50},
                    'rsi_14': {'type': 'rsi', 'period': 14},
                    'macd': {'type': 'macd', 'fast': 12, 'slow': 26, 'signal': 9}
                }

        Returns:
            DataFrame with added indicator columns
        """
        df_copy = df.copy()

        for indicator_name, config in indicators_config.items():
            indicator_type = config.get('type', '').lower()

            try:
                if indicator_type == 'sma':
                    period = config.get('period', 20)
                    column = config.get('column', 'Close')
                    df_copy[indicator_name] = TechnicalIndicators.calculate_sma(df_copy, column, period)

                elif indicator_type == 'ema':
                    period = config.get('period', 20)
                    column = config.get('column', 'Close')
                    df_copy[indicator_name] = TechnicalIndicators.calculate_ema(df_copy, column, period)

                elif indicator_type == 'rsi':
                    period = config.get('period', 14)
                    column = config.get('column', 'Close')
                    df_copy[indicator_name] = TechnicalIndicators.calculate_rsi(df_copy, column, period)

                elif indicator_type == 'macd':
                    fast = config.get('fast', 12)
                    slow = config.get('slow', 26)
                    signal = config.get('signal', 9)
                    column = config.get('column', 'Close')
                    macd_data = TechnicalIndicators.calculate_macd(df_copy, column, fast, slow, signal)
                    df_copy[f'{indicator_name}_line'] = macd_data['macd']
                    df_copy[f'{indicator_name}_signal'] = macd_data['signal']
                    df_copy[f'{indicator_name}_histogram'] = macd_data['histogram']

                elif indicator_type == 'bollinger' or indicator_type == 'bbands':
                    period = config.get('period', 20)
                    std_dev = config.get('std_dev', 2.0)
                    column = config.get('column', 'Close')
                    bb_data = TechnicalIndicators.calculate_bollinger_bands(df_copy, column, period, std_dev)
                    df_copy[f'{indicator_name}_upper'] = bb_data['upper']
                    df_copy[f'{indicator_name}_middle'] = bb_data['middle']
                    df_copy[f'{indicator_name}_lower'] = bb_data['lower']

                elif indicator_type == 'atr':
                    period = config.get('period', 14)
                    df_copy[indicator_name] = TechnicalIndicators.calculate_atr(df_copy, period)

                elif indicator_type == 'stochastic' or indicator_type == 'stoch':
                    k_period = config.get('k_period', 14)
                    d_period = config.get('d_period', 3)
                    smooth_k = config.get('smooth_k', 3)
                    stoch_data = TechnicalIndicators.calculate_stochastic(df_copy, k_period, d_period, smooth_k)
                    df_copy[f'{indicator_name}_k'] = stoch_data['k']
                    df_copy[f'{indicator_name}_d'] = stoch_data['d']

                else:
                    logger.warning(f"Unknown indicator type: {indicator_type}")

            except Exception as e:
                logger.error(f"Error calculating indicator {indicator_name}: {e}")

        return df_copy
