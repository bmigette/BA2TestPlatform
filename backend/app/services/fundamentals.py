"""
Fundamentals Service

Fetches fundamental data (FCF, P/E, EPS, Revenue) for tickers and creates
derived features for ML model training.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging
import yfinance as yf

logger = logging.getLogger(__name__)


class FundamentalsService:
    """
    Service for fetching and processing fundamental financial data.

    Creates features like:
    - days_to_last_FCF
    - last_FCF value
    - last_FCF_percent change
    - days_to_next_FCF (estimated)
    - next_FCF_forecast (estimated)
    """

    # Fundamental metrics to fetch
    FUNDAMENTAL_METRICS = [
        'FreeCashFlow',
        'TrailingPE',
        'ForwardPE',
        'EarningsPerShare',
        'TotalRevenue',
        'DebtToEquity',
        'ReturnOnEquity',
        'PriceToBook',
        'DividendYield'
    ]

    @staticmethod
    def get_fundamental_data(ticker: str) -> Dict[str, Any]:
        """
        Fetch fundamental data for a ticker using yfinance.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary with fundamental metrics and dates
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            # Get quarterly financials for historical data
            quarterly_cashflow = stock.quarterly_cashflow
            quarterly_income = stock.quarterly_income_stmt

            fundamentals = {
                'ticker': ticker,
                'fetch_date': datetime.now().isoformat(),
                'current': {},
                'historical': {}
            }

            # Current values from info
            fundamentals['current'] = {
                'free_cash_flow': info.get('freeCashflow'),
                'trailing_pe': info.get('trailingPE'),
                'forward_pe': info.get('forwardPE'),
                'eps': info.get('trailingEps'),
                'forward_eps': info.get('forwardEps'),
                'revenue': info.get('totalRevenue'),
                'debt_to_equity': info.get('debtToEquity'),
                'roe': info.get('returnOnEquity'),
                'price_to_book': info.get('priceToBook'),
                'dividend_yield': info.get('dividendYield'),
                'market_cap': info.get('marketCap')
            }

            # Historical quarterly data
            if not quarterly_cashflow.empty:
                fcf_data = []
                if 'Free Cash Flow' in quarterly_cashflow.index:
                    for date in quarterly_cashflow.columns:
                        value = quarterly_cashflow.loc['Free Cash Flow', date]
                        if pd.notna(value):
                            fcf_data.append({
                                'date': date.isoformat() if hasattr(date, 'isoformat') else str(date),
                                'value': float(value)
                            })
                fundamentals['historical']['free_cash_flow'] = fcf_data

            if not quarterly_income.empty:
                # Revenue history
                revenue_data = []
                if 'Total Revenue' in quarterly_income.index:
                    for date in quarterly_income.columns:
                        value = quarterly_income.loc['Total Revenue', date]
                        if pd.notna(value):
                            revenue_data.append({
                                'date': date.isoformat() if hasattr(date, 'isoformat') else str(date),
                                'value': float(value)
                            })
                fundamentals['historical']['revenue'] = revenue_data

                # EPS history
                eps_data = []
                if 'Basic EPS' in quarterly_income.index:
                    for date in quarterly_income.columns:
                        value = quarterly_income.loc['Basic EPS', date]
                        if pd.notna(value):
                            eps_data.append({
                                'date': date.isoformat() if hasattr(date, 'isoformat') else str(date),
                                'value': float(value)
                            })
                fundamentals['historical']['eps'] = eps_data

            logger.info(f"Fetched fundamental data for {ticker}")
            return fundamentals

        except Exception as e:
            logger.error(f"Error fetching fundamentals for {ticker}: {e}")
            raise

    @staticmethod
    def create_fundamental_features(
        df: pd.DataFrame,
        ticker: str,
        metrics: List[str] = None
    ) -> pd.DataFrame:
        """
        Create fundamental-derived features for a dataset.

        For each metric, creates:
        - days_to_last_{metric}: Days since last reported value
        - last_{metric}: Most recent value
        - last_{metric}_percent: Percent change from previous period
        - days_to_next_{metric}: Estimated days to next report
        - next_{metric}_forecast: Simple forecast based on trend

        Args:
            df: DataFrame with Date column
            ticker: Stock ticker
            metrics: List of metrics to process (default: ['fcf', 'pe', 'eps', 'revenue'])

        Returns:
            DataFrame with added fundamental features
        """
        if metrics is None:
            metrics = ['fcf', 'pe', 'eps', 'revenue']

        result_df = df.copy()

        # Ensure Date is datetime
        if 'Date' in result_df.columns:
            result_df['Date'] = pd.to_datetime(result_df['Date'])

        try:
            # Fetch fundamental data
            fund_data = FundamentalsService.get_fundamental_data(ticker)

            # Process each metric
            for metric in metrics:
                result_df = FundamentalsService._add_metric_features(
                    result_df, fund_data, metric
                )

            logger.info(f"Created fundamental features for {len(metrics)} metrics")
            return result_df

        except Exception as e:
            logger.error(f"Error creating fundamental features: {e}")
            # Return original dataframe with NaN features
            for metric in metrics:
                result_df[f'days_to_last_{metric}'] = np.nan
                result_df[f'last_{metric}'] = np.nan
                result_df[f'last_{metric}_percent'] = np.nan
                result_df[f'days_to_next_{metric}'] = np.nan
                result_df[f'next_{metric}_forecast'] = np.nan
            return result_df

    @staticmethod
    def _add_metric_features(
        df: pd.DataFrame,
        fund_data: Dict,
        metric: str
    ) -> pd.DataFrame:
        """
        Add features for a specific fundamental metric.

        Args:
            df: DataFrame with Date column
            fund_data: Fundamental data dictionary
            metric: Metric name (fcf, pe, eps, revenue)

        Returns:
            DataFrame with added features
        """
        result_df = df.copy()

        # Map metric names to data keys
        metric_map = {
            'fcf': 'free_cash_flow',
            'pe': 'trailing_pe',
            'eps': 'eps',
            'revenue': 'revenue',
            'de': 'debt_to_equity',
            'roe': 'roe'
        }

        data_key = metric_map.get(metric, metric)

        # Get historical data if available
        historical = fund_data.get('historical', {}).get(data_key, [])
        current_value = fund_data.get('current', {}).get(data_key)

        if not historical and current_value is None:
            # No data available
            result_df[f'days_to_last_{metric}'] = np.nan
            result_df[f'last_{metric}'] = np.nan
            result_df[f'last_{metric}_percent'] = np.nan
            result_df[f'days_to_next_{metric}'] = np.nan
            result_df[f'next_{metric}_forecast'] = np.nan
            return result_df

        # Process historical data
        if historical:
            # Sort by date
            historical = sorted(historical, key=lambda x: x['date'], reverse=True)

            # Create a series for looking up values by date
            dates = [pd.to_datetime(h['date']) for h in historical]
            values = [h['value'] for h in historical]

            # Calculate percent changes
            percent_changes = [0]  # First value has no previous
            for i in range(1, len(values)):
                if values[i] != 0:
                    pct = (values[i-1] - values[i]) / abs(values[i]) * 100
                    percent_changes.append(pct)
                else:
                    percent_changes.append(0)

            # For each date in the dataset, find the most recent fundamental date
            def get_days_since_last(row_date):
                for i, d in enumerate(dates):
                    if d <= row_date:
                        return (row_date - d).days
                return np.nan

            def get_last_value(row_date):
                for i, d in enumerate(dates):
                    if d <= row_date:
                        return values[i]
                return np.nan

            def get_last_percent(row_date):
                for i, d in enumerate(dates):
                    if d <= row_date:
                        return percent_changes[i]
                return np.nan

            def get_days_to_next(row_date):
                # Estimate based on quarterly reports (90 days)
                last_days = get_days_since_last(row_date)
                if pd.isna(last_days):
                    return np.nan
                return max(0, 90 - last_days)

            def get_next_forecast(row_date):
                # Simple forecast: last value * (1 + average percent change)
                last_val = get_last_value(row_date)
                if pd.isna(last_val):
                    return np.nan
                avg_change = np.mean([p for p in percent_changes if p != 0])
                if pd.isna(avg_change):
                    return last_val
                return last_val * (1 + avg_change / 100)

            result_df[f'days_to_last_{metric}'] = result_df['Date'].apply(get_days_since_last)
            result_df[f'last_{metric}'] = result_df['Date'].apply(get_last_value)
            result_df[f'last_{metric}_percent'] = result_df['Date'].apply(get_last_percent)
            result_df[f'days_to_next_{metric}'] = result_df['Date'].apply(get_days_to_next)
            result_df[f'next_{metric}_forecast'] = result_df['Date'].apply(get_next_forecast)

        else:
            # Only current value available
            result_df[f'days_to_last_{metric}'] = np.nan
            result_df[f'last_{metric}'] = current_value
            result_df[f'last_{metric}_percent'] = np.nan
            result_df[f'days_to_next_{metric}'] = np.nan
            result_df[f'next_{metric}_forecast'] = np.nan

        return result_df
