"""Tests for multi-dataset training: compatibility checker, data preparation, training."""
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def make_synthetic_dataset(ticker: str, n_rows: int = 200, seed: int = 42) -> pd.DataFrame:
    """Create a synthetic OHLCV dataset with indicators for testing."""
    rng = np.random.RandomState(seed)
    dates = pd.date_range('2023-01-01', periods=n_rows, freq='h')
    close = 100 + rng.randn(n_rows).cumsum()
    df = pd.DataFrame({
        'Date': dates,
        'Open': close + rng.randn(n_rows) * 0.5,
        'High': close + abs(rng.randn(n_rows)),
        'Low': close - abs(rng.randn(n_rows)),
        'Close': close,
        'Volume': (rng.rand(n_rows) * 1e6).astype(int),
        'SMA_20': close + rng.randn(n_rows) * 2,
        'RSI_14': 50 + rng.randn(n_rows) * 15,
    })
    return df


class TestDatasetCompatibility:
    """Tests for dataset compatibility checking."""

    def test_compatible_datasets(self):
        """Datasets with same columns in same order are compatible."""
        from app.api.datasets import check_dataset_compatibility
        df1 = make_synthetic_dataset('AAPL', seed=1)
        df2 = make_synthetic_dataset('MSFT', seed=2)
        result = check_dataset_compatibility([df1, df2])
        assert result['compatible'] is True

    def test_incompatible_columns(self):
        """Datasets with different columns are incompatible."""
        from app.api.datasets import check_dataset_compatibility
        df1 = make_synthetic_dataset('AAPL')
        df2 = make_synthetic_dataset('MSFT')
        df2['ExtraIndicator'] = 0
        result = check_dataset_compatibility([df1, df2])
        assert result['compatible'] is False
        assert 'ExtraIndicator' in result['message']

    def test_incompatible_column_order(self):
        """Datasets with same columns but different order are incompatible."""
        from app.api.datasets import check_dataset_compatibility
        df1 = make_synthetic_dataset('AAPL')
        df2 = make_synthetic_dataset('MSFT')
        df2 = df2[list(reversed(df2.columns))]
        result = check_dataset_compatibility([df1, df2])
        assert result['compatible'] is False

    def test_single_dataset_always_compatible(self):
        """A single dataset is always compatible with itself."""
        from app.api.datasets import check_dataset_compatibility
        df1 = make_synthetic_dataset('AAPL')
        result = check_dataset_compatibility([df1])
        assert result['compatible'] is True


from app.services.darts_training import DartsTrainingService, DARTS_AVAILABLE


class TestDartsMultiSeries:
    """Tests for Darts multi-series data preparation."""

    @pytest.fixture
    def service(self):
        return DartsTrainingService()

    @pytest.mark.skipif(not DARTS_AVAILABLE, reason="darts not available")
    def test_prepare_multi_series(self, service):
        """prepare_multi_series returns list of TimeSeries."""
        dfs = [make_synthetic_dataset('AAPL', seed=1), make_synthetic_dataset('MSFT', seed=2)]
        series_list, cov_list = service.prepare_multi_series(
            dfs, target_column='Close', feature_columns=['SMA_20', 'RSI_14'], timeframe='1h'
        )
        assert len(series_list) == 2
        assert len(cov_list) == 2

    @pytest.mark.skipif(not DARTS_AVAILABLE, reason="darts not available")
    def test_prepare_multi_series_split(self, service):
        """prepare_multi_series_split returns train/test lists."""
        dfs = [make_synthetic_dataset('AAPL', seed=1), make_synthetic_dataset('MSFT', seed=2)]
        train_s, test_s, train_c, test_c = service.prepare_multi_series_split(
            dfs, train_ratio=0.8, target_column='Close',
            feature_columns=['SMA_20', 'RSI_14'], timeframe='1h'
        )
        assert len(train_s) == 2
        assert len(test_s) == 2
