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
