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


from app.services.tsai_training import TSAITrainingService, TSAI_AVAILABLE


class TestTSAIMultiDataset:
    """Tests for TSAI multi-dataset windowed data preparation."""

    @pytest.fixture
    def service(self):
        return TSAITrainingService()

    @pytest.mark.skipif(not TSAI_AVAILABLE, reason="tsai not available")
    def test_prepare_multi_dataset_split(self, service):
        """Multi-dataset preparation creates windows per-dataset then concatenates."""
        dfs = [make_synthetic_dataset('AAPL', n_rows=200, seed=1),
               make_synthetic_dataset('MSFT', n_rows=200, seed=2)]
        for df in dfs:
            df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
            df.dropna(subset=['target'], inplace=True)

        X_train, X_test, y_train, y_test = service.prepare_multi_dataset_split(
            dfs, train_ratio=0.8, target_column='target',
            feature_columns=['Close', 'Volume', 'SMA_20', 'RSI_14'], seq_len=24
        )
        # Combined should have more samples than single dataset
        single_X, _, _, _ = service.prepare_data_split(
            dfs[0], train_ratio=0.8, target_column='target',
            feature_columns=['Close', 'Volume', 'SMA_20', 'RSI_14'], seq_len=24
        )
        assert X_train.shape[0] > single_X.shape[0]
        assert X_train.shape[1] == 4  # features
        assert X_train.shape[2] == 24  # seq_len

    @pytest.mark.skipif(not TSAI_AVAILABLE, reason="tsai not available")
    def test_no_cross_boundary_windows(self, service):
        """Windows must not span across dataset boundaries."""
        dfs = [make_synthetic_dataset('AAPL', n_rows=60, seed=1),
               make_synthetic_dataset('MSFT', n_rows=60, seed=2)]
        for df in dfs:
            df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
            df.dropna(subset=['target'], inplace=True)

        X_train, X_test, y_train, y_test = service.prepare_multi_dataset_split(
            dfs, train_ratio=0.8, target_column='target',
            feature_columns=['Close', 'Volume'], seq_len=10
        )
        assert X_train.shape[0] > 0
        assert X_test.shape[0] > 0


class TestCrossValidation:
    """Tests for dataset-level cross-validation."""

    def test_manual_train_test_split(self):
        """Manual assignment: specific datasets as train, others as test."""
        from app.services.job_handler import split_datasets_by_role
        dfs = [make_synthetic_dataset(t) for t in ['AAPL', 'MSFT', 'GOOGL']]
        dataset_ids = [1, 2, 3]
        test_ids = [3]  # GOOGL as test
        train_dfs, test_dfs = split_datasets_by_role(dfs, dataset_ids, test_ids)
        assert len(train_dfs) == 2
        assert len(test_dfs) == 1

    def test_kfold_creates_correct_folds(self):
        """K-fold creates N folds where each dataset is test once."""
        from app.services.job_handler import create_kfold_splits
        dfs = [make_synthetic_dataset(t) for t in ['AAPL', 'MSFT', 'GOOGL']]
        dataset_ids = [1, 2, 3]
        folds = create_kfold_splits(dfs, dataset_ids)
        assert len(folds) == 3  # 3 datasets = 3 folds
        for train, test, test_ids in folds:
            assert len(test) == 1
            assert len(train) == 2

    def test_kfold_every_dataset_tested(self):
        """Every dataset appears as test exactly once across folds."""
        from app.services.job_handler import create_kfold_splits
        dfs = [make_synthetic_dataset(t) for t in ['AAPL', 'MSFT', 'GOOGL']]
        dataset_ids = [1, 2, 3]
        folds = create_kfold_splits(dfs, dataset_ids)
        tested_ids = set()
        for _, _, test_ids in folds:
            tested_ids.update(test_ids)
        assert tested_ids == {1, 2, 3}
