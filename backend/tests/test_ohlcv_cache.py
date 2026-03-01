"""
Tests for OHLCV cache handler and cache-related endpoints.
"""

import pytest
from unittest.mock import patch, MagicMock, PropertyMock
from datetime import datetime
import pandas as pd
import os
import tempfile


class TestHandleOHLCVCacheFetch:
    """Tests for handle_ohlcv_cache_fetch handler."""

    @pytest.fixture
    def mock_task_queue(self):
        """Create a mock task queue."""
        tq = MagicMock()
        tq.update_progress = MagicMock()
        return tq

    @pytest.fixture
    def mock_provider(self):
        """Create a mock OHLCV provider."""
        provider = MagicMock()
        provider.get_ohlcv_data.return_value = pd.DataFrame({
            'Date': pd.date_range('2020-01-01', periods=100),
            'Open': range(100),
            'High': range(100),
            'Low': range(100),
            'Close': range(100),
            'Volume': range(100)
        })
        return provider

    def test_successful_cache_fetch(self, mock_task_queue, mock_provider):
        """Test successful OHLCV cache fetch for a symbol."""
        with patch('app.services.ohlcv_cache_handler.get_task_queue', return_value=mock_task_queue), \
             patch('app.api.datasets.get_ohlcv_provider', return_value=mock_provider):
            from app.services.ohlcv_cache_handler import handle_ohlcv_cache_fetch

            result = handle_ohlcv_cache_fetch('task-123', {
                'provider': 'yfinance',
                'symbol': 'AAPL',
                'timeframes': ['1d', '1h']
            })

            assert result['status'] == 'completed'
            assert result['symbol'] == 'AAPL'
            assert '1d' in result['results']
            assert '1h' in result['results']
            assert result['results']['1d']['status'] == 'success'
            assert result['results']['1d']['rows'] == 100

    def test_cache_fetch_with_provider_error(self, mock_task_queue, mock_provider):
        """Test cache fetch handles provider errors gracefully."""
        mock_provider.get_ohlcv_data.side_effect = Exception("API limit reached")

        with patch('app.services.ohlcv_cache_handler.get_task_queue', return_value=mock_task_queue), \
             patch('app.api.datasets.get_ohlcv_provider', return_value=mock_provider):
            from app.services.ohlcv_cache_handler import handle_ohlcv_cache_fetch

            result = handle_ohlcv_cache_fetch('task-123', {
                'provider': 'yfinance',
                'symbol': 'AAPL',
                'timeframes': ['1d']
            })

            assert result['status'] == 'completed'
            assert result['results']['1d']['status'] == 'error'
            assert 'API limit reached' in result['results']['1d']['error']

    def test_cache_fetch_missing_symbol(self, mock_task_queue):
        """Test cache fetch fails without a symbol."""
        with patch('app.services.ohlcv_cache_handler.get_task_queue', return_value=mock_task_queue):
            from app.services.ohlcv_cache_handler import handle_ohlcv_cache_fetch

            result = handle_ohlcv_cache_fetch('task-123', {
                'provider': 'yfinance',
                'symbol': '',
                'timeframes': ['1d']
            })

            assert result['status'] == 'failed'
            assert 'symbol is required' in result['error']

    def test_cache_fetch_progress_updates(self, mock_task_queue, mock_provider):
        """Test that progress updates are called during fetch."""
        with patch('app.services.ohlcv_cache_handler.get_task_queue', return_value=mock_task_queue), \
             patch('app.api.datasets.get_ohlcv_provider', return_value=mock_provider):
            from app.services.ohlcv_cache_handler import handle_ohlcv_cache_fetch

            handle_ohlcv_cache_fetch('task-123', {
                'provider': 'yfinance',
                'symbol': 'AAPL',
                'timeframes': ['1d', '4h', '1h']
            })

            # Should update progress for each timeframe + final
            assert mock_task_queue.update_progress.call_count == 4  # 3 timeframes + final 100%

    def test_cache_fetch_calls_use_cache_false(self, mock_task_queue, mock_provider):
        """Test that cache fetch forces refresh (use_cache=False)."""
        with patch('app.services.ohlcv_cache_handler.get_task_queue', return_value=mock_task_queue), \
             patch('app.api.datasets.get_ohlcv_provider', return_value=mock_provider):
            from app.services.ohlcv_cache_handler import handle_ohlcv_cache_fetch

            handle_ohlcv_cache_fetch('task-123', {
                'provider': 'yfinance',
                'symbol': 'AAPL',
                'timeframes': ['1d']
            })

            # Verify use_cache=False was passed
            call_kwargs = mock_provider.get_ohlcv_data.call_args
            assert call_kwargs.kwargs.get('use_cache') is False or \
                   (len(call_kwargs.args) > 3 and call_kwargs.kwargs.get('use_cache', None) is False) or \
                   'use_cache' in str(call_kwargs) and 'False' in str(call_kwargs)


class TestOHLCVProviderEndpoint:
    """Tests for the OHLCV providers listing endpoint."""

    def test_providers_list_structure(self):
        """Test that providers endpoint returns expected structure."""
        # Import and call the endpoint function directly
        import asyncio
        from app.api.tools import list_ohlcv_providers

        result = asyncio.get_event_loop().run_until_complete(list_ohlcv_providers())

        assert 'providers' in result
        assert 'default' in result
        assert result['default'] == 'yfinance'
        assert len(result['providers']) >= 1

        # Check yfinance provider
        yf = next(p for p in result['providers'] if p['id'] == 'yfinance')
        assert yf['available'] is True
        assert yf['requires_api_key'] is False


class TestOHLCVCacheStatusEndpoint:
    """Tests for the cache status endpoint."""

    def test_cache_status_empty_dir(self):
        """Test cache status with empty or non-existent cache directory."""
        import asyncio
        from app.api.tools import get_ohlcv_cache_status

        with patch('app.api.tools.Path') as MockPath:
            mock_path = MagicMock()
            mock_path.exists.return_value = False
            MockPath.return_value = mock_path

            result = asyncio.get_event_loop().run_until_complete(get_ohlcv_cache_status())

            assert result['count'] == 0
            assert result['cache_files'] == []

    def test_cache_status_with_files(self):
        """Test cache status correctly parses cache files."""
        import asyncio
        from app.api.tools import get_ohlcv_cache_status

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a fake cache file
            cache_file = os.path.join(tmpdir, "AAPL_1d.csv")
            with open(cache_file, 'w') as f:
                f.write("Date,Open,High,Low,Close,Volume\n")
                f.write("2024-01-01,100,105,99,103,1000000\n")
                f.write("2024-01-02,103,108,102,107,1200000\n")

            with patch('app.api.tools.Path') as MockPath:
                from pathlib import Path as RealPath
                real_path = RealPath(tmpdir)
                MockPath.return_value = real_path

                result = asyncio.get_event_loop().run_until_complete(get_ohlcv_cache_status())

                assert result['count'] == 1
                assert result['cache_files'][0]['symbol'] == 'AAPL'
                assert result['cache_files'][0]['interval'] == '1d'
                assert result['cache_files'][0]['rows'] == 2


import pathlib


class TestCacheFilePerProvider:
    """Test that cache files are stored per-provider in subdirectories."""

    def test_cache_file_is_per_provider(self):
        """Cache file path must include provider name as subdirectory."""
        from dataproviders.base import MarketDataProviderInterface

        class _Stub(MarketDataProviderInterface):
            def _get_ohlcv_data_impl(self, *a, **kw):
                return pd.DataFrame()
            def get_provider_name(self):
                return "testprov"
            def get_supported_features(self):
                return []
            def validate_config(self):
                return True

        with tempfile.TemporaryDirectory() as tmp:
            s = _Stub()
            s.cache_folder = pathlib.Path(tmp)
            p = s._get_cache_file("AAPL", "1h")
            assert p == pathlib.Path(tmp) / "testprov" / "AAPL_1h.csv"

    def test_cache_file_creates_directory(self):
        """_get_cache_file must create the provider subdirectory if it does not exist."""
        from dataproviders.base import MarketDataProviderInterface

        class _Stub(MarketDataProviderInterface):
            def _get_ohlcv_data_impl(self, *a, **kw):
                return pd.DataFrame()
            def get_provider_name(self):
                return "myprov"
            def get_supported_features(self):
                return []
            def validate_config(self):
                return True

        with tempfile.TemporaryDirectory() as tmp:
            s = _Stub()
            s.cache_folder = pathlib.Path(tmp)
            p = s._get_cache_file("MSFT", "1d")
            assert p.parent.exists(), "Provider subdirectory should have been created"
