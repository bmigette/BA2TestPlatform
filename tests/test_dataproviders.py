"""
Test script to verify data providers import and basic functionality.
"""

import sys
from pathlib import Path

# Add parent directory to path so we can import dataproviders
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing data provider imports...")

    try:
        from dataproviders import MarketDataProviderInterface, MarketDataPoint
        print("[OK] Successfully imported base classes")
    except Exception as e:
        print(f"[FAIL] Failed to import base classes: {e}")
        return False

    try:
        from dataproviders import YFinanceDataProvider
        print("[OK] Successfully imported YFinanceDataProvider")
    except Exception as e:
        print(f"[FAIL] Failed to import YFinanceDataProvider: {e}")
        return False

    try:
        from dataproviders import AlphaVantageOHLCVProvider
        print("[OK] Successfully imported AlphaVantageOHLCVProvider")
    except Exception as e:
        print(f"[FAIL] Failed to import AlphaVantageOHLCVProvider: {e}")
        return False

    return True


def test_yfinance_provider():
    """Test YFinance provider basic functionality."""
    print("\nTesting YFinance provider...")

    try:
        from dataproviders import YFinanceDataProvider
        from datetime import datetime, timedelta

        provider = YFinanceDataProvider()
        print(f"[OK] Created YFinanceDataProvider instance")
        print(f"  Provider name: {provider.get_provider_name()}")
        print(f"  Supported features: {provider.get_supported_features()}")

        # Test configuration
        if provider.validate_config():
            print(f"[OK] Provider configuration is valid")
        else:
            print(f"[FAIL] Provider configuration is invalid")
            return False

        # Test fetching a small amount of data (5 days)
        print("\nFetching 5 days of AAPL data...")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5)

        df = provider.get_ohlcv_data(
            symbol='AAPL',
            start_date=start_date,
            end_date=end_date,
            interval='1d',
            use_cache=False  # Don't use cache for test
        )

        if not df.empty:
            print(f"[OK] Successfully fetched {len(df)} data points")
            print(f"  Columns: {list(df.columns)}")
            print(f"  Date range: {df['Date'].min()} to {df['Date'].max()}")
            print(f"\n  Sample data (first row):")
            print(f"    Date: {df.iloc[0]['Date']}")
            print(f"    Open: ${df.iloc[0]['Open']:.2f}")
            print(f"    High: ${df.iloc[0]['High']:.2f}")
            print(f"    Low: ${df.iloc[0]['Low']:.2f}")
            print(f"    Close: ${df.iloc[0]['Close']:.2f}")
            print(f"    Volume: {df.iloc[0]['Volume']:,.0f}")
            return True
        else:
            print(f"[FAIL] No data returned")
            return False

    except Exception as e:
        print(f"[FAIL] YFinance provider test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Data Providers Test Suite")
    print("=" * 60)
    print()

    # Test 1: Imports
    imports_ok = test_imports()

    # Test 2: YFinance provider (doesn't require API key)
    if imports_ok:
        yfinance_ok = test_yfinance_provider()
    else:
        yfinance_ok = False

    # Summary
    print()
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"Imports: {'PASS' if imports_ok else 'FAIL'}")
    print(f"YFinance Provider: {'PASS' if yfinance_ok else 'FAIL'}")
    print()

    if imports_ok and yfinance_ok:
        print("[OK] All tests passed!")
        sys.exit(0)
    else:
        print("[FAIL] Some tests failed")
        sys.exit(1)
