#!/usr/bin/env python
"""
Database Migration Script: Create Indicator Collections

Creates new indicator collections with ALL available technical indicators.
Creates both standard (conservative) and aggressive parameter versions.

Run from backend directory:
    ./venv/bin/python scripts/create_indicator_collections.py
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
from app.models.database import SessionLocal
from app.models.indicator_collection import IndicatorCollection
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_comprehensive_standard() -> dict:
    """Create a comprehensive collection with all indicators using standard parameters."""
    indicators = []

    # SMA variations
    for period in [10, 20, 50, 100, 200]:
        indicators.append({
            "type": "sma",
            "name": f"SMA {period}",
            "period": period,
        })

    # EMA variations
    for period in [12, 26, 50, 100, 200]:
        indicators.append({
            "type": "ema",
            "name": f"EMA {period}",
            "period": period,
        })

    # RSI
    indicators.append({
        "type": "rsi",
        "name": "RSI 14",
        "period": 14,
    })

    # MACD
    indicators.append({
        "type": "macd",
        "name": "MACD (12,26,9)",
        "fast": 12,
        "slow": 26,
        "signal": 9,
    })

    # Bollinger Bands
    indicators.append({
        "type": "bbands",
        "name": "Bollinger Bands (20,2)",
        "period": 20,
        "std_dev": 2.0,
    })

    # ATR
    indicators.append({
        "type": "atr",
        "name": "ATR 14",
        "period": 14,
    })

    # Stochastic
    indicators.append({
        "type": "stochastic",
        "name": "Stochastic (14,3)",
        "k_period": 14,
        "d_period": 3,
    })

    # SAR (Parabolic SAR)
    indicators.append({
        "type": "sar",
        "name": "SAR (0.02,0.2)",
        "af_start": 0.02,
        "af_max": 0.2,
    })

    # ZigZag
    indicators.append({
        "type": "zigzag",
        "name": "ZigZag 5%",
        "deviation_pct": 5.0,
    })

    # Donchian Channel
    indicators.append({
        "type": "donchian",
        "name": "Donchian 20",
        "period": 20,
    })

    # ADX (Average Directional Index)
    indicators.append({
        "type": "adx",
        "name": "ADX 14",
        "period": 14,
    })

    # OBV (On-Balance Volume)
    indicators.append({
        "type": "obv",
        "name": "OBV",
    })

    # Pivot Points
    indicators.append({
        "type": "pivot_points",
        "name": "Pivot Points (Standard)",
        "method": "standard",
    })

    return {
        "name": "All Indicators - Standard",
        "description": "Comprehensive collection with all available indicators using standard/conservative parameters. "
                      "Includes moving averages, momentum, volatility, trend, and volume indicators.",
        "indicators": indicators,
        "is_default": True,
    }


def create_comprehensive_aggressive() -> dict:
    """Create a comprehensive collection with all indicators using aggressive parameters."""
    indicators = []

    # SMA - shorter periods
    for period in [5, 10, 20, 50]:
        indicators.append({
            "type": "sma",
            "name": f"SMA {period}",
            "period": period,
        })

    # EMA - shorter periods
    for period in [8, 13, 21, 34, 55]:  # Fibonacci-based
        indicators.append({
            "type": "ema",
            "name": f"EMA {period}",
            "period": period,
        })

    # RSI - shorter period for faster signals
    indicators.append({
        "type": "rsi",
        "name": "RSI 7",
        "period": 7,
    })

    # MACD - faster settings
    indicators.append({
        "type": "macd",
        "name": "MACD (8,17,9)",
        "fast": 8,
        "slow": 17,
        "signal": 9,
    })

    # Bollinger Bands - tighter bands
    indicators.append({
        "type": "bbands",
        "name": "Bollinger Bands (10,1.5)",
        "period": 10,
        "std_dev": 1.5,
    })

    # ATR - shorter period
    indicators.append({
        "type": "atr",
        "name": "ATR 7",
        "period": 7,
    })

    # Stochastic - faster settings
    indicators.append({
        "type": "stochastic",
        "name": "Stochastic (9,3)",
        "k_period": 9,
        "d_period": 3,
    })

    # SAR - more aggressive acceleration
    indicators.append({
        "type": "sar",
        "name": "SAR (0.04,0.4)",
        "af_start": 0.04,
        "af_max": 0.4,
    })

    # ZigZag - tighter deviation
    indicators.append({
        "type": "zigzag",
        "name": "ZigZag 3%",
        "deviation_pct": 3.0,
    })

    # Donchian Channel - shorter lookback
    indicators.append({
        "type": "donchian",
        "name": "Donchian 10",
        "period": 10,
    })

    # ADX - shorter period
    indicators.append({
        "type": "adx",
        "name": "ADX 7",
        "period": 7,
    })

    # OBV
    indicators.append({
        "type": "obv",
        "name": "OBV",
    })

    # Pivot Points
    indicators.append({
        "type": "pivot_points",
        "name": "Pivot Points (Fibonacci)",
        "method": "fibonacci",
    })

    return {
        "name": "All Indicators - Aggressive",
        "description": "Comprehensive collection with all indicators using aggressive parameters (faster signals). "
                      "Shorter periods, tighter thresholds. Good for shorter timeframes or scalping strategies.",
        "indicators": indicators,
        "is_default": True,
    }


def create_momentum_collection() -> dict:
    """Create a collection focused on momentum indicators."""
    indicators = [
        {"type": "rsi", "name": "RSI 14", "period": 14},
        {"type": "rsi", "name": "RSI 7", "period": 7},
        {"type": "macd", "name": "MACD (12,26,9)", "fast": 12, "slow": 26, "signal": 9},
        {"type": "stochastic", "name": "Stochastic (14,3)", "k_period": 14, "d_period": 3},
        {"type": "stochastic", "name": "Stochastic (9,3)", "k_period": 9, "d_period": 3},
        {"type": "adx", "name": "ADX 14", "period": 14},
        {"type": "obv", "name": "OBV"},
    ]

    return {
        "name": "Momentum Indicators",
        "description": "Momentum-focused indicators: RSI, MACD, Stochastic, ADX, and OBV. "
                      "Best for identifying trend strength and overbought/oversold conditions.",
        "indicators": indicators,
        "is_default": False,
    }


def create_trend_collection() -> dict:
    """Create a collection focused on trend-following indicators."""
    indicators = [
        {"type": "sma", "name": "SMA 20", "period": 20},
        {"type": "sma", "name": "SMA 50", "period": 50},
        {"type": "sma", "name": "SMA 200", "period": 200},
        {"type": "ema", "name": "EMA 12", "period": 12},
        {"type": "ema", "name": "EMA 26", "period": 26},
        {"type": "ema", "name": "EMA 50", "period": 50},
        {"type": "sar", "name": "SAR (0.02,0.2)", "af_start": 0.02, "af_max": 0.2},
        {"type": "zigzag", "name": "ZigZag 5%", "deviation_pct": 5.0},
        {"type": "donchian", "name": "Donchian 20", "period": 20},
        {"type": "adx", "name": "ADX 14", "period": 14},
    ]

    return {
        "name": "Trend-Following Indicators",
        "description": "Trend-following indicators: Moving averages, Parabolic SAR, ZigZag, Donchian Channels, and ADX. "
                      "Best for identifying trend direction and reversals.",
        "indicators": indicators,
        "is_default": False,
    }


def create_volatility_collection() -> dict:
    """Create a collection focused on volatility indicators."""
    indicators = [
        {"type": "atr", "name": "ATR 14", "period": 14},
        {"type": "atr", "name": "ATR 7", "period": 7},
        {"type": "bbands", "name": "Bollinger Bands (20,2)", "period": 20, "std_dev": 2.0},
        {"type": "bbands", "name": "Bollinger Bands (20,2.5)", "period": 20, "std_dev": 2.5},
        {"type": "donchian", "name": "Donchian 20", "period": 20},
        {"type": "donchian", "name": "Donchian 10", "period": 10},
    ]

    return {
        "name": "Volatility Indicators",
        "description": "Volatility-focused indicators: ATR, Bollinger Bands, and Donchian Channels. "
                      "Best for measuring price volatility and identifying breakout opportunities.",
        "indicators": indicators,
        "is_default": False,
    }


def create_minimal_collection() -> dict:
    """Create a minimal collection with essential indicators only."""
    indicators = [
        {"type": "sma", "name": "SMA 20", "period": 20},
        {"type": "ema", "name": "EMA 20", "period": 20},
        {"type": "rsi", "name": "RSI 14", "period": 14},
        {"type": "macd", "name": "MACD (12,26,9)", "fast": 12, "slow": 26, "signal": 9},
        {"type": "atr", "name": "ATR 14", "period": 14},
    ]

    return {
        "name": "Essential Indicators",
        "description": "Minimal set of essential indicators: SMA, EMA, RSI, MACD, and ATR. "
                      "Good for quick dataset generation and avoiding feature bloat.",
        "indicators": indicators,
        "is_default": False,
    }


def main():
    """Main migration function."""
    db = SessionLocal()

    try:
        # Check existing collections
        existing_names = [c.name for c in db.query(IndicatorCollection).all()]
        logger.info(f"Found {len(existing_names)} existing indicator collections")

        # Define collections to create
        collections_to_create = [
            create_comprehensive_standard(),
            create_comprehensive_aggressive(),
            create_momentum_collection(),
            create_trend_collection(),
            create_volatility_collection(),
            create_minimal_collection(),
        ]

        created_count = 0
        skipped_count = 0

        for coll_data in collections_to_create:
            if coll_data['name'] in existing_names:
                logger.info(f"Skipping '{coll_data['name']}' - already exists")
                skipped_count += 1
                continue

            collection = IndicatorCollection(
                name=coll_data['name'],
                description=coll_data['description'],
                indicators=coll_data['indicators'],
                is_default=coll_data.get('is_default', False),
            )
            db.add(collection)
            logger.info(f"Created collection: '{coll_data['name']}' with {len(coll_data['indicators'])} indicators")
            created_count += 1

        db.commit()

        logger.info(f"\n=== Migration Complete ===")
        logger.info(f"Created: {created_count} indicator collections")
        logger.info(f"Skipped: {skipped_count} (already existed)")

        # List all collections
        all_collections = db.query(IndicatorCollection).all()
        logger.info(f"\nAll indicator collections ({len(all_collections)}):")
        for coll in all_collections:
            default_marker = " [DEFAULT]" if coll.is_default else ""
            logger.info(f"  - {coll.name}: {len(coll.indicators)} indicators{default_marker}")

    except Exception as e:
        db.rollback()
        logger.error(f"Migration failed: {e}", exc_info=True)
        raise
    finally:
        db.close()


if __name__ == '__main__':
    main()
