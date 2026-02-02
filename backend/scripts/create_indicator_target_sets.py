#!/usr/bin/env python
"""
Database Migration Script: Create Indicator Target Sets

Creates new target sets with ALL available indicator-based prediction targets.
Creates both standard (conservative) and aggressive parameter versions.

Run from backend directory:
    ./venv/bin/python scripts/create_indicator_target_sets.py
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
from app.models.database import SessionLocal
from app.models.target_set import TargetSet
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# All available indicators with their parameters
INDICATOR_CONFIGS = {
    'rsi': {
        'name': 'RSI',
        'standard': {'period': 14},
        'aggressive': {'period': 7},
        'threshold_bullish': 30,
        'threshold_bearish': 70,
        'aggressive_threshold_bullish': 25,
        'aggressive_threshold_bearish': 75,
    },
    'macd': {
        'name': 'MACD',
        'standard': {'fast': 12, 'slow': 26, 'signal': 9},
        'aggressive': {'fast': 8, 'slow': 17, 'signal': 9},
        'threshold_bullish': 0,
        'threshold_bearish': 0,
    },
    'sar': {
        'name': 'Parabolic SAR',
        'standard': {'afStart': 0.02, 'afMax': 0.2},
        'aggressive': {'afStart': 0.04, 'afMax': 0.4},
        'threshold_bullish': 0,
        'threshold_bearish': 0,
    },
    'zigzag': {
        'name': 'ZigZag',
        'standard': {'deviationPct': 5.0},
        'aggressive': {'deviationPct': 3.0},
        'threshold_bullish': 0,
        'threshold_bearish': 0,
    },
    'donchian': {
        'name': 'Donchian Channel',
        'standard': {'period': 20},
        'aggressive': {'period': 10},
        'threshold_bullish': 0,
        'threshold_bearish': 0,
    },
    'adx': {
        'name': 'ADX',
        'standard': {'period': 14},
        'aggressive': {'period': 7},
        'threshold_bullish': 25,
        'threshold_bearish': 25,
        'aggressive_threshold_bullish': 20,
        'aggressive_threshold_bearish': 20,
    },
    'stochastic': {
        'name': 'Stochastic',
        'standard': {'kPeriod': 14, 'dPeriod': 3},
        'aggressive': {'kPeriod': 9, 'dPeriod': 3},
        'threshold_bullish': 20,
        'threshold_bearish': 80,
        'aggressive_threshold_bullish': 15,
        'aggressive_threshold_bearish': 85,
    },
}


def create_target_config(indicator: str, direction: str, params: dict, threshold: int) -> dict:
    """Create a single target configuration."""
    return {
        'type': 'trend_reversal',
        'category': 'binary_classification',
        'indicator': indicator,
        'indicatorParams': params,
        'threshold': threshold,
        'direction': direction,
        'enabled': True,
    }


def create_standard_target_set() -> dict:
    """Create a target set with all indicators using standard parameters."""
    targets = []

    for ind_key, config in INDICATOR_CONFIGS.items():
        # Bullish target
        threshold_bull = config.get('threshold_bullish', 0)
        targets.append(create_target_config(
            indicator=ind_key,
            direction='bullish',
            params=config['standard'],
            threshold=threshold_bull
        ))

        # Bearish target
        threshold_bear = config.get('threshold_bearish', 0)
        targets.append(create_target_config(
            indicator=ind_key,
            direction='bearish',
            params=config['standard'],
            threshold=threshold_bear
        ))

    return {
        'name': 'All Indicators - Standard',
        'description': 'All available indicator-based targets with standard/conservative parameters. '
                      'Includes RSI, MACD, SAR, ZigZag, Donchian, ADX, and Stochastic oscillators.',
        'targets': targets,
    }


def create_aggressive_target_set() -> dict:
    """Create a target set with all indicators using aggressive parameters."""
    targets = []

    for ind_key, config in INDICATOR_CONFIGS.items():
        # Bullish target
        threshold_bull = config.get('aggressive_threshold_bullish', config.get('threshold_bullish', 0))
        targets.append(create_target_config(
            indicator=ind_key,
            direction='bullish',
            params=config['aggressive'],
            threshold=threshold_bull
        ))

        # Bearish target
        threshold_bear = config.get('aggressive_threshold_bearish', config.get('threshold_bearish', 0))
        targets.append(create_target_config(
            indicator=ind_key,
            direction='bearish',
            params=config['aggressive'],
            threshold=threshold_bear
        ))

    return {
        'name': 'All Indicators - Aggressive',
        'description': 'All available indicator-based targets with aggressive parameters (faster signals). '
                      'Shorter periods, tighter thresholds. Good for shorter timeframes or scalping.',
        'targets': targets,
    }


def create_momentum_target_set() -> dict:
    """Create a target set focused on momentum indicators."""
    targets = []
    momentum_indicators = ['rsi', 'macd', 'stochastic', 'adx']

    for ind_key in momentum_indicators:
        config = INDICATOR_CONFIGS[ind_key]

        # Bullish
        threshold_bull = config.get('threshold_bullish', 0)
        targets.append(create_target_config(
            indicator=ind_key,
            direction='bullish',
            params=config['standard'],
            threshold=threshold_bull
        ))

        # Bearish
        threshold_bear = config.get('threshold_bearish', 0)
        targets.append(create_target_config(
            indicator=ind_key,
            direction='bearish',
            params=config['standard'],
            threshold=threshold_bear
        ))

    return {
        'name': 'Momentum Indicators',
        'description': 'Momentum-focused indicators: RSI, MACD, Stochastic, and ADX. '
                      'Best for identifying trend strength and overbought/oversold conditions.',
        'targets': targets,
    }


def create_trend_target_set() -> dict:
    """Create a target set focused on trend-following indicators."""
    targets = []
    trend_indicators = ['sar', 'zigzag', 'donchian']

    for ind_key in trend_indicators:
        config = INDICATOR_CONFIGS[ind_key]

        # Bullish
        threshold_bull = config.get('threshold_bullish', 0)
        targets.append(create_target_config(
            indicator=ind_key,
            direction='bullish',
            params=config['standard'],
            threshold=threshold_bull
        ))

        # Bearish
        threshold_bear = config.get('threshold_bearish', 0)
        targets.append(create_target_config(
            indicator=ind_key,
            direction='bearish',
            params=config['standard'],
            threshold=threshold_bear
        ))

    return {
        'name': 'Trend-Following Indicators',
        'description': 'Trend-following indicators: Parabolic SAR, ZigZag, and Donchian Channels. '
                      'Best for identifying trend reversals and breakouts.',
        'targets': targets,
    }


def main():
    """Main migration function."""
    db = SessionLocal()

    try:
        # Check existing target sets
        existing_names = [ts.name for ts in db.query(TargetSet).all()]
        logger.info(f"Found {len(existing_names)} existing target sets")

        # Define target sets to create
        target_sets_to_create = [
            create_standard_target_set(),
            create_aggressive_target_set(),
            create_momentum_target_set(),
            create_trend_target_set(),
        ]

        created_count = 0
        skipped_count = 0

        for ts_data in target_sets_to_create:
            if ts_data['name'] in existing_names:
                logger.info(f"Skipping '{ts_data['name']}' - already exists")
                skipped_count += 1
                continue

            target_set = TargetSet(
                name=ts_data['name'],
                description=ts_data['description'],
                targets=ts_data['targets'],
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            )
            db.add(target_set)
            logger.info(f"Created target set: '{ts_data['name']}' with {len(ts_data['targets'])} targets")
            created_count += 1

        db.commit()

        logger.info(f"\n=== Migration Complete ===")
        logger.info(f"Created: {created_count} target sets")
        logger.info(f"Skipped: {skipped_count} (already existed)")

        # List all target sets
        all_sets = db.query(TargetSet).all()
        logger.info(f"\nAll target sets ({len(all_sets)}):")
        for ts in all_sets:
            logger.info(f"  - {ts.name}: {len(ts.targets)} targets")

    except Exception as e:
        db.rollback()
        logger.error(f"Migration failed: {e}", exc_info=True)
        raise
    finally:
        db.close()


if __name__ == '__main__':
    main()
