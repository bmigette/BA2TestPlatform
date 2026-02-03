"""
Backtest Handler Service

Executes backtests using the strategy executor and updates results in the database.
"""

import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from app.models.database import SessionLocal
from app.models import Dataset, TrainedModel, Strategy, Backtest
from app.services.strategy_executor import StrategyExecutor, evaluate_condition_tree, ConfirmationTracker, StrategyExecutionError, reset_evaluation_stats, get_evaluation_stats
from app.services.data_preparation import DataPreparationService
from app.services.tsai_training import TSAITrainingService
from app.services.job_handler import ffill_sparse_indicators

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Represents a completed trade."""
    entry_time: datetime
    exit_time: datetime
    direction: str  # "buy" or "sell"
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    pnl_pct: float
    bars_held: int


@dataclass
class OpenPosition:
    """Represents an open position during backtest."""
    entry_time: datetime
    direction: str  # "buy" or "sell"
    entry_price: float
    size: float
    bars_held: int = 0


def run_backtest(
    model: TrainedModel,
    pred_df: pd.DataFrame,
    exec_df: pd.DataFrame,
    strategy_params: Dict[str, Any],
    initial_capital: float = 10000.0,
    position_sizing_type: str = "fixed",
    position_sizing_value: float = 1000.0,
    commission: float = 0.0,
    slippage: float = 0.0,
    buy_entry_conditions: Optional[Dict] = None,
    sell_entry_conditions: Optional[Dict] = None,
    exit_conditions: Optional[List[Dict]] = None,
) -> Dict[str, Any]:
    """
    Run a backtest simulation.

    Args:
        model: TrainedModel record with model file path and params
        pred_df: DataFrame for model predictions (features)
        exec_df: DataFrame for trade execution (OHLCV)
        strategy_params: Strategy configuration
        initial_capital: Starting capital
        position_sizing_type: "fixed" or "percent"
        position_sizing_value: Position size in $ or %
        commission: Commission per trade (%)
        slippage: Slippage per trade (%)
        buy_entry_conditions: Condition tree for buy entries
        sell_entry_conditions: Condition tree for sell entries
        exit_conditions: List of exit condition rules

    Returns:
        Dict with backtest results and metrics
    """
    import torch
    from pathlib import Path
    from app.services.tsai_models import TSAIModelService

    # Get model parameters
    hyperparameters = model.hyperparameters or {}
    seq_len = hyperparameters.get('seq_len', hyperparameters.get('seqLen', 24))
    prediction_mode = model.prediction_mode or 'shift'
    threshold = model.threshold or 0.5
    model_type = model.model_type.lower() if model.model_type else 'lstm'

    # Try to load metadata early for feature_columns
    import json
    stored_feature_columns = hyperparameters.get('featureColumns')
    file_path = model.file_path
    file_path_obj = Path(file_path) if file_path else None

    logger.info(f"Model {model.model_id}: file_path={file_path}, featureColumns in hyperparams={stored_feature_columns is not None}, hyperparams keys={list(hyperparameters.keys()) if hyperparameters else 'None'}")

    # If no stored feature_columns in hyperparameters, try metadata file
    if not stored_feature_columns and file_path_obj:
        if not file_path_obj.exists():
            logger.warning(f"Model file does not exist: {file_path_obj}")
        else:
            meta_patterns = [
                file_path_obj.with_name(file_path_obj.stem + '_meta.json'),
                file_path_obj.with_suffix('.json'),
            ]
            logger.debug(f"Trying metadata patterns: {[str(p) for p in meta_patterns]}")
            for meta_path in meta_patterns:
                if meta_path.exists():
                    try:
                        with open(meta_path, 'r') as f:
                            meta = json.load(f)
                        stored_feature_columns = meta.get('feature_columns')
                        if stored_feature_columns:
                            logger.info(f"Loaded feature_columns from {meta_path}: {len(stored_feature_columns)} features")
                            break
                        else:
                            logger.warning(f"Metadata file {meta_path} exists but has no feature_columns key")
                    except Exception as e:
                        logger.warning(f"Failed to load metadata from {meta_path}: {e}")
                else:
                    logger.debug(f"Metadata file not found: {meta_path}")

    # Get feature columns - prefer stored columns from training
    c_in = hyperparameters.get('c_in')

    if stored_feature_columns:
        logger.info(f"Using {len(stored_feature_columns)} stored feature columns from training (c_in={c_in})")
        # Use only features that exist in current dataset
        feature_cols = [col for col in stored_feature_columns if col in pred_df.columns]

        # Validate that feature count matches c_in (model architecture)
        if c_in and len(feature_cols) != c_in:
            logger.warning(f"Feature count mismatch: featureColumns has {len(feature_cols)} but model expects c_in={c_in}. "
                          f"Model may have been saved with incorrect featureColumns. Retrain to fix.")

        logger.info(f"After filtering for dataset columns: {len(feature_cols)} features (dataset has {len(pred_df.columns)} total columns)")
        if len(feature_cols) != len(stored_feature_columns):
            missing = set(stored_feature_columns) - set(feature_cols)
            if len(missing) < 20:  # Only log if not too many
                logger.warning(f"Some training features not in dataset: {missing}")
            else:
                logger.warning(f"Some training features not in dataset: {len(missing)} missing")
        if not feature_cols:
            logger.error(f"None of the training features found in dataset")
            return _empty_results(initial_capital)
    else:
        # Fall back to computing from dataset - WARN about this as it may cause issues
        logger.warning(f"No stored feature_columns found for model {model.model_id}. "
                      f"Falling back to computing features from dataset. "
                      f"Tried hyperparameters.featureColumns and metadata file at {file_path_obj}")
        exclude_cols = {'Date', 'target', 'Open', 'High', 'Low', 'Close', 'Volume'}
        feature_cols = [c for c in pred_df.columns if c not in exclude_cols]
        logger.info(f"Computed {len(feature_cols)} feature columns from dataset")

    if not feature_cols:
        logger.error("No feature columns found in prediction dataset")
        return _empty_results(initial_capital)

    # Forward-fill sparse indicators (e.g., zigzag) like training does
    # These have NaN between pivots which causes prediction issues
    pred_df = ffill_sparse_indicators(pred_df)

    # Apply normalization if available
    # Track which columns are actually used for NaN error reporting
    used_feature_cols = feature_cols
    if model.normalization_params:
        data_prep = DataPreparationService()
        data_prep.load_params(model.normalization_params)
        # Transform expects DataFrame, not numpy array
        df_normalized = data_prep.transform(pred_df[feature_cols])
        # Use valid columns (excludes zero-variance columns dropped during training)
        valid_cols = data_prep.get_valid_columns()
        if valid_cols:
            # Filter to columns that exist in our normalized DataFrame
            valid_cols = [c for c in valid_cols if c in df_normalized.columns]
            features = df_normalized[valid_cols].values
            used_feature_cols = valid_cols
            logger.info(f"Using {len(valid_cols)} valid columns after zero-variance filtering")
        else:
            # Fallback if valid_columns not set
            features = df_normalized[feature_cols].values
            logger.warning(f"No valid_columns info in normalization params, using all {len(feature_cols)} features")
    else:
        features = pred_df[feature_cols].values

    # Create sliding windows for prediction
    n_samples = len(features) - seq_len + 1
    if n_samples <= 0:
        logger.error(f"Not enough data for seq_len={seq_len}")
        return _empty_results(initial_capital)

    X = np.array([features[i:i+seq_len] for i in range(n_samples)])
    X = X.transpose(0, 2, 1)  # (samples, features, seq_len) for tsai
    logger.info(f"Created input tensor X with shape {X.shape} (samples, features, seq_len)")

    # Check for NaN in input features
    nan_count = np.isnan(X).sum()
    if nan_count > 0:
        nan_pct = nan_count / X.size * 100
        # Find which features have NaN
        nan_per_feature = np.isnan(X).any(axis=(0, 2))
        nan_features = [used_feature_cols[i] for i, has_nan in enumerate(nan_per_feature) if has_nan]
        logger.error(f"Input features contain {nan_count} NaN values ({nan_pct:.1f}%). "
                    f"Features with NaN: {nan_features[:10]}{'...' if len(nan_features) > 10 else ''}")
        result = _empty_results(initial_capital)
        result['error'] = f"Input features contain NaN values in: {nan_features[:5]}"
        result['status'] = 'failed'
        return result

    # Load the trained model
    training_service = TSAITrainingService()
    file_path = model.file_path
    if not file_path:
        logger.error("Model has no file path")
        return _empty_results(initial_capital)

    file_path_obj = Path(file_path)
    if not file_path_obj.exists():
        logger.error(f"Model file not found: {file_path}")
        return _empty_results(initial_capital)

    try:
        if file_path_obj.suffix == '.pkl':
            # Full learner export
            from tsai.all import load_learner
            learner = load_learner(file_path)
            model_obj = learner.model
        else:
            # State dict (.pt file) - need to recreate model architecture
            # Get c_in from metadata (stored during training) - NOT from current dataset
            import json
            c_in = hyperparameters.get('c_in')
            c_out = hyperparameters.get('c_out', 2)
            model_params = hyperparameters.get('modelParams', {})

            # If c_in not in DB hyperparameters, check for _meta.json file
            if c_in is None:
                meta_patterns = [
                    file_path_obj.with_name(file_path_obj.stem + '_meta.json'),
                    file_path_obj.with_suffix('.json'),
                ]
                for meta_path in meta_patterns:
                    if meta_path.exists():
                        try:
                            with open(meta_path, 'r') as f:
                                meta = json.load(f)
                            c_in = meta.get('c_in')
                            if c_out == 2:
                                c_out = meta.get('c_out', c_out)
                            if not model_params:
                                model_params = meta.get('params', {})
                            logger.info(f"Loaded model metadata: c_in={c_in}, c_out={c_out}")
                            break
                        except Exception as e:
                            logger.warning(f"Failed to load metadata from {meta_path}: {e}")

            if c_in is None:
                logger.error(f"Cannot determine c_in for model {file_path}")
                return _empty_results(initial_capital)

            model_service = TSAIModelService()
            model_obj = model_service.create_model(
                model_type=model_type,
                params=model_params,
                c_in=c_in,
                c_out=c_out,
                seq_len=seq_len
            )

            # Load state dict
            state_dict = torch.load(file_path, map_location='cpu', weights_only=True)
            model_obj.load_state_dict(state_dict)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return _empty_results(initial_capital)

    # Run predictions
    try:
        predictions = training_service.predict(
            model=model_obj,
            data=X,
            prediction_mode=prediction_mode
        )
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return _empty_results(initial_capital)

    # Check for NaN predictions
    nan_count = np.isnan(predictions).sum()
    if nan_count > 0:
        nan_pct = nan_count / predictions.size * 100
        logger.error(f"Predictions contain {nan_count} NaN values ({nan_pct:.1f}%). "
                    f"This usually indicates NaN values in input features or model issues.")
        # Check input data for NaN
        input_nan_count = np.isnan(X).sum()
        if input_nan_count > 0:
            logger.error(f"Input data X contains {input_nan_count} NaN values - this is the cause.")
        result = _empty_results(initial_capital)
        result['error'] = f"Model produced {nan_count} NaN predictions. Check input data for NaN values."
        result['status'] = 'failed'
        return result

    # Align predictions with execution data
    # Predictions start at index seq_len-1 (after we have enough history)
    pred_start_idx = seq_len - 1
    pred_dates = pred_df['Date'].iloc[pred_start_idx:pred_start_idx + len(predictions)].values

    # predictions is now 2D: (samples, n_classes) for all modes
    # Create prediction lookup by date - stores full probability array per date
    pred_lookup = dict(zip(pred_dates, predictions))
    n_classes = predictions.shape[1] if len(predictions.shape) > 1 else 1
    logger.info(f"Predictions shape: {predictions.shape}, n_classes={n_classes}")

    # Run simulation
    equity = initial_capital
    open_positions: List[OpenPosition] = []
    completed_trades: List[Trade] = []
    equity_curve = [{'date': exec_df['Date'].iloc[0].isoformat() if len(exec_df) > 0 else '', 'equity': equity}]

    exit_conditions = exit_conditions or []

    # Create confirmation tracker for condition history
    confirmation_tracker = ConfirmationTracker()
    logged_context_fields = False

    # Reset evaluation stats for fresh aggregated logging
    reset_evaluation_stats()

    # Track last trade entry for "no trade in past X bars/days" conditions
    last_buy_bar_idx: Optional[int] = None
    last_buy_date: Optional[Any] = None
    last_sell_bar_idx: Optional[int] = None
    last_sell_date: Optional[Any] = None

    # Track trade counts for summary
    buy_trades_opened = 0
    sell_trades_opened = 0
    bars_processed = 0

    for idx in range(len(exec_df)):
        row = exec_df.iloc[idx]
        current_date = row['Date']
        current_price = row['Close']

        # Get prediction for this bar (now a probability array for all classes)
        probs = pred_lookup.get(current_date, None)
        if probs is None:
            # No prediction available for this bar
            for pos in open_positions:
                pos.bars_held += 1
            continue

        # Count positions by direction
        buy_positions = [p for p in open_positions if p.direction == 'buy']
        sell_positions = [p for p in open_positions if p.direction == 'sell']

        # Build context for condition evaluation
        # probs is now an array of probabilities for each class
        predicted_class = int(np.argmax(probs))
        max_prob = float(np.max(probs))

        # Calculate bars/days since last buy/sell trade was opened
        bars_since_last_buy = (idx - last_buy_bar_idx) if last_buy_bar_idx is not None else 999999
        bars_since_last_sell = (idx - last_sell_bar_idx) if last_sell_bar_idx is not None else 999999

        # Calculate days since last trade (handle both datetime and Timestamp)
        if last_buy_date is not None:
            try:
                days_since_last_buy = (current_date - last_buy_date).days
            except (TypeError, AttributeError):
                days_since_last_buy = bars_since_last_buy  # Fallback to bars
        else:
            days_since_last_buy = 999999

        if last_sell_date is not None:
            try:
                days_since_last_sell = (current_date - last_sell_date).days
            except (TypeError, AttributeError):
                days_since_last_sell = bars_since_last_sell  # Fallback to bars
        else:
            days_since_last_sell = 999999

        context = {
            'model:prediction': predicted_class,
            'model:predicted_class': predicted_class,
            'model:probability': max_prob,  # Probability of predicted class
            'model:max_probability': max_prob,
            'Open': row.get('Open', current_price),
            'High': row.get('High', current_price),
            'Low': row.get('Low', current_price),
            'Close': current_price,
            'Volume': row.get('Volume', 0),
            'position:in_position': len(open_positions) > 0,
            'position:buy_count': len(buy_positions),
            'position:sell_count': len(sell_positions),
            'position:total_count': len(open_positions),
            # Bars/days since last trade was opened
            'trade:bars_since_last_buy': bars_since_last_buy,
            'trade:bars_since_last_sell': bars_since_last_sell,
            'trade:days_since_last_buy': days_since_last_buy,
            'trade:days_since_last_sell': days_since_last_sell,
        }

        # Add probability and class indicator for each class
        for class_idx in range(len(probs)):
            context[f'model:probability_{class_idx}'] = float(probs[class_idx])
            context[f'model:class_{class_idx}'] = 1 if predicted_class == class_idx else 0

        # Add any additional columns from exec_df
        for col in exec_df.columns:
            if col not in context and col != 'Date':
                context[col] = row[col]

        # Log available context fields once
        if not logged_context_fields:
            logger.info(f"Available context fields: {sorted(context.keys())}")
            logger.info(f"Buy entry conditions: {buy_entry_conditions}")
            logger.info(f"Sell entry conditions: {sell_entry_conditions}")
            logged_context_fields = True

        # Check exit conditions for each open position
        positions_to_close = []
        for i, pos in enumerate(open_positions):
            # Calculate position P&L
            if pos.direction == 'buy':
                pnl_pct = (current_price - pos.entry_price) / pos.entry_price * 100
            else:
                pnl_pct = (pos.entry_price - current_price) / pos.entry_price * 100

            pos_context = context.copy()
            pos_context['position:is_buy'] = pos.direction == 'buy'
            pos_context['position:is_sell'] = pos.direction == 'sell'
            pos_context['bars_in_trade'] = pos.bars_held
            pos_context['position_pnl_pct'] = pnl_pct

            # Check each exit rule
            for exit_rule in exit_conditions:
                conditions = exit_rule.get('conditions', {})
                if evaluate_condition_tree(conditions, pos_context, confirmation_tracker, label=f"Exit-{pos.direction}"):
                    positions_to_close.append(i)
                    break

            pos.bars_held += 1

        # Close positions (in reverse order to preserve indices)
        for i in sorted(positions_to_close, reverse=True):
            pos = open_positions.pop(i)
            exit_price = current_price * (1 - slippage / 100)

            if pos.direction == 'buy':
                pnl = (exit_price - pos.entry_price) * pos.size - (commission / 100 * pos.size * exit_price)
                pnl_pct = (exit_price - pos.entry_price) / pos.entry_price * 100
            else:
                pnl = (pos.entry_price - exit_price) * pos.size - (commission / 100 * pos.size * exit_price)
                pnl_pct = (pos.entry_price - exit_price) / pos.entry_price * 100

            completed_trades.append(Trade(
                entry_time=pos.entry_time,
                exit_time=current_date,
                direction=pos.direction,
                entry_price=pos.entry_price,
                exit_price=exit_price,
                size=pos.size,
                pnl=pnl,
                pnl_pct=pnl_pct,
                bars_held=pos.bars_held
            ))
            equity += pnl

        # Check entry conditions (users can add position:total_count == 0 to limit entries)
        # Check buy entry
        if buy_entry_conditions and evaluate_condition_tree(buy_entry_conditions, context, confirmation_tracker, label="BuyEntry"):
            entry_price = current_price * (1 + slippage / 100)
            size = _calculate_position_size(equity, position_sizing_type, position_sizing_value, entry_price)
            if size > 0:
                open_positions.append(OpenPosition(
                    entry_time=current_date,
                    direction='buy',
                    entry_price=entry_price,
                    size=size
                ))
                # Track last buy trade for "no trade in past X" conditions
                last_buy_bar_idx = idx
                last_buy_date = current_date
                buy_trades_opened += 1

        # Check sell entry
        elif sell_entry_conditions and evaluate_condition_tree(sell_entry_conditions, context, confirmation_tracker, label="SellEntry"):
            entry_price = current_price * (1 - slippage / 100)
            size = _calculate_position_size(equity, position_sizing_type, position_sizing_value, entry_price)
            if size > 0:
                open_positions.append(OpenPosition(
                    entry_time=current_date,
                    direction='sell',
                    entry_price=entry_price,
                    size=size
                ))
                # Track last sell trade for "no trade in past X" conditions
                last_sell_bar_idx = idx
                last_sell_date = current_date
                sell_trades_opened += 1

        bars_processed += 1

        # Record equity
        equity_curve.append({
            'date': current_date.isoformat() if hasattr(current_date, 'isoformat') else str(current_date),
            'equity': equity
        })

    # Log backtest summary
    eval_stats = get_evaluation_stats()
    total_trades = len(completed_trades)
    winning = sum(1 for t in completed_trades if t.pnl > 0)
    losing = total_trades - winning

    logger.info(f"=== Backtest Summary ===")
    logger.info(f"Bars processed: {bars_processed}, Total condition evaluations: {eval_stats['total_evaluations']}")
    logger.info(f"Trades opened: {buy_trades_opened} buy, {sell_trades_opened} sell")
    logger.info(f"Trades completed: {total_trades} ({winning} winning, {losing} losing)")

    # Log condition hit rates
    tree_results = eval_stats.get('tree_results', {})
    if tree_results:
        hit_summary = []
        for label, counts in tree_results.items():
            total = counts['true'] + counts['false']
            hit_rate = counts['true'] / total * 100 if total > 0 else 0
            hit_summary.append(f"{label}: {counts['true']}/{total} ({hit_rate:.1f}%)")
        logger.info(f"Entry/Exit hit rates: {', '.join(hit_summary)}")

    # Calculate metrics
    return _calculate_metrics(completed_trades, equity_curve, initial_capital, equity)


def _empty_results(initial_capital: float) -> Dict[str, Any]:
    """Return empty results structure."""
    return {
        'total_trades': 0,
        'winning_trades': 0,
        'losing_trades': 0,
        'win_rate': 0.0,
        'total_return': 0.0,
        'sharpe_ratio': 0.0,
        'max_drawdown': 0.0,
        'profit_factor': 0.0,
        'avg_trade_duration': 0.0,
        'final_equity': initial_capital,
        'equity_curve': [],
        'drawdown_curve': [],
        'trades': []
    }


def _calculate_position_size(
    equity: float,
    sizing_type: str,
    sizing_value: float,
    price: float
) -> float:
    """Calculate position size based on sizing type."""
    if sizing_type == 'percent':
        return (equity * sizing_value / 100) / price
    else:  # fixed
        return sizing_value / price


def _calculate_metrics(
    trades: List[Trade],
    equity_curve: List[Dict],
    initial_capital: float,
    final_equity: float
) -> Dict[str, Any]:
    """Calculate backtest performance metrics."""
    total_trades = len(trades)

    if total_trades == 0:
        return _empty_results(initial_capital)

    winning_trades = sum(1 for t in trades if t.pnl > 0)
    losing_trades = sum(1 for t in trades if t.pnl <= 0)
    win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0.0

    total_return = (final_equity - initial_capital) / initial_capital * 100

    # Profit factor
    gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
    gross_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0.0

    # Average trade duration
    avg_trade_duration = sum(t.bars_held for t in trades) / total_trades if total_trades > 0 else 0.0

    # Calculate drawdown curve and max drawdown
    equities = [e['equity'] for e in equity_curve]
    peak = initial_capital
    drawdowns = []
    for eq in equities:
        if eq > peak:
            peak = eq
        drawdown = (peak - eq) / peak * 100 if peak > 0 else 0.0
        drawdowns.append(drawdown)

    max_drawdown = max(drawdowns) if drawdowns else 0.0
    drawdown_curve = [
        {'date': equity_curve[i]['date'], 'drawdown': drawdowns[i]}
        for i in range(len(drawdowns))
    ]

    # Sharpe ratio (simplified - using trade returns)
    if len(trades) > 1:
        returns = [t.pnl_pct for t in trades]
        avg_return = np.mean(returns)
        std_return = np.std(returns)
        sharpe_ratio = (avg_return / std_return * np.sqrt(252)) if std_return > 0 else 0.0
    else:
        sharpe_ratio = 0.0

    # Format trades for storage
    trades_list = [
        {
            'entry_time': t.entry_time.isoformat() if hasattr(t.entry_time, 'isoformat') else str(t.entry_time),
            'exit_time': t.exit_time.isoformat() if hasattr(t.exit_time, 'isoformat') else str(t.exit_time),
            'direction': t.direction,
            'entry_price': t.entry_price,
            'exit_price': t.exit_price,
            'size': t.size,
            'pnl': t.pnl,
            'pnl_pct': t.pnl_pct,
            'bars_held': t.bars_held
        }
        for t in trades
    ]

    return {
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'win_rate': round(win_rate, 2),
        'total_return': round(total_return, 2),
        'sharpe_ratio': round(sharpe_ratio, 2),
        'max_drawdown': round(max_drawdown, 2),
        'profit_factor': round(profit_factor, 2) if profit_factor != float('inf') else 999.99,
        'avg_trade_duration': round(avg_trade_duration, 1),
        'final_equity': round(final_equity, 2),
        'equity_curve': equity_curve,
        'drawdown_curve': drawdown_curve,
        'trades': trades_list
    }


def handle_backtest(task_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute a backtest.

    Args:
        task_id: The background task ID
        payload: Dict containing backtest_id and configuration

    Returns:
        Result dict with status and metrics
    """
    backtest_id = payload.get('backtest_id')
    if not backtest_id:
        return {'status': 'failed', 'error': 'backtest_id is required'}

    logger.info(f"Starting backtest execution for backtest_id={backtest_id}")

    db = SessionLocal()
    try:
        # Get backtest record
        backtest = db.query(Backtest).filter(Backtest.id == backtest_id).first()
        if not backtest:
            return {'status': 'failed', 'error': f'Backtest {backtest_id} not found'}

        # Update status to running
        backtest.status = 'running'
        db.commit()

        # Get related records
        model = db.query(TrainedModel).filter(TrainedModel.id == backtest.model_id).first()
        pred_dataset = db.query(Dataset).filter(Dataset.id == backtest.prediction_dataset_id).first()
        exec_dataset = db.query(Dataset).filter(Dataset.id == backtest.execution_dataset_id).first()

        if not model:
            backtest.status = 'failed'
            backtest.error_message = f'Model {backtest.model_id} not found'
            db.commit()
            return {'status': 'failed', 'error': backtest.error_message}

        if not pred_dataset or not exec_dataset:
            backtest.status = 'failed'
            backtest.error_message = 'Dataset not found'
            db.commit()
            return {'status': 'failed', 'error': backtest.error_message}

        # Get strategy and conditions
        strategy_params = backtest.strategy_params or {}
        buy_entry_conditions = None
        sell_entry_conditions = None
        exit_conditions = None

        if backtest.strategy_id:
            # Load conditions from saved strategy in database
            strategy = db.query(Strategy).filter(Strategy.id == backtest.strategy_id).first()
            if strategy:
                buy_entry_conditions = strategy.buy_entry_conditions
                sell_entry_conditions = strategy.sell_entry_conditions
                exit_conditions = strategy.exit_conditions
                # Merge strategy TP/SL params with backtest-specific overrides
                strategy_base_params = {
                    'initial_tp_percent': strategy.initial_tp_percent or 5.0,
                    'initial_sl_percent': strategy.initial_sl_percent or 2.0,
                }
                strategy_params = {**strategy_base_params, **strategy_params}
        else:
            # Extract conditions from inline strategy_params (frontend sends camelCase)
            buy_entry_conditions = strategy_params.get('buyEntryConditions') or strategy_params.get('buy_entry_conditions')
            sell_entry_conditions = strategy_params.get('sellEntryConditions') or strategy_params.get('sell_entry_conditions')
            exit_conditions = strategy_params.get('exitConditions') or strategy_params.get('exit_conditions')

        logger.info(f"Strategy conditions loaded: buy={buy_entry_conditions is not None}, sell={sell_entry_conditions is not None}, exit={len(exit_conditions) if exit_conditions else 0} rules")

        # Load prediction dataset
        try:
            pred_df = pd.read_csv(pred_dataset.file_path)
            if 'Date' in pred_df.columns:
                pred_df['Date'] = pd.to_datetime(pred_df['Date'])
        except Exception as e:
            backtest.status = 'failed'
            backtest.error_message = f'Failed to load prediction dataset: {e}'
            db.commit()
            return {'status': 'failed', 'error': backtest.error_message}

        # Load execution dataset
        try:
            exec_df = pd.read_csv(exec_dataset.file_path)
            if 'Date' in exec_df.columns:
                exec_df['Date'] = pd.to_datetime(exec_df['Date'])
        except Exception as e:
            backtest.status = 'failed'
            backtest.error_message = f'Failed to load execution dataset: {e}'
            db.commit()
            return {'status': 'failed', 'error': backtest.error_message}

        # Filter by date range
        if backtest.start_date:
            pred_df = pred_df[pred_df['Date'] >= backtest.start_date]
            exec_df = exec_df[exec_df['Date'] >= backtest.start_date]
        if backtest.end_date:
            pred_df = pred_df[pred_df['Date'] <= backtest.end_date]
            exec_df = exec_df[exec_df['Date'] <= backtest.end_date]

        # Run the backtest
        results = run_backtest(
            model=model,
            pred_df=pred_df,
            exec_df=exec_df,
            strategy_params=strategy_params,
            initial_capital=backtest.initial_capital,
            position_sizing_type=backtest.position_sizing_type,
            position_sizing_value=backtest.position_sizing_value,
            commission=backtest.commission or 0.0,
            slippage=backtest.slippage or 0.0,
            buy_entry_conditions=buy_entry_conditions,
            sell_entry_conditions=sell_entry_conditions,
            exit_conditions=exit_conditions,
        )

        # Update backtest with results
        backtest.status = 'completed'
        backtest.completed_at = datetime.now()
        backtest.total_trades = results['total_trades']
        backtest.winning_trades = results['winning_trades']
        backtest.losing_trades = results['losing_trades']
        backtest.win_rate = results['win_rate']
        backtest.total_return = results['total_return']
        backtest.sharpe_ratio = results['sharpe_ratio']
        backtest.max_drawdown = results['max_drawdown']
        backtest.profit_factor = results['profit_factor']
        backtest.avg_trade_duration = results['avg_trade_duration']
        backtest.final_equity = results['final_equity']
        backtest.equity_curve = results['equity_curve']
        backtest.drawdown_curve = results['drawdown_curve']
        backtest.trades = results['trades']

        db.commit()

        logger.info(f"Backtest {backtest_id} completed: {results['total_trades']} trades")

        return {
            'status': 'completed',
            'backtest_id': backtest_id,
            'results': results
        }

    except Exception as e:
        logger.error(f"Backtest {backtest_id} failed: {e}")
        try:
            backtest = db.query(Backtest).filter(Backtest.id == backtest_id).first()
            if backtest:
                backtest.status = 'failed'
                backtest.error_message = str(e)
                db.commit()
        except:
            pass
        return {'status': 'failed', 'error': str(e)}
    finally:
        db.close()
