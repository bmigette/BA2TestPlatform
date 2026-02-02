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
from app.services.strategy_executor import StrategyExecutor, evaluate_condition_tree
from app.services.data_preparation import DataPreparationService
from app.services.tsai_training import TSAITrainingService

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

    # Prepare features for prediction first (need dimensions for model creation)
    # Exclude non-feature columns
    exclude_cols = {'Date', 'target', 'Open', 'High', 'Low', 'Close', 'Volume'}
    feature_cols = [c for c in pred_df.columns if c not in exclude_cols]

    if not feature_cols:
        logger.error("No feature columns found in prediction dataset")
        return _empty_results(initial_capital)

    # Apply normalization if available
    if model.normalization_params:
        data_prep = DataPreparationService()
        data_prep.load_params(model.normalization_params)
        features = data_prep.transform(pred_df[feature_cols].values)
    else:
        features = pred_df[feature_cols].values

    # Create sliding windows for prediction
    n_samples = len(features) - seq_len + 1
    if n_samples <= 0:
        logger.error(f"Not enough data for seq_len={seq_len}")
        return _empty_results(initial_capital)

    X = np.array([features[i:i+seq_len] for i in range(n_samples)])
    X = X.transpose(0, 2, 1)  # (samples, features, seq_len) for tsai

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
            c_in = X.shape[1]  # number of features
            c_out = hyperparameters.get('c_out', 2)  # number of classes
            model_params = hyperparameters.get('modelParams', {})

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

    # Align predictions with execution data
    # Predictions start at index seq_len-1 (after we have enough history)
    pred_start_idx = seq_len - 1
    pred_dates = pred_df['Date'].iloc[pred_start_idx:pred_start_idx + len(predictions)].values

    # Create prediction lookup by date
    pred_lookup = dict(zip(pred_dates, predictions))

    # Run simulation
    equity = initial_capital
    open_positions: List[OpenPosition] = []
    completed_trades: List[Trade] = []
    equity_curve = [{'date': exec_df['Date'].iloc[0].isoformat() if len(exec_df) > 0 else '', 'equity': equity}]

    exit_conditions = exit_conditions or []

    for idx in range(len(exec_df)):
        row = exec_df.iloc[idx]
        current_date = row['Date']
        current_price = row['Close']

        # Get prediction for this bar
        prob = pred_lookup.get(current_date, None)
        if prob is None:
            # No prediction available for this bar
            for pos in open_positions:
                pos.bars_held += 1
            continue

        # Build context for condition evaluation
        context = {
            'model:prediction': 1 if prob >= threshold else 0,
            'model:probability': prob,
            'model:probability_0': 1 - prob,
            'model:probability_1': prob,
            'Open': row.get('Open', current_price),
            'High': row.get('High', current_price),
            'Low': row.get('Low', current_price),
            'Close': current_price,
            'Volume': row.get('Volume', 0),
            'position:in_position': len(open_positions) > 0,
        }

        # Add any additional columns from exec_df
        for col in exec_df.columns:
            if col not in context and col != 'Date':
                context[col] = row[col]

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
                if evaluate_condition_tree(conditions, pos_context):
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

        # Check entry conditions
        if len(open_positions) == 0:  # Only enter if no position
            # Check buy entry
            if buy_entry_conditions and evaluate_condition_tree(buy_entry_conditions, context):
                entry_price = current_price * (1 + slippage / 100)
                size = _calculate_position_size(equity, position_sizing_type, position_sizing_value, entry_price)
                if size > 0:
                    open_positions.append(OpenPosition(
                        entry_time=current_date,
                        direction='buy',
                        entry_price=entry_price,
                        size=size
                    ))

            # Check sell entry
            elif sell_entry_conditions and evaluate_condition_tree(sell_entry_conditions, context):
                entry_price = current_price * (1 - slippage / 100)
                size = _calculate_position_size(equity, position_sizing_type, position_sizing_value, entry_price)
                if size > 0:
                    open_positions.append(OpenPosition(
                        entry_time=current_date,
                        direction='sell',
                        entry_price=entry_price,
                        size=size
                    ))

        # Record equity
        equity_curve.append({
            'date': current_date.isoformat() if hasattr(current_date, 'isoformat') else str(current_date),
            'equity': equity
        })

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
            strategy = db.query(Strategy).filter(Strategy.id == backtest.strategy_id).first()
            if strategy:
                buy_entry_conditions = strategy.buy_entry_conditions
                sell_entry_conditions = strategy.sell_entry_conditions
                exit_conditions = strategy.exit_conditions
                # Merge strategy params with backtest-specific overrides
                if strategy.strategy_params:
                    strategy_params = {**strategy.strategy_params, **strategy_params}

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
