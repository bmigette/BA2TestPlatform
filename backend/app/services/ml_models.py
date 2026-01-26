"""
ML Models Service

Provides machine learning model architectures using PyTorch and Darts library.
Supports LSTM, N-BEATS, RNN for timeseries forecasting.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
import os

logger = logging.getLogger(__name__)

# Check for PyTorch availability
try:
    import torch
    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
    GPU_NAME = torch.cuda.get_device_name(0) if CUDA_AVAILABLE else None
    logger.info(f"PyTorch available. CUDA: {CUDA_AVAILABLE}, GPU: {GPU_NAME}")
except ImportError:
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False
    GPU_NAME = None
    logger.warning("PyTorch not available. Install with: pip install torch")

# Check for Darts availability
try:
    from darts import TimeSeries
    from darts.models import (
        RNNModel,
        NBEATSModel,
        TFTModel,
        BlockRNNModel
    )
    from darts.dataprocessing.transformers import Scaler
    DARTS_AVAILABLE = True
    logger.info("Darts library available")
except ImportError:
    DARTS_AVAILABLE = False
    logger.warning("Darts not available. Install with: pip install darts")


class MLModelsService:
    """
    Service for creating and managing ML model architectures.

    Supports:
    - LSTM (Long Short-Term Memory)
    - N-BEATS (Neural Basis Expansion Analysis)
    - RNN (Recurrent Neural Network)
    - TFT (Temporal Fusion Transformer)
    """

    # Model architecture configurations
    MODEL_ARCHITECTURES = {
        'lstm': {
            'name': 'LSTM',
            'description': 'Long Short-Term Memory network for sequence modeling',
            'default_params': {
                'input_chunk_length': 30,
                'output_chunk_length': 7,
                'hidden_dim': 64,
                'n_rnn_layers': 2,
                'dropout': 0.1,
                'batch_size': 32,
                'n_epochs': 100
            }
        },
        'nbeats': {
            'name': 'N-BEATS',
            'description': 'Neural Basis Expansion Analysis for Time Series',
            'default_params': {
                'input_chunk_length': 30,
                'output_chunk_length': 7,
                'num_stacks': 30,
                'num_blocks': 1,
                'num_layers': 4,
                'layer_widths': 256,
                'batch_size': 32,
                'n_epochs': 100
            }
        },
        'rnn': {
            'name': 'RNN',
            'description': 'Recurrent Neural Network for sequence prediction',
            'default_params': {
                'input_chunk_length': 30,
                'output_chunk_length': 7,
                'hidden_dim': 64,
                'n_rnn_layers': 2,
                'dropout': 0.1,
                'batch_size': 32,
                'n_epochs': 100
            }
        },
        'tft': {
            'name': 'Temporal Fusion Transformer',
            'description': 'Transformer-based model for multi-horizon forecasting',
            'default_params': {
                'input_chunk_length': 30,
                'output_chunk_length': 7,
                'hidden_size': 64,
                'lstm_layers': 1,
                'num_attention_heads': 4,
                'dropout': 0.1,
                'batch_size': 32,
                'n_epochs': 100
            }
        }
    }

    def __init__(self, use_gpu: bool = True):
        """
        Initialize MLModelsService.

        Args:
            use_gpu: Whether to use GPU if available
        """
        self.use_gpu = use_gpu and CUDA_AVAILABLE
        self.device = 'cuda' if self.use_gpu else 'cpu'

        if self.use_gpu:
            logger.info(f"Using GPU: {GPU_NAME}")
        else:
            logger.info("Using CPU for model training")

    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        """
        Get system information for ML training.

        Returns:
            Dictionary with system capabilities
        """
        info = {
            'pytorch_available': TORCH_AVAILABLE,
            'darts_available': DARTS_AVAILABLE,
            'cuda_available': CUDA_AVAILABLE,
            'gpu_name': GPU_NAME,
            'gpu_memory_total': None,
            'gpu_memory_free': None
        }

        if TORCH_AVAILABLE and CUDA_AVAILABLE:
            try:
                info['gpu_memory_total'] = torch.cuda.get_device_properties(0).total_memory
                info['gpu_memory_free'] = torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0)
            except Exception:
                pass

        return info

    def create_lstm_model(self, params: Dict = None) -> Any:
        """
        Create LSTM model architecture using Darts.

        Args:
            params: Model parameters (uses defaults if not provided)
                   - hidden_dim: Can be int (same for all layers) or list/tuple (per-layer)

        Returns:
            Darts RNNModel configured as LSTM
        """
        if not DARTS_AVAILABLE:
            raise RuntimeError("Darts library not available")

        p = {**self.MODEL_ARCHITECTURES['lstm']['default_params'], **(params or {})}

        # Handle per-layer hidden dimensions
        hidden_dim = p['hidden_dim']
        n_rnn_layers = p['n_rnn_layers']

        # If hidden_dim is a list/tuple, validate length matches n_rnn_layers
        if isinstance(hidden_dim, (list, tuple)):
            if len(hidden_dim) != n_rnn_layers:
                # Truncate or extend to match n_rnn_layers
                if len(hidden_dim) > n_rnn_layers:
                    hidden_dim = hidden_dim[:n_rnn_layers]
                else:
                    # Extend with last value
                    hidden_dim = list(hidden_dim) + [hidden_dim[-1]] * (n_rnn_layers - len(hidden_dim))
            hidden_dim = tuple(hidden_dim)
            logger.info(f"Using per-layer hidden dimensions: {hidden_dim}")

        model = RNNModel(
            model='LSTM',
            input_chunk_length=p['input_chunk_length'],
            output_chunk_length=p['output_chunk_length'],
            hidden_dim=hidden_dim,
            n_rnn_layers=n_rnn_layers,
            dropout=p['dropout'],
            batch_size=p['batch_size'],
            n_epochs=p['n_epochs'],
            optimizer_kwargs={'lr': p.get('learning_rate', 1e-3)},
            pl_trainer_kwargs={
                'accelerator': 'gpu' if self.use_gpu else 'cpu',
                'devices': 1 if self.use_gpu else 'auto'
            }
        )

        logger.info(f"Created LSTM model with params: {p}")
        return model

    def create_nbeats_model(self, params: Dict = None) -> Any:
        """
        Create N-BEATS model architecture using Darts.

        Args:
            params: Model parameters (uses defaults if not provided)
                   - layer_widths: Can be int (same for all layers) or list (per-layer)

        Returns:
            Darts NBEATSModel
        """
        if not DARTS_AVAILABLE:
            raise RuntimeError("Darts library not available")

        p = {**self.MODEL_ARCHITECTURES['nbeats']['default_params'], **(params or {})}

        # Handle per-layer widths
        layer_widths = p['layer_widths']
        num_layers = p['num_layers']

        # If layer_widths is a list, validate length matches num_layers
        if isinstance(layer_widths, (list, tuple)):
            if len(layer_widths) != num_layers:
                # Truncate or extend to match num_layers
                if len(layer_widths) > num_layers:
                    layer_widths = list(layer_widths[:num_layers])
                else:
                    # Extend with last value
                    layer_widths = list(layer_widths) + [layer_widths[-1]] * (num_layers - len(layer_widths))
            logger.info(f"Using per-layer widths: {layer_widths}")

        model = NBEATSModel(
            input_chunk_length=p['input_chunk_length'],
            output_chunk_length=p['output_chunk_length'],
            num_stacks=p['num_stacks'],
            num_blocks=p['num_blocks'],
            num_layers=num_layers,
            layer_widths=layer_widths,
            batch_size=p['batch_size'],
            n_epochs=p['n_epochs'],
            optimizer_kwargs={'lr': p.get('learning_rate', 1e-3)},
            pl_trainer_kwargs={
                'accelerator': 'gpu' if self.use_gpu else 'cpu',
                'devices': 1 if self.use_gpu else 'auto'
            }
        )

        logger.info(f"Created N-BEATS model with params: {p}")
        return model

    def create_rnn_model(self, params: Dict = None) -> Any:
        """
        Create RNN model architecture using Darts.

        Args:
            params: Model parameters (uses defaults if not provided)
                   - hidden_dim: Can be int (same for all layers) or list/tuple (per-layer)

        Returns:
            Darts RNNModel configured as vanilla RNN
        """
        if not DARTS_AVAILABLE:
            raise RuntimeError("Darts library not available")

        p = {**self.MODEL_ARCHITECTURES['rnn']['default_params'], **(params or {})}

        # Handle per-layer hidden dimensions
        hidden_dim = p['hidden_dim']
        n_rnn_layers = p['n_rnn_layers']

        # If hidden_dim is a list/tuple, validate length matches n_rnn_layers
        if isinstance(hidden_dim, (list, tuple)):
            if len(hidden_dim) != n_rnn_layers:
                # Truncate or extend to match n_rnn_layers
                if len(hidden_dim) > n_rnn_layers:
                    hidden_dim = hidden_dim[:n_rnn_layers]
                else:
                    # Extend with last value
                    hidden_dim = list(hidden_dim) + [hidden_dim[-1]] * (n_rnn_layers - len(hidden_dim))
            hidden_dim = tuple(hidden_dim)
            logger.info(f"Using per-layer hidden dimensions: {hidden_dim}")

        model = RNNModel(
            model='RNN',
            input_chunk_length=p['input_chunk_length'],
            output_chunk_length=p['output_chunk_length'],
            hidden_dim=hidden_dim,
            n_rnn_layers=n_rnn_layers,
            dropout=p['dropout'],
            batch_size=p['batch_size'],
            n_epochs=p['n_epochs'],
            optimizer_kwargs={'lr': p.get('learning_rate', 1e-3)},
            pl_trainer_kwargs={
                'accelerator': 'gpu' if self.use_gpu else 'cpu',
                'devices': 1 if self.use_gpu else 'auto'
            }
        )

        logger.info(f"Created RNN model with params: {p}")
        return model

    def create_model(self, model_type: str, params: Dict = None) -> Any:
        """
        Create a model of the specified type.

        Args:
            model_type: One of 'lstm', 'nbeats', 'rnn', 'tft'
            params: Model parameters

        Returns:
            Configured Darts model
        """
        creators = {
            'lstm': self.create_lstm_model,
            'nbeats': self.create_nbeats_model,
            'rnn': self.create_rnn_model
        }

        if model_type not in creators:
            raise ValueError(f"Unknown model type: {model_type}. Supported: {list(creators.keys())}")

        return creators[model_type](params)

    @staticmethod
    def get_available_models() -> Dict[str, Dict]:
        """
        Get list of available model architectures.

        Returns:
            Dictionary of model configurations
        """
        return MLModelsService.MODEL_ARCHITECTURES.copy()


class PredictionTargetService:
    """
    Service for calculating prediction targets for ML training.

    Creates binary classification targets like:
    - price_up_10pct_5dd_7d: Price goes up 10% with max 5% drawdown in 7 days
    - price_down_10pct_5dd_7d: Price goes down 10% with max 5% drawup in 7 days
    """

    def __init__(self):
        pass

    def calculate_prediction_targets(
        self,
        df: pd.DataFrame,
        targets: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Calculate prediction targets for a dataset.

        Args:
            df: DataFrame with Date, Close columns
            targets: List of target configurations, e.g.:
                [{'profit_pct': 10, 'max_dd': 5, 'days': 7, 'direction': 'up'}]

        Returns:
            DataFrame with target columns added
        """
        result_df = df.copy()
        result_df = result_df.sort_values('Date').reset_index(drop=True)

        for target in targets:
            profit_pct = target.get('profit_pct', 10)
            max_dd = target.get('max_dd', 5)
            days = target.get('days', 7)
            direction = target.get('direction', 'up')

            col_name = f"price_{direction}_{profit_pct}pct_{max_dd}dd_{days}d"

            result_df[col_name] = self._calculate_single_target(
                result_df, profit_pct, max_dd, days, direction
            )

            logger.info(f"Calculated target: {col_name}")

        return result_df

    def _calculate_single_target(
        self,
        df: pd.DataFrame,
        profit_pct: float,
        max_dd: float,
        days: int,
        direction: str
    ) -> pd.Series:
        """
        Calculate a single prediction target.

        Args:
            df: DataFrame with Close column
            profit_pct: Required profit percentage
            max_dd: Maximum drawdown allowed
            days: Time horizon in days
            direction: 'up' or 'down'

        Returns:
            Series with 1 where target is met, 0 otherwise
        """
        n = len(df)
        targets = np.zeros(n)

        close_prices = df['Close'].values

        for i in range(n - days):
            entry_price = close_prices[i]
            future_prices = close_prices[i+1:i+days+1]

            if len(future_prices) < days:
                continue

            if direction == 'up':
                # Check if price goes up by profit_pct without drawdown exceeding max_dd
                max_price = np.max(future_prices)
                min_price = np.min(future_prices)

                profit = (max_price - entry_price) / entry_price * 100
                drawdown = (entry_price - min_price) / entry_price * 100

                if profit >= profit_pct and drawdown <= max_dd:
                    targets[i] = 1

            else:  # down
                # Check if price goes down by profit_pct without drawup exceeding max_dd
                max_price = np.max(future_prices)
                min_price = np.min(future_prices)

                profit = (entry_price - min_price) / entry_price * 100
                drawup = (max_price - entry_price) / entry_price * 100

                if profit >= profit_pct and drawup <= max_dd:
                    targets[i] = 1

        return pd.Series(targets, index=df.index)

    def create_symmetric_targets(
        self,
        df: pd.DataFrame,
        profit_pct: float = 10,
        max_dd: float = 5,
        days: int = 7
    ) -> pd.DataFrame:
        """
        Create symmetric up/down prediction targets.

        Args:
            df: DataFrame with Date, Close columns
            profit_pct: Profit target percentage
            max_dd: Maximum drawdown percentage
            days: Time horizon in days

        Returns:
            DataFrame with both up and down target columns
        """
        targets = [
            {'profit_pct': profit_pct, 'max_dd': max_dd, 'days': days, 'direction': 'up'},
            {'profit_pct': profit_pct, 'max_dd': max_dd, 'days': days, 'direction': 'down'}
        ]

        return self.calculate_prediction_targets(df, targets)

    def verify_symmetry(self, df: pd.DataFrame, targets: List[Dict]) -> bool:
        """
        Verify that prediction targets maintain symmetry constraint.

        Args:
            df: DataFrame with target columns
            targets: List of target configurations

        Returns:
            True if symmetry is maintained
        """
        up_targets = [t for t in targets if t.get('direction') == 'up']
        down_targets = [t for t in targets if t.get('direction') == 'down']

        if len(up_targets) != len(down_targets):
            return False

        # Check that each up target has a matching down target
        for up in up_targets:
            matching_down = False
            for down in down_targets:
                if (up['profit_pct'] == down['profit_pct'] and
                    up['max_dd'] == down['max_dd'] and
                    up['days'] == down['days']):
                    matching_down = True
                    break
            if not matching_down:
                return False

        return True


class ClassImbalanceConfig:
    """
    Configuration for handling class imbalance in binary classification.

    Provides factory methods for getting loss functions and fitness metrics
    optimized for imbalanced datasets (common with prediction targets).
    """

    # Available loss functions
    LOSS_FUNCTIONS = {
        'cross_entropy': {
            'name': 'Cross-Entropy',
            'description': 'Standard cross-entropy loss. NOT recommended for imbalanced data.',
            'recommended_for': 'Balanced datasets only'
        },
        'weighted_cross_entropy': {
            'name': 'Weighted Cross-Entropy',
            'description': 'Cross-entropy with class weights based on frequency.',
            'recommended_for': 'Moderate imbalance (10-30% minority class)'
        },
        'focal_loss': {
            'name': 'Focal Loss',
            'description': 'Down-weights easy examples, focuses on hard ones. Best for severe imbalance.',
            'recommended_for': 'Severe imbalance (<10% minority class)',
            'default_gamma': 2.0
        }
    }

    # Available fitness metrics for genetic algorithm
    FITNESS_METRICS = {
        'accuracy': {
            'name': 'Accuracy',
            'description': 'Proportion of correct predictions. NOT recommended for imbalanced data.',
            'recommended_for': 'Balanced datasets only'
        },
        'f1_score': {
            'name': 'F1 Score',
            'description': 'Harmonic mean of precision and recall. Best general-purpose metric.',
            'recommended_for': 'Most imbalanced scenarios (DEFAULT)'
        },
        'precision': {
            'name': 'Precision',
            'description': 'Minimize false positives. Use when false alarms are costly.',
            'recommended_for': 'When avoiding false positives is critical'
        },
        'recall': {
            'name': 'Recall',
            'description': 'Minimize false negatives. Use when catching all positives is critical.',
            'recommended_for': 'When missing positive cases is critical'
        },
        'auc_roc': {
            'name': 'AUC-ROC',
            'description': 'Area under ROC curve. Threshold-independent metric.',
            'recommended_for': 'When threshold will be tuned later'
        },
        'balanced_accuracy': {
            'name': 'Balanced Accuracy',
            'description': 'Average of per-class recall. Simple balanced metric.',
            'recommended_for': 'Quick balanced evaluation'
        }
    }

    @staticmethod
    def get_loss_function(
        loss_type: str,
        positive_count: int = None,
        negative_count: int = None,
        gamma: float = 2.0,
        alpha: float = None
    ):
        """
        Get a PyTorch loss function for training.

        Args:
            loss_type: One of 'cross_entropy', 'weighted_cross_entropy', 'focal_loss'
            positive_count: Number of positive samples (for auto-weighting)
            negative_count: Number of negative samples (for auto-weighting)
            gamma: Focal loss gamma parameter (focusing strength)
            alpha: Optional alpha override for focal loss

        Returns:
            PyTorch loss module
        """
        from app.services.losses import get_loss_function
        return get_loss_function(loss_type, positive_count, negative_count, gamma, alpha)

    @staticmethod
    def get_recommended_config(positive_count: int, negative_count: int) -> dict:
        """
        Get recommended loss function and fitness metric based on class distribution.

        Args:
            positive_count: Number of positive samples
            negative_count: Number of negative samples

        Returns:
            Dictionary with recommended configuration
        """
        total = positive_count + negative_count
        positive_pct = (positive_count / total * 100) if total > 0 else 50

        if positive_pct < 5:
            # Extreme imbalance
            return {
                'loss_function': 'focal_loss',
                'gamma': 2.5,  # Higher gamma for extreme imbalance
                'fitness_metric': 'f1_score',
                'warning': f'Extreme class imbalance ({positive_pct:.1f}% positive). '
                          f'Model may struggle. Consider adjusting targets.'
            }
        elif positive_pct < 10:
            # Severe imbalance
            return {
                'loss_function': 'focal_loss',
                'gamma': 2.0,
                'fitness_metric': 'f1_score',
                'warning': f'Severe class imbalance ({positive_pct:.1f}% positive). '
                          f'Using Focal Loss with F1 metric.'
            }
        elif positive_pct < 30:
            # Moderate imbalance
            return {
                'loss_function': 'weighted_cross_entropy',
                'fitness_metric': 'f1_score',
                'warning': None
            }
        else:
            # Relatively balanced
            return {
                'loss_function': 'cross_entropy',
                'fitness_metric': 'accuracy',
                'warning': None
            }

    @staticmethod
    def get_available_configs() -> dict:
        """Get all available loss functions and fitness metrics."""
        return {
            'loss_functions': ClassImbalanceConfig.LOSS_FUNCTIONS,
            'fitness_metrics': ClassImbalanceConfig.FITNESS_METRICS,
            'defaults': {
                'loss_function': 'focal_loss',
                'fitness_metric': 'f1_score',
                'gamma': 2.0
            }
        }


class DatasetSplitter:
    """
    Service for splitting datasets into train/test sets.
    """

    @staticmethod
    def train_test_split(
        df: pd.DataFrame,
        train_ratio: float = 0.8,
        shuffle: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split dataset into train and test sets.

        Args:
            df: DataFrame to split
            train_ratio: Proportion of data for training (default: 0.8)
            shuffle: Whether to shuffle before splitting (default: False for timeseries)

        Returns:
            Tuple of (train_df, test_df)
        """
        df_sorted = df.sort_values('Date').reset_index(drop=True)

        if shuffle:
            df_sorted = df_sorted.sample(frac=1, random_state=42).reset_index(drop=True)

        split_idx = int(len(df_sorted) * train_ratio)

        train_df = df_sorted.iloc[:split_idx].copy()
        test_df = df_sorted.iloc[split_idx:].copy()

        logger.info(f"Split dataset: train={len(train_df)} rows, test={len(test_df)} rows")

        return train_df, test_df

    @staticmethod
    def walk_forward_split(
        df: pd.DataFrame,
        n_splits: int = 5,
        test_ratio: float = 0.2
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Create walk-forward validation splits for time series.

        Args:
            df: DataFrame to split
            n_splits: Number of train/test splits
            test_ratio: Proportion of data for each test set

        Returns:
            List of (train_df, test_df) tuples
        """
        df_sorted = df.sort_values('Date').reset_index(drop=True)
        n = len(df_sorted)

        splits = []
        test_size = int(n * test_ratio)

        for i in range(n_splits):
            # Growing training set
            train_end = int(n * (1 - test_ratio * (n_splits - i) / n_splits))
            test_end = min(train_end + test_size, n)

            train_df = df_sorted.iloc[:train_end].copy()
            test_df = df_sorted.iloc[train_end:test_end].copy()

            if len(test_df) > 0:
                splits.append((train_df, test_df))

        logger.info(f"Created {len(splits)} walk-forward splits")
        return splits
