"""
tsai Training Service for classification models.

Provides training, evaluation, and prediction functionality using the tsai/fastai library.
Supports focal loss, class weighting, and comprehensive classification metrics.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Check for tsai availability
TSAI_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    from tsai.all import (
        TSClassifier, Learner, DataLoaders,
        get_ts_dls, TSStandardize, TSClassification,
        accuracy, F1Score, Precision, Recall,
    )
    from fastai.callback.core import Callback
    from fastai.losses import CrossEntropyLossFlat, FocalLossFlat
    TSAI_AVAILABLE = True
    logger.info("tsai training service available")
except ImportError as e:
    logger.warning(f"tsai training not available: {e}")

from app.services.model_interface import ITrainingService
from app.services.tsai_models import DEVICE, MPS_AVAILABLE, CUDA_AVAILABLE


class TSAITrainingService(ITrainingService):
    """
    tsai-based training service for time series classification.
    """

    def __init__(self, models_dir: str = "trained_models"):
        """Initialize TSAITrainingService."""
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)

    def prepare_data(
        self,
        df: pd.DataFrame,
        target_column: str,
        feature_columns: List[str],
        timeframe: str = 'daily',
        seq_len: int = 24
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for tsai model training.

        Args:
            df: DataFrame with features and target
            target_column: Name of target column
            feature_columns: List of feature column names
            timeframe: Data timeframe (not used for tsai but kept for interface)
            seq_len: Sequence length for sliding window

        Returns:
            Tuple of (X, y) numpy arrays
        """
        if not TSAI_AVAILABLE:
            raise RuntimeError("tsai library not available")

        # Extract features and target
        X_data = df[feature_columns].values
        y_data = df[target_column].values.astype(np.int64)

        # Create sliding window sequences
        X, y = self._create_sequences(X_data, y_data, seq_len)

        logger.info(f"Prepared data: X shape {X.shape}, y shape {y.shape}")
        return X, y

    def prepare_data_split(
        self,
        df: pd.DataFrame,
        train_ratio: float,
        target_column: str,
        feature_columns: List[str],
        timeframe: str = 'daily',
        seq_len: int = 24
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare and split data into train/test sets.

        Args:
            df: Full DataFrame
            train_ratio: Fraction for training (0.0-1.0)
            target_column: Name of target column
            feature_columns: List of feature column names
            timeframe: Data timeframe
            seq_len: Sequence length for sliding window

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        X, y = self.prepare_data(df, target_column, feature_columns, timeframe, seq_len)

        # Split by time (no shuffling for time series)
        split_idx = int(len(X) * train_ratio)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        logger.info(f"Split data: train={len(X_train)}, test={len(X_test)}")
        return X_train, X_test, y_train, y_test

    def _create_sequences(
        self,
        X: np.ndarray,
        y: np.ndarray,
        seq_len: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sliding window sequences from data."""
        n_samples = len(X) - seq_len + 1
        n_features = X.shape[1]

        # Shape: (samples, features, seq_len) for tsai
        X_seq = np.zeros((n_samples, n_features, seq_len), dtype=np.float32)
        y_seq = np.zeros(n_samples, dtype=np.int64)

        for i in range(n_samples):
            # Transpose to get (features, seq_len)
            X_seq[i] = X[i:i+seq_len].T
            y_seq[i] = y[i + seq_len - 1]  # Target at end of sequence

        return X_seq, y_seq

    def get_loss_function(
        self,
        loss_type: str = 'focal',
        alpha: float = None,
        gamma: float = 2.0,
        class_weights: np.ndarray = None
    ) -> Any:
        """
        Get loss function for classification.

        Args:
            loss_type: 'focal', 'ce' (cross-entropy), 'weighted_ce'
            alpha: Alpha for focal loss (class balance)
            gamma: Gamma for focal loss (focusing parameter)
            class_weights: Class weights for weighted CE

        Returns:
            Loss function
        """
        if not TSAI_AVAILABLE:
            raise RuntimeError("tsai library not available")

        if loss_type == 'focal':
            return FocalLossFlat(gamma=gamma)
        elif loss_type == 'ce':
            return CrossEntropyLossFlat()
        elif loss_type == 'weighted_ce' and class_weights is not None:
            weights = torch.tensor(class_weights, dtype=torch.float32)
            if DEVICE:
                weights = weights.to(DEVICE)
            return CrossEntropyLossFlat(weight=weights)
        else:
            return CrossEntropyLossFlat()

    def train_model(
        self,
        model: Any,
        train_data: Tuple[np.ndarray, np.ndarray],
        val_data: Tuple[np.ndarray, np.ndarray] = None,
        epochs: int = 50,
        batch_size: int = 64,
        learning_rate: float = 0.001,
        loss_fn: Any = None,
        epoch_callback: callable = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train a tsai model.

        Args:
            model: tsai model architecture (nn.Module)
            train_data: Tuple of (X_train, y_train)
            val_data: Optional tuple of (X_val, y_val)
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            loss_fn: Optional custom loss function
            epoch_callback: Optional callback for progress updates
            **kwargs: Additional options

        Returns:
            Training result with status, metrics, trained learner
        """
        if not TSAI_AVAILABLE:
            return {'status': 'failed', 'error': 'tsai not available'}

        try:
            X_train, y_train = train_data

            # Combine train and val for dataloaders
            if val_data is not None:
                X_val, y_val = val_data
                X_all = np.concatenate([X_train, X_val])
                y_all = np.concatenate([y_train, y_val])
                # Splits: train indices, then val indices
                splits = (list(range(len(X_train))),
                         list(range(len(X_train), len(X_all))))
            else:
                X_all, y_all = X_train, y_train
                # 80/20 split if no validation provided
                split_idx = int(len(X_all) * 0.8)
                splits = (list(range(split_idx)),
                         list(range(split_idx, len(X_all))))

            # Create dataloaders
            dls = get_ts_dls(
                X_all, y_all,
                splits=splits,
                bs=batch_size,
                batch_tfms=[TSStandardize()],
            )

            # Get loss function
            if loss_fn is None:
                loss_fn = self.get_loss_function('focal')

            # Create learner with model
            learn = Learner(
                dls,
                model,
                loss_func=loss_fn,
                metrics=[accuracy, F1Score(), Precision(), Recall()],
            )

            # Move to device
            if DEVICE:
                learn.model = learn.model.to(DEVICE)

            # Add epoch callback if provided
            callbacks = []
            if epoch_callback:
                callbacks.append(EpochProgressCallback(epoch_callback))

            # Train
            logger.info(f"Training for {epochs} epochs...")
            learn.fit_one_cycle(epochs, learning_rate, cbs=callbacks)

            # Extract final metrics
            final_metrics = {}
            if hasattr(learn, 'recorder') and learn.recorder.values:
                last_epoch = learn.recorder.values[-1]
                # Metrics order: train_loss, valid_loss, accuracy, f1, precision, recall
                if len(last_epoch) >= 6:
                    final_metrics = {
                        'train_loss': float(last_epoch[0]),
                        'valid_loss': float(last_epoch[1]),
                        'accuracy': float(last_epoch[2]),
                        'f1_score': float(last_epoch[3]),
                        'precision': float(last_epoch[4]),
                        'recall': float(last_epoch[5]),
                    }

            return {
                'status': 'success',
                'metrics': final_metrics,
                'learner': learn,
                'model': learn.model,
            }

        except Exception as e:
            logger.error(f"Training failed: {e}")
            import traceback
            traceback.print_exc()
            return {'status': 'failed', 'error': str(e)}

    def assess_model(
        self,
        model: Any,
        test_data: Tuple[np.ndarray, np.ndarray],
        metric: str = 'f1_score',
        threshold: float = 0.5,
        **kwargs
    ) -> Dict[str, float]:
        """
        Assess a trained model (aliased from evaluate_model for interface).

        Args:
            model: Trained model or Learner
            test_data: Tuple of (X_test, y_test)
            metric: Primary metric to optimize
            threshold: Classification threshold
            **kwargs: Additional options (e.g., 'learner' for Learner object)

        Returns:
            Dictionary of assessment metrics
        """
        if not TSAI_AVAILABLE:
            return {'error': 'tsai not available'}

        try:
            X_test, y_test = test_data
            learner = kwargs.get('learner')

            if learner is None:
                # If no learner, just do inference
                model.eval()
                with torch.no_grad():
                    X_tensor = torch.tensor(X_test, dtype=torch.float32)
                    if DEVICE:
                        X_tensor = X_tensor.to(DEVICE)
                        model = model.to(DEVICE)
                    outputs = model(X_tensor)
                    probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            else:
                # Use learner for prediction
                probs, _, preds = learner.get_X_preds(X_test)
                probs = probs[:, 1] if len(probs.shape) > 1 else probs

            # Convert to binary predictions
            y_pred = (np.array(probs) > threshold).astype(int)
            y_true = np.array(y_test)

            # Calculate metrics
            from sklearn.metrics import (
                f1_score, accuracy_score, precision_score, recall_score,
                roc_auc_score, matthews_corrcoef, confusion_matrix
            )

            metrics = {
                'f1_score': f1_score(y_true, y_pred, zero_division=0),
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, zero_division=0),
                'recall': recall_score(y_true, y_pred, zero_division=0),
                'mcc': matthews_corrcoef(y_true, y_pred),
            }

            # AUC-ROC if we have both classes
            if len(np.unique(y_true)) > 1:
                metrics['auc_roc'] = roc_auc_score(y_true, probs)

            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            metrics['true_negatives'] = int(cm[0, 0])
            metrics['false_positives'] = int(cm[0, 1]) if cm.shape[1] > 1 else 0
            metrics['false_negatives'] = int(cm[1, 0]) if cm.shape[0] > 1 else 0
            metrics['true_positives'] = int(cm[1, 1]) if cm.shape[0] > 1 and cm.shape[1] > 1 else 0

            logger.info(f"Assessment metrics: {metrics}")
            return metrics

        except Exception as e:
            logger.error(f"Assessment failed: {e}")
            import traceback
            traceback.print_exc()
            return {'error': str(e)}

    # Interface method - calls assess_model
    def evaluate_model(
        self,
        model: Any,
        test_data: Tuple[np.ndarray, np.ndarray],
        metric: str = 'f1_score',
        threshold: float = 0.5,
        **kwargs
    ) -> Dict[str, float]:
        """Interface method that calls assess_model."""
        return self.assess_model(model, test_data, metric, threshold, **kwargs)

    def predict(
        self,
        model: Any,
        data: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        """
        Generate predictions from a trained model.

        Args:
            model: Trained model or Learner
            data: Input data (X array or tuple with X)
            **kwargs: Additional options

        Returns:
            Predictions (probabilities for positive class)
        """
        if not TSAI_AVAILABLE:
            raise RuntimeError("tsai not available")

        # Handle tuple input
        if isinstance(data, tuple):
            X = data[0]
        else:
            X = data

        learner = kwargs.get('learner')

        if learner is not None:
            probs, _, _ = learner.get_X_preds(X)
            return probs[:, 1] if len(probs.shape) > 1 else probs
        else:
            model.eval()
            with torch.no_grad():
                X_tensor = torch.tensor(X, dtype=torch.float32)
                if DEVICE:
                    X_tensor = X_tensor.to(DEVICE)
                    model = model.to(DEVICE)
                outputs = model(X_tensor)
                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            return probs

    def save_model(self, learner: Any, name: str, metadata: Dict = None) -> str:
        """Save a trained model."""
        model_path = self.models_dir / f"{name}.pkl"
        learner.export(model_path)
        logger.info(f"Saved model to {model_path}")
        return str(model_path)

    def load_model(self, name: str) -> Any:
        """Load a saved model."""
        from tsai.all import load_learner
        model_path = self.models_dir / f"{name}.pkl"
        learner = load_learner(model_path)
        logger.info(f"Loaded model from {model_path}")
        return learner


class EpochProgressCallback:
    """Callback to report epoch progress during training."""

    def __init__(self, on_epoch_end: callable):
        self.on_epoch_end_fn = on_epoch_end
        self.learn = None

    def after_epoch(self):
        """Called after each epoch."""
        if self.on_epoch_end_fn and self.learn:
            metrics = {}
            if hasattr(self.learn, 'recorder') and self.learn.recorder.values:
                last = self.learn.recorder.values[-1]
                metrics = {
                    'epoch': self.learn.epoch,
                    'train_loss': float(last[0]) if len(last) > 0 else None,
                    'valid_loss': float(last[1]) if len(last) > 1 else None,
                }
            self.on_epoch_end_fn(self.learn.epoch, metrics)
