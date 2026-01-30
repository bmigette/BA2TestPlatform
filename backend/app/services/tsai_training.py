"""
tsai Training Service for classification models.

Provides training, evaluation, and prediction functionality using the tsai/fastai library.
Supports focal loss, class weighting, and comprehensive classification metrics.

Uses DataPreparationService for 35% buffered normalization (same as Darts),
with exportable scaler parameters for inference.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json

from app.services.data_preparation import DataPreparationService

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

    Uses 35% buffered normalization (same as Darts) for consistency.
    Scaler parameters are saved with the model for inference.
    """

    def __init__(self, models_dir: str = "trained_models", normalize: bool = True, buffer_pct: float = 0.35):
        """Initialize TSAITrainingService.

        Args:
            models_dir: Directory to save trained models
            normalize: Whether to apply per-feature normalization (required for MiniRocket)
            buffer_pct: Extra room above/below observed min/max for price normalization (default 35%)
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.normalize = normalize
        self.buffer_pct = buffer_pct
        self.data_prep = None  # DataPreparationService instance, fitted on training data

    def prepare_data(
        self,
        df: pd.DataFrame,
        target_column: str,
        feature_columns: List[str],
        timeframe: str = 'daily',
        seq_len: int = 24,
        prediction_horizon: int = 0,
        fit_scaler: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for tsai model training.

        Uses 35% buffered MinMax normalization (same as Darts) for consistency.
        This handles the huge range difference between Volume (millions) and prices (~200).

        Args:
            df: DataFrame with features and target
            target_column: Name of target column
            feature_columns: List of feature column names
            timeframe: Data timeframe (not used for tsai but kept for interface)
            seq_len: Sequence length for sliding window
            prediction_horizon: How many bars ahead to predict (0 = predict at end of sequence)
            fit_scaler: Whether to fit the scaler (True for train, False for test)

        Returns:
            Tuple of (X, y) numpy arrays

        Example with seq_len=24, prediction_horizon=3:
            Input: Bars T-23 to T (24 bars)
            Target: Class label at bar T+3
        """
        if not TSAI_AVAILABLE:
            raise RuntimeError("tsai library not available")

        # Extract target
        y_data = df[target_column].values.astype(np.int64)

        # Apply 35% buffered normalization (same as Darts)
        if self.normalize:
            if fit_scaler:
                self.data_prep = DataPreparationService(buffer_pct=self.buffer_pct)
                df_normalized = self.data_prep.fit_transform(df, feature_columns, method="minmax_buffered")
                logger.info(f"Fitted 35% buffered scaler on {len(df)} samples, {len(feature_columns)} features")
            elif self.data_prep is not None:
                df_normalized = self.data_prep.transform(df)
            else:
                # No scaler fitted yet, create one
                self.data_prep = DataPreparationService(buffer_pct=self.buffer_pct)
                df_normalized = self.data_prep.fit_transform(df, feature_columns, method="minmax_buffered")
                logger.warning("Scaler not fitted, fitting on current data")

            X_data = df_normalized[feature_columns].values.astype(np.float32)
        else:
            X_data = df[feature_columns].values.astype(np.float32)

        # Create sliding window sequences with prediction horizon
        X, y = self._create_sequences(X_data, y_data, seq_len, prediction_horizon)

        logger.info(f"Prepared data: X shape {X.shape}, y shape {y.shape}, horizon={prediction_horizon}, normalized={self.normalize}")
        return X, y

    def prepare_data_split(
        self,
        df: pd.DataFrame,
        train_ratio: float,
        target_column: str,
        feature_columns: List[str],
        timeframe: str = 'daily',
        seq_len: int = 24,
        prediction_horizon: int = 0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare and split data into train/test sets.

        Applies per-feature normalization: fits scaler on train data,
        transforms both train and test. This is critical for MiniRocket
        and models sensitive to feature scale differences (e.g., Volume vs Price).

        Args:
            df: Full DataFrame
            train_ratio: Fraction for training (0.0-1.0)
            target_column: Name of target column
            feature_columns: List of feature column names
            timeframe: Data timeframe
            seq_len: Sequence length for sliding window
            prediction_horizon: How many bars ahead to predict (0 = predict at end of sequence)

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        if not TSAI_AVAILABLE:
            raise RuntimeError("tsai library not available")

        # Split DataFrame first (before normalization to avoid data leakage)
        split_idx = int(len(df) * train_ratio)
        df_train = df.iloc[:split_idx]
        df_test = df.iloc[split_idx:]

        # Prepare train data (fit scaler)
        X_train, y_train = self.prepare_data(
            df_train, target_column, feature_columns, timeframe, seq_len, prediction_horizon, fit_scaler=True
        )

        # Prepare test data (use fitted scaler)
        X_test, y_test = self.prepare_data(
            df_test, target_column, feature_columns, timeframe, seq_len, prediction_horizon, fit_scaler=False
        )

        logger.info(f"Split data: train={len(X_train)}, test={len(X_test)}, horizon={prediction_horizon}")
        return X_train, X_test, y_train, y_test

    def _create_sequences(
        self,
        X: np.ndarray,
        y: np.ndarray,
        seq_len: int,
        prediction_horizon: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sliding window sequences from data.

        Args:
            X: Feature array of shape (n_rows, n_features)
            y: Target array of shape (n_rows,)
            seq_len: Length of input sequence (lookback window)
            prediction_horizon: How many bars ahead to predict (0 = predict at end of sequence)

        Returns:
            X_seq: Sequences of shape (n_samples, n_features, seq_len)
            y_seq: Targets of shape (n_samples,)

        Example with seq_len=24, prediction_horizon=3:
            Input: Bars T-23 to T (24 bars)
            Target: Class label at bar T+3
        """
        # Account for prediction_horizon when calculating valid samples
        # We need seq_len bars for input, plus prediction_horizon bars for the target
        n_samples = len(X) - seq_len - prediction_horizon + 1

        if n_samples <= 0:
            raise ValueError(
                f"Not enough data: need at least {seq_len + prediction_horizon} rows, got {len(X)}"
            )

        n_features = X.shape[1]

        # Shape: (samples, features, seq_len) for tsai
        X_seq = np.zeros((n_samples, n_features, seq_len), dtype=np.float32)
        y_seq = np.zeros(n_samples, dtype=np.int64)

        for i in range(n_samples):
            # Transpose to get (features, seq_len)
            X_seq[i] = X[i:i+seq_len].T
            # Target is prediction_horizon bars AFTER the end of the input sequence
            # End of sequence is at index i+seq_len-1, so target is at i+seq_len-1+prediction_horizon
            y_seq[i] = y[i + seq_len - 1 + prediction_horizon]

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
        force_cpu: bool = False,
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
            force_cpu: Force CPU training (for MPS-limited models like xception, patchtst)
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

            # Determine device (force_cpu only applies on Apple Silicon MPS)
            import torch
            if force_cpu and MPS_AVAILABLE:
                # Force CPU only on Apple Silicon - some models have MPS limitations
                device = torch.device('cpu')
                logger.info("Training on CPU (force_cpu=True, MPS available but skipped)")
            elif DEVICE:
                device = DEVICE
            else:
                device = torch.device('cpu')

            # Auto-adjust batch size for small datasets to avoid empty batches
            val_size = len(splits[1]) if len(splits) > 1 else int(len(X_all) * 0.2)
            effective_bs = min(batch_size, val_size, len(splits[0]))
            if effective_bs < batch_size:
                logger.info(f"Reduced batch size from {batch_size} to {effective_bs} for small dataset")

            # Create dataloaders with explicit device
            dls = get_ts_dls(
                X_all, y_all,
                splits=splits,
                bs=effective_bs,
                batch_tfms=[TSStandardize()],
                device=device,  # Set device for dataloaders
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

            # Move model to device
            learn.model = learn.model.to(device)

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

            # Always use direct inference to avoid batch size issues with learner.get_X_preds
            # which can drop incomplete last batches
            if learner is not None:
                model = learner.model

            model.eval()
            with torch.no_grad():
                X_tensor = torch.tensor(X_test, dtype=torch.float32)
                if DEVICE:
                    X_tensor = X_tensor.to(DEVICE)
                    model = model.to(DEVICE)
                outputs = model(X_tensor)
                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()

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

        # Always use direct inference to avoid batch size issues with learner.get_X_preds
        # which can drop incomplete last batches
        if learner is not None:
            model = learner.model

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
        """
        Save a trained model with normalization parameters.

        Saves both the model (.pkl) and the normalization params (.norm.json)
        so that inference can apply the same data transformation.

        Args:
            learner: Trained tsai Learner
            name: Model name
            metadata: Optional additional metadata

        Returns:
            Path to saved model
        """
        model_path = self.models_dir / f"{name}.pkl"
        norm_path = self.models_dir / f"{name}.norm.json"

        # Save the model
        learner.export(model_path)
        logger.info(f"Saved model to {model_path}")

        # Save normalization parameters if available
        if self.data_prep is not None:
            params = self.data_prep.export_params()
            if metadata:
                params["metadata"] = metadata
            with open(norm_path, 'w') as f:
                json.dump(params, f, indent=2)
            logger.info(f"Saved normalization params to {norm_path}")

        return str(model_path)

    def load_model(self, name: str) -> Any:
        """
        Load a saved model with its normalization parameters.

        Args:
            name: Model name

        Returns:
            Loaded tsai Learner
        """
        from tsai.all import load_learner

        model_path = self.models_dir / f"{name}.pkl"
        norm_path = self.models_dir / f"{name}.norm.json"

        # Load the model
        learner = load_learner(model_path)
        logger.info(f"Loaded model from {model_path}")

        # Load normalization parameters if available
        if norm_path.exists():
            self.data_prep = DataPreparationService()
            self.data_prep.load_params_from_file(str(norm_path))
            logger.info(f"Loaded normalization params from {norm_path}")
        else:
            logger.warning(f"No normalization params found at {norm_path}")

        return learner

    def get_normalization_params(self) -> Optional[Dict[str, Any]]:
        """
        Get the current normalization parameters.

        Returns:
            Dictionary with normalization params or None if not fitted
        """
        if self.data_prep is not None:
            return self.data_prep.export_params()
        return None


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
