"""
Model Training Service

Provides model training, evaluation, and saving functionality.
Integrates with Darts library for timeseries model training.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import logging
import os
from pathlib import Path
import json

logger = logging.getLogger(__name__)

# Check for required libraries
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from darts import TimeSeries
    from darts.models import RNNModel, NBEATSModel
    from darts.dataprocessing.transformers import Scaler
    from darts.metrics import mape, mae, rmse
    DARTS_AVAILABLE = True
except ImportError:
    DARTS_AVAILABLE = False


class TrainingService:
    """
    Service for training and evaluating ML models.
    """

    def __init__(self, models_dir: str = "trained_models"):
        """
        Initialize TrainingService.

        Args:
            models_dir: Directory to save trained models
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.scaler = None

    def prepare_data(
        self,
        df: pd.DataFrame,
        target_column: str = 'Close',
        feature_columns: List[str] = None,
        timeframe: str = 'daily'
    ) -> Tuple[Any, Any]:
        """
        Prepare data for Darts model training.

        Args:
            df: DataFrame with Date and target columns
            target_column: Column to predict
            feature_columns: Optional covariate columns
            timeframe: Dataset timeframe for frequency inference

        Returns:
            Tuple of (target_series, covariates_series)
        """
        if not DARTS_AVAILABLE:
            raise RuntimeError("Darts library not available")

        df_sorted = df.sort_values('Date').copy()
        df_sorted['Date'] = pd.to_datetime(df_sorted['Date'])
        df_sorted = df_sorted.set_index('Date')

        # Infer frequency based on timeframe
        freq = self._infer_frequency(timeframe, df_sorted)

        # Drop rows with NaN in target column
        if target_column in df_sorted.columns:
            df_sorted = df_sorted.dropna(subset=[target_column])

        # Create target series with fill_missing_dates to handle gaps (weekends/holidays)
        target_series = TimeSeries.from_dataframe(
            df_sorted[[target_column]],
            value_cols=target_column,
            fill_missing_dates=True,
            freq=freq
        )

        # Scale the data
        self.scaler = Scaler()
        target_series = self.scaler.fit_transform(target_series)

        # Create covariates if specified
        covariates = None
        if feature_columns:
            available_cols = [c for c in feature_columns if c in df_sorted.columns]
            if available_cols:
                # Drop NaN values from covariates
                cov_df = df_sorted[available_cols].dropna()
                if len(cov_df) > 0:
                    covariates = TimeSeries.from_dataframe(
                        cov_df,
                        value_cols=available_cols,
                        fill_missing_dates=True,
                        freq=freq
                    )
                    # Scale covariates
                    cov_scaler = Scaler()
                    covariates = cov_scaler.fit_transform(covariates)

        return target_series, covariates

    def _infer_frequency(self, timeframe: str, df: pd.DataFrame) -> str:
        """
        Infer pandas frequency string from timeframe.

        Args:
            timeframe: Dataset timeframe (e.g., 'daily', '1h', '4h', '1d')
            df: DataFrame with DatetimeIndex

        Returns:
            Pandas frequency string
        """
        timeframe_lower = timeframe.lower()

        # Map common timeframes to pandas frequencies
        freq_map = {
            '1m': 'T',      # 1 minute
            '5m': '5T',     # 5 minutes
            '15m': '15T',   # 15 minutes
            '30m': '30T',   # 30 minutes
            '1h': 'H',      # 1 hour
            '4h': '4H',     # 4 hours
            '1d': 'D',      # 1 day
            'daily': 'D',   # daily
            '1w': 'W',      # 1 week
            'weekly': 'W',  # weekly
        }

        if timeframe_lower in freq_map:
            return freq_map[timeframe_lower]

        # Try to infer from data
        if len(df) >= 2:
            try:
                # Calculate median difference between consecutive rows
                time_diffs = df.index.to_series().diff().dropna()
                median_diff = time_diffs.median()

                if median_diff <= pd.Timedelta(minutes=5):
                    return '5T'
                elif median_diff <= pd.Timedelta(hours=1):
                    return 'H'
                elif median_diff <= pd.Timedelta(hours=4):
                    return '4H'
                elif median_diff <= pd.Timedelta(days=1):
                    return 'D'
                else:
                    return 'W'
            except Exception:
                pass

        # Default to daily
        logger.warning(f"Could not infer frequency for timeframe '{timeframe}', using daily")
        return 'D'

    def train_model(
        self,
        model: Any,
        train_series: Any,
        val_series: Any = None,
        covariates: Any = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Train a Darts model.

        Args:
            model: Darts model instance
            train_series: Training TimeSeries
            val_series: Optional validation TimeSeries
            covariates: Optional covariate TimeSeries
            verbose: Whether to print training progress

        Returns:
            Training metrics
        """
        if not DARTS_AVAILABLE:
            raise RuntimeError("Darts library not available")

        start_time = datetime.now()

        logger.info(f"Starting training with {len(train_series)} data points")

        try:
            if covariates is not None:
                model.fit(train_series, past_covariates=covariates, verbose=verbose)
            else:
                model.fit(train_series, verbose=verbose)

            training_time = (datetime.now() - start_time).total_seconds()

            metrics = {
                'training_time_seconds': training_time,
                'train_samples': len(train_series),
                'status': 'completed'
            }

            # Calculate training error if possible
            try:
                train_pred = model.predict(n=len(train_series) - model.input_chunk_length)
                train_target = train_series[model.input_chunk_length:]

                # Align lengths
                min_len = min(len(train_pred), len(train_target))
                train_pred = train_pred[:min_len]
                train_target = train_target[:min_len]

                metrics['train_mape'] = float(mape(train_target, train_pred))
                metrics['train_mae'] = float(mae(train_target, train_pred))
            except Exception as e:
                logger.warning(f"Could not calculate training metrics: {e}")

            logger.info(f"Training completed in {training_time:.2f} seconds")
            return metrics

        except Exception as e:
            logger.error(f"Training failed: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }

    def evaluate_model(
        self,
        model: Any,
        test_series: Any,
        covariates: Any = None
    ) -> Dict[str, float]:
        """
        Evaluate model on test set.

        Args:
            model: Trained Darts model
            test_series: Test TimeSeries
            covariates: Optional covariate TimeSeries

        Returns:
            Evaluation metrics
        """
        if not DARTS_AVAILABLE:
            raise RuntimeError("Darts library not available")

        try:
            # Make predictions
            n_predict = len(test_series) - model.input_chunk_length
            if n_predict <= 0:
                return {'error': 'Test series too short'}

            predictions = model.predict(n=n_predict, past_covariates=covariates)
            actuals = test_series[model.input_chunk_length:]

            # Align lengths
            min_len = min(len(predictions), len(actuals))
            predictions = predictions[:min_len]
            actuals = actuals[:min_len]

            metrics = {
                'mape': float(mape(actuals, predictions)),
                'mae': float(mae(actuals, predictions)),
                'rmse': float(rmse(actuals, predictions)),
                'test_samples': len(test_series),
                'predictions_made': len(predictions)
            }

            logger.info(f"Evaluation complete: MAPE={metrics['mape']:.4f}")
            return metrics

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return {'error': str(e)}

    def save_model(
        self,
        model: Any,
        model_name: str,
        metadata: Dict = None
    ) -> str:
        """
        Save trained model to disk.

        Args:
            model: Trained Darts model
            model_name: Name for the model file
            metadata: Optional metadata to save alongside

        Returns:
            Path to saved model
        """
        model_path = self.models_dir / f"{model_name}.pt"

        # Save model
        model.save(str(model_path))
        logger.info(f"Model saved to {model_path}")

        # Save metadata
        if metadata:
            meta_path = self.models_dir / f"{model_name}_meta.json"
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            logger.info(f"Metadata saved to {meta_path}")

        # Save scaler if available
        if self.scaler:
            scaler_path = self.models_dir / f"{model_name}_scaler.pt"
            if TORCH_AVAILABLE:
                torch.save(self.scaler, str(scaler_path))

        return str(model_path)

    def load_model(self, model_path: str) -> Any:
        """
        Load a trained model from disk.

        Args:
            model_path: Path to model file

        Returns:
            Loaded model
        """
        if not DARTS_AVAILABLE:
            raise RuntimeError("Darts library not available")

        # Determine model type from metadata if available
        meta_path = Path(model_path).with_suffix('').with_name(
            Path(model_path).stem + '_meta.json'
        )

        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
                model_type = meta.get('model_type', 'lstm')
        else:
            model_type = 'lstm'

        # Load model based on type
        if model_type == 'nbeats':
            model = NBEATSModel.load(model_path)
        else:
            model = RNNModel.load(model_path)

        logger.info(f"Loaded model from {model_path}")
        return model


class ModelEvaluator:
    """
    Utility class for model evaluation and metrics.

    For imbalanced datasets (which is common with prediction targets),
    use the ClassificationMetrics from app.services.metrics instead.
    """

    @staticmethod
    def calculate_classification_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Calculate classification metrics for binary predictions.

        For comprehensive metrics including AUC-ROC, use:
        from app.services.metrics import ClassificationMetrics
        ClassificationMetrics.calculate_all(y_true, y_pred_proba, threshold)

        Args:
            y_true: True labels
            y_pred: Predicted probabilities
            threshold: Classification threshold

        Returns:
            Dictionary of metrics
        """
        # Import and delegate to new comprehensive metrics module
        try:
            from app.services.metrics import ClassificationMetrics
            return ClassificationMetrics.calculate_all(y_true, y_pred, threshold)
        except ImportError:
            # Fallback to basic implementation
            pass

        y_pred_binary = (y_pred >= threshold).astype(int)

        # True/False Positives/Negatives
        tp = np.sum((y_true == 1) & (y_pred_binary == 1))
        tn = np.sum((y_true == 0) & (y_pred_binary == 0))
        fp = np.sum((y_true == 0) & (y_pred_binary == 1))
        fn = np.sum((y_true == 1) & (y_pred_binary == 0))

        # Calculate metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        balanced_accuracy = (recall + specificity) / 2

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'balanced_accuracy': balanced_accuracy,
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn)
        }

    @staticmethod
    def get_fitness_score(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metric: str = 'f1_score',
        threshold: float = 0.5
    ) -> float:
        """
        Get a single fitness score for optimization.

        Args:
            y_true: True labels
            y_pred: Predicted probabilities
            metric: One of 'accuracy', 'f1_score', 'precision', 'recall',
                   'balanced_accuracy', 'auc_roc'
            threshold: Classification threshold

        Returns:
            Fitness score (higher is better)
        """
        try:
            from app.services.metrics import ClassificationMetrics
            return ClassificationMetrics.get_fitness_score(y_true, y_pred, metric, threshold)
        except ImportError:
            metrics = ModelEvaluator.calculate_classification_metrics(y_true, y_pred, threshold)
            return metrics.get(metric, 0.0)
