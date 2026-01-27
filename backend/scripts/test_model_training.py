#!/usr/bin/env python3
"""
Test script for model training service.

Tests different model architectures (LSTM, GRU, RNN, N-BEATS, TCN) with small
configurations (1 layer, 128 neurons) on the test dataset.

Run from backend directory:
    python scripts/test_model_training.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import logging
import time

# Setup logging
from app.logging_config import setup_logging
setup_logging(console_level=logging.INFO)

logger = logging.getLogger(__name__)

# Check for required libraries
try:
    import torch
    TORCH_AVAILABLE = True
    # Check for MPS (Apple Silicon) or CUDA
    if torch.backends.mps.is_available():
        DEVICE = "mps"
    elif torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = "cpu"
    print(f"PyTorch available, using device: {DEVICE}")
except ImportError:
    TORCH_AVAILABLE = False
    DEVICE = "cpu"
    print("PyTorch not available")

try:
    from darts import TimeSeries
    from darts.models import RNNModel, NBEATSModel, TCNModel, BlockRNNModel
    from darts.dataprocessing.transformers import Scaler
    from darts.metrics import mape, mae, rmse
    DARTS_AVAILABLE = True
    print("Darts library available")
except ImportError as e:
    DARTS_AVAILABLE = False
    print(f"Darts library not available: {e}")


# Model configurations to test
# Using smaller input_chunk_length (12) to work with limited test data
MODEL_CONFIGS = {
    "LSTM": {
        "class": "RNNModel",
        "params": {
            "model": "LSTM",
            "hidden_dim": 128,
            "n_rnn_layers": 1,
            "input_chunk_length": 12,  # 12 bars lookback (3 days at 4h)
            "output_chunk_length": 1,  # RNNModel only supports output_chunk_length=1
            "training_length": 24,  # Training sequence length
            "n_epochs": 10,
            "batch_size": 16,
            "dropout": 0.1,
            "random_state": 42,
        }
    },
    "GRU": {
        "class": "RNNModel",
        "params": {
            "model": "GRU",
            "hidden_dim": 128,
            "n_rnn_layers": 1,
            "input_chunk_length": 12,
            "output_chunk_length": 1,
            "training_length": 24,
            "n_epochs": 10,
            "batch_size": 16,
            "dropout": 0.1,
            "random_state": 42,
        }
    },
    "RNN": {
        "class": "RNNModel",
        "params": {
            "model": "RNN",
            "hidden_dim": 128,
            "n_rnn_layers": 1,
            "input_chunk_length": 12,
            "output_chunk_length": 1,
            "training_length": 24,
            "n_epochs": 10,
            "batch_size": 16,
            "dropout": 0.1,
            "random_state": 42,
        }
    },
    "NBEATS": {
        "class": "NBEATSModel",
        "params": {
            "input_chunk_length": 12,
            "output_chunk_length": 6,
            "num_stacks": 2,
            "num_blocks": 1,
            "num_layers": 1,
            "layer_widths": 128,
            "n_epochs": 10,
            "batch_size": 16,
            "random_state": 42,
        }
    },
    "TCN": {
        "class": "TCNModel",
        "params": {
            "input_chunk_length": 12,
            "output_chunk_length": 6,
            "num_layers": 1,
            "num_filters": 128,
            "kernel_size": 3,
            "dilation_base": 2,
            "n_epochs": 10,
            "batch_size": 16,
            "dropout": 0.1,
            "random_state": 42,
        }
    },
}


def load_test_dataset(dataset_path: str) -> pd.DataFrame:
    """Load the test dataset."""
    df = pd.read_csv(dataset_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    print(f"Loaded dataset: {len(df)} rows, {len(df.columns)} columns")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    return df


def prepare_data(df: pd.DataFrame, target_col: str = 'Close'):
    """Prepare data for Darts models."""
    # Select numeric columns only, excluding date columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Remove any columns with all NaN
    valid_cols = [c for c in numeric_cols if df[c].notna().any()]

    # Create a clean dataframe
    df_clean = df[['Date'] + valid_cols].copy()

    # Fill NaN values with forward fill then backward fill
    for col in valid_cols:
        df_clean[col] = df_clean[col].ffill().bfill()

    # Drop any remaining rows with NaN
    df_clean = df_clean.dropna()

    print(f"Clean dataset: {len(df_clean)} rows, {len(valid_cols)} numeric columns")

    # Remove timezone info and set Date as index
    df_clean['Date'] = pd.to_datetime(df_clean['Date']).dt.tz_localize(None)
    df_clean = df_clean.set_index('Date')

    # For intraday data with gaps (weekends, market hours), use integer index
    # This avoids issues with frequency detection
    df_clean = df_clean.reset_index(drop=True)

    # Create target series using values directly (integer index)
    target_values = df_clean[[target_col]].values
    target_series = TimeSeries.from_values(target_values)

    # Scale the data
    scaler = Scaler()
    target_series_scaled = scaler.fit_transform(target_series)

    # Create covariates from other numeric columns
    covariate_cols = [c for c in valid_cols if c != target_col]
    if covariate_cols:
        # Limit covariates to avoid memory issues (use only technical indicators)
        tech_cols = [c for c in covariate_cols if any(c.startswith(p) for p in
                     ['sma_', 'ema_', 'rsi_', 'macd_', 'bbands_', 'atr_', 'stochastic_'])]
        if tech_cols:
            covariate_cols = tech_cols[:10]  # Limit to 10 covariates
        else:
            covariate_cols = covariate_cols[:10]

        # Use values directly for covariates (integer index)
        cov_values = df_clean[covariate_cols].values
        covariates = TimeSeries.from_values(cov_values)
        cov_scaler = Scaler()
        covariates_scaled = cov_scaler.fit_transform(covariates)
        print(f"Using {len(covariate_cols)} covariates: {covariate_cols[:5]}...")
    else:
        covariates_scaled = None

    return target_series_scaled, covariates_scaled, scaler


def create_model(model_name: str, config: dict):
    """Create a Darts model from configuration."""
    model_class = config["class"]
    params = config["params"].copy()

    # Add device configuration for PyTorch models
    if TORCH_AVAILABLE:
        # For MPS, use CPU for now as some operations may not be supported
        pl_trainer_kwargs = {
            "accelerator": "cpu",  # Use CPU for compatibility
            "enable_progress_bar": True,
        }
        params["pl_trainer_kwargs"] = pl_trainer_kwargs

    if model_class == "RNNModel":
        return RNNModel(**params)
    elif model_class == "NBEATSModel":
        return NBEATSModel(**params)
    elif model_class == "TCNModel":
        return TCNModel(**params)
    elif model_class == "BlockRNNModel":
        return BlockRNNModel(**params)
    else:
        raise ValueError(f"Unknown model class: {model_class}")


def train_and_evaluate(
    model_name: str,
    model,
    train_series,
    val_series,
    covariates=None
) -> dict:
    """Train a model and evaluate it."""
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")

    result = {
        "model_name": model_name,
        "status": "pending",
        "train_time": 0,
        "metrics": {}
    }

    start_time = time.time()

    try:
        # Train the model
        print(f"  Training on {len(train_series)} samples...")

        # Train without covariates for simplicity (RNNModel doesn't support past_covariates)
        # In production, use future_covariates for RNNModel if needed
        model.fit(train_series, verbose=True)

        train_time = time.time() - start_time
        result["train_time"] = round(train_time, 2)
        print(f"  Training completed in {train_time:.2f} seconds")

        # Evaluate on validation set
        print(f"  Evaluating on {len(val_series)} samples...")

        # Make predictions
        input_length = model.input_chunk_length
        output_length = model.output_chunk_length

        # We need at least input_length + output_length samples for prediction
        if len(val_series) > input_length:
            n_predictions = min(output_length, len(val_series) - input_length)

            try:
                predictions = model.predict(n=n_predictions)
                actuals = val_series[input_length:input_length + n_predictions]

                # Calculate metrics
                result["metrics"] = {
                    "mape": float(mape(actuals, predictions)),
                    "mae": float(mae(actuals, predictions)),
                    "rmse": float(rmse(actuals, predictions)),
                }
                print(f"  MAPE: {result['metrics']['mape']:.4f}")
                print(f"  MAE:  {result['metrics']['mae']:.4f}")
                print(f"  RMSE: {result['metrics']['rmse']:.4f}")
            except Exception as e:
                print(f"  Prediction failed: {e}")
                result["metrics"] = {"error": str(e)}
        else:
            print(f"  Validation set too small for prediction (need > {input_length} samples)")
            result["metrics"] = {"error": "Validation set too small"}

        result["status"] = "completed"

    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)
        print(f"  Training failed: {e}")
        import traceback
        traceback.print_exc()

    return result


def main():
    """Main test function."""
    print("=" * 80)
    print("MODEL TRAINING TEST")
    print("=" * 80)
    print(f"Date: {datetime.now()}")
    print(f"Device: {DEVICE}")
    print()

    if not DARTS_AVAILABLE:
        print("ERROR: Darts library not available. Cannot run tests.")
        return

    if not TORCH_AVAILABLE:
        print("WARNING: PyTorch not available. Some models may not work.")

    # Load test dataset
    dataset_path = Path("test/test_dataset.csv")
    if not dataset_path.exists():
        print(f"ERROR: Test dataset not found at {dataset_path}")
        print("Run scripts/test_dataset_generation.py first to generate test data")
        return

    df = load_test_dataset(str(dataset_path))

    # Prepare data
    print("\nPreparing data...")
    target_series, covariates, scaler = prepare_data(df, target_col='Close')

    # Split into train/validation (80/20)
    split_idx = int(len(target_series) * 0.8)
    train_series = target_series[:split_idx]
    val_series = target_series[split_idx:]

    if covariates is not None:
        train_covariates = covariates[:split_idx]
        val_covariates = covariates[split_idx:]
    else:
        train_covariates = None
        val_covariates = None

    print(f"\nTrain samples: {len(train_series)}")
    print(f"Validation samples: {len(val_series)}")

    # Check if we have enough data
    min_samples = 30  # Minimum for training
    if len(train_series) < min_samples:
        print(f"\nERROR: Not enough training samples (need at least {min_samples})")
        return

    # Test each model
    results = []
    for model_name, config in MODEL_CONFIGS.items():
        try:
            model = create_model(model_name, config)
            result = train_and_evaluate(
                model_name,
                model,
                train_series,
                val_series,
                train_covariates
            )
            results.append(result)
        except Exception as e:
            print(f"\n{model_name} failed to initialize: {e}")
            results.append({
                "model_name": model_name,
                "status": "init_failed",
                "error": str(e)
            })

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Model':<12} {'Status':<12} {'Time (s)':<10} {'MAPE':<10} {'MAE':<10} {'RMSE':<10}")
    print("-" * 80)

    for r in results:
        status = r.get("status", "unknown")
        train_time = r.get("train_time", 0)
        metrics = r.get("metrics", {})

        mape_val = metrics.get("mape", "N/A")
        mae_val = metrics.get("mae", "N/A")
        rmse_val = metrics.get("rmse", "N/A")

        if isinstance(mape_val, float):
            mape_val = f"{mape_val:.4f}"
        if isinstance(mae_val, float):
            mae_val = f"{mae_val:.4f}"
        if isinstance(rmse_val, float):
            rmse_val = f"{rmse_val:.4f}"

        print(f"{r['model_name']:<12} {status:<12} {train_time:<10} {mape_val:<10} {mae_val:<10} {rmse_val:<10}")

    # Check results
    successful = sum(1 for r in results if r.get("status") == "completed")
    failed = sum(1 for r in results if r.get("status") in ["failed", "init_failed"])

    print("-" * 80)
    print(f"Completed: {successful}/{len(results)}, Failed: {failed}/{len(results)}")

    if successful == len(results):
        print("\nTEST PASSED: All models trained successfully")
    elif successful > 0:
        print(f"\nTEST PARTIAL: {successful} models trained, {failed} failed")
    else:
        print("\nTEST FAILED: No models trained successfully")

    print("=" * 80)


if __name__ == "__main__":
    main()
