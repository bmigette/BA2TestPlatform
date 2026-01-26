"""
ML Models API endpoints

Provides endpoints for ML model configuration, prediction targets,
and dataset splitting for training.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import pandas as pd
from pathlib import Path

from app.models.database import get_db
from app.models.dataset import Dataset
from app.services.ml_models import (
    MLModelsService,
    PredictionTargetService,
    DatasetSplitter
)
from app.services.training import TrainingService, ModelEvaluator
from app.services.genetic import GeneticOptimizer, FitnessEvaluator, DEAP_AVAILABLE
from app.services.genetic_optimizer_base import (
    GeneticOptimizerFactory,
    GeneticLibrary,
    OptimizationResult
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/system-info")
async def get_system_info():
    """
    Get system information for ML training.

    Returns:
        Dictionary with PyTorch, CUDA, GPU availability
    """
    return MLModelsService.get_system_info()


@router.get("/models")
async def get_available_models():
    """
    Get list of available ML model architectures.

    Returns:
        Dictionary of model configurations with parameters
    """
    return {
        "models": MLModelsService.get_available_models(),
        "description": "Available ML model architectures for timeseries forecasting"
    }


@router.get("/models/{model_type}")
async def get_model_details(model_type: str):
    """
    Get details for a specific model type.

    Args:
        model_type: Model type (lstm, nbeats, rnn, tft)

    Returns:
        Model configuration details
    """
    models = MLModelsService.get_available_models()

    if model_type not in models:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model type '{model_type}' not found. Available: {list(models.keys())}"
        )

    return {
        "model_type": model_type,
        **models[model_type]
    }


@router.post("/datasets/{dataset_id}/calculate-targets")
async def calculate_prediction_targets(
    dataset_id: int,
    targets: List[Dict[str, Any]] = None,
    db: Session = Depends(get_db)
):
    """
    Calculate prediction targets for a dataset.

    Creates binary classification targets like price_up_10pct_5dd_7d.

    Args:
        dataset_id: Dataset ID
        targets: List of target configurations, e.g.:
            [{"profit_pct": 10, "max_dd": 5, "days": 7, "direction": "up"}]
        db: Database session

    Returns:
        Updated dataset with target columns
    """
    try:
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()

        if not dataset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dataset with ID {dataset_id} not found"
            )

        # Load dataset
        file_path = Path(dataset.file_path)
        if not file_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dataset file not found: {file_path}"
            )

        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'])

        # Default targets if not provided
        if targets is None:
            targets = [
                {'profit_pct': 10, 'max_dd': 5, 'days': 7, 'direction': 'up'},
                {'profit_pct': 10, 'max_dd': 5, 'days': 7, 'direction': 'down'},
                {'profit_pct': 20, 'max_dd': 10, 'days': 30, 'direction': 'up'},
                {'profit_pct': 20, 'max_dd': 10, 'days': 30, 'direction': 'down'}
            ]

        logger.info(f"Calculating {len(targets)} prediction targets for dataset {dataset_id}")

        # Calculate targets
        target_service = PredictionTargetService()
        result_df = target_service.calculate_prediction_targets(df, targets)

        # Verify symmetry
        is_symmetric = target_service.verify_symmetry(result_df, targets)

        # Save updated dataset
        result_df.to_csv(file_path, index=False)

        # Count added columns
        added_columns = [col for col in result_df.columns if col not in df.columns]

        logger.info(f"Successfully calculated {len(added_columns)} prediction targets")

        return {
            "dataset_id": dataset_id,
            "targets_calculated": added_columns,
            "is_symmetric": is_symmetric,
            "total_columns": len(result_df.columns),
            "rows": len(result_df),
            "message": f"Calculated {len(added_columns)} prediction targets"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error calculating prediction targets: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to calculate prediction targets: {str(e)}"
        )


@router.post("/datasets/{dataset_id}/split")
async def split_dataset(
    dataset_id: int,
    train_ratio: float = 0.8,
    shuffle: bool = False,
    db: Session = Depends(get_db)
):
    """
    Split dataset into train and test sets.

    Args:
        dataset_id: Dataset ID
        train_ratio: Proportion for training (default: 0.8)
        shuffle: Whether to shuffle (default: False for timeseries)
        db: Database session

    Returns:
        Split information and file paths
    """
    try:
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()

        if not dataset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dataset with ID {dataset_id} not found"
            )

        # Load dataset
        file_path = Path(dataset.file_path)
        if not file_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dataset file not found: {file_path}"
            )

        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'])

        logger.info(f"Splitting dataset {dataset_id} with ratio {train_ratio}")

        # Split dataset
        train_df, test_df = DatasetSplitter.train_test_split(
            df, train_ratio=train_ratio, shuffle=shuffle
        )

        # Save split files
        train_path = file_path.parent / f"{file_path.stem}_train.csv"
        test_path = file_path.parent / f"{file_path.stem}_test.csv"

        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        logger.info(f"Saved train set to {train_path}, test set to {test_path}")

        return {
            "dataset_id": dataset_id,
            "train_ratio": train_ratio,
            "shuffle": shuffle,
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "train_date_range": {
                "start": train_df['Date'].min().isoformat(),
                "end": train_df['Date'].max().isoformat()
            },
            "test_date_range": {
                "start": test_df['Date'].min().isoformat(),
                "end": test_df['Date'].max().isoformat()
            },
            "train_file": str(train_path),
            "test_file": str(test_path),
            "message": f"Split into {len(train_df)} train and {len(test_df)} test rows"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error splitting dataset: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to split dataset: {str(e)}"
        )


@router.get("/gpu-status")
async def get_gpu_status():
    """
    Get GPU utilization status.

    Returns:
        GPU memory and utilization info
    """
    info = MLModelsService.get_system_info()

    if not info['cuda_available']:
        return {
            "available": False,
            "message": "CUDA not available. Using CPU for training."
        }

    return {
        "available": True,
        "gpu_name": info['gpu_name'],
        "memory_total_gb": round(info['gpu_memory_total'] / 1e9, 2) if info['gpu_memory_total'] else None,
        "memory_free_gb": round(info['gpu_memory_free'] / 1e9, 2) if info['gpu_memory_free'] else None,
        "message": f"GPU available: {info['gpu_name']}"
    }


@router.get("/genetic/status")
async def get_genetic_optimizer_status():
    """
    Get genetic optimization library status.

    Returns:
        DEAP availability and configuration
    """
    return {
        "deap_available": DEAP_AVAILABLE,
        "default_param_ranges": GeneticOptimizer.DEFAULT_PARAM_RANGES if DEAP_AVAILABLE else {},
        "description": "DEAP genetic algorithm optimization for hyperparameters"
    }


@router.post("/genetic/optimize")
async def run_genetic_optimization(
    population_size: int = 10,
    n_generations: int = 5,
    crossover_prob: float = 0.7,
    mutation_prob: float = 0.2,
    param_ranges: Dict = None
):
    """
    Run genetic algorithm optimization with dummy fitness function.

    This is a demo endpoint that runs optimization with a test fitness function.
    For real training, use the optimization job system.

    Args:
        population_size: Population size
        n_generations: Number of generations
        crossover_prob: Crossover probability
        mutation_prob: Mutation probability
        param_ranges: Optional custom parameter ranges

    Returns:
        Optimization results
    """
    if not DEAP_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="DEAP library not available. Install with: pip install deap"
        )

    try:
        logger.info(f"Starting genetic optimization: pop={population_size}, gen={n_generations}")

        optimizer = GeneticOptimizer(
            param_ranges=param_ranges,
            population_size=population_size,
            n_generations=n_generations,
            crossover_prob=crossover_prob,
            mutation_prob=mutation_prob
        )

        # Run with dummy fitness function
        results = optimizer.optimize(FitnessEvaluator.dummy_fitness)

        return {
            "status": "completed",
            "best_params": results['best_params'],
            "best_fitness": results['best_fitness'],
            "generations_run": results['generations_run'],
            "history_summary": [
                {
                    "generation": h['generation'],
                    "best_fitness": h['best_fitness']
                }
                for h in results['history']
            ]
        }

    except Exception as e:
        logger.error(f"Genetic optimization failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Optimization failed: {str(e)}"
        )


@router.post("/train/{model_type}")
async def train_model_demo(
    model_type: str,
    epochs: int = 10,
    batch_size: int = 32
):
    """
    Demo endpoint for model training configuration.

    This returns the configuration that would be used for training.
    Actual training should be done through the optimization job system.

    Args:
        model_type: Model type (lstm, nbeats, rnn)
        epochs: Number of training epochs
        batch_size: Batch size

    Returns:
        Training configuration
    """
    models = MLModelsService.get_available_models()

    if model_type not in models:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model type '{model_type}' not found. Available: {list(models.keys())}"
        )

    config = models[model_type]
    params = config['default_params'].copy()
    params['n_epochs'] = epochs
    params['batch_size'] = batch_size

    system_info = MLModelsService.get_system_info()

    return {
        "model_type": model_type,
        "model_name": config['name'],
        "description": config['description'],
        "training_params": params,
        "system_info": {
            "pytorch_available": system_info['pytorch_available'],
            "darts_available": system_info['darts_available'],
            "cuda_available": system_info['cuda_available'],
            "gpu_name": system_info['gpu_name']
        },
        "message": f"Training configuration ready for {config['name']}"
    }


@router.get("/training/saved-models")
async def list_saved_models():
    """
    List all saved trained models.

    Returns:
        List of saved model files
    """
    models_dir = Path("trained_models")

    if not models_dir.exists():
        return {"models": [], "count": 0}

    models = []
    for model_file in models_dir.glob("*.pt"):
        # Skip scaler files
        if "_scaler" in model_file.name:
            continue

        meta_file = model_file.with_suffix('').with_name(
            model_file.stem + '_meta.json'
        )

        model_info = {
            "name": model_file.stem,
            "path": str(model_file),
            "size_mb": round(model_file.stat().st_size / 1e6, 2)
        }

        if meta_file.exists():
            import json
            with open(meta_file) as f:
                model_info["metadata"] = json.load(f)

        models.append(model_info)

    return {
        "models": models,
        "count": len(models)
    }


@router.get("/genetic/libraries")
async def list_genetic_libraries():
    """
    List available genetic optimization libraries.

    Returns:
        List of library info including availability status
    """
    libraries = GeneticOptimizerFactory.get_available_libraries()

    return {
        "libraries": libraries,
        "default": GeneticLibrary.DEAP.value,
        "description": "Available genetic optimization libraries for hyperparameter tuning"
    }


@router.post("/genetic/optimize-with-library")
async def run_genetic_optimization_with_library(
    library: str = "deap",
    population_size: int = 10,
    n_generations: int = 5,
    crossover_prob: float = 0.7,
    mutation_prob: float = 0.2,
    param_ranges: Dict = None
):
    """
    Run genetic algorithm optimization with a specific library.

    Args:
        library: Library to use (deap, pygad, shinka_evolve)
        population_size: Population size
        n_generations: Number of generations
        crossover_prob: Crossover probability
        mutation_prob: Mutation probability
        param_ranges: Optional custom parameter ranges

    Returns:
        Optimization results
    """
    try:
        # Convert library string to enum
        try:
            lib_enum = GeneticLibrary(library.lower())
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unknown library: {library}. Available: {[l.value for l in GeneticLibrary]}"
            )

        logger.info(f"Starting genetic optimization with {library}: pop={population_size}, gen={n_generations}")

        # Create optimizer with factory
        optimizer = GeneticOptimizerFactory.create(
            library=lib_enum,
            param_ranges=param_ranges,
            population_size=population_size,
            n_generations=n_generations,
            crossover_prob=crossover_prob,
            mutation_prob=mutation_prob
        )

        # Run with dummy fitness function
        result = optimizer.optimize(FitnessEvaluator.dummy_fitness)

        return {
            "status": "completed",
            "library": library,
            "best_params": result.best_params,
            "best_fitness": result.best_fitness,
            "generations_run": result.generations_run,
            "early_stopped": result.early_stopped,
            "history_summary": [
                {
                    "generation": h.get('generation', i),
                    "best_fitness": h.get('best_fitness', 0)
                }
                for i, h in enumerate(result.history)
            ]
        }

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except NotImplementedError as e:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Genetic optimization failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Optimization failed: {str(e)}"
        )
