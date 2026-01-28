"""
Job Training Handler

Background task handler for ML model training jobs.
Uses real ML services instead of simulation.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path

from app.models.database import SessionLocal
from app.models.dataset import Dataset

logger = logging.getLogger(__name__)

# Check for ML libraries
try:
    from app.services.training import TrainingService, DARTS_AVAILABLE
    from app.services.ml_models import MLModelsService, PredictionTargetService, DatasetSplitter
    from app.services.genetic import GeneticOptimizer, DEAP_AVAILABLE
    ML_AVAILABLE = DARTS_AVAILABLE and DEAP_AVAILABLE
except ImportError as e:
    logger.warning(f"ML libraries not fully available: {e}")
    ML_AVAILABLE = False


def update_job_progress(task_id: str, progress: float, message: str):
    """Update job progress in the task queue."""
    from app.services.task_queue import get_task_queue
    try:
        task_queue = get_task_queue()
        task_queue.update_progress(task_id, progress, message)
        logger.info(f"Job {task_id}: {message} ({progress:.1f}%)")
    except Exception as e:
        logger.warning(f"Failed to update job progress: {e}")


def load_dataset(dataset_id: int) -> Optional[pd.DataFrame]:
    """Load dataset from file."""
    db = SessionLocal()
    try:
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not dataset:
            logger.error(f"Dataset {dataset_id} not found")
            return None

        if not dataset.file_path:
            logger.error(f"Dataset {dataset_id} has no file path")
            return None

        file_path = Path(dataset.file_path)
        if not file_path.exists():
            logger.error(f"Dataset file not found: {file_path}")
            return None

        df = pd.read_csv(file_path)
        logger.info(f"Loaded dataset {dataset_id}: {len(df)} rows, {len(df.columns)} columns")
        return df

    finally:
        db.close()


def get_dataset_info(dataset_id: int) -> Dict[str, Any]:
    """Get dataset metadata including timeframe."""
    db = SessionLocal()
    try:
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if dataset:
            return {
                'id': dataset.id,
                'name': dataset.name,
                'ticker': dataset.ticker,
                'timeframe': dataset.timeframe or 'daily'
            }
        return {'timeframe': 'daily'}
    finally:
        db.close()


def handle_training_job(task_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle ML training job.

    This is the main entry point for training jobs, called by TaskQueueService.

    Args:
        task_id: Unique task identifier
        payload: Job configuration containing:
            - dataset_ids: List of dataset IDs to train on
            - selected_models: Model types to train (lstm, nbeats, rnn)
            - parameter_ranges: Hyperparameter search ranges
            - prediction_targets: Target configurations
            - train_test_split: Train/test split percentage
            - genetic_config: Genetic algorithm configuration
            - metrics_config: Metrics configuration

    Returns:
        Result dictionary with trained model info and metrics
    """
    if not ML_AVAILABLE:
        return {
            'status': 'failed',
            'error': 'ML libraries not available. Install darts and deap.'
        }

    try:
        update_job_progress(task_id, 0, "Starting training job...")

        # Extract configuration
        dataset_ids = payload.get('dataset_ids', [])
        if not dataset_ids and payload.get('dataset_id'):
            dataset_ids = [payload['dataset_id']]

        if not dataset_ids:
            return {'status': 'failed', 'error': 'No datasets specified'}

        selected_models = payload.get('selected_models', ['lstm'])
        parameter_ranges = payload.get('parameter_ranges', {})
        prediction_targets = payload.get('prediction_targets', [])
        train_test_split = payload.get('train_test_split', 80)
        genetic_config = payload.get('genetic_config', {})
        metrics_config = payload.get('metrics_config', {})

        # Load and combine datasets
        update_job_progress(task_id, 5, "Loading datasets...")
        combined_df = None
        dataset_infos = []

        for ds_id in dataset_ids:
            df = load_dataset(ds_id)
            if df is None:
                return {'status': 'failed', 'error': f'Failed to load dataset {ds_id}'}

            info = get_dataset_info(ds_id)
            dataset_infos.append(info)

            # Add ticker column if combining multiple datasets
            if len(dataset_ids) > 1:
                df['ticker'] = info.get('ticker', f'TICKER{ds_id}')

            if combined_df is None:
                combined_df = df
            else:
                combined_df = pd.concat([combined_df, df], ignore_index=True)

        # Sort by date
        combined_df = combined_df.sort_values('Date').reset_index(drop=True)
        update_job_progress(task_id, 10, f"Loaded {len(combined_df)} rows from {len(dataset_ids)} dataset(s)")

        # Calculate prediction targets
        if prediction_targets:
            update_job_progress(task_id, 15, "Calculating prediction targets...")
            target_service = PredictionTargetService()

            targets = []
            for pt in prediction_targets:
                targets.append({
                    'profit_pct': pt.get('profitPercent', 10),
                    'max_dd': pt.get('maxDrawdownPercent', 5),
                    'days': pt.get('timePeriodDays', 7),
                    'direction': 'up'
                })
                # Add symmetric down target
                targets.append({
                    'profit_pct': pt.get('profitPercent', 10),
                    'max_dd': pt.get('maxDrawdownPercent', 5),
                    'days': pt.get('timePeriodDays', 7),
                    'direction': 'down'
                })

            combined_df = target_service.calculate_prediction_targets(combined_df, targets)

            # Get the first target column name for training
            target_column = f"price_up_{targets[0]['profit_pct']}pct_{targets[0]['max_dd']}dd_{targets[0]['days']}d"
        else:
            # Default: use Close for regression
            target_column = 'Close'

        # Train/test split
        update_job_progress(task_id, 20, "Splitting train/test data...")
        train_ratio = train_test_split / 100.0
        train_df, test_df = DatasetSplitter.train_test_split(combined_df, train_ratio=train_ratio)

        logger.info(f"Train: {len(train_df)} rows, Test: {len(test_df)} rows")

        # Get feature columns (exclude Date, OHLCV, and target columns)
        exclude_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'ticker']
        target_cols = [c for c in combined_df.columns if c.startswith('price_')]
        exclude_cols.extend(target_cols)
        feature_columns = [c for c in combined_df.columns if c not in exclude_cols]

        # Get timeframe from first dataset (for frequency inference)
        timeframe = dataset_infos[0].get('timeframe', 'daily') if dataset_infos else 'daily'

        # Train each model type
        results = []
        total_models = len(selected_models)

        for model_idx, model_type in enumerate(selected_models):
            model_progress_base = 25 + (model_idx / total_models) * 60

            update_job_progress(task_id, model_progress_base, f"Training {model_type.upper()} model...")

            try:
                model_result = train_single_model(
                    task_id=task_id,
                    model_type=model_type,
                    train_df=train_df,
                    test_df=test_df,
                    target_column=target_column,
                    feature_columns=feature_columns,
                    parameter_ranges=parameter_ranges,
                    genetic_config=genetic_config,
                    metrics_config=metrics_config,
                    progress_base=model_progress_base,
                    progress_range=60.0 / total_models,
                    timeframe=timeframe
                )
                results.append(model_result)

            except Exception as e:
                logger.error(f"Failed to train {model_type}: {e}")
                results.append({
                    'model_type': model_type,
                    'status': 'failed',
                    'error': str(e)
                })

        # Find best model
        update_job_progress(task_id, 90, "Analyzing results...")

        successful_results = [r for r in results if r.get('status') == 'completed']
        best_result = None
        if successful_results:
            best_result = max(successful_results, key=lambda x: x.get('best_fitness', 0))

        # Determine overall job status
        if len(successful_results) == 0:
            # All models failed
            error_messages = [r.get('error', 'Unknown error') for r in results if r.get('status') == 'failed']
            combined_error = "; ".join(set(error_messages[:3]))  # Dedupe and limit
            update_job_progress(task_id, 100, f"Training failed: {combined_error}")

            return {
                'status': 'failed',
                'error': f"All {total_models} model(s) failed to train. Errors: {combined_error}",
                'models_trained': 0,
                'total_models': total_models,
                'results': results,
                'datasets': dataset_infos,
                'train_rows': len(train_df),
                'test_rows': len(test_df),
                'completed_at': datetime.now().isoformat()
            }
        elif len(successful_results) < total_models:
            # Some models failed
            update_job_progress(task_id, 100, f"Training partially completed ({len(successful_results)}/{total_models} models)")

            return {
                'status': 'partial',
                'models_trained': len(successful_results),
                'total_models': total_models,
                'results': results,
                'best_model': best_result,
                'datasets': dataset_infos,
                'train_rows': len(train_df),
                'test_rows': len(test_df),
                'completed_at': datetime.now().isoformat()
            }
        else:
            # All models succeeded
            update_job_progress(task_id, 100, "Training completed successfully")

            return {
                'status': 'completed',
                'models_trained': len(successful_results),
                'total_models': total_models,
                'results': results,
                'best_model': best_result,
                'datasets': dataset_infos,
                'train_rows': len(train_df),
                'test_rows': len(test_df),
                'completed_at': datetime.now().isoformat()
            }

    except Exception as e:
        logger.error(f"Training job {task_id} failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            'status': 'failed',
            'error': str(e)
        }


def train_single_model(
    task_id: str,
    model_type: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_column: str,
    feature_columns: List[str],
    parameter_ranges: Dict[str, Any],
    genetic_config: Dict[str, Any],
    metrics_config: Dict[str, Any],
    progress_base: float,
    progress_range: float,
    timeframe: str = 'daily'
) -> Dict[str, Any]:
    """
    Train a single model type with genetic optimization.

    Provides real-time progress updates for:
    - Each generation of genetic optimization
    - Each individual model being trained within a generation

    Args:
        task_id: Task ID for progress updates
        model_type: Model type (lstm, nbeats, rnn)
        train_df: Training data
        test_df: Test data
        target_column: Column to predict
        feature_columns: Feature columns
        parameter_ranges: Hyperparameter ranges from job config
        genetic_config: Genetic algorithm config
        metrics_config: Metrics configuration
        progress_base: Base progress percentage
        progress_range: Progress range for this model
        timeframe: Dataset timeframe for frequency inference

    Returns:
        Training result dictionary
    """
    from app.services.task_queue import get_task_queue

    # Initialize services
    ml_service = MLModelsService()
    training_service = TrainingService()

    # Build parameter ranges for genetic optimizer
    ga_param_ranges = build_param_ranges(model_type, parameter_ranges)

    # Get genetic config
    population_size = genetic_config.get('populationSize', 20)
    generations = genetic_config.get('generations', 50)
    crossover_prob = genetic_config.get('crossoverProb', 0.7)
    mutation_prob = genetic_config.get('mutationProb', 0.2)
    early_stopping = genetic_config.get('earlyStoppingGenerations', 5)

    # Optimize metric
    optimize_metric = metrics_config.get('optimizeMetric', 'f1_score')

    # Prepare data for Darts
    try:
        train_series, train_covariates = training_service.prepare_data(
            train_df,
            target_column=target_column,
            feature_columns=feature_columns[:10],  # Limit features for stability
            timeframe=timeframe
        )
        test_series, test_covariates = training_service.prepare_data(
            test_df,
            target_column=target_column,
            feature_columns=feature_columns[:10],
            timeframe=timeframe
        )
    except Exception as e:
        logger.error(f"Failed to prepare data for {model_type}: {e}")
        return {
            'model_type': model_type,
            'status': 'failed',
            'error': f'Data preparation failed: {e}'
        }

    # Progress tracking state - mutable to allow updates from nested functions
    progress_state = {
        'current_generation': 0,
        'current_individual': 0,
        'best_fitness': 0.0,
        'cancelled': False
    }

    # Best model tracking
    best_model = [None]
    best_metrics = [{}]

    def check_cancelled() -> bool:
        """Check if task was cancelled - uses short-lived DB session."""
        task_queue = get_task_queue()
        status = task_queue.get_task_status(task_id)
        if status and status.get('status') in ['cancelled', 'paused']:
            progress_state['cancelled'] = True
            return True
        return False

    def fitness_function(params: Dict) -> float:
        """
        Evaluate model with given parameters.
        Updates progress for each individual being trained.
        """
        # Check for cancellation before training
        if progress_state['cancelled'] or check_cancelled():
            raise InterruptedError("Task cancelled")

        # Update progress for this individual
        progress_state['current_individual'] += 1
        individual_num = progress_state['current_individual']
        gen = progress_state['current_generation']

        # Calculate fine-grained progress:
        # Each generation covers (progress_range / generations) percent
        # Within each generation, each individual covers a fraction of that
        gen_progress = (gen / generations) * progress_range * 0.9  # 90% for generations
        individual_progress = (individual_num / population_size) * (progress_range / generations) * 0.9

        current_progress = progress_base + gen_progress + individual_progress

        update_job_progress(
            task_id,
            current_progress,
            f"{model_type.upper()}: Gen {gen}/{generations}, Training individual {individual_num}/{population_size}"
        )

        try:
            # Create model
            model = ml_service.create_model(model_type, params)

            # Train - this is the time-consuming part
            train_result = training_service.train_model(
                model,
                train_series,
                covariates=train_covariates,
                verbose=False
            )

            if train_result.get('status') == 'failed':
                return 0.0

            # Check cancellation after training
            if check_cancelled():
                raise InterruptedError("Task cancelled")

            # Evaluate
            eval_result = training_service.evaluate_model(
                model,
                test_series,
                covariates=test_covariates
            )

            if 'error' in eval_result:
                return 0.0

            # Calculate fitness based on metric
            if optimize_metric == 'mape':
                # Lower MAPE is better
                mape = eval_result.get('mape', 100)
                fitness = 1.0 / (1.0 + mape / 100)
            else:
                # For other metrics, use directly or calculate
                fitness = 1.0 - eval_result.get('mape', 100) / 100

            # Track best
            if best_model[0] is None or fitness > best_metrics[0].get('fitness', 0):
                best_model[0] = model
                best_metrics[0] = {**eval_result, 'fitness': fitness, 'params': params}
                progress_state['best_fitness'] = fitness

                # Update progress with new best
                update_job_progress(
                    task_id,
                    current_progress,
                    f"{model_type.upper()}: Gen {gen}/{generations}, New best fitness: {fitness:.4f}"
                )

            return fitness

        except InterruptedError:
            raise
        except Exception as e:
            logger.warning(f"Fitness evaluation failed: {e}")
            return 0.0

    def ga_callback(generation: int, best_fitness: float, best_params: Dict):
        """Called after each generation completes."""
        # Reset individual counter for next generation
        progress_state['current_generation'] = generation + 1
        progress_state['current_individual'] = 0

        # Update progress at generation boundary
        progress = progress_base + ((generation + 1) / generations) * progress_range * 0.9
        update_job_progress(
            task_id,
            progress,
            f"{model_type.upper()}: Completed Gen {generation + 1}/{generations}, Best: {best_fitness:.4f}"
        )

        # Check if task is paused/cancelled
        if check_cancelled():
            raise InterruptedError("Task paused/cancelled")

    # Run genetic optimization
    logger.info(f"Starting genetic optimization for {model_type}")
    logger.info(f"Population: {population_size}, Generations: {generations}")

    update_job_progress(
        task_id,
        progress_base,
        f"{model_type.upper()}: Starting optimization (pop={population_size}, gens={generations})"
    )

    optimizer = GeneticOptimizer(
        param_ranges=ga_param_ranges,
        population_size=population_size,
        n_generations=generations,
        crossover_prob=crossover_prob,
        mutation_prob=mutation_prob,
        early_stopping_generations=early_stopping
    )

    try:
        opt_result = optimizer.optimize(
            fitness_function=fitness_function,
            callback=ga_callback
        )
    except InterruptedError:
        update_job_progress(
            task_id,
            progress_base + progress_range * 0.5,
            f"{model_type.upper()}: Training interrupted"
        )
        return {
            'model_type': model_type,
            'status': 'paused',
            'generations_run': progress_state['current_generation'],
            'best_fitness': progress_state['best_fitness']
        }

    # Save best model if found
    model_path = None
    if best_model[0] is not None:
        update_job_progress(
            task_id,
            progress_base + progress_range * 0.95,
            f"{model_type.upper()}: Saving best model..."
        )
        try:
            model_name = f"job_{task_id}_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            model_path = training_service.save_model(
                best_model[0],
                model_name,
                metadata={
                    'model_type': model_type,
                    'task_id': task_id,
                    'best_params': opt_result.get('best_params'),
                    'best_fitness': opt_result.get('best_fitness'),
                    'metrics': best_metrics[0]
                }
            )
            logger.info(f"Saved best {model_type} model to {model_path}")
        except Exception as e:
            logger.warning(f"Failed to save model: {e}")

    update_job_progress(
        task_id,
        progress_base + progress_range,
        f"{model_type.upper()}: Completed with fitness {opt_result.get('best_fitness', 0):.4f}"
    )

    return {
        'model_type': model_type,
        'status': 'completed',
        'best_params': opt_result.get('best_params'),
        'best_fitness': opt_result.get('best_fitness'),
        'generations_run': opt_result.get('generations_run'),
        'metrics': best_metrics[0],
        'model_path': model_path,
        'history': opt_result.get('history', [])[-5:]  # Last 5 generations
    }


def build_param_ranges(model_type: str, ranges: Dict[str, Any]) -> Dict[str, Dict]:
    """
    Build genetic algorithm parameter ranges from job config.

    Args:
        model_type: Model type
        ranges: Parameter ranges from job config

    Returns:
        Parameter ranges for GeneticOptimizer
    """
    # Extract ranges from job config
    layers_min = ranges.get('layersMin', 1)
    layers_max = ranges.get('layersMax', 4)
    layer_size_min = ranges.get('layerSizeMin', 32)
    layer_size_max = ranges.get('layerSizeMax', 256)
    lr_min = ranges.get('learningRateMin', 0.0001)
    lr_max = ranges.get('learningRateMax', 0.01)
    dropout_min = ranges.get('dropoutMin', 0.0)
    dropout_max = ranges.get('dropoutMax', 0.5)

    param_ranges = {
        'n_rnn_layers': {'min': layers_min, 'max': layers_max, 'step': 1, 'type': 'int'},
        'dropout': {'min': dropout_min, 'max': dropout_max, 'step': 0.1, 'type': 'float'},
        'learning_rate': {'min': lr_min, 'max': lr_max, 'step': 0.0001, 'type': 'float'},
        'batch_size': {'min': 16, 'max': 128, 'step': 16, 'type': 'int'},
        'input_chunk_length': {'min': 10, 'max': 60, 'step': 5, 'type': 'int'}
    }

    # Add per-layer hidden dimensions
    for i in range(1, 5):  # Up to 4 layers
        param_ranges[f'hidden_dim_layer_{i}'] = {
            'min': layer_size_min,
            'max': layer_size_max,
            'step': 16,
            'type': 'int'
        }

    # Model-specific adjustments
    if model_type == 'nbeats':
        param_ranges['num_stacks'] = {'min': 10, 'max': 50, 'step': 10, 'type': 'int'}
        param_ranges['num_blocks'] = {'min': 1, 'max': 3, 'step': 1, 'type': 'int'}
        param_ranges['num_layers'] = {'min': 2, 'max': 6, 'step': 1, 'type': 'int'}

    return param_ranges
