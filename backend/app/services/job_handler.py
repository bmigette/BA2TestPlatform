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
    """Update job progress in the task queue and add to logs."""
    from app.services.task_queue import get_task_queue
    try:
        task_queue = get_task_queue()
        task_queue.update_progress(task_id, progress, message)
        logger.info(f"Job {task_id}: {message} ({progress:.1f}%)")
        # Also add to job_progress_data for UI logs
        add_job_log(task_id, message)
    except Exception as e:
        logger.warning(f"Failed to update job progress: {e}")


def add_job_log(task_id: str, message: str):
    """Add a log entry to the job's progress data."""
    from datetime import datetime
    try:
        # Import here to avoid circular imports
        from app.api.jobs import job_progress_data
        if task_id in job_progress_data:
            job_progress_data[task_id]["logs"].append(
                f"[{datetime.now().isoformat()}] {message}"
            )
    except Exception as e:
        # Silently fail if job_progress_data not available
        pass


def update_job_training_state(
    task_id: str,
    current_generation: int = None,
    total_generations: int = None,
    current_individual: int = None,
    population_size: int = None,
    current_model_type: str = None,
    current_epoch: int = None,
    total_epochs: int = None,
    best_fitness: float = None,
    error_count: int = None,
    success_count: int = None,
    current_model_params: Dict[str, Any] = None,
    epoch_loss: float = None
):
    """Update job training state for real-time progress tracking."""
    try:
        from app.api.jobs import jobs_store
        if task_id in jobs_store:
            job = jobs_store[task_id]
            if current_generation is not None:
                job["currentGeneration"] = current_generation
            if total_generations is not None:
                job["totalGenerations"] = total_generations
            if current_individual is not None:
                job["currentIndividual"] = current_individual
            if population_size is not None:
                job["populationSize"] = population_size
            if current_model_type is not None:
                job["currentModelType"] = current_model_type
            if current_epoch is not None:
                job["currentEpoch"] = current_epoch
            if total_epochs is not None:
                job["totalEpochs"] = total_epochs
            if best_fitness is not None:
                job["bestFitness"] = best_fitness
            if error_count is not None:
                job["errorCount"] = error_count
            if success_count is not None:
                job["successCount"] = success_count
            if current_model_params is not None:
                job["currentModelParams"] = current_model_params
            if epoch_loss is not None:
                # Append to epoch history for graphing
                if "epochHistory" not in job:
                    job["epochHistory"] = []
                job["epochHistory"].append({
                    "epoch": current_epoch or len(job["epochHistory"]) + 1,
                    "loss": epoch_loss
                })
                # Keep only last 100 epochs to prevent memory bloat
                if len(job["epochHistory"]) > 100:
                    job["epochHistory"] = job["epochHistory"][-100:]
    except Exception as e:
        logger.warning(f"Failed to update job training state: {e}")


def save_ga_checkpoint(task_id: str, checkpoint_data: Dict[str, Any]):
    """Save genetic algorithm checkpoint to database for crash recovery."""
    from app.models.task_queue import TaskQueue
    db = SessionLocal()
    try:
        task = db.query(TaskQueue).filter(TaskQueue.task_id == task_id).first()
        if task:
            task.checkpoint_data = checkpoint_data
            db.commit()
            logger.debug(f"Saved GA checkpoint for task {task_id}, gen {checkpoint_data.get('generation', 0)}")
    except Exception as e:
        logger.error(f"Failed to save GA checkpoint: {e}")
        db.rollback()
    finally:
        db.close()


def load_ga_checkpoint(task_id: str) -> Optional[Dict[str, Any]]:
    """Load genetic algorithm checkpoint from database."""
    from app.models.task_queue import TaskQueue
    db = SessionLocal()
    try:
        task = db.query(TaskQueue).filter(TaskQueue.task_id == task_id).first()
        if task and task.checkpoint_data:
            logger.info(f"Found GA checkpoint for task {task_id}, gen {task.checkpoint_data.get('generation', 0)}")
            return task.checkpoint_data
        return None
    except Exception as e:
        logger.error(f"Failed to load GA checkpoint: {e}")
        return None
    finally:
        db.close()


def clear_ga_checkpoint(task_id: str):
    """Clear checkpoint data after successful completion."""
    from app.models.task_queue import TaskQueue
    db = SessionLocal()
    try:
        task = db.query(TaskQueue).filter(TaskQueue.task_id == task_id).first()
        if task:
            task.checkpoint_data = None
            db.commit()
    except Exception as e:
        logger.error(f"Failed to clear GA checkpoint: {e}")
    finally:
        db.close()


def get_job_models_dir(task_id: str) -> Path:
    """Get the directory for storing job models."""
    models_dir = Path("trained_models") / task_id
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


def save_generation_model(
    task_id: str,
    model: Any,
    generation: int,
    individual: int,
    model_type: str,
    fitness: float,
    params: Dict[str, Any],
    metrics: Dict[str, Any],
    training_service: Any
) -> Optional[str]:
    """
    Save a model from a generation.

    Args:
        task_id: Job ID
        model: Trained model
        generation: Generation number
        individual: Individual number within generation
        model_type: Type of model (lstm, nbeats, etc.)
        fitness: Model fitness score
        params: Model parameters
        metrics: Evaluation metrics
        training_service: TrainingService instance for saving

    Returns:
        Path to saved model or None if failed
    """
    try:
        models_dir = get_job_models_dir(task_id)
        model_filename = f"gen{generation:03d}_ind{individual:03d}_{model_type}_f{fitness:.4f}"
        full_path = models_dir / f"{model_filename}.pt"

        metadata = {
            'task_id': task_id,
            'generation': generation,
            'individual': individual,
            'model_type': model_type,
            'fitness': fitness,
            'params': params,
            'metrics': metrics
        }

        # Save model directly to the job models directory
        model.save(str(full_path))
        logger.debug(f"Saved model: {full_path}")

        # Save metadata
        meta_path = models_dir / f"{model_filename}_meta.json"
        import json
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        model_path = str(full_path)
        logger.debug(f"Saved model: {model_path}")
        return model_path

    except Exception as e:
        logger.error(f"Failed to save generation model: {e}")
        return None


def cleanup_generation_models(task_id: str, generation: int):
    """
    Remove all models from a specific generation.

    Args:
        task_id: Job ID
        generation: Generation number to clean up
    """
    try:
        models_dir = get_job_models_dir(task_id)
        pattern = f"gen{generation:03d}_*"

        import glob
        files_to_remove = list(models_dir.glob(pattern))

        for file_path in files_to_remove:
            try:
                file_path.unlink()
                logger.debug(f"Removed: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to remove {file_path}: {e}")

        if files_to_remove:
            logger.info(f"Cleaned up {len(files_to_remove)} files from generation {generation}")

    except Exception as e:
        logger.warning(f"Failed to cleanup generation {generation} models: {e}")


def save_best_model(
    task_id: str,
    model: Any,
    model_type: str,
    fitness: float,
    params: Dict[str, Any],
    metrics: Dict[str, Any],
    training_service: Any
) -> Optional[str]:
    """
    Save the best model to a permanent location.

    Args:
        task_id: Job ID
        model: Best trained model
        model_type: Type of model
        fitness: Model fitness score
        params: Model parameters
        metrics: Evaluation metrics
        training_service: TrainingService instance

    Returns:
        Path to saved model or None if failed
    """
    try:
        models_dir = get_job_models_dir(task_id)
        model_name = f"best_{model_type}_f{fitness:.4f}"

        metadata = {
            'task_id': task_id,
            'model_type': model_type,
            'fitness': fitness,
            'params': params,
            'metrics': metrics,
            'is_best': True
        }

        model_path = training_service.save_model(model, str(models_dir / model_name), metadata)
        logger.info(f"Saved best model: {model_path}")
        return model_path

    except Exception as e:
        logger.error(f"Failed to save best model: {e}")
        return None


def cleanup_job_models(task_id: str, keep_best: bool = True):
    """
    Clean up all models for a job, optionally keeping the best model.

    Args:
        task_id: Job ID
        keep_best: If True, keep files starting with 'best_' or 'elite_'
    """
    try:
        models_dir = get_job_models_dir(task_id)

        for file_path in models_dir.iterdir():
            if keep_best and (file_path.name.startswith('best_') or file_path.name.startswith('elite_')):
                continue
            try:
                file_path.unlink()
            except Exception as e:
                logger.warning(f"Failed to remove {file_path}: {e}")

        logger.info(f"Cleaned up job {task_id} models (keep_best={keep_best})")

    except Exception as e:
        logger.warning(f"Failed to cleanup job models: {e}")


def save_elite_models(
    task_id: str,
    all_individuals: List[Dict[str, Any]],
    elitism_percent: float = 10.0,
    population_size: int = 20,
    default_elite_count: int = 10
) -> List[str]:
    """
    Rename/mark the top N models as elite models to preserve them.

    The number of elite models is determined by:
    - If elitism_percent > 0: (elitism_percent / 100) * population_size
    - Otherwise: default_elite_count (default 10)

    Args:
        task_id: Job ID
        all_individuals: List of all evaluated individuals with their info
        elitism_percent: Percentage of population to keep as elite
        population_size: Size of population for calculating elite count
        default_elite_count: Default number of models to keep if no elitism

    Returns:
        List of paths to elite models
    """
    try:
        models_dir = get_job_models_dir(task_id)

        # Calculate number of elite models to keep
        if elitism_percent > 0:
            elite_count = max(1, int((elitism_percent / 100.0) * population_size))
        else:
            elite_count = default_elite_count

        # Sort individuals by fitness (descending) and get top N
        sorted_individuals = sorted(
            all_individuals,
            key=lambda x: x.get('fitness', 0),
            reverse=True
        )
        elite_individuals = sorted_individuals[:elite_count]

        logger.info(f"Saving {len(elite_individuals)} elite models (elitism={elitism_percent}%, pop={population_size})")

        elite_paths = []
        for rank, ind in enumerate(elite_individuals, 1):
            gen = ind.get('generation', 0)
            individual_num = ind.get('individual', 0)
            model_type = ind.get('model_type', 'unknown')
            fitness = ind.get('fitness', 0)

            # Find the original model file
            original_pattern = f"gen{gen:03d}_ind{individual_num:03d}_{model_type}_*"
            matching_files = list(models_dir.glob(original_pattern))

            if not matching_files:
                logger.warning(f"Could not find model for gen{gen}_ind{individual_num}_{model_type}")
                continue

            for original_path in matching_files:
                # Create new elite filename with rank
                suffix = original_path.suffix
                if suffix == '.json':
                    new_name = f"elite_{rank:02d}_{model_type}_f{fitness:.4f}_meta.json"
                else:
                    new_name = f"elite_{rank:02d}_{model_type}_f{fitness:.4f}{suffix}"

                new_path = models_dir / new_name

                try:
                    # Rename to elite
                    original_path.rename(new_path)
                    if not suffix == '.json':
                        elite_paths.append(str(new_path))
                    logger.debug(f"Renamed {original_path.name} -> {new_name}")
                except Exception as e:
                    logger.warning(f"Failed to rename {original_path}: {e}")

        logger.info(f"Saved {len(elite_paths)} elite models")
        return elite_paths

    except Exception as e:
        logger.error(f"Failed to save elite models: {e}")
        return []


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

        # Train with unified optimization (model type is an optimization parameter)
        update_job_progress(task_id, 25, f"Starting unified optimization across {len(selected_models)} model types...")

        try:
            model_result = train_unified_optimization(
                task_id=task_id,
                selected_models=selected_models,
                full_df=combined_df,
                train_ratio=train_ratio,
                target_column=target_column,
                feature_columns=feature_columns,
                parameter_ranges=parameter_ranges,
                genetic_config=genetic_config,
                metrics_config=metrics_config,
                progress_base=25,
                progress_range=65,
                timeframe=timeframe
            )
            results = [model_result]

        except Exception as e:
            logger.error(f"Failed unified optimization: {e}")
            import traceback
            traceback.print_exc()
            results = [{
                'model_type': 'unified',
                'status': 'failed',
                'error': str(e)
            }]

        # Find best model
        update_job_progress(task_id, 90, "Analyzing results...")

        successful_results = [r for r in results if r.get('status') == 'completed']
        best_result = None
        if successful_results:
            best_result = max(successful_results, key=lambda x: x.get('best_fitness', 0))

        # Determine overall job status (unified optimization counts as 1 model)
        total_models = 1
        if len(successful_results) == 0:
            # Training failed
            error_messages = [r.get('error', 'Unknown error') for r in results if r.get('status') == 'failed']
            combined_error = "; ".join(set(error_messages[:3]))  # Dedupe and limit
            update_job_progress(task_id, 100, f"Training failed: {combined_error}")

            return {
                'status': 'failed',
                'error': f"Training failed: {combined_error}",
                'models_trained': 0,
                'total_models': total_models,
                'results': results,
                'datasets': dataset_infos,
                'train_rows': len(train_df),
                'test_rows': len(test_df),
                'completed_at': datetime.now().isoformat()
            }
        else:
            # Training succeeded
            update_job_progress(task_id, 100, "Training completed successfully")

            # Include all_individuals and error/success counts for visualization
            all_individuals = []
            total_error_count = 0
            total_success_count = 0
            for r in results:
                if 'all_individuals' in r:
                    all_individuals.extend(r['all_individuals'])
                total_error_count += r.get('error_count', 0)
                total_success_count += r.get('success_count', 0)

            return {
                'status': 'completed',
                'models_trained': len(successful_results),
                'total_models': total_models,
                'results': results,
                'best_model': best_result,
                'all_individuals': all_individuals,  # For UI visualization
                'error_count': total_error_count,
                'success_count': total_success_count,
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

    # Prepare data for Darts FIRST to know data length for constraints
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

    # Build parameter ranges with data length constraints
    # For RNN: training_length defaults to 3 * input_chunk_length
    # Ensure input_chunk_length <= train_series_length / 4 to have enough data
    train_length = len(train_series)
    max_input_chunk = min(60, max(10, train_length // 4))
    ga_param_ranges = build_param_ranges(model_type, parameter_ranges, max_input_chunk=max_input_chunk)

    # Get genetic config
    population_size = genetic_config.get('populationSize', 20)
    generations = genetic_config.get('generations', 50)
    crossover_prob = genetic_config.get('crossoverProb', 0.7)
    mutation_prob = genetic_config.get('mutationProb', 0.2)
    early_stopping = genetic_config.get('earlyStoppingGenerations', 5)

    # Optimize metric
    optimize_metric = metrics_config.get('optimizeMetric', 'f1_score')

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

            # Evaluate with the selected optimization metric
            eval_result = training_service.evaluate_model(
                model,
                test_series,
                covariates=test_covariates,
                optimize_metric=optimize_metric
            )

            if 'error' in eval_result:
                return 0.0

            # Calculate fitness based on metric type
            if optimize_metric == 'mape':
                # Lower MAPE is better - convert to fitness (higher is better)
                mape_val = eval_result.get('mape', 100)
                if mape_val is None:
                    mape_val = 100
                fitness = 1.0 / (1.0 + mape_val / 100)
            elif optimize_metric in {'mae', 'rmse'}:
                # Lower is better for regression metrics
                metric_val = eval_result.get(optimize_metric, 100)
                fitness = 1.0 / (1.0 + metric_val)
            else:
                # Classification metrics (f1_score, accuracy, etc.) - higher is better
                fitness = eval_result.get(optimize_metric, 0.0)

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
            logger.error(f"Fitness evaluation failed: {e}")
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
            logger.error(f"Failed to save model: {e}")

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


def train_unified_optimization(
    task_id: str,
    selected_models: List[str],
    full_df: pd.DataFrame,
    train_ratio: float,
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
    Unified optimization where model type is an optimization parameter.

    Each individual in the population can be a different model type,
    allowing the GA to compare and optimize across model architectures.
    """
    from app.services.task_queue import get_task_queue

    # Initialize services
    ml_service = MLModelsService()
    training_service = TrainingService()

    # Prepare data once with proper train/test split
    # Using prepare_data_split ensures train and test series share the same
    # index space, which is required for Darts metric functions to work correctly
    try:
        train_series, test_series, train_covariates, test_covariates = training_service.prepare_data_split(
            full_df,
            train_ratio=train_ratio,
            target_column=target_column,
            feature_columns=feature_columns[:10],
            timeframe=timeframe
        )
    except Exception as e:
        logger.error(f"Failed to prepare data: {e}")
        return {
            'model_type': 'unified',
            'status': 'failed',
            'error': f'Data preparation failed: {e}'
        }

    # Build unified parameter ranges including model_type_idx
    train_length = len(train_series)
    max_input_chunk = min(60, max(10, train_length // 4))
    ga_param_ranges = build_unified_param_ranges(selected_models, parameter_ranges, max_input_chunk)

    # Get genetic config
    population_size = genetic_config.get('populationSize', 20)
    generations = genetic_config.get('generations', 50)
    crossover_prob = genetic_config.get('crossoverProb', 0.7)
    mutation_prob = genetic_config.get('mutationProb', 0.2)
    early_stopping = genetic_config.get('earlyStoppingGenerations', 5)
    training_epochs = genetic_config.get('trainingEpochs', 10)
    optimize_metric = metrics_config.get('optimizeMetric', 'f1_score')

    # Progress tracking
    progress_state = {
        'current_generation': 0,
        'current_individual': 0,
        'best_fitness': 0.0,
        'best_model_type': None,
        'best_model_params': {},
        'cancelled': False,
        'all_individuals': [],  # Track all evaluated individuals for visualization
        'error_count': 0,  # Track training/evaluation errors
        'success_count': 0  # Track successful evaluations
    }

    best_model = [None]
    best_metrics = [{}]

    def check_cancelled() -> bool:
        task_queue = get_task_queue()
        status = task_queue.get_task_status(task_id)
        if status and status.get('status') in ['cancelled', 'paused']:
            progress_state['cancelled'] = True
            return True
        return False

    def fitness_function(params: Dict) -> float:
        """Evaluate model with given parameters including model type."""
        if progress_state['cancelled'] or check_cancelled():
            raise InterruptedError("Task cancelled")

        progress_state['current_individual'] += 1
        individual_num = progress_state['current_individual']
        gen = progress_state['current_generation']

        # Get model type from params
        model_type_idx = int(params.get('model_type_idx', 0))
        model_type = selected_models[model_type_idx % len(selected_models)]

        # Calculate progress
        gen_progress = (gen / generations) * progress_range
        individual_progress = (individual_num / population_size) * (progress_range / generations) * 0.8
        current_progress = progress_base + gen_progress + individual_progress

        # Update training state for real-time UI
        update_job_training_state(
            task_id,
            current_generation=gen,
            total_generations=generations,
            current_individual=individual_num,
            population_size=population_size,
            current_model_type=model_type,
            current_epoch=0,
            total_epochs=10,  # Will be updated during training
            best_fitness=progress_state.get('best_fitness'),
            error_count=progress_state.get('error_count', 0),
            success_count=progress_state.get('success_count', 0)
        )

        update_job_progress(
            task_id,
            current_progress,
            f"Gen {gen}/{generations}, {model_type.upper()} #{individual_num}/{population_size}"
        )

        try:
            # Get model-specific params
            model_params = get_model_params(model_type, params, training_epochs)
            n_epochs = model_params.get('n_epochs', 10)

            # Update epoch info and current model params before training
            update_job_training_state(
                task_id,
                current_epoch=0,
                total_epochs=n_epochs,
                current_model_type=model_type,
                current_model_params=model_params
            )

            # Create epoch callback to update UI during training with loss
            def epoch_callback(current_epoch: int, total_epochs: int, loss: float = None):
                update_job_training_state(
                    task_id,
                    current_epoch=current_epoch,
                    total_epochs=total_epochs,
                    epoch_loss=loss
                )

            model = ml_service.create_model(model_type, model_params, epoch_callback=epoch_callback)

            # Train
            training_result = training_service.train_model(
                model, train_series, covariates=train_covariates, verbose=False
            )

            # Update epoch to complete after training
            update_job_training_state(task_id, current_epoch=n_epochs)

            if training_result.get('status') == 'failed':
                logger.error(f"Training failed: {training_result.get('error')}")
                progress_state['error_count'] += 1
                update_job_training_state(
                    task_id,
                    error_count=progress_state['error_count'],
                    success_count=progress_state['success_count']
                )
                return 0.0

            # Evaluate with the selected optimization metric
            eval_result = training_service.evaluate_model(
                model, test_series,
                covariates=test_covariates,
                optimize_metric=optimize_metric
            )

            if 'error' in eval_result:
                progress_state['error_count'] += 1
                update_job_training_state(
                    task_id,
                    error_count=progress_state['error_count'],
                    success_count=progress_state['success_count']
                )
                return 0.0

            # Calculate fitness based on metric type
            if optimize_metric == 'mape':
                # Lower MAPE is better - convert to fitness (higher is better)
                mape_val = eval_result.get('mape', 100)
                if mape_val is None:
                    mape_val = 100
                fitness = 1.0 / (1.0 + mape_val / 100)
            elif optimize_metric in {'mae', 'rmse'}:
                # Lower is better for regression metrics
                metric_val = eval_result.get(optimize_metric, 100)
                fitness = 1.0 / (1.0 + metric_val)
            else:
                # Classification metrics (f1_score, accuracy, etc.) - higher is better
                fitness = eval_result.get(optimize_metric, 0.0)

            # Track this individual for visualization
            individual_record = {
                'generation': gen,
                'individual': individual_num,
                'model_type': model_type,
                'params': model_params,
                'fitness': fitness,
                'metrics': eval_result
            }
            progress_state['all_individuals'].append(individual_record)

            # Save model for this generation
            save_generation_model(
                task_id=task_id,
                model=model,
                generation=gen,
                individual=individual_num,
                model_type=model_type,
                fitness=fitness,
                params=model_params,
                metrics=eval_result,
                training_service=training_service
            )

            # Track best
            if fitness > progress_state['best_fitness']:
                progress_state['best_fitness'] = fitness
                progress_state['best_model_type'] = model_type
                progress_state['best_model_params'] = model_params
                best_model[0] = model
                best_metrics[0] = eval_result
                update_job_progress(
                    task_id, current_progress,
                    f"Gen {gen}/{generations}, New best: {model_type.upper()} fitness={fitness:.4f}"
                )

            progress_state['success_count'] += 1
            update_job_training_state(
                task_id,
                error_count=progress_state['error_count'],
                success_count=progress_state['success_count']
            )
            return fitness

        except Exception as e:
            logger.error(f"Fitness evaluation failed for {model_type}: {e}")
            progress_state['error_count'] += 1
            update_job_training_state(
                task_id,
                error_count=progress_state['error_count'],
                success_count=progress_state['success_count']
            )
            return 0.0

    def ga_callback(gen: int, best_fitness: float, best_params: Dict):
        """Called after each generation completes."""
        if check_cancelled():
            raise InterruptedError("Task cancelled")
        progress_state['current_generation'] = gen + 1
        progress_state['current_individual'] = 0

        # Note: We keep all models during training and cleanup at the end
        # to preserve elite models from any generation

        # Update training state for UI
        update_job_training_state(
            task_id,
            current_generation=gen + 1,
            current_individual=0,
            best_fitness=best_fitness
        )

        update_job_progress(
            task_id,
            progress_base + ((gen + 1) / generations) * progress_range,
            f"Gen {gen + 1}/{generations} complete, best fitness: {best_fitness:.4f}"
        )

    def checkpoint_callback(gen: int, population: list):
        """Save checkpoint after each generation for crash recovery."""
        checkpoint_data = optimizer.get_checkpoint_data(gen, population)
        checkpoint_data['all_individuals'] = progress_state['all_individuals']
        save_ga_checkpoint(task_id, checkpoint_data)

    # Check for existing checkpoint (for resume)
    checkpoint = load_ga_checkpoint(task_id)
    start_generation = 0
    initial_population = None

    if checkpoint:
        logger.info(f"Resuming from checkpoint: gen {checkpoint.get('generation', 0)}")
        update_job_progress(task_id, progress_base, f"Resuming from generation {checkpoint.get('generation', 0)}")
        progress_state['all_individuals'] = checkpoint.get('all_individuals', [])
        start_generation = checkpoint.get('generation', 0) + 1
        initial_population = checkpoint.get('population', [])
    else:
        update_job_progress(task_id, progress_base, f"Starting unified optimization (pop={population_size}, gens={generations})")

    optimizer = GeneticOptimizer(
        param_ranges=ga_param_ranges,
        population_size=population_size,
        n_generations=generations,
        crossover_prob=crossover_prob,
        mutation_prob=mutation_prob,
        early_stopping_generations=early_stopping
    )

    # Restore optimizer state if resuming
    if checkpoint:
        optimizer.resume_from_checkpoint(checkpoint)

    try:
        opt_result = optimizer.optimize(
            fitness_function=fitness_function,
            callback=ga_callback,
            start_generation=start_generation,
            initial_population=initial_population,
            checkpoint_callback=checkpoint_callback
        )
        # Clear checkpoint on successful completion
        clear_ga_checkpoint(task_id)
    except InterruptedError:
        logger.info(f"Unified optimization cancelled for task {task_id}")
        return {
            'model_type': 'unified',
            'status': 'cancelled',
            'best_fitness': progress_state['best_fitness'],
            'all_individuals': progress_state['all_individuals']
        }

    # Get best model type from best params
    best_params = opt_result.get('best_params', {})
    best_model_type_idx = int(best_params.get('model_type_idx', 0))
    best_model_type = selected_models[best_model_type_idx % len(selected_models)]

    update_job_progress(
        task_id,
        progress_base + progress_range,
        f"Unified optimization complete. Best: {best_model_type.upper()} fitness={opt_result.get('best_fitness', 0):.4f}"
    )

    # Save elite models (top N based on elitism percent or default 10)
    elitism_percent = genetic_config.get('elitismPercent', 10.0)
    elite_paths = save_elite_models(
        task_id=task_id,
        all_individuals=progress_state['all_individuals'],
        elitism_percent=elitism_percent,
        population_size=population_size,
        default_elite_count=10
    )

    # Cleanup all generation models, keep only elite models
    cleanup_job_models(task_id, keep_best=True)

    # Get path to the best model (elite_01)
    model_path = elite_paths[0] if elite_paths else None

    return {
        'model_type': best_model_type,
        'status': 'completed',
        'best_params': best_params,
        'best_fitness': opt_result.get('best_fitness'),
        'generations_run': opt_result.get('generations_run'),
        'metrics': best_metrics[0],
        'model_path': model_path,
        'history': opt_result.get('history', [])[-5:],
        'all_individuals': progress_state['all_individuals'],  # For UI visualization
        'error_count': progress_state.get('error_count', 0),
        'success_count': progress_state.get('success_count', 0)
    }


def get_model_params(model_type: str, params: Dict, training_epochs: int = 10) -> Dict:
    """Extract model-specific parameters from unified params.

    Args:
        model_type: Type of model (lstm, nbeats, etc.)
        params: Unified parameters from genetic optimization
        training_epochs: Number of epochs for training (from geneticConfig)
    """
    model_params = {
        'input_chunk_length': int(params.get('input_chunk_length', 24)),
        'output_chunk_length': 7,
        'n_epochs': training_epochs,
        'batch_size': int(params.get('batch_size', 32)),
        'learning_rate': params.get('learning_rate', 0.001),
        'dropout': params.get('dropout', 0.1),
    }

    model_type_lower = model_type.lower()

    if model_type_lower in ['lstm', 'gru']:
        # RNN models use hidden_dim (single int)
        model_params['hidden_dim'] = int(params.get('hidden_dim_layer_1', 128))
        model_params['n_rnn_layers'] = int(params.get('n_rnn_layers', 2))
    elif model_type_lower == 'nbeats':
        model_params['num_stacks'] = int(params.get('num_stacks', 30))
        model_params['num_blocks'] = int(params.get('num_blocks', 1))
        model_params['num_layers'] = int(params.get('num_layers', 4))
        model_params['layer_widths'] = int(params.get('hidden_dim_layer_1', 256))
    elif model_type_lower == 'tcn':
        model_params['kernel_size'] = int(params.get('kernel_size', 3))
        model_params['num_filters'] = int(params.get('num_filters', 64))
        model_params['dilation_base'] = int(params.get('dilation_base', 2))
    elif model_type_lower == 'transformer':
        model_params['d_model'] = int(params.get('d_model', 64))
        model_params['nhead'] = int(params.get('nhead', 4))
        model_params['num_encoder_layers'] = int(params.get('num_encoder_layers', 2))
        model_params['num_decoder_layers'] = int(params.get('num_decoder_layers', 2))
        model_params['dim_feedforward'] = int(params.get('hidden_dim_layer_1', 128))
    elif model_type_lower == 'tft':
        model_params['hidden_size'] = int(params.get('hidden_dim_layer_1', 64))
        model_params['lstm_layers'] = int(params.get('n_rnn_layers', 1))
        model_params['num_attention_heads'] = int(params.get('nhead', 4))

    return model_params


def build_unified_param_ranges(
    selected_models: List[str],
    ranges: Dict[str, Any],
    max_input_chunk: int = 60
) -> Dict[str, Dict]:
    """
    Build parameter ranges for unified optimization including model_type_idx.
    """
    layers_min = ranges.get('layersMin', 1)
    layers_max = ranges.get('layersMax', 4)
    layer_size_min = ranges.get('layerSizeMin', 32)
    layer_size_max = ranges.get('layerSizeMax', 256)
    lr_min = ranges.get('learningRateMin', 0.0001)
    lr_max = ranges.get('learningRateMax', 0.01)
    dropout_min = ranges.get('dropoutMin', 0.0)
    dropout_max = ranges.get('dropoutMax', 0.5)

    # input_chunk_length constraints (RNN models now set training_length dynamically)
    input_chunk_max = max_input_chunk
    input_chunk_min = min(10, input_chunk_max)

    param_ranges = {
        # Model type selection (index into selected_models)
        'model_type_idx': {'min': 0, 'max': len(selected_models) - 1, 'step': 1, 'type': 'int'},
        # Common parameters
        'n_rnn_layers': {'min': layers_min, 'max': layers_max, 'step': 1, 'type': 'int'},
        'dropout': {'min': dropout_min, 'max': dropout_max, 'step': 0.1, 'type': 'float'},
        'learning_rate': {'min': lr_min, 'max': lr_max, 'step': 0.0001, 'type': 'float'},
        'batch_size': {'min': 16, 'max': 128, 'step': 16, 'type': 'int'},
        'input_chunk_length': {'min': input_chunk_min, 'max': input_chunk_max, 'step': 5, 'type': 'int'},
    }

    # Add hidden dim layers (scaled appropriately per model during evaluation)
    for i in range(1, 5):
        param_ranges[f'hidden_dim_layer_{i}'] = {
            'min': layer_size_min,
            'max': layer_size_max,
            'step': 16,
            'type': 'int'
        }

    # Model-specific params (used by respective models)
    param_ranges['num_stacks'] = {'min': 10, 'max': 50, 'step': 10, 'type': 'int'}
    param_ranges['num_blocks'] = {'min': 1, 'max': 3, 'step': 1, 'type': 'int'}
    param_ranges['num_layers'] = {'min': 2, 'max': 6, 'step': 1, 'type': 'int'}
    param_ranges['kernel_size'] = {'min': 2, 'max': 7, 'step': 1, 'type': 'int'}
    param_ranges['num_filters'] = {'min': 32, 'max': 128, 'step': 16, 'type': 'int'}
    param_ranges['dilation_base'] = {'min': 2, 'max': 4, 'step': 1, 'type': 'int'}
    param_ranges['d_model'] = {'min': 32, 'max': 256, 'step': 32, 'type': 'int'}
    param_ranges['nhead'] = {'min': 2, 'max': 8, 'step': 2, 'type': 'int'}
    param_ranges['num_encoder_layers'] = {'min': 1, 'max': 4, 'step': 1, 'type': 'int'}
    param_ranges['num_decoder_layers'] = {'min': 1, 'max': 4, 'step': 1, 'type': 'int'}

    return param_ranges


def build_param_ranges(model_type: str, ranges: Dict[str, Any], max_input_chunk: int = 60) -> Dict[str, Dict]:
    """
    Build genetic algorithm parameter ranges from job config.

    Layer size scaling from base (user specifies Transformer/TCN/TFT base size):
    - LSTM/GRU: 4x base (e.g., base=128 -> hidden_dim=512)
    - N-BEATS: 2x base (e.g., base=128 -> layer_widths=256)
    - Transformer/TCN/TFT: 1x base (e.g., base=128 -> d_model/num_filters/hidden_size=128)

    Args:
        model_type: Model type
        ranges: Parameter ranges from job config
        max_input_chunk: Maximum allowed input_chunk_length based on data length

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

    # Constrain input_chunk_length based on data length and model type
    # For RNN models (LSTM/GRU): training_length defaults to 24, so input_chunk_length must be <= 24
    model_type_lower = model_type.lower()
    if model_type_lower in ['lstm', 'gru']:
        # RNNModel has training_length=24 by default, input_chunk must be <= training_length
        input_chunk_max = min(max_input_chunk, 24)
    else:
        input_chunk_max = min(max_input_chunk, 60)
    input_chunk_min = min(10, input_chunk_max)

    # Apply model-specific layer size scaling
    # User specifies layer size for Transformer (base), then:
    # - Transformer/TCN/TFT: use as-is (1x)
    # - N-BEATS: multiply by 2 (2x)
    # - LSTM/GRU: multiply by 4 (4x)
    if model_type_lower in ['lstm', 'gru']:
        # LSTM/GRU: 4x the base layer size
        layer_size_multiplier = 4
    elif model_type_lower in ['nbeats']:
        # N-BEATS: 2x the base layer size
        layer_size_multiplier = 2
    else:
        # Transformer/TCN/TFT: use base layer size (1x)
        layer_size_multiplier = 1

    effective_layer_size_min = layer_size_min * layer_size_multiplier
    effective_layer_size_max = layer_size_max * layer_size_multiplier

    param_ranges = {
        'n_rnn_layers': {'min': layers_min, 'max': layers_max, 'step': 1, 'type': 'int'},
        'dropout': {'min': dropout_min, 'max': dropout_max, 'step': 0.1, 'type': 'float'},
        'learning_rate': {'min': lr_min, 'max': lr_max, 'step': 0.0001, 'type': 'float'},
        'batch_size': {'min': 16, 'max': 128, 'step': 16, 'type': 'int'},
        'input_chunk_length': {'min': input_chunk_min, 'max': input_chunk_max, 'step': 5, 'type': 'int'}
    }

    # Add per-layer hidden dimensions with model-appropriate sizes
    for i in range(1, 5):  # Up to 4 layers
        param_ranges[f'hidden_dim_layer_{i}'] = {
            'min': effective_layer_size_min,
            'max': effective_layer_size_max,
            'step': 16,
            'type': 'int'
        }

    # Model-specific adjustments
    if model_type_lower == 'nbeats':
        param_ranges['num_stacks'] = {'min': 10, 'max': 50, 'step': 10, 'type': 'int'}
        param_ranges['num_blocks'] = {'min': 1, 'max': 3, 'step': 1, 'type': 'int'}
        param_ranges['num_layers'] = {'min': 2, 'max': 6, 'step': 1, 'type': 'int'}
    elif model_type_lower == 'tcn':
        param_ranges['kernel_size'] = {'min': 2, 'max': 7, 'step': 1, 'type': 'int'}
        param_ranges['num_filters'] = {'min': 32, 'max': 128, 'step': 16, 'type': 'int'}
        param_ranges['dilation_base'] = {'min': 2, 'max': 4, 'step': 1, 'type': 'int'}
    elif model_type_lower == 'transformer':
        param_ranges['d_model'] = {'min': 32, 'max': 256, 'step': 32, 'type': 'int'}
        param_ranges['nhead'] = {'min': 2, 'max': 8, 'step': 2, 'type': 'int'}
        param_ranges['num_encoder_layers'] = {'min': 1, 'max': 4, 'step': 1, 'type': 'int'}
        param_ranges['num_decoder_layers'] = {'min': 1, 'max': 4, 'step': 1, 'type': 'int'}

    return param_ranges
