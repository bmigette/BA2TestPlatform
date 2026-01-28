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
                train_df=train_df,
                test_df=test_df,
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

            # Include all_individuals for visualization
            all_individuals = []
            for r in results:
                if 'all_individuals' in r:
                    all_individuals.extend(r['all_individuals'])

            return {
                'status': 'completed',
                'models_trained': len(successful_results),
                'total_models': total_models,
                'results': results,
                'best_model': best_result,
                'all_individuals': all_individuals,  # For UI visualization
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


def train_unified_optimization(
    task_id: str,
    selected_models: List[str],
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
    Unified optimization where model type is an optimization parameter.

    Each individual in the population can be a different model type,
    allowing the GA to compare and optimize across model architectures.
    """
    from app.services.task_queue import get_task_queue

    # Initialize services
    ml_service = MLModelsService()
    training_service = TrainingService()

    # Prepare data once (shared across all model types)
    try:
        train_series, train_covariates = training_service.prepare_data(
            train_df,
            target_column=target_column,
            feature_columns=feature_columns[:10],
            timeframe=timeframe
        )
        test_series, test_covariates = training_service.prepare_data(
            test_df,
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
    optimize_metric = metrics_config.get('optimizeMetric', 'f1_score')

    # Progress tracking
    progress_state = {
        'current_generation': 0,
        'current_individual': 0,
        'best_fitness': 0.0,
        'cancelled': False,
        'all_individuals': []  # Track all evaluated individuals for visualization
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

        update_job_progress(
            task_id,
            current_progress,
            f"Gen {gen}/{generations}, {model_type.upper()} #{individual_num}/{population_size}"
        )

        try:
            # Get model-specific params
            model_params = get_model_params(model_type, params)
            model = ml_service.create_model(model_type, model_params)

            # Train
            training_result = training_service.train_model(
                model, train_series, covariates=train_covariates, verbose=False
            )

            if training_result.get('status') == 'failed':
                logger.warning(f"Training failed: {training_result.get('error')}")
                return 0.0

            # Evaluate
            eval_result = training_service.evaluate_model(model, test_series, covariates=test_covariates)

            if 'error' in eval_result:
                return 0.0

            # Calculate fitness
            fitness = eval_result.get(optimize_metric, eval_result.get('mape', 0))
            if optimize_metric == 'mape':
                fitness = max(0, 100 - fitness) / 100

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

            # Track best
            if fitness > progress_state['best_fitness']:
                progress_state['best_fitness'] = fitness
                best_model[0] = model
                best_metrics[0] = eval_result
                update_job_progress(
                    task_id, current_progress,
                    f"Gen {gen}/{generations}, New best: {model_type.upper()} fitness={fitness:.4f}"
                )

            return fitness

        except Exception as e:
            logger.warning(f"Fitness evaluation failed for {model_type}: {e}")
            return 0.0

    def ga_callback(gen: int, best_fitness: float, best_params: Dict):
        """Called after each generation completes."""
        if check_cancelled():
            raise InterruptedError("Task cancelled")
        progress_state['current_generation'] = gen + 1
        progress_state['current_individual'] = 0
        update_job_progress(
            task_id,
            progress_base + ((gen + 1) / generations) * progress_range,
            f"Gen {gen + 1}/{generations} complete, best fitness: {best_fitness:.4f}"
        )

    # Run optimization
    update_job_progress(task_id, progress_base, f"Starting unified optimization (pop={population_size}, gens={generations})")

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

    return {
        'model_type': best_model_type,
        'status': 'completed',
        'best_params': best_params,
        'best_fitness': opt_result.get('best_fitness'),
        'generations_run': opt_result.get('generations_run'),
        'metrics': best_metrics[0],
        'model_path': None,
        'history': opt_result.get('history', [])[-5:],
        'all_individuals': progress_state['all_individuals']  # For UI visualization
    }


def get_model_params(model_type: str, params: Dict) -> Dict:
    """Extract model-specific parameters from unified params."""
    model_params = {
        'input_chunk_length': int(params.get('input_chunk_length', 24)),
        'output_chunk_length': 7,
        'n_epochs': 10,  # Reduced for faster GA evaluation
        'batch_size': int(params.get('batch_size', 32)),
        'learning_rate': params.get('learning_rate', 0.001),
        'dropout': params.get('dropout', 0.1),
    }

    model_type_lower = model_type.lower()

    if model_type_lower in ['lstm', 'gru', 'rnn']:
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

    # Constrain input_chunk_length (use min of all model constraints)
    input_chunk_max = min(max_input_chunk, 24)  # RNN constraint
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

    Different model types have different appropriate layer size ranges:
    - LSTM/GRU/RNN: hidden_dim 32-1024 (recurrent units)
    - N-BEATS: layer_widths 128-512 (FC layers per stack)
    - TCN: num_filters 32-256 (convolutional filters)
    - Transformer: d_model 32-256 (embedding dimension)

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
    if model_type_lower in ['lstm', 'gru', 'rnn', 'rcnn']:
        # RNNModel has training_length=24 by default, input_chunk must be <= training_length
        input_chunk_max = min(max_input_chunk, 24)
    else:
        input_chunk_max = min(max_input_chunk, 60)
    input_chunk_min = min(10, input_chunk_max)

    # Apply model-specific layer size scaling
    # User specifies layer size for Transformer (base), then:
    # - Transformer/TCN: use as-is (1x)
    # - N-BEATS: multiply by 2 (2x)
    # - LSTM/GRU/RNN: multiply by 4 (4x)
    if model_type_lower in ['lstm', 'gru', 'rnn', 'rcnn']:
        # LSTM/GRU/RNN: 4x the base layer size
        layer_size_multiplier = 4
    elif model_type_lower in ['nbeats']:
        # N-BEATS: 2x the base layer size
        layer_size_multiplier = 2
    else:
        # Transformer/TCN: use base layer size (1x)
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
