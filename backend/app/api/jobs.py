"""
Optimization Jobs API endpoints

Uses TaskQueueService for background job processing with real ML training.
"""

from fastapi import APIRouter, HTTPException, status, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging
import uuid
import asyncio
import json

from sqlalchemy.orm import Session
from app.services.task_queue import get_task_queue
from app.models.database import SessionLocal, get_db
from app.models.dataset import Dataset
from app.models.task_queue import TaskQueue

logger = logging.getLogger(__name__)

router = APIRouter()

# In-memory job store for quick access (synced with task queue)
jobs_store: dict = {}

# Flag to track if we've loaded jobs from DB
_jobs_loaded_from_db = False

# Training progress data (metrics over time)
job_progress_data: Dict[str, Dict[str, Any]] = {}


class PredictionTarget(BaseModel):
    profitPercent: float
    maxDrawdownPercent: float
    timePeriodDays: int


class ParameterRanges(BaseModel):
    layersMin: int
    layersMax: int
    layersStep: int = 1
    layerSizeMin: int
    layerSizeMax: int
    layerSizeStep: int = 64
    learningRateMin: float
    learningRateMax: float
    learningRateStep: float = 0.001
    dropoutMin: float = 0.0
    dropoutMax: float = 0.5
    dropoutStep: float = 0.1
    activationFunctions: List[str]


class GeneticConfig(BaseModel):
    """Genetic algorithm optimization configuration"""
    populationSize: int = 20
    generations: int = 50
    elitismPercent: float = 10.0  # Percentage of best individuals to keep
    crossoverProb: float = 0.7
    mutationProb: float = 0.2
    earlyStoppingGenerations: int = 5  # Stop if no improvement for N generations
    trainingEpochs: int = 10  # Number of epochs for training each model


class MetricsConfig(BaseModel):
    """Metrics configuration for model optimization"""
    optimizeMetric: str = "f1_score"  # f1_score, accuracy, balanced_accuracy, precision, recall, auc_roc, mcc


class CrossValidationConfig(BaseModel):
    enabled: bool = False
    folds: int = 5
    useDatasetAsFold: bool = True  # Use each dataset as a fold


class JobCreate(BaseModel):
    datasetId: Optional[int] = None  # Single dataset (backwards compatible)
    datasetIds: Optional[List[int]] = None  # Multiple datasets
    selectedModels: List[str]
    parameterRanges: ParameterRanges
    predictionTargets: List[PredictionTarget]
    trainTestSplit: int
    crossValidation: Optional[CrossValidationConfig] = None
    geneticConfig: Optional[GeneticConfig] = None
    metricsConfig: Optional[MetricsConfig] = None


class DatasetProgress(BaseModel):
    datasetId: int
    datasetName: str
    ticker: str
    status: str  # 'pending', 'processing', 'completed'
    progress: float = 0.0
    rowsProcessed: int = 0
    totalRows: int = 0


class JobResponse(BaseModel):
    id: str
    datasetId: Optional[int] = None  # Single dataset (backwards compatible)
    datasetIds: Optional[List[int]] = None  # Multiple datasets
    datasetNames: Optional[List[str]] = None  # Dataset names for display
    selectedModels: List[str]
    parameterRanges: ParameterRanges
    predictionTargets: List[PredictionTarget]
    trainTestSplit: int
    crossValidation: Optional[CrossValidationConfig] = None
    geneticConfig: Optional[GeneticConfig] = None
    metricsConfig: Optional[MetricsConfig] = None
    status: str  # 'queued', 'running', 'paused', 'completed', 'failed', 'cancelled'
    progress: float  # 0-100
    createdAt: str
    startedAt: Optional[str] = None
    completedAt: Optional[str] = None
    error: Optional[str] = None
    # Training metrics
    currentGeneration: int = 0
    totalGenerations: int = 50
    currentLoss: Optional[float] = None
    currentAccuracy: Optional[float] = None
    bestFitness: Optional[float] = None
    gpuUtilization: Optional[float] = None
    estimatedTimeRemaining: Optional[str] = None
    optimizeMetric: Optional[str] = None  # The metric being optimized
    # Training progress details
    currentEpoch: Optional[int] = None
    totalEpochs: Optional[int] = None
    currentIndividual: Optional[int] = None
    populationSize: Optional[int] = None
    currentModelType: Optional[str] = None
    currentModelParams: Optional[Dict[str, Any]] = None  # Current model hyperparameters
    epochHistory: Optional[List[Dict[str, Any]]] = None  # Epoch-level loss history
    errorCount: Optional[int] = None  # Number of training errors
    successCount: Optional[int] = None  # Number of successful trainings
    # Multi-dataset progress
    datasetProgress: Optional[List[DatasetProgress]] = None
    currentDatasetId: Optional[int] = None
    # Cross-validation results
    foldResults: Optional[List[Dict[str, Any]]] = None
    # Parameter combinations count
    totalCombinations: Optional[int] = None


class TrainingMetrics(BaseModel):
    generation: int
    loss: float
    accuracy: float
    valLoss: float
    valAccuracy: float
    fitness: float
    timestamp: str


class JobProgressResponse(BaseModel):
    job: JobResponse
    metrics: List[TrainingMetrics]
    logs: List[str]


class JobListResponse(BaseModel):
    jobs: List[JobResponse]
    total: int


def load_jobs_from_database():
    """
    Load jobs from database into jobs_store on startup or first access.
    This ensures jobs persist across app restarts.
    """
    global _jobs_loaded_from_db
    if _jobs_loaded_from_db:
        return

    db = SessionLocal()
    try:
        # Load all training_job tasks from database
        tasks = db.query(TaskQueue).filter(
            TaskQueue.task_type == 'training_job'
        ).order_by(TaskQueue.created_at.desc()).limit(100).all()

        for task in tasks:
            if task.task_id not in jobs_store:
                # Reconstruct job from task payload and status
                payload = task.payload or {}

                # Get dataset info
                dataset_ids = payload.get('dataset_ids', [])
                if not dataset_ids and payload.get('dataset_id'):
                    dataset_ids = [payload['dataset_id']]

                dataset_names = []
                dataset_progress = []
                for ds_id in dataset_ids:
                    ds_info = get_dataset_info(ds_id)
                    dataset_names.append(ds_info['datasetName'])
                    dataset_progress.append(ds_info)

                # Get genetic config with defaults
                genetic_config = payload.get('genetic_config', {})
                metrics_config = payload.get('metrics_config', {})
                param_ranges = payload.get('parameter_ranges', {})

                job_data = {
                    'id': task.task_id,
                    'datasetId': dataset_ids[0] if len(dataset_ids) == 1 else None,
                    'datasetIds': dataset_ids if len(dataset_ids) > 1 else None,
                    'datasetNames': dataset_names if len(dataset_ids) > 1 else None,
                    'selectedModels': payload.get('selected_models', []),
                    'parameterRanges': param_ranges,
                    'predictionTargets': payload.get('prediction_targets', []),
                    'trainTestSplit': payload.get('train_test_split', 80),
                    'crossValidation': payload.get('cross_validation'),
                    'geneticConfig': genetic_config,
                    'metricsConfig': metrics_config,
                    'status': task.status,
                    'progress': task.progress or 0.0,
                    'createdAt': task.created_at.isoformat() if task.created_at else datetime.now().isoformat(),
                    'startedAt': task.started_at.isoformat() if task.started_at else None,
                    'completedAt': task.completed_at.isoformat() if task.completed_at else None,
                    'error': task.error_message,
                    'currentGeneration': 0,
                    'totalGenerations': genetic_config.get('generations', 50),
                    'currentLoss': None,
                    'currentAccuracy': None,
                    'bestFitness': None,
                    'gpuUtilization': None,
                    'estimatedTimeRemaining': None,
                    'optimizeMetric': metrics_config.get('optimizeMetric', 'f1_score'),
                    'datasetProgress': dataset_progress if len(dataset_ids) > 1 else None,
                    'currentDatasetId': None,
                    'foldResults': None,
                    'totalCombinations': None
                }

                # Extract result data if available
                if task.result:
                    result = task.result
                    if result.get('best_model'):
                        best = result['best_model']
                        job_data['bestFitness'] = best.get('best_fitness')
                        if best.get('metrics'):
                            job_data['currentAccuracy'] = best['metrics'].get('fitness')

                jobs_store[task.task_id] = job_data

                # Initialize progress data
                if task.task_id not in job_progress_data:
                    job_progress_data[task.task_id] = {
                        "metrics": [],
                        "logs": [f"[{task.created_at.isoformat() if task.created_at else datetime.now().isoformat()}] Job loaded from database"]
                    }

        _jobs_loaded_from_db = True
        logger.info(f"Loaded {len(tasks)} jobs from database")

    except Exception as e:
        logger.error(f"Failed to load jobs from database: {e}")
    finally:
        db.close()


def get_dataset_info(dataset_id: int) -> Dict[str, Any]:
    """Get dataset information from database."""
    db = SessionLocal()
    try:
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if dataset:
            return {
                "datasetId": dataset.id,
                "datasetName": dataset.name,
                "ticker": dataset.ticker or "UNKNOWN",
                "status": "pending",
                "progress": 0.0,
                "rowsProcessed": 0,
                "totalRows": dataset.rows_count or 0
            }
        return {
            "datasetId": dataset_id,
            "datasetName": f"Dataset_{dataset_id}",
            "ticker": f"TICKER{dataset_id}",
            "status": "pending",
            "progress": 0.0,
            "rowsProcessed": 0,
            "totalRows": 1000
        }
    finally:
        db.close()


def sync_job_from_task(job_id: str) -> Optional[Dict[str, Any]]:
    """Sync job data from task queue."""
    if job_id not in jobs_store:
        return None

    task_queue = get_task_queue()
    task_status = task_queue.get_task_status(job_id)

    if task_status:
        # Update local job store from task queue
        jobs_store[job_id]["status"] = task_status.get("status", "queued")
        jobs_store[job_id]["progress"] = task_status.get("progress", 0)

        if task_status.get("started_at"):
            jobs_store[job_id]["startedAt"] = task_status["started_at"]
        if task_status.get("completed_at"):
            jobs_store[job_id]["completedAt"] = task_status["completed_at"]
        if task_status.get("error_message"):
            jobs_store[job_id]["error"] = task_status["error_message"]
        if task_status.get("progress_message"):
            # Parse progress message for current generation info
            msg = task_status["progress_message"]
            if "Gen " in msg:
                try:
                    # Extract generation from message like "LSTM: Gen 5/50, Fitness: 0.85"
                    gen_part = msg.split("Gen ")[1].split(",")[0]
                    current, total = gen_part.split("/")
                    jobs_store[job_id]["currentGeneration"] = int(current)
                except (IndexError, ValueError):
                    pass

        # Get result data if completed or failed
        if task_status.get("result"):
            result = task_status["result"]

            # Check result status for failures
            if result.get("status") == "failed":
                jobs_store[job_id]["error"] = result.get("error", "Training failed")

            # Extract best model info if available
            if result.get("best_model"):
                best = result["best_model"]
                jobs_store[job_id]["bestFitness"] = best.get("best_fitness")
                if best.get("metrics"):
                    jobs_store[job_id]["currentAccuracy"] = best["metrics"].get("fitness")

            # Store models trained count
            if "models_trained" in result:
                jobs_store[job_id]["modelsTrained"] = result["models_trained"]
            if "total_models" in result:
                jobs_store[job_id]["totalModels"] = result["total_models"]

    return jobs_store.get(job_id)


@router.post("", response_model=JobResponse, status_code=status.HTTP_201_CREATED)
async def create_job(job_create: JobCreate):
    """
    Create a new optimization job.

    Supports both single dataset (datasetId) and multiple datasets (datasetIds).
    For multi-dataset training:
    - Datasets are combined chronologically
    - Ticker column is added to distinguish data from different tickers
    - Cross-validation can use each dataset as a fold

    Uses TaskQueueService for background ML training.

    Args:
        job_create: Job creation parameters

    Returns:
        Created job with ID and status
    """
    try:
        # Determine dataset IDs (backwards compatible)
        if job_create.datasetIds and len(job_create.datasetIds) > 0:
            dataset_ids = job_create.datasetIds
        elif job_create.datasetId:
            dataset_ids = [job_create.datasetId]
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Either datasetId or datasetIds must be provided"
            )

        logger.info(f"Creating optimization job for {len(dataset_ids)} dataset(s)")

        # Build dataset progress tracking from real database
        dataset_progress = []
        dataset_names = []
        for ds_id in dataset_ids:
            ds_info = get_dataset_info(ds_id)
            dataset_progress.append(ds_info)
            dataset_names.append(ds_info["datasetName"])

        # Calculate parameter combinations
        params = job_create.parameterRanges
        layers_count = max(1, (params.layersMax - params.layersMin) // params.layersStep + 1)
        layer_size_count = max(1, (params.layerSizeMax - params.layerSizeMin) // params.layerSizeStep + 1)
        lr_count = max(1, int((params.learningRateMax - params.learningRateMin) / params.learningRateStep) + 1)
        dropout_count = max(1, int((params.dropoutMax - params.dropoutMin) / params.dropoutStep) + 1)
        activation_count = len(params.activationFunctions)
        total_combinations = layers_count * layer_size_count * lr_count * dropout_count * activation_count * len(job_create.selectedModels)

        # Get genetic config with defaults
        genetic_config = job_create.geneticConfig or GeneticConfig()
        metrics_config = job_create.metricsConfig or MetricsConfig()

        # Build payload for background task
        task_payload = {
            'dataset_ids': dataset_ids,
            'selected_models': job_create.selectedModels,
            'parameter_ranges': params.dict(),
            'prediction_targets': [pt.dict() for pt in job_create.predictionTargets],
            'train_test_split': job_create.trainTestSplit,
            'cross_validation': job_create.crossValidation.dict() if job_create.crossValidation else None,
            'genetic_config': genetic_config.dict(),
            'metrics_config': metrics_config.dict()
        }

        # Queue background training task
        task_queue = get_task_queue()
        task_id = task_queue.queue_task(
            task_type='training_job',
            name=f'Training job: {", ".join(job_create.selectedModels)} on {len(dataset_ids)} dataset(s)',
            payload=task_payload,
            description=f'Genetic optimization with {genetic_config.generations} generations'
        )

        # Use task_id as job_id
        job_id = task_id

        job = JobResponse(
            id=job_id,
            datasetId=dataset_ids[0] if len(dataset_ids) == 1 else None,
            datasetIds=dataset_ids if len(dataset_ids) > 1 else None,
            datasetNames=dataset_names if len(dataset_ids) > 1 else None,
            selectedModels=job_create.selectedModels,
            parameterRanges=job_create.parameterRanges,
            predictionTargets=job_create.predictionTargets,
            trainTestSplit=job_create.trainTestSplit,
            crossValidation=job_create.crossValidation,
            geneticConfig=genetic_config,
            metricsConfig=metrics_config,
            status="queued",
            progress=0.0,
            createdAt=datetime.now().isoformat(),
            totalGenerations=genetic_config.generations,
            optimizeMetric=metrics_config.optimizeMetric,
            totalCombinations=total_combinations,
            datasetProgress=dataset_progress if len(dataset_ids) > 1 else None,
        )

        # Store in memory for quick access
        jobs_store[job_id] = job.dict()

        # Initialize progress data
        job_progress_data[job_id] = {
            "metrics": [],
            "logs": [f"[{datetime.now().isoformat()}] Job queued for processing"]
        }

        logger.info(f"Created job {job_id} with {len(dataset_ids)} dataset(s) - queued for background processing")

        return job

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create job: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create job: {str(e)}"
        )


@router.get("", response_model=JobListResponse)
async def list_jobs():
    """
    List all optimization jobs.

    Loads jobs from database on first access, then syncs status from task queue.

    Returns:
        List of jobs with status
    """
    try:
        # Load jobs from database if not already loaded (for persistence across restarts)
        load_jobs_from_database()

        # Sync all jobs from task queue
        for job_id in list(jobs_store.keys()):
            sync_job_from_task(job_id)

        jobs = [JobResponse(**job) for job in jobs_store.values()]
        # Sort by createdAt descending
        jobs.sort(key=lambda x: x.createdAt, reverse=True)

        return JobListResponse(
            jobs=jobs,
            total=len(jobs)
        )

    except Exception as e:
        logger.error(f"Failed to list jobs: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list jobs: {str(e)}"
        )


@router.get("/{job_id}", response_model=JobResponse)
async def get_job(job_id: str):
    """
    Get a specific job by ID.

    Loads from database if needed, then syncs status from task queue.

    Args:
        job_id: Job ID

    Returns:
        Job details
    """
    # Load jobs from database first to ensure we have all jobs
    load_jobs_from_database()

    if job_id not in jobs_store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found"
        )

    # Sync from task queue
    sync_job_from_task(job_id)

    return JobResponse(**jobs_store[job_id])


@router.delete("/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_job(job_id: str):
    """
    Delete a job by ID.

    Deletes from both in-memory store and database.

    Args:
        job_id: Job ID
    """
    # Load jobs from database first to ensure we have all jobs
    load_jobs_from_database()

    if job_id not in jobs_store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found"
        )

    # Cancel task if running
    task_queue = get_task_queue()
    task_queue.cancel_task(job_id)

    # Delete from database
    db = SessionLocal()
    try:
        db.query(TaskQueue).filter(TaskQueue.task_id == job_id).delete()
        db.commit()
        logger.info(f"Deleted job {job_id} from database")
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to delete job {job_id} from database: {e}")
    finally:
        db.close()

    # Delete from in-memory stores
    del jobs_store[job_id]
    if job_id in job_progress_data:
        del job_progress_data[job_id]
    logger.info(f"Deleted job {job_id}")


@router.get("/{job_id}/progress", response_model=JobProgressResponse)
async def get_job_progress(job_id: str):
    """
    Get detailed progress information for a job including metrics and logs.

    Syncs status from task queue before returning.

    Args:
        job_id: Job ID

    Returns:
        Job with progress data, metrics history, and logs
    """
    if job_id not in jobs_store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found"
        )

    # Sync from task queue
    sync_job_from_task(job_id)

    # Get progress message from task queue
    task_queue = get_task_queue()
    task_progress = task_queue.get_task_progress(job_id)
    if task_progress and task_progress.get("progress_message"):
        # Add task progress message to logs if new
        progress_msg = task_progress["progress_message"]
        if job_id in job_progress_data:
            logs = job_progress_data[job_id].get("logs", [])
            if not logs or progress_msg not in logs[-1]:
                job_progress_data[job_id]["logs"].append(
                    f"[{datetime.now().isoformat()}] {progress_msg}"
                )

    job = JobResponse(**jobs_store[job_id])
    progress_data = job_progress_data.get(job_id, {"metrics": [], "logs": []})

    # Convert metrics to TrainingMetrics objects
    metrics = [TrainingMetrics(**m) for m in progress_data.get("metrics", [])]
    logs = progress_data.get("logs", [])

    return JobProgressResponse(
        job=job,
        metrics=metrics,
        logs=logs
    )


@router.post("/{job_id}/pause")
async def pause_job(job_id: str):
    """
    Pause a running job.

    Args:
        job_id: Job ID
    """
    if job_id not in jobs_store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found"
        )

    # Sync and check status
    sync_job_from_task(job_id)
    job = jobs_store[job_id]

    if job["status"] != "running":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot pause job in status: {job['status']}"
        )

    # Pause via task queue
    task_queue = get_task_queue()
    if task_queue.pause_task(job_id):
        jobs_store[job_id]["status"] = "paused"
        if job_id in job_progress_data:
            job_progress_data[job_id]["logs"].append(
                f"[{datetime.now().isoformat()}] Job paused"
            )
        logger.info(f"Paused job {job_id}")
        return {"status": "paused", "message": f"Job {job_id} paused"}
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to pause job"
        )


@router.post("/{job_id}/resume")
async def resume_job(job_id: str):
    """
    Resume a paused or stopped (crashed) job.

    Args:
        job_id: Job ID
    """
    if job_id not in jobs_store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found"
        )

    # Sync and check status
    sync_job_from_task(job_id)
    job = jobs_store[job_id]

    # Allow resuming paused or stopped (crashed) jobs
    if job["status"] not in ["paused", "stopped"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot resume job in status: {job['status']}"
        )

    # Resume via task queue
    task_queue = get_task_queue()
    if task_queue.resume_task(job_id):
        jobs_store[job_id]["status"] = "queued"  # Re-queued for processing
        if job_id in job_progress_data:
            job_progress_data[job_id]["logs"].append(
                f"[{datetime.now().isoformat()}] Job resumed"
            )
        logger.info(f"Resumed job {job_id}")
        return {"status": "running", "message": f"Job {job_id} resumed"}
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to resume job"
        )


@router.post("/{job_id}/cancel")
async def cancel_job(job_id: str):
    """
    Cancel a running or paused job.

    Args:
        job_id: Job ID
    """
    if job_id not in jobs_store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found"
        )

    # Sync and check status
    sync_job_from_task(job_id)
    job = jobs_store[job_id]

    if job["status"] not in ["running", "paused", "queued"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot cancel job in status: {job['status']}"
        )

    # Cancel via task queue
    task_queue = get_task_queue()
    task_queue.cancel_task(job_id)

    jobs_store[job_id]["status"] = "cancelled"
    if job_id in job_progress_data:
        job_progress_data[job_id]["logs"].append(
            f"[{datetime.now().isoformat()}] Job cancelled by user"
        )

    logger.info(f"Cancelled job {job_id}")
    return {"status": "cancelled", "message": f"Job {job_id} cancelled"}


# ============================================================================
# SSE (Server-Sent Events) for Live Progress
# ============================================================================

async def generate_sse_events(job_id: str):
    """
    Generator for SSE events for job progress.

    Syncs with task queue for real-time status.

    Yields:
        SSE formatted events with job progress data
    """
    last_progress = -1

    while True:
        if job_id not in jobs_store:
            yield f"data: {json.dumps({'type': 'error', 'message': 'Job not found'})}\n\n"
            break

        # Sync from task queue
        sync_job_from_task(job_id)

        job = jobs_store[job_id]
        current_progress = job.get("progress", 0)
        current_gen = job.get("currentGeneration", 0)

        # Send update if progress changed or status changed
        if current_progress != last_progress or job["status"] in ["completed", "cancelled", "failed"]:
            event_data = {
                "type": "progress",
                "job_id": job_id,
                "status": job["status"],
                "progress": current_progress,
                "currentGeneration": current_gen,
                "totalGenerations": job.get("totalGenerations", 50),
                "currentLoss": job.get("currentLoss"),
                "currentAccuracy": job.get("currentAccuracy"),
                "bestFitness": job.get("bestFitness"),
                "gpuUtilization": job.get("gpuUtilization"),
                "estimatedTimeRemaining": job.get("estimatedTimeRemaining"),
                "timestamp": datetime.now().isoformat()
            }
            yield f"data: {json.dumps(event_data)}\n\n"
            last_progress = current_progress

        # Exit if job finished
        if job["status"] in ["completed", "cancelled", "failed"]:
            yield f"data: {json.dumps({'type': 'complete', 'status': job['status']})}\n\n"
            break

        await asyncio.sleep(0.5)  # Poll every 500ms


@router.get("/{job_id}/sse")
async def get_job_progress_sse(job_id: str):
    """
    Get live job progress via Server-Sent Events (SSE).

    This endpoint streams real-time updates about job progress.
    Connect with EventSource in the browser.

    Args:
        job_id: Job ID

    Returns:
        SSE stream with progress events
    """
    if job_id not in jobs_store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found"
        )

    return StreamingResponse(
        generate_sse_events(job_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@router.get("/{job_id}/logs")
async def get_job_logs(job_id: str, limit: int = 100):
    """
    Get training logs for a job.

    Args:
        job_id: Job ID
        limit: Maximum number of log entries to return

    Returns:
        List of log entries
    """
    if job_id not in jobs_store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found"
        )

    progress_data = job_progress_data.get(job_id, {"logs": []})
    logs = progress_data.get("logs", [])

    # Return last N logs
    return {
        "job_id": job_id,
        "logs": logs[-limit:],
        "total": len(logs)
    }


# ============================================================================
# Optimization Profiles (Database-backed)
# ============================================================================

from app.models.optimization_profile import OptimizationProfile as OptimizationProfileModel


class OptimizationProfileCreate(BaseModel):
    name: str
    description: Optional[str] = None
    selectedModels: List[str]
    parameterRanges: ParameterRanges
    predictionTargets: List[PredictionTarget]
    trainTestSplit: float = 80.0
    geneticConfig: Optional[GeneticConfig] = None
    metricsConfig: Optional[MetricsConfig] = None


class ProfileResponse(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    selectedModels: List[str]
    parameterRanges: Dict[str, Any]
    predictionTargets: List[Dict[str, Any]]
    trainTestSplit: float
    geneticConfig: Optional[Dict[str, Any]] = None
    metricsConfig: Optional[Dict[str, Any]] = None
    createdAt: str
    updatedAt: Optional[str] = None


def _profile_to_response(profile: OptimizationProfileModel) -> ProfileResponse:
    """Convert database model to response."""
    return ProfileResponse(
        id=profile.id,
        name=profile.name,
        description=profile.description,
        selectedModels=profile.model_types or [],
        parameterRanges=profile.parameter_ranges or {},
        predictionTargets=profile.prediction_targets or [],
        trainTestSplit=profile.train_test_split or 80.0,
        geneticConfig=profile.genetic_config,
        metricsConfig=profile.metrics_config,
        createdAt=profile.created_at.isoformat() if profile.created_at else datetime.now().isoformat(),
        updatedAt=profile.updated_at.isoformat() if profile.updated_at else None
    )


@router.post("/profiles", response_model=ProfileResponse, status_code=status.HTTP_201_CREATED)
async def create_profile(profile: OptimizationProfileCreate, db: Session = Depends(get_db)):
    """
    Create a new optimization profile.

    Profiles save optimization settings that can be reused for multiple jobs.

    Args:
        profile: Profile settings

    Returns:
        Created profile with ID
    """
    try:
        db_profile = OptimizationProfileModel(
            name=profile.name,
            description=profile.description,
            model_types=profile.selectedModels,
            parameter_ranges=profile.parameterRanges.dict() if profile.parameterRanges else {},
            prediction_targets=[t.dict() for t in profile.predictionTargets] if profile.predictionTargets else [],
            train_test_split=profile.trainTestSplit,
            genetic_config=profile.geneticConfig.dict() if profile.geneticConfig else None,
            metrics_config=profile.metricsConfig.dict() if profile.metricsConfig else None
        )

        db.add(db_profile)
        db.commit()
        db.refresh(db_profile)

        logger.info(f"Created optimization profile: {profile.name} (id={db_profile.id})")
        return _profile_to_response(db_profile)

    except Exception as e:
        db.rollback()
        logger.error(f"Failed to create profile: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create profile: {str(e)}"
        )


@router.get("/profiles")
async def list_profiles(db: Session = Depends(get_db)):
    """
    List all optimization profiles.

    Returns:
        List of profiles
    """
    profiles = db.query(OptimizationProfileModel).order_by(OptimizationProfileModel.created_at.desc()).all()

    return {
        "profiles": [_profile_to_response(p) for p in profiles],
        "total": len(profiles)
    }


@router.get("/profiles/{profile_id}", response_model=ProfileResponse)
async def get_profile(profile_id: int, db: Session = Depends(get_db)):
    """
    Get a specific optimization profile.

    Args:
        profile_id: Profile ID

    Returns:
        Profile details
    """
    profile = db.query(OptimizationProfileModel).filter(OptimizationProfileModel.id == profile_id).first()
    if not profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Profile {profile_id} not found"
        )

    return _profile_to_response(profile)


@router.delete("/profiles/{profile_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_profile(profile_id: int, db: Session = Depends(get_db)):
    """
    Delete an optimization profile.

    Args:
        profile_id: Profile ID
    """
    profile = db.query(OptimizationProfileModel).filter(OptimizationProfileModel.id == profile_id).first()
    if not profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Profile {profile_id} not found"
        )

    db.delete(profile)
    db.commit()
    logger.info(f"Deleted profile {profile_id}")


@router.post("/profiles/{profile_id}/apply")
async def apply_profile_to_job(profile_id: int, dataset_id: int, db: Session = Depends(get_db)):
    """
    Create a new job using settings from a profile.

    Args:
        profile_id: Profile ID to apply
        dataset_id: Dataset ID for the new job

    Returns:
        Created job
    """
    profile = db.query(OptimizationProfileModel).filter(OptimizationProfileModel.id == profile_id).first()
    if not profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Profile {profile_id} not found"
        )

    # Create job from profile
    job_create = JobCreate(
        datasetId=dataset_id,
        selectedModels=profile.model_types or [],
        parameterRanges=ParameterRanges(**(profile.parameter_ranges or {})),
        predictionTargets=[PredictionTarget(**t) for t in (profile.prediction_targets or [])],
        trainTestSplit=int(profile.train_test_split or 80),
        geneticConfig=GeneticConfig(**(profile.genetic_config or {})) if profile.genetic_config else None,
        metricsConfig=MetricsConfig(**(profile.metrics_config or {})) if profile.metrics_config else None
    )

    # Reuse create_job logic
    return await create_job(job_create, db)


@router.put("/profiles/{profile_id}", response_model=ProfileResponse)
async def update_profile(profile_id: int, profile: OptimizationProfileCreate, db: Session = Depends(get_db)):
    """
    Update an existing optimization profile.

    Args:
        profile_id: Profile ID
        profile: Updated profile settings

    Returns:
        Updated profile
    """
    db_profile = db.query(OptimizationProfileModel).filter(OptimizationProfileModel.id == profile_id).first()
    if not db_profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Profile {profile_id} not found"
        )

    db_profile.name = profile.name
    db_profile.description = profile.description
    db_profile.model_types = profile.selectedModels
    db_profile.parameter_ranges = profile.parameterRanges.dict() if profile.parameterRanges else {}
    db_profile.prediction_targets = [t.dict() for t in profile.predictionTargets] if profile.predictionTargets else []
    db_profile.train_test_split = profile.trainTestSplit
    db_profile.genetic_config = profile.geneticConfig.dict() if profile.geneticConfig else None
    db_profile.metrics_config = profile.metricsConfig.dict() if profile.metricsConfig else None

    db.commit()
    db.refresh(db_profile)
    logger.info(f"Updated profile {profile_id}")

    return _profile_to_response(db_profile)


@router.get("/profiles/{profile_id}/export")
async def export_profile(profile_id: int, db: Session = Depends(get_db)):
    """
    Export optimization profile to JSON format.

    Args:
        profile_id: Profile ID

    Returns:
        JSON representation of the profile
    """
    profile = db.query(OptimizationProfileModel).filter(OptimizationProfileModel.id == profile_id).first()
    if not profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Profile {profile_id} not found"
        )

    # Create exportable format (exclude internal IDs)
    export_data = {
        "name": profile.name,
        "description": profile.description,
        "selectedModels": profile.model_types or [],
        "parameterRanges": profile.parameter_ranges or {},
        "predictionTargets": profile.prediction_targets or [],
        "trainTestSplit": profile.train_test_split or 80,
        "geneticConfig": profile.genetic_config,
        "metricsConfig": profile.metrics_config,
        "exportedAt": datetime.now().isoformat(),
        "version": "1.0"
    }

    return export_data


@router.get("/{job_id}/individuals")
async def get_job_individuals(job_id: str, generation: Optional[int] = None, model_type: Optional[str] = None):
    """
    Get all individuals evaluated during optimization for visualization.

    Returns data for each individual including model type, parameters, fitness, and metrics.
    Used to visualize optimization progress across generations and model types.

    Args:
        job_id: Job ID
        generation: Filter by specific generation (optional)
        model_type: Filter by model type (optional)

    Returns:
        List of individual evaluations with parameters and metrics
    """
    # Load jobs from database if needed
    load_jobs_from_database()

    if job_id not in jobs_store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found"
        )

    # Get result from task queue
    task_queue = get_task_queue()
    task_status = task_queue.get_task_status(job_id)

    all_individuals = []
    if task_status and task_status.get("result"):
        result = task_status["result"]
        all_individuals = result.get("all_individuals", [])

        # Also check in results array
        if not all_individuals and "results" in result:
            for r in result["results"]:
                if "all_individuals" in r:
                    all_individuals.extend(r["all_individuals"])

    # Apply filters
    if generation is not None:
        all_individuals = [i for i in all_individuals if i.get("generation") == generation]
    if model_type:
        all_individuals = [i for i in all_individuals if i.get("model_type", "").lower() == model_type.lower()]

    # Calculate summary stats
    summary = {
        "total_individuals": len(all_individuals),
        "generations": sorted(set(i.get("generation", 0) for i in all_individuals)),
        "model_types": sorted(set(i.get("model_type", "unknown") for i in all_individuals)),
        "best_fitness": max((i.get("fitness", 0) for i in all_individuals), default=0),
        "avg_fitness": sum(i.get("fitness", 0) for i in all_individuals) / len(all_individuals) if all_individuals else 0
    }

    # Find best individual
    best_individual = None
    if all_individuals:
        best_individual = max(all_individuals, key=lambda x: x.get("fitness", 0))

    return {
        "job_id": job_id,
        "summary": summary,
        "best_individual": best_individual,
        "individuals": all_individuals
    }


@router.get("/{job_id}/generations")
async def get_job_generations(job_id: str):
    """
    Get generation-by-generation summary of optimization progress.

    Returns aggregated stats for each generation including best/avg fitness,
    model type distribution, and top individuals.

    Args:
        job_id: Job ID

    Returns:
        List of generation summaries
    """
    # Get all individuals
    individuals_response = await get_job_individuals(job_id)
    all_individuals = individuals_response["individuals"]

    # Group by generation
    generations_data = {}
    for ind in all_individuals:
        gen = ind.get("generation", 0)
        if gen not in generations_data:
            generations_data[gen] = {
                "generation": gen,
                "individuals": [],
                "model_types": {},
                "best_fitness": 0,
                "avg_fitness": 0
            }
        generations_data[gen]["individuals"].append(ind)

        # Count model types
        model_type = ind.get("model_type", "unknown")
        generations_data[gen]["model_types"][model_type] = \
            generations_data[gen]["model_types"].get(model_type, 0) + 1

    # Calculate stats per generation
    generations = []
    for gen, data in sorted(generations_data.items()):
        individuals = data["individuals"]
        fitnesses = [i.get("fitness", 0) for i in individuals]

        gen_summary = {
            "generation": gen,
            "individual_count": len(individuals),
            "best_fitness": max(fitnesses) if fitnesses else 0,
            "avg_fitness": sum(fitnesses) / len(fitnesses) if fitnesses else 0,
            "min_fitness": min(fitnesses) if fitnesses else 0,
            "model_types": data["model_types"],
            "best_individual": max(individuals, key=lambda x: x.get("fitness", 0)) if individuals else None
        }
        generations.append(gen_summary)

    return {
        "job_id": job_id,
        "total_generations": len(generations),
        "generations": generations
    }


@router.post("/profiles/import", response_model=ProfileResponse)
async def import_profile(profile_data: Dict[str, Any], db: Session = Depends(get_db)):
    """
    Import optimization profile from JSON format.

    Args:
        profile_data: JSON profile data

    Returns:
        Created profile with new ID
    """
    try:
        # Validate required fields
        required_fields = ["name", "selectedModels", "parameterRanges", "predictionTargets", "trainTestSplit"]
        for field in required_fields:
            if field not in profile_data:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Missing required field: {field}"
                )

        # Create profile from import data
        profile = OptimizationProfileCreate(
            name=profile_data["name"],
            description=profile_data.get("description"),
            selectedModels=profile_data["selectedModels"],
            parameterRanges=ParameterRanges(**profile_data["parameterRanges"]),
            predictionTargets=[PredictionTarget(**t) for t in profile_data["predictionTargets"]],
            trainTestSplit=profile_data["trainTestSplit"],
            geneticConfig=GeneticConfig(**profile_data["geneticConfig"]) if profile_data.get("geneticConfig") else None,
            metricsConfig=MetricsConfig(**profile_data["metricsConfig"]) if profile_data.get("metricsConfig") else None
        )

        # Create as new profile
        return await create_profile(profile, db)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to import profile: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid profile format: {str(e)}"
        )
