"""
Optimization Jobs API endpoints
"""

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging
import uuid
import asyncio
import random
import threading
import time
import json

logger = logging.getLogger(__name__)

router = APIRouter()

# In-memory job store (would be replaced with database in production)
jobs_store: dict = {}

# Training progress data (metrics over time)
job_progress_data: Dict[str, Dict[str, Any]] = {}

# Background training simulation threads
training_threads: Dict[str, bool] = {}  # job_id -> should_stop flag


class PredictionTarget(BaseModel):
    profitPercent: float
    maxDrawdownPercent: float
    timePeriodDays: int


class ParameterRanges(BaseModel):
    layersMin: int
    layersMax: int
    layerSizeMin: int
    layerSizeMax: int
    learningRateMin: float
    learningRateMax: float
    activationFunctions: List[str]


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
    # Multi-dataset progress
    datasetProgress: Optional[List[DatasetProgress]] = None
    currentDatasetId: Optional[int] = None
    # Cross-validation results
    foldResults: Optional[List[Dict[str, Any]]] = None


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


def simulate_multi_dataset_training(job_id: str, dataset_progress: List[Dict[str, Any]]):
    """
    Simulates training with multiple datasets, processing them chronologically.
    Updates per-dataset progress during training.
    """
    if job_id not in jobs_store:
        return

    job = jobs_store[job_id]
    cross_validation = job.get("crossValidation", {})
    use_cv = cross_validation.get("enabled", False) if cross_validation else False

    # Process each dataset
    for idx, ds_progress in enumerate(dataset_progress):
        if training_threads.get(job_id, False):
            return

        ds_id = ds_progress["datasetId"]
        ds_name = ds_progress["datasetName"]
        ticker = ds_progress["ticker"]

        # Update current dataset
        jobs_store[job_id]["currentDatasetId"] = ds_id
        jobs_store[job_id]["datasetProgress"][idx]["status"] = "processing"

        job_progress_data[job_id]["logs"].append(
            f"[{datetime.now().isoformat()}] Processing dataset: {ds_name} ({ticker})"
        )

        # Simulate processing this dataset's data
        total_rows = ds_progress["totalRows"]
        for row_batch in range(0, total_rows, max(1, total_rows // 10)):
            if training_threads.get(job_id, False):
                return

            processed = min(row_batch + total_rows // 10, total_rows)
            jobs_store[job_id]["datasetProgress"][idx]["rowsProcessed"] = processed
            jobs_store[job_id]["datasetProgress"][idx]["progress"] = (processed / total_rows) * 100

            time.sleep(0.1)

        # Mark dataset as completed
        jobs_store[job_id]["datasetProgress"][idx]["status"] = "completed"
        jobs_store[job_id]["datasetProgress"][idx]["rowsProcessed"] = total_rows
        jobs_store[job_id]["datasetProgress"][idx]["progress"] = 100.0

        # If cross-validation, add fold results
        if use_cv:
            fold_result = {
                "fold": idx + 1,
                "datasetId": ds_id,
                "datasetName": ds_name,
                "ticker": ticker,
                "metrics": {
                    "mape": round(random.uniform(5, 15), 2),
                    "mae": round(random.uniform(0.01, 0.1), 4),
                    "rmse": round(random.uniform(0.02, 0.15), 4),
                    "accuracy": round(random.uniform(0.7, 0.9), 4)
                }
            }
            if jobs_store[job_id].get("foldResults") is None:
                jobs_store[job_id]["foldResults"] = []
            jobs_store[job_id]["foldResults"].append(fold_result)

            job_progress_data[job_id]["logs"].append(
                f"[{datetime.now().isoformat()}] CV Fold {idx + 1} completed - "
                f"Accuracy: {fold_result['metrics']['accuracy']:.4f}"
            )

    job_progress_data[job_id]["logs"].append(
        f"[{datetime.now().isoformat()}] All datasets combined chronologically"
    )


def simulate_training(job_id: str):
    """
    Simulates training progress for a job.
    Updates job metrics over time to simulate genetic optimization.
    Handles both single and multi-dataset training.
    """
    if job_id not in jobs_store:
        return

    # Initialize progress data
    job_progress_data[job_id] = {
        "metrics": [],
        "logs": [f"[{datetime.now().isoformat()}] Starting optimization job {job_id}"],
    }

    # Update job to running
    jobs_store[job_id]["status"] = "running"
    jobs_store[job_id]["startedAt"] = datetime.now().isoformat()

    # Handle multi-dataset training first
    dataset_progress = jobs_store[job_id].get("datasetProgress")
    if dataset_progress and len(dataset_progress) > 1:
        job_progress_data[job_id]["logs"].append(
            f"[{datetime.now().isoformat()}] Multi-dataset training: {len(dataset_progress)} datasets"
        )
        simulate_multi_dataset_training(job_id, dataset_progress)

    total_generations = jobs_store[job_id].get("totalGenerations", 50)
    start_time = time.time()

    # Simulate training over generations
    base_loss = 2.5
    base_accuracy = 0.25
    best_fitness = 0.0

    for gen in range(1, total_generations + 1):
        # Check if we should stop (paused, cancelled)
        if job_id in training_threads and training_threads[job_id]:
            job_progress_data[job_id]["logs"].append(
                f"[{datetime.now().isoformat()}] Training stopped at generation {gen}"
            )
            return

        # Check if job is paused
        if job_id in jobs_store and jobs_store[job_id]["status"] == "paused":
            while jobs_store[job_id]["status"] == "paused":
                time.sleep(0.5)
                if job_id in training_threads and training_threads[job_id]:
                    return

        # Simulate improvement over generations
        noise = random.uniform(-0.1, 0.1)
        improvement = gen / total_generations
        loss = max(0.1, base_loss * (1 - improvement * 0.8) + noise * 0.3)
        accuracy = min(0.95, base_accuracy + improvement * 0.65 + noise * 0.1)
        val_loss = loss + random.uniform(0.05, 0.2)
        val_accuracy = accuracy - random.uniform(0.02, 0.08)
        fitness = accuracy * 100 - loss * 10

        if fitness > best_fitness:
            best_fitness = fitness

        # Calculate ETA
        elapsed = time.time() - start_time
        avg_time_per_gen = elapsed / gen if gen > 0 else 1
        remaining_gens = total_generations - gen
        eta_seconds = int(avg_time_per_gen * remaining_gens)
        eta_str = f"{eta_seconds // 60}m {eta_seconds % 60}s" if eta_seconds > 60 else f"{eta_seconds}s"

        # Update job store
        jobs_store[job_id]["progress"] = (gen / total_generations) * 100
        jobs_store[job_id]["currentGeneration"] = gen
        jobs_store[job_id]["currentLoss"] = round(loss, 4)
        jobs_store[job_id]["currentAccuracy"] = round(accuracy, 4)
        jobs_store[job_id]["bestFitness"] = round(best_fitness, 2)
        jobs_store[job_id]["gpuUtilization"] = random.uniform(75, 95)
        jobs_store[job_id]["estimatedTimeRemaining"] = eta_str

        # Add metrics
        metric = {
            "generation": gen,
            "loss": round(loss, 4),
            "accuracy": round(accuracy, 4),
            "valLoss": round(val_loss, 4),
            "valAccuracy": round(val_accuracy, 4),
            "fitness": round(fitness, 2),
            "timestamp": datetime.now().isoformat(),
        }
        job_progress_data[job_id]["metrics"].append(metric)

        # Add log entry every 5 generations
        if gen % 5 == 0 or gen == 1:
            job_progress_data[job_id]["logs"].append(
                f"[{datetime.now().isoformat()}] Gen {gen}/{total_generations}: "
                f"loss={loss:.4f}, accuracy={accuracy:.4f}, fitness={fitness:.2f}"
            )

        # Simulate training time (0.5-1s per generation for demo)
        time.sleep(random.uniform(0.5, 1.0))

    # Training completed
    jobs_store[job_id]["status"] = "completed"
    jobs_store[job_id]["completedAt"] = datetime.now().isoformat()
    jobs_store[job_id]["progress"] = 100
    job_progress_data[job_id]["logs"].append(
        f"[{datetime.now().isoformat()}] Training completed! Best fitness: {best_fitness:.2f}"
    )


@router.post("", response_model=JobResponse, status_code=status.HTTP_201_CREATED)
async def create_job(job_create: JobCreate):
    """
    Create a new optimization job.

    Supports both single dataset (datasetId) and multiple datasets (datasetIds).
    For multi-dataset training:
    - Datasets are combined chronologically
    - Ticker column is added to distinguish data from different tickers
    - Cross-validation can use each dataset as a fold

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

        job_id = str(uuid.uuid4())[:8]

        # Build dataset progress tracking (simulated - real implementation would load from DB)
        dataset_progress = []
        dataset_names = []
        for idx, ds_id in enumerate(dataset_ids):
            # Simulate dataset info (in production, would query database)
            ds_info = {
                "datasetId": ds_id,
                "datasetName": f"Dataset_{ds_id}",
                "ticker": f"TICKER{ds_id}",
                "status": "pending",
                "progress": 0.0,
                "rowsProcessed": 0,
                "totalRows": 1000 + idx * 500  # Simulated
            }
            dataset_progress.append(ds_info)
            dataset_names.append(ds_info["datasetName"])

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
            status="queued",
            progress=0.0,
            createdAt=datetime.now().isoformat(),
            totalGenerations=50,
            datasetProgress=dataset_progress if len(dataset_ids) > 1 else None,
        )

        # Store in memory
        jobs_store[job_id] = job.dict()

        logger.info(f"Created job {job_id} with {len(dataset_ids)} dataset(s)")

        # Start training simulation in background thread
        training_threads[job_id] = False  # should_stop = False
        thread = threading.Thread(target=simulate_training, args=(job_id,), daemon=True)
        thread.start()

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

    Returns:
        List of jobs with status
    """
    try:
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

    Args:
        job_id: Job ID

    Returns:
        Job details
    """
    if job_id not in jobs_store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found"
        )

    return JobResponse(**jobs_store[job_id])


@router.delete("/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_job(job_id: str):
    """
    Delete a job by ID.

    Args:
        job_id: Job ID
    """
    if job_id not in jobs_store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found"
        )

    # Stop training thread if running
    training_threads[job_id] = True

    del jobs_store[job_id]
    if job_id in job_progress_data:
        del job_progress_data[job_id]
    logger.info(f"Deleted job {job_id}")


@router.get("/{job_id}/progress", response_model=JobProgressResponse)
async def get_job_progress(job_id: str):
    """
    Get detailed progress information for a job including metrics and logs.

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

    job = jobs_store[job_id]
    if job["status"] != "running":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot pause job in status: {job['status']}"
        )

    jobs_store[job_id]["status"] = "paused"
    if job_id in job_progress_data:
        job_progress_data[job_id]["logs"].append(
            f"[{datetime.now().isoformat()}] Job paused"
        )

    logger.info(f"Paused job {job_id}")
    return {"status": "paused", "message": f"Job {job_id} paused"}


@router.post("/{job_id}/resume")
async def resume_job(job_id: str):
    """
    Resume a paused job.

    Args:
        job_id: Job ID
    """
    if job_id not in jobs_store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found"
        )

    job = jobs_store[job_id]
    if job["status"] != "paused":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot resume job in status: {job['status']}"
        )

    jobs_store[job_id]["status"] = "running"
    if job_id in job_progress_data:
        job_progress_data[job_id]["logs"].append(
            f"[{datetime.now().isoformat()}] Job resumed"
        )

    logger.info(f"Resumed job {job_id}")
    return {"status": "running", "message": f"Job {job_id} resumed"}


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

    job = jobs_store[job_id]
    if job["status"] not in ["running", "paused", "queued"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot cancel job in status: {job['status']}"
        )

    # Signal the training thread to stop
    training_threads[job_id] = True
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

    Yields:
        SSE formatted events with job progress data
    """
    last_generation = -1

    while True:
        if job_id not in jobs_store:
            yield f"data: {json.dumps({'type': 'error', 'message': 'Job not found'})}\n\n"
            break

        job = jobs_store[job_id]
        current_gen = job.get("currentGeneration", 0)

        # Send update if generation changed or status changed
        if current_gen != last_generation or job["status"] in ["completed", "cancelled", "failed"]:
            event_data = {
                "type": "progress",
                "job_id": job_id,
                "status": job["status"],
                "progress": job.get("progress", 0),
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
            last_generation = current_gen

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
# Optimization Profiles
# ============================================================================

# In-memory profile store (would be database in production)
profiles_store: Dict[str, Dict[str, Any]] = {}


class OptimizationProfile(BaseModel):
    name: str
    description: Optional[str] = None
    selectedModels: List[str]
    parameterRanges: ParameterRanges
    predictionTargets: List[PredictionTarget]
    trainTestSplit: int


class ProfileResponse(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    selectedModels: List[str]
    parameterRanges: ParameterRanges
    predictionTargets: List[PredictionTarget]
    trainTestSplit: int
    createdAt: str
    updatedAt: str


@router.post("/profiles", response_model=ProfileResponse, status_code=status.HTTP_201_CREATED)
async def create_profile(profile: OptimizationProfile):
    """
    Create a new optimization profile.

    Profiles save optimization settings that can be reused for multiple jobs.

    Args:
        profile: Profile settings

    Returns:
        Created profile with ID
    """
    try:
        profile_id = str(uuid.uuid4())[:8]
        now = datetime.now().isoformat()

        profile_data = ProfileResponse(
            id=profile_id,
            name=profile.name,
            description=profile.description,
            selectedModels=profile.selectedModels,
            parameterRanges=profile.parameterRanges,
            predictionTargets=profile.predictionTargets,
            trainTestSplit=profile.trainTestSplit,
            createdAt=now,
            updatedAt=now
        )

        profiles_store[profile_id] = profile_data.dict()

        logger.info(f"Created optimization profile: {profile.name} ({profile_id})")
        return profile_data

    except Exception as e:
        logger.error(f"Failed to create profile: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create profile: {str(e)}"
        )


@router.get("/profiles")
async def list_profiles():
    """
    List all optimization profiles.

    Returns:
        List of profiles
    """
    profiles = [ProfileResponse(**p) for p in profiles_store.values()]
    profiles.sort(key=lambda x: x.createdAt, reverse=True)

    return {
        "profiles": profiles,
        "total": len(profiles)
    }


@router.get("/profiles/{profile_id}", response_model=ProfileResponse)
async def get_profile(profile_id: str):
    """
    Get a specific optimization profile.

    Args:
        profile_id: Profile ID

    Returns:
        Profile details
    """
    if profile_id not in profiles_store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Profile {profile_id} not found"
        )

    return ProfileResponse(**profiles_store[profile_id])


@router.delete("/profiles/{profile_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_profile(profile_id: str):
    """
    Delete an optimization profile.

    Args:
        profile_id: Profile ID
    """
    if profile_id not in profiles_store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Profile {profile_id} not found"
        )

    del profiles_store[profile_id]
    logger.info(f"Deleted profile {profile_id}")


@router.post("/profiles/{profile_id}/apply")
async def apply_profile_to_job(profile_id: str, dataset_id: int):
    """
    Create a new job using settings from a profile.

    Args:
        profile_id: Profile ID to apply
        dataset_id: Dataset ID for the new job

    Returns:
        Created job
    """
    if profile_id not in profiles_store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Profile {profile_id} not found"
        )

    profile = profiles_store[profile_id]

    # Create job from profile
    job_create = JobCreate(
        datasetId=dataset_id,
        selectedModels=profile["selectedModels"],
        parameterRanges=ParameterRanges(**profile["parameterRanges"]),
        predictionTargets=[PredictionTarget(**t) for t in profile["predictionTargets"]],
        trainTestSplit=profile["trainTestSplit"]
    )

    # Reuse create_job logic
    return await create_job(job_create)


@router.put("/profiles/{profile_id}", response_model=ProfileResponse)
async def update_profile(profile_id: str, profile: OptimizationProfile):
    """
    Update an existing optimization profile.

    Args:
        profile_id: Profile ID
        profile: Updated profile settings

    Returns:
        Updated profile
    """
    if profile_id not in profiles_store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Profile {profile_id} not found"
        )

    existing = profiles_store[profile_id]

    updated = ProfileResponse(
        id=profile_id,
        name=profile.name,
        description=profile.description,
        selectedModels=profile.selectedModels,
        parameterRanges=profile.parameterRanges,
        predictionTargets=profile.predictionTargets,
        trainTestSplit=profile.trainTestSplit,
        createdAt=existing["createdAt"],
        updatedAt=datetime.now().isoformat()
    )

    profiles_store[profile_id] = updated.dict()
    logger.info(f"Updated profile {profile_id}")

    return updated


@router.get("/profiles/{profile_id}/export")
async def export_profile(profile_id: str):
    """
    Export optimization profile to JSON format.

    Args:
        profile_id: Profile ID

    Returns:
        JSON representation of the profile
    """
    if profile_id not in profiles_store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Profile {profile_id} not found"
        )

    profile = profiles_store[profile_id]

    # Create exportable format (exclude internal IDs)
    export_data = {
        "name": profile["name"],
        "description": profile.get("description"),
        "selectedModels": profile["selectedModels"],
        "parameterRanges": profile["parameterRanges"],
        "predictionTargets": profile["predictionTargets"],
        "trainTestSplit": profile["trainTestSplit"],
        "exportedAt": datetime.now().isoformat(),
        "version": "1.0"
    }

    return export_data


@router.post("/profiles/import", response_model=ProfileResponse)
async def import_profile(profile_data: Dict[str, Any]):
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
        profile = OptimizationProfile(
            name=profile_data["name"],
            description=profile_data.get("description"),
            selectedModels=profile_data["selectedModels"],
            parameterRanges=ParameterRanges(**profile_data["parameterRanges"]),
            predictionTargets=[PredictionTarget(**t) for t in profile_data["predictionTargets"]],
            trainTestSplit=profile_data["trainTestSplit"]
        )

        # Create as new profile
        return await create_profile(profile)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to import profile: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid profile format: {str(e)}"
        )
