"""
Dashboard API endpoints.

Provides aggregated data for the dashboard view including job stats,
recent activity, and system resources.
"""

import logging
import psutil
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.models import get_db, Worker
from app.models.dataset import Dataset
from app.api.jobs import jobs_store

logger = logging.getLogger(__name__)

router = APIRouter()


class JobStats(BaseModel):
    running: int
    completed: int
    failed: int
    paused: int
    queued: int
    cancelled: int
    total: int


class ActivityItem(BaseModel):
    id: str
    type: str  # 'job', 'dataset', 'backtest', 'model'
    action: str  # 'created', 'completed', 'failed', 'started'
    title: str
    timestamp: str
    status: Optional[str] = None


class SystemResources(BaseModel):
    cpuPercent: float
    memoryUsedMB: int
    memoryTotalMB: int
    memoryPercent: float
    gpuUtilization: Optional[float] = None
    gpuMemoryUsedMB: Optional[int] = None
    gpuMemoryTotalMB: Optional[int] = None


class DashboardResponse(BaseModel):
    jobStats: JobStats
    recentActivity: List[ActivityItem]
    systemResources: SystemResources


@router.get("/stats", response_model=DashboardResponse)
async def get_dashboard_stats(db: Session = Depends(get_db)):
    """
    Get aggregated dashboard statistics.

    Returns:
        Job stats, recent activity timeline, and system resources
    """
    # Calculate job stats
    job_stats = {
        "running": 0,
        "completed": 0,
        "failed": 0,
        "paused": 0,
        "queued": 0,
        "cancelled": 0,
        "total": 0
    }

    for job in jobs_store.values():
        status = job.get("status", "unknown")
        if status in job_stats:
            job_stats[status] += 1
        job_stats["total"] += 1

    # Get recent activity from jobs and datasets
    activities: List[ActivityItem] = []

    # Add job activities
    for job_id, job in jobs_store.items():
        status = job.get("status", "unknown")

        # Job created activity
        created_at = job.get("createdAt")
        if created_at:
            activities.append(ActivityItem(
                id=f"job-create-{job_id}",
                type="job",
                action="created",
                title=f"Optimization job {job_id} created",
                timestamp=created_at,
                status=status
            ))

        # Job completed/failed activity
        if status == "completed":
            completed_at = job.get("completedAt")
            if completed_at:
                activities.append(ActivityItem(
                    id=f"job-complete-{job_id}",
                    type="job",
                    action="completed",
                    title=f"Optimization job {job_id} completed",
                    timestamp=completed_at,
                    status=status
                ))
        elif status == "failed":
            activities.append(ActivityItem(
                id=f"job-fail-{job_id}",
                type="job",
                action="failed",
                title=f"Optimization job {job_id} failed",
                timestamp=job.get("completedAt", created_at),
                status=status
            ))
        elif status == "running":
            started_at = job.get("startedAt")
            if started_at:
                activities.append(ActivityItem(
                    id=f"job-start-{job_id}",
                    type="job",
                    action="started",
                    title=f"Optimization job {job_id} started",
                    timestamp=started_at,
                    status=status
                ))

    # Add dataset activities (handle case when table doesn't exist)
    try:
        datasets = db.query(Dataset).order_by(Dataset.created_at.desc()).limit(20).all()
        for ds in datasets:
            activities.append(ActivityItem(
                id=f"dataset-{ds.id}",
                type="dataset",
                action="created",
                title=f"Dataset '{ds.name}' created for {ds.ticker}",
                timestamp=ds.created_at.isoformat() if ds.created_at else datetime.now().isoformat(),
                status="active"
            ))
    except Exception as e:
        logger.warning(f"Could not fetch datasets for activity: {e}")

    # Sort by timestamp descending and limit to 20
    activities.sort(key=lambda x: x.timestamp, reverse=True)
    activities = activities[:20]

    # Get system resources
    cpu_percent = psutil.cpu_percent()
    memory = psutil.virtual_memory()

    gpu_utilization = None
    gpu_memory_used = None
    gpu_memory_total = None

    try:
        import torch
        if torch.cuda.is_available():
            gpu_utilization = 0  # Placeholder - real GPU util requires pynvml
            props = torch.cuda.get_device_properties(0)
            gpu_memory_total = props.total_memory // (1024 * 1024)
            # Get current memory usage
            gpu_memory_used = torch.cuda.memory_allocated(0) // (1024 * 1024)
    except ImportError:
        pass

    system_resources = SystemResources(
        cpuPercent=round(cpu_percent, 1),
        memoryUsedMB=memory.used // (1024 * 1024),
        memoryTotalMB=memory.total // (1024 * 1024),
        memoryPercent=round(memory.percent, 1),
        gpuUtilization=gpu_utilization,
        gpuMemoryUsedMB=gpu_memory_used,
        gpuMemoryTotalMB=gpu_memory_total
    )

    return DashboardResponse(
        jobStats=JobStats(**job_stats),
        recentActivity=activities,
        systemResources=system_resources
    )
