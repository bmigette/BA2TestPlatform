"""
WebSocket API endpoints for real-time updates
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Dict, List, Set
import logging
import asyncio
import json
from datetime import datetime

logger = logging.getLogger(__name__)

router = APIRouter()

# Connection management
class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""

    def __init__(self):
        # Map of job_id -> set of connected websockets
        self.job_connections: Dict[str, Set[WebSocket]] = {}
        # All active connections
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket, job_id: str = None):
        """Accept a new WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)

        if job_id:
            if job_id not in self.job_connections:
                self.job_connections[job_id] = set()
            self.job_connections[job_id].add(websocket)

        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket, job_id: str = None):
        """Remove a WebSocket connection."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

        if job_id and job_id in self.job_connections:
            self.job_connections[job_id].discard(websocket)
            if not self.job_connections[job_id]:
                del self.job_connections[job_id]

        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")

    async def send_to_job(self, job_id: str, message: dict):
        """Send message to all connections watching a specific job."""
        if job_id in self.job_connections:
            dead_connections = []
            for connection in self.job_connections[job_id]:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    logger.warning(f"Failed to send to connection: {e}")
                    dead_connections.append(connection)

            # Clean up dead connections
            for conn in dead_connections:
                self.disconnect(conn, job_id)

    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients."""
        dead_connections = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.warning(f"Failed to broadcast to connection: {e}")
                dead_connections.append(connection)

        # Clean up dead connections
        for conn in dead_connections:
            if conn in self.active_connections:
                self.active_connections.remove(conn)


# Global connection manager
manager = ConnectionManager()


@router.websocket("/ws/jobs/{job_id}")
async def websocket_job_progress(websocket: WebSocket, job_id: str):
    """
    WebSocket endpoint for real-time job progress updates.

    Connect to receive live updates for a specific job.
    Messages include:
    - type: 'connected' - Initial connection confirmation
    - type: 'progress' - Job progress update
    - type: 'log' - New log entry
    - type: 'complete' - Job finished
    - type: 'error' - Error occurred

    Args:
        websocket: WebSocket connection
        job_id: Job ID to monitor
    """
    await manager.connect(websocket, job_id)

    try:
        # Send connection confirmation
        await websocket.send_json({
            "type": "connected",
            "job_id": job_id,
            "message": f"Connected to job {job_id} updates",
            "timestamp": datetime.now().isoformat()
        })

        # Import here to avoid circular imports
        from app.api.jobs import jobs_store, job_progress_data

        last_progress = -1
        last_log_count = 0

        # Keep connection alive and send updates
        while True:
            try:
                # Check for job updates
                if job_id in jobs_store:
                    job = jobs_store[job_id]
                    current_progress = job.get("progress", 0)

                    # Send progress update if changed
                    if current_progress != last_progress:
                        await websocket.send_json({
                            "type": "progress",
                            "job_id": job_id,
                            "status": job.get("status"),
                            "progress": current_progress,
                            "currentGeneration": job.get("currentGeneration", 0),
                            "totalGenerations": job.get("totalGenerations", 50),
                            "currentLoss": job.get("currentLoss"),
                            "currentAccuracy": job.get("currentAccuracy"),
                            "bestFitness": job.get("bestFitness"),
                            "gpuUtilization": job.get("gpuUtilization"),
                            "estimatedTimeRemaining": job.get("estimatedTimeRemaining"),
                            "datasetProgress": job.get("datasetProgress"),
                            "currentDatasetId": job.get("currentDatasetId"),
                            "timestamp": datetime.now().isoformat()
                        })
                        last_progress = current_progress

                    # Send new log entries
                    if job_id in job_progress_data:
                        logs = job_progress_data[job_id].get("logs", [])
                        if len(logs) > last_log_count:
                            new_logs = logs[last_log_count:]
                            for log_entry in new_logs:
                                await websocket.send_json({
                                    "type": "log",
                                    "job_id": job_id,
                                    "message": log_entry,
                                    "timestamp": datetime.now().isoformat()
                                })
                            last_log_count = len(logs)

                    # Check if job is complete
                    if job.get("status") in ["completed", "cancelled", "failed"]:
                        await websocket.send_json({
                            "type": "complete",
                            "job_id": job_id,
                            "status": job.get("status"),
                            "message": f"Job {job.get('status')}",
                            "timestamp": datetime.now().isoformat()
                        })
                        break
                else:
                    await websocket.send_json({
                        "type": "error",
                        "job_id": job_id,
                        "message": "Job not found",
                        "timestamp": datetime.now().isoformat()
                    })
                    break

                # Wait before next update
                await asyncio.sleep(0.5)

                # Handle incoming messages (for keep-alive or commands)
                try:
                    data = await asyncio.wait_for(
                        websocket.receive_text(),
                        timeout=0.1
                    )
                    # Process ping/pong for keep-alive
                    if data == "ping":
                        await websocket.send_text("pong")
                except asyncio.TimeoutError:
                    pass

            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"Error in WebSocket loop: {e}")
                await websocket.send_json({
                    "type": "error",
                    "message": str(e),
                    "timestamp": datetime.now().isoformat()
                })
                break

    except WebSocketDisconnect:
        logger.info(f"Client disconnected from job {job_id}")
    except Exception as e:
        logger.error(f"WebSocket error for job {job_id}: {e}")
    finally:
        manager.disconnect(websocket, job_id)


@router.websocket("/ws/all-jobs")
async def websocket_all_jobs(websocket: WebSocket):
    """
    WebSocket endpoint for monitoring all jobs.

    Connect to receive updates for all running jobs.
    Useful for dashboard views.

    Args:
        websocket: WebSocket connection
    """
    await manager.connect(websocket)

    try:
        # Send connection confirmation
        await websocket.send_json({
            "type": "connected",
            "message": "Connected to all jobs updates",
            "timestamp": datetime.now().isoformat()
        })

        from app.api.jobs import jobs_store

        last_status = {}

        while True:
            try:
                # Send updates for all jobs
                current_jobs = {}
                for job_id, job in jobs_store.items():
                    current_jobs[job_id] = {
                        "id": job_id,
                        "status": job.get("status"),
                        "progress": job.get("progress", 0),
                        "currentGeneration": job.get("currentGeneration", 0)
                    }

                # Only send if something changed
                if current_jobs != last_status:
                    await websocket.send_json({
                        "type": "jobs_update",
                        "jobs": list(current_jobs.values()),
                        "count": len(current_jobs),
                        "timestamp": datetime.now().isoformat()
                    })
                    last_status = current_jobs.copy()

                await asyncio.sleep(1.0)

                # Handle keep-alive
                try:
                    data = await asyncio.wait_for(
                        websocket.receive_text(),
                        timeout=0.1
                    )
                    if data == "ping":
                        await websocket.send_text("pong")
                except asyncio.TimeoutError:
                    pass

            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"Error in all-jobs WebSocket: {e}")
                break

    except WebSocketDisconnect:
        logger.info("Client disconnected from all-jobs feed")
    except Exception as e:
        logger.error(f"WebSocket error for all-jobs: {e}")
    finally:
        manager.disconnect(websocket)


# Helper function for external use
async def notify_job_update(job_id: str, update: dict):
    """
    Send an update to all clients watching a specific job.
    Can be called from other modules.

    Args:
        job_id: Job ID
        update: Update data to send
    """
    await manager.send_to_job(job_id, {
        "type": "update",
        "job_id": job_id,
        **update,
        "timestamp": datetime.now().isoformat()
    })
