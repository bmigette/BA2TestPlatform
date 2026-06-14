"""Cache-management API. Brand-new router.

No /api/cache existed before this phase (providers.router is commented out in
app/main.py). This router exposes per-type disk usage + drill-down over every
cache root tracked by app.services.cache_manager. Deletion endpoints (clean-all /
by-type / by-date) are added in a later task.
"""
import logging

from fastapi import APIRouter, HTTPException

from app.services import cache_manager

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/usage")
async def cache_usage():
    """Per-type disk usage (bytes, file count, oldest/newest mtime, destructive flag, TTL)."""
    return {"types": cache_manager.get_usage()}


@router.get("/usage/{cache_type}")
async def cache_drill_down(cache_type: str):
    """Per-item breakdown for one cache type."""
    try:
        return {"type": cache_type, "items": cache_manager.drill_down(cache_type)}
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown cache type: {cache_type}")
