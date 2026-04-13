"""
Admin API endpoints.

Provides a secure endpoint for CLI-driven server updates (git pull + restart).
Protected by a bearer token configured via BA2_ADMIN_TOKEN environment variable.
"""

import logging
import os
import subprocess
import sys
import threading

from fastapi import APIRouter, Header, HTTPException
from pathlib import Path

logger = logging.getLogger(__name__)

router = APIRouter()

# Project root is three levels up from this file: admin.py -> api/ -> app/ -> backend/
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def _schedule_restart():
    """Replace the current process after a short delay to allow the response to be sent."""
    import time
    time.sleep(1)
    logger.info("Restarting server via os.execv...")
    os.execv(sys.executable, [sys.executable] + sys.argv)


@router.post("/update")
async def update_server(authorization: str = Header(default=None)):
    """
    Pull latest code from git and schedule a server restart.

    Requires BA2_ADMIN_TOKEN to be set in the environment.
    The request must include an Authorization: Bearer <token> header.
    """
    admin_token = os.environ.get("BA2_ADMIN_TOKEN")

    # Token not configured on the server side
    if not admin_token:
        raise HTTPException(
            status_code=503,
            detail="Admin token is not configured on the server (BA2_ADMIN_TOKEN not set).",
        )

    # Missing Authorization header
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Missing Authorization header.",
        )

    # Parse and validate Bearer token
    parts = authorization.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=401,
            detail="Invalid Authorization header format. Expected 'Bearer <token>'.",
        )

    token = parts[1]
    if token != admin_token:
        raise HTTPException(
            status_code=403,
            detail="Invalid admin token.",
        )

    # Run git pull in the project root
    logger.info(f"Running git pull in {PROJECT_ROOT}")
    try:
        result = subprocess.run(
            ["git", "pull"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=30,
        )
        git_output = result.stdout.strip() or result.stderr.strip()
    except subprocess.TimeoutExpired:
        raise HTTPException(
            status_code=504,
            detail="git pull timed out after 30 seconds.",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"git pull failed: {str(e)}",
        )

    logger.info(f"git pull output: {git_output}")

    # Schedule restart in a background thread so the response can be sent first
    threading.Thread(target=_schedule_restart, daemon=True).start()

    return {
        "git_pull": git_output,
        "restart": "scheduled",
        "message": "Server will restart in ~1 second.",
    }
