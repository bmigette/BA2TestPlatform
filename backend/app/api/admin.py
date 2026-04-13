"""
Admin API endpoints.

Provides secure endpoints for CLI-driven server updates (git pull + restart)
and log file reading.
Protected by a bearer token configured via BA2_ADMIN_TOKEN environment variable.
"""

import logging
import os
import subprocess
import sys
import threading
from typing import Optional

from fastapi import APIRouter, Header, HTTPException, Query
from pathlib import Path

logger = logging.getLogger(__name__)

router = APIRouter()

# Project root is three levels up from this file: admin.py -> api/ -> app/ -> backend/
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# Backend root (where logs/ directory lives)
BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent


def verify_admin_token(authorization: str):
    """Validate the Authorization header against BA2_ADMIN_TOKEN.

    Raises HTTPException on failure.
    """
    admin_token = os.environ.get("BA2_ADMIN_TOKEN")

    if not admin_token:
        raise HTTPException(
            status_code=503,
            detail="Admin token is not configured on the server (BA2_ADMIN_TOKEN not set).",
        )

    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Missing Authorization header.",
        )

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
    verify_admin_token(authorization)

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


@router.get("/logs/{level}")
async def read_logs(
    level: str,
    lines: int = Query(default=100),
    search: Optional[str] = Query(default=None),
    authorization: str = Header(default=None),
):
    """
    Read the last N lines from a log file.

    *level* must be one of ``info``, ``error``, or ``debug``.
    Optionally filter lines with a case-insensitive *search* string.
    """
    verify_admin_token(authorization)

    valid_levels = ("info", "error", "debug")
    if level not in valid_levels:
        raise HTTPException(
            status_code=400,
            detail="Invalid log level '{}'. Must be one of: {}".format(
                level, ", ".join(valid_levels)
            ),
        )

    log_file = BACKEND_ROOT / "logs" / "{}.log".format(level)

    if not log_file.exists():
        raise HTTPException(
            status_code=404,
            detail="Log file not found: logs/{}.log".format(level),
        )

    try:
        with open(log_file, "r", encoding="utf-8", errors="replace") as fh:
            all_lines = fh.readlines()
    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail="Failed to read log file: {}".format(str(exc)),
        )

    # Strip trailing newlines
    all_lines = [line.rstrip("\n") for line in all_lines]

    if search:
        search_lower = search.lower()
        all_lines = [line for line in all_lines if search_lower in line.lower()]

    total = len(all_lines)
    result_lines = all_lines[-lines:] if lines < total else all_lines

    return {
        "level": level,
        "lines": result_lines,
        "total_lines": total,
        "file": "logs/{}.log".format(level),
    }
