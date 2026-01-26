"""
Deep Learning Financial Forecasting Platform - Main API Application

This is the entry point for the FastAPI application.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pathlib import Path

# Import and initialize logging configuration
from app.logging_config import setup_logging, get_logger

# Set up logging with separate files for debug, info, and error
setup_logging(
    log_dir="logs",
    debug_log="debug.log",
    info_log="info.log",
    error_log="error.log"
)

logger = get_logger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Deep Learning Financial Forecasting Platform",
    description="Train and evaluate deep learning models for financial forecasting",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint - API information"""
    return {
        "name": "Deep Learning Financial Forecasting Platform API",
        "version": "0.1.0",
        "status": "operational",
        "docs": "/docs",
        "redoc": "/redoc"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "0.1.0"
    }


@app.on_event("startup")
async def startup_event():
    """Run on application startup"""
    logger.info("Starting Deep Learning Financial Forecasting Platform API")

    # Create necessary directories
    directories = [
        "datasets",
        "trained_models",
        "logs"
    ]

    for directory in directories:
        Path(directory).mkdir(exist_ok=True)

    # Initialize database tables
    from app.models.database import init_db
    init_db()

    # Initialize task queue
    from app.services.task_queue import init_task_queue
    init_task_queue(max_workers=2)
    logger.info("Task queue initialized with 2 workers")

    logger.info("Application startup complete")


@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown"""
    logger.info("Shutting down Deep Learning Financial Forecasting Platform API")


# Exception handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc)
        }
    )


# Import and include routers
from app.api import datasets, jobs, workers, dashboard, models, backtests, ml, settings, websocket, tasks

app.include_router(datasets.router, prefix="/api/datasets", tags=["datasets"])
app.include_router(jobs.router, prefix="/api/jobs", tags=["optimization"])
app.include_router(workers.router, prefix="/api/workers", tags=["workers"])
app.include_router(dashboard.router, prefix="/api/dashboard", tags=["dashboard"])
app.include_router(models.router, prefix="/api/models", tags=["models"])
app.include_router(backtests.router, prefix="/api/backtests", tags=["backtesting"])
app.include_router(ml.router, prefix="/api/ml", tags=["machine-learning"])
app.include_router(settings.router, prefix="/api/settings", tags=["settings"])
app.include_router(websocket.router, prefix="/api", tags=["websocket"])
app.include_router(tasks.router, prefix="/api/tasks", tags=["task-queue"])

# Additional routers (will be added as we build features)
# from app.api import models, backtests, profiles, providers, settings
# app.include_router(models.router, prefix="/api/models", tags=["models"])
# app.include_router(backtests.router, prefix="/api/backtests", tags=["backtesting"])
# app.include_router(profiles.router, prefix="/api/profiles", tags=["profiles"])
# app.include_router(providers.router, prefix="/api/providers", tags=["data-providers"])
# app.include_router(settings.router, prefix="/api/settings", tags=["settings"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
