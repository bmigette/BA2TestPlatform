"""
Models API endpoints.

Manages trained ML models from optimization jobs.
"""

import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import uuid
import random

logger = logging.getLogger(__name__)

router = APIRouter()

# In-memory model store (would be replaced with database in production)
models_store: Dict[str, dict] = {}


class HyperParameters(BaseModel):
    layers: int
    layerSize: int
    learningRate: float
    activationFunction: str
    dropout: float
    batchSize: int
    epochs: int


class TrainingHistory(BaseModel):
    epoch: int
    loss: float
    accuracy: float
    valLoss: float
    valAccuracy: float


class PerformanceMetrics(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1Score: float
    auc: float
    sharpeRatio: Optional[float] = None
    maxDrawdown: Optional[float] = None


class ModelResponse(BaseModel):
    id: str
    name: str
    modelType: str  # LSTM, GRU, N-BEATS, Transformer, TCN, RNN
    datasetId: int
    datasetName: Optional[str] = None  # Dataset name for display
    symbol: Optional[str] = None  # Trading symbol (e.g., AAPL)
    timeframe: Optional[str] = None  # Timeframe (e.g., 1d, 1h)
    trainPeriod: Optional[str] = None  # Training period (e.g., 2020-01-01 to 2023-12-31)
    jobId: str
    status: str  # trained, failed, exported
    hyperparameters: HyperParameters
    trainingHistory: List[TrainingHistory]
    performanceMetrics: PerformanceMetrics
    confusionMatrix: Optional[List[List[int]]] = None  # Confusion matrix data
    allMetrics: Optional[Dict[str, Any]] = None  # All metrics from training
    trainingDateRange: Optional[Dict[str, str]] = None  # Training date range used
    predictionTargets: Optional[List[Dict[str, Any]]] = None  # Target configs used during training
    createdAt: str
    trainedAt: Optional[str] = None
    filePath: Optional[str] = None
    fileSize: Optional[int] = None  # bytes
    generations: int
    bestGeneration: int
    fitness: float


class ModelListResponse(BaseModel):
    models: List[ModelResponse]
    total: int


# Initialize with some sample models for demonstration
def init_sample_models():
    """Create sample models for demo purposes."""
    if models_store:
        return

    sample_models = [
        {
            "id": "mdl-001",
            "name": "LSTM_AAPL_Predictor",
            "modelType": "LSTM",
            "datasetId": 1,
            "jobId": "job-001",
            "status": "trained",
            "hyperparameters": {
                "layers": 3,
                "layerSize": 128,
                "learningRate": 0.001,
                "activationFunction": "relu",
                "dropout": 0.2,
                "batchSize": 32,
                "epochs": 100
            },
            "trainingHistory": [
                {"epoch": i, "loss": 2.5 * (0.95 ** i), "accuracy": 0.5 + 0.4 * (1 - 0.95 ** i),
                 "valLoss": 2.6 * (0.95 ** i), "valAccuracy": 0.48 + 0.38 * (1 - 0.95 ** i)}
                for i in range(1, 21)
            ],
            "performanceMetrics": {
                "accuracy": 0.87,
                "precision": 0.85,
                "recall": 0.89,
                "f1Score": 0.87,
                "auc": 0.92,
                "sharpeRatio": 1.45,
                "maxDrawdown": 0.12
            },
            "createdAt": "2026-01-24T10:30:00",
            "trainedAt": "2026-01-24T11:45:00",
            "filePath": "trained_models/mdl-001.pt",
            "fileSize": 15234567,
            "generations": 50,
            "bestGeneration": 42,
            "fitness": 87.5
        },
        {
            "id": "mdl-002",
            "name": "NBEATS_MSFT_Forecast",
            "modelType": "N-BEATS",
            "datasetId": 2,
            "jobId": "job-002",
            "status": "trained",
            "hyperparameters": {
                "layers": 4,
                "layerSize": 256,
                "learningRate": 0.0005,
                "activationFunction": "relu",
                "dropout": 0.15,
                "batchSize": 64,
                "epochs": 150
            },
            "trainingHistory": [
                {"epoch": i, "loss": 2.2 * (0.94 ** i), "accuracy": 0.52 + 0.38 * (1 - 0.94 ** i),
                 "valLoss": 2.3 * (0.94 ** i), "valAccuracy": 0.50 + 0.36 * (1 - 0.94 ** i)}
                for i in range(1, 21)
            ],
            "performanceMetrics": {
                "accuracy": 0.89,
                "precision": 0.87,
                "recall": 0.91,
                "f1Score": 0.89,
                "auc": 0.94,
                "sharpeRatio": 1.62,
                "maxDrawdown": 0.09
            },
            "createdAt": "2026-01-23T14:00:00",
            "trainedAt": "2026-01-23T16:30:00",
            "filePath": "trained_models/mdl-002.pt",
            "fileSize": 28456789,
            "generations": 60,
            "bestGeneration": 55,
            "fitness": 92.3
        },
        {
            "id": "mdl-003",
            "name": "RNN_GOOGL_Trend",
            "modelType": "RNN",
            "datasetId": 3,
            "jobId": "job-003",
            "status": "trained",
            "hyperparameters": {
                "layers": 2,
                "layerSize": 64,
                "learningRate": 0.002,
                "activationFunction": "tanh",
                "dropout": 0.25,
                "batchSize": 32,
                "epochs": 80
            },
            "trainingHistory": [
                {"epoch": i, "loss": 2.8 * (0.93 ** i), "accuracy": 0.48 + 0.35 * (1 - 0.93 ** i),
                 "valLoss": 2.9 * (0.93 ** i), "valAccuracy": 0.46 + 0.33 * (1 - 0.93 ** i)}
                for i in range(1, 21)
            ],
            "performanceMetrics": {
                "accuracy": 0.82,
                "precision": 0.80,
                "recall": 0.84,
                "f1Score": 0.82,
                "auc": 0.88,
                "sharpeRatio": 1.21,
                "maxDrawdown": 0.15
            },
            "createdAt": "2026-01-22T09:00:00",
            "trainedAt": "2026-01-22T10:15:00",
            "filePath": "trained_models/mdl-003.pt",
            "fileSize": 8234567,
            "generations": 40,
            "bestGeneration": 35,
            "fitness": 78.4
        }
    ]

    for model in sample_models:
        models_store[model["id"]] = model


# Initialize sample models on module load
init_sample_models()


@router.get("", response_model=ModelListResponse)
async def list_models(
    dataset_id: Optional[int] = None,
    model_type: Optional[str] = None,
    sort_by: Optional[str] = "createdAt",
    sort_order: Optional[str] = "desc"
):
    """
    List all trained models with filtering and sorting.

    Args:
        dataset_id: Filter by dataset ID
        model_type: Filter by model type (LSTM, N-BEATS, RNN)
        sort_by: Sort field (accuracy, fitness, createdAt, name)
        sort_order: Sort order (asc, desc)

    Returns:
        List of models
    """
    from app.models.database import SessionLocal
    from app.models.dataset import Dataset

    models = list(models_store.values())

    # Filter by dataset_id if provided
    if dataset_id is not None:
        models = [m for m in models if m.get("datasetId") == dataset_id]

    # Filter by model_type if provided
    if model_type is not None:
        models = [m for m in models if m.get("modelType", "").upper() == model_type.upper()]

    # Enrich models with dataset info
    db = SessionLocal()
    try:
        dataset_cache = {}
        for model in models:
            ds_id = model.get("datasetId")
            if ds_id and ds_id not in dataset_cache:
                dataset = db.query(Dataset).filter(Dataset.id == ds_id).first()
                if dataset:
                    dataset_cache[ds_id] = {
                        'name': dataset.name,
                        'symbol': dataset.ticker,
                        'timeframe': dataset.timeframe,
                        'start': dataset.data_range_start.strftime('%Y-%m-%d') if dataset.data_range_start else None,
                        'end': dataset.data_range_end.strftime('%Y-%m-%d') if dataset.data_range_end else None
                    }
                else:
                    dataset_cache[ds_id] = None

            ds_info = dataset_cache.get(ds_id)
            if ds_info:
                model['datasetName'] = ds_info['name']
                model['symbol'] = ds_info['symbol']
                model['timeframe'] = ds_info['timeframe']
                if ds_info['start'] and ds_info['end']:
                    model['trainPeriod'] = f"{ds_info['start']} to {ds_info['end']}"
    finally:
        db.close()

    models = [ModelResponse(**m) for m in models]

    # Sort based on sort_by field
    reverse = sort_order.lower() == "desc"

    if sort_by == "accuracy":
        models.sort(key=lambda x: x.performanceMetrics.accuracy if x.performanceMetrics else 0, reverse=reverse)
    elif sort_by == "fitness":
        models.sort(key=lambda x: x.fitness if x.fitness else 0, reverse=reverse)
    elif sort_by == "name":
        models.sort(key=lambda x: x.name.lower(), reverse=reverse)
    elif sort_by == "date" or sort_by == "createdAt":
        models.sort(key=lambda x: x.createdAt, reverse=reverse)
    else:
        models.sort(key=lambda x: x.createdAt, reverse=True)

    return ModelListResponse(
        models=models,
        total=len(models)
    )


@router.get("/{model_id}", response_model=ModelResponse)
async def get_model(model_id: str):
    """Get model details by ID."""
    if model_id not in models_store:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

    return ModelResponse(**models_store[model_id])


@router.delete("/{model_id}")
async def delete_model(model_id: str):
    """Delete a model."""
    if model_id not in models_store:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

    del models_store[model_id]
    logger.info(f"Deleted model {model_id}")

    return {"message": f"Model {model_id} deleted"}


@router.post("/{model_id}/export")
async def export_model(model_id: str, format: str = "pytorch"):
    """
    Export a model in the specified format.

    Args:
        model_id: Model ID
        format: Export format - "pytorch" (default) or "onnx"

    Returns:
        Export details including path
    """
    if model_id not in models_store:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

    supported_formats = ["pytorch", "onnx", "pt", "pth"]
    if format.lower() not in supported_formats:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format: {format}. Supported: {supported_formats}"
        )

    model = models_store[model_id]

    # Map format to extension
    extension_map = {
        "pytorch": "pt",
        "pt": "pt",
        "pth": "pth",
        "onnx": "onnx"
    }
    ext = extension_map.get(format.lower(), "pt")

    # Simulate export (in production, would actually convert and save the model)
    export_path = f"exports/{model_id}.{ext}"

    # For ONNX, add additional conversion info
    export_info = {
        "message": f"Model exported successfully to {format.upper()} format",
        "format": format,
        "path": export_path,
        "size": model.get("fileSize", 0)
    }

    if format.lower() == "onnx":
        export_info["opset_version"] = 13
        export_info["input_names"] = ["input"]
        export_info["output_names"] = ["output"]
        export_info["dynamic_axes"] = {"input": {0: "batch_size"}, "output": {0: "batch_size"}}

    logger.info(f"Exported model {model_id} to {export_path}")

    return export_info


@router.post("/{model_id}/export/pytorch")
async def export_model_pytorch(model_id: str):
    """
    Export model to PyTorch checkpoint format (.pt).

    Args:
        model_id: Model ID

    Returns:
        Export details
    """
    return await export_model(model_id, format="pytorch")


@router.post("/{model_id}/export/onnx")
async def export_model_onnx(model_id: str):
    """
    Export model to ONNX format for cross-platform deployment.

    Args:
        model_id: Model ID

    Returns:
        Export details including ONNX configuration
    """
    return await export_model(model_id, format="onnx")


@router.post("/{model_id}/clone")
async def clone_model(model_id: str):
    """Clone a model with a new ID."""
    if model_id not in models_store:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

    original = models_store[model_id].copy()
    new_id = f"mdl-{uuid.uuid4().hex[:6]}"

    cloned = {
        **original,
        "id": new_id,
        "name": f"{original['name']}_clone",
        "createdAt": datetime.now().isoformat(),
        "trainedAt": None,
        "status": "cloned",
        "filePath": None,
        "fileSize": None
    }

    models_store[new_id] = cloned
    logger.info(f"Cloned model {model_id} to {new_id}")

    return ModelResponse(**cloned)


@router.get("/{model_id}/predictions")
async def get_model_predictions(model_id: str, limit: int = 100):
    """Get prediction visualization data for a model."""
    if model_id not in models_store:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

    # Generate sample prediction data for visualization
    predictions = []
    for i in range(limit):
        actual = 100 + random.gauss(0, 10) + i * 0.1
        predicted = actual + random.gauss(0, 3)
        predictions.append({
            "index": i,
            "actual": round(actual, 2),
            "predicted": round(predicted, 2),
            "error": round(abs(actual - predicted), 2)
        })

    return {
        "modelId": model_id,
        "predictions": predictions,
        "mse": round(sum(p["error"] ** 2 for p in predictions) / len(predictions), 4),
        "mae": round(sum(p["error"] for p in predictions) / len(predictions), 4)
    }


@router.get("/{model_id}/confusion-matrix")
async def get_confusion_matrix(model_id: str):
    """Get confusion matrix data for a classification model."""
    if model_id not in models_store:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

    # Generate sample confusion matrix data
    # For binary classification: up/down prediction
    tp = random.randint(80, 100)  # True positive
    tn = random.randint(75, 95)   # True negative
    fp = random.randint(10, 25)   # False positive
    fn = random.randint(12, 28)   # False negative

    return {
        "modelId": model_id,
        "labels": ["Down", "Up"],
        "matrix": [
            [tn, fp],
            [fn, tp]
        ],
        "metrics": {
            "accuracy": round((tp + tn) / (tp + tn + fp + fn), 4),
            "precision": round(tp / (tp + fp), 4),
            "recall": round(tp / (tp + fn), 4),
            "specificity": round(tn / (tn + fp), 4)
        }
    }
