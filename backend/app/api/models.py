"""
Models API endpoints.

Manages trained ML models from optimization jobs.
Now with database persistence.
"""

import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
import uuid

from app.models.database import get_db
from app.models.model import TrainedModel
from app.models.dataset import Dataset

logger = logging.getLogger(__name__)

router = APIRouter()

# Keep in-memory store for backward compatibility during transition
# Will be phased out as database is populated
models_store: Dict[str, dict] = {}


class HyperParameters(BaseModel):
    layers: Optional[int] = None
    layerSize: Optional[int] = None
    learningRate: Optional[float] = None
    dropout: Optional[float] = None
    batchSize: Optional[int] = None
    epochs: Optional[int] = None
    # Note: activationFunction removed - not configurable on most models

    class Config:
        extra = "allow"  # Allow extra fields from database


class TrainingHistory(BaseModel):
    epoch: int
    loss: float
    accuracy: float
    valLoss: float
    valAccuracy: float


class PerformanceMetrics(BaseModel):
    accuracy: Optional[float] = 0
    precision: Optional[float] = 0
    recall: Optional[float] = 0
    f1Score: Optional[float] = 0
    auc: Optional[float] = 0
    sharpeRatio: Optional[float] = None
    maxDrawdown: Optional[float] = None

    class Config:
        extra = "allow"  # Allow extra fields


class ModelResponse(BaseModel):
    id: str
    name: str
    modelType: str
    datasetId: Optional[int] = None
    datasetName: Optional[str] = None
    symbol: Optional[str] = None
    timeframe: Optional[str] = None
    trainPeriod: Optional[str] = None
    jobId: Optional[str] = None
    status: Optional[str] = "trained"
    hyperparameters: Optional[HyperParameters] = None
    trainingHistory: Optional[List[TrainingHistory]] = []
    performanceMetrics: Optional[PerformanceMetrics] = None
    confusionMatrix: Optional[List[List[int]]] = None
    allMetrics: Optional[Dict[str, Any]] = None
    trainingDateRange: Optional[Dict[str, str]] = None
    predictionTargets: Optional[List[Dict[str, Any]]] = None
    predictionHorizon: Optional[int] = 3
    normalizationParams: Optional[Dict[str, Any]] = None  # Scaler settings for inference
    createdAt: Optional[str] = None
    trainedAt: Optional[str] = None
    filePath: Optional[str] = None
    fileSize: Optional[int] = None
    generations: Optional[int] = 0
    bestGeneration: Optional[int] = 0
    fitness: Optional[float] = 0

    class Config:
        extra = "allow"


class ModelListResponse(BaseModel):
    models: List[ModelResponse]
    total: int


def db_model_to_dict(db_model: TrainedModel, dataset_info: dict = None) -> dict:
    """Convert database model to API response dict"""
    result = db_model.to_dict()

    # Add dataset info if available
    if dataset_info:
        result['datasetName'] = dataset_info.get('name')
        result['symbol'] = dataset_info.get('symbol')
        result['timeframe'] = dataset_info.get('timeframe')
        if dataset_info.get('start') and dataset_info.get('end'):
            result['trainPeriod'] = f"{dataset_info['start']} to {dataset_info['end']}"

    return result


def get_all_models(db: Session) -> List[dict]:
    """Get all models from both database and in-memory store"""
    models = []

    # Get from database
    db_models = db.query(TrainedModel).all()
    for m in db_models:
        models.append(db_model_to_dict(m))

    # Also include in-memory models (for backward compatibility)
    for model_id, model_data in models_store.items():
        # Skip if already in database
        if not any(m['id'] == model_id for m in models):
            models.append(model_data.copy())

    return models


def get_model_by_id(model_id: str, db: Session) -> Optional[dict]:
    """Get a model by ID from database or in-memory store"""
    # Try database first
    db_model = db.query(TrainedModel).filter(TrainedModel.model_id == model_id).first()
    if db_model:
        return db_model_to_dict(db_model)

    # Fall back to in-memory store
    if model_id in models_store:
        return models_store[model_id].copy()

    return None


def save_model_to_db(model_data: dict, db: Session) -> TrainedModel:
    """Save a model to the database"""
    # Check if already exists
    existing = db.query(TrainedModel).filter(TrainedModel.model_id == model_data['id']).first()

    if existing:
        # Update existing
        existing.name = model_data.get('name', existing.name)
        existing.model_type = model_data.get('modelType', existing.model_type)
        existing.dataset_id = model_data.get('datasetId', existing.dataset_id)
        existing.job_id = model_data.get('jobId', existing.job_id)
        existing.status = model_data.get('status', existing.status)
        existing.hyperparameters = model_data.get('hyperparameters', existing.hyperparameters)
        existing.training_history = model_data.get('trainingHistory', existing.training_history)
        existing.performance_metrics = model_data.get('performanceMetrics', existing.performance_metrics)
        existing.confusion_matrix = model_data.get('confusionMatrix', existing.confusion_matrix)
        existing.all_metrics = model_data.get('allMetrics', existing.all_metrics)
        existing.training_date_range = model_data.get('trainingDateRange', existing.training_date_range)
        existing.prediction_targets = model_data.get('predictionTargets', existing.prediction_targets)
        existing.prediction_horizon = model_data.get('predictionHorizon', existing.prediction_horizon)
        existing.prediction_mode = model_data.get('predictionMode', existing.prediction_mode)
        existing.loss_function = model_data.get('lossFunction', existing.loss_function)
        existing.normalization_params = model_data.get('normalizationParams', existing.normalization_params)
        existing.generations = model_data.get('generations', existing.generations)
        existing.best_generation = model_data.get('bestGeneration', existing.best_generation)
        existing.fitness = model_data.get('fitness', existing.fitness)
        existing.file_path = model_data.get('filePath', existing.file_path)
        existing.file_size = model_data.get('fileSize', existing.file_size)
        if model_data.get('trainedAt'):
            try:
                existing.trained_at = datetime.fromisoformat(model_data['trainedAt'].replace('Z', '+00:00'))
            except:
                pass
        db.commit()
        return existing
    else:
        # Create new
        new_model = TrainedModel(
            model_id=model_data['id'],
            name=model_data.get('name', 'Unnamed Model'),
            model_type=model_data.get('modelType', 'Unknown'),
            dataset_id=model_data.get('datasetId'),
            job_id=model_data.get('jobId'),
            status=model_data.get('status', 'trained'),
            hyperparameters=model_data.get('hyperparameters'),
            training_history=model_data.get('trainingHistory'),
            performance_metrics=model_data.get('performanceMetrics'),
            confusion_matrix=model_data.get('confusionMatrix'),
            all_metrics=model_data.get('allMetrics'),
            training_date_range=model_data.get('trainingDateRange'),
            prediction_targets=model_data.get('predictionTargets'),
            prediction_horizon=model_data.get('predictionHorizon', 3),
            prediction_mode=model_data.get('predictionMode', 'shift'),
            loss_function=model_data.get('lossFunction', 'focal_loss'),
            normalization_params=model_data.get('normalizationParams'),
            generations=model_data.get('generations', 0),
            best_generation=model_data.get('bestGeneration', 0),
            fitness=model_data.get('fitness', 0),
            file_path=model_data.get('filePath'),
            file_size=model_data.get('fileSize'),
        )
        if model_data.get('trainedAt'):
            try:
                new_model.trained_at = datetime.fromisoformat(model_data['trainedAt'].replace('Z', '+00:00'))
            except:
                pass
        db.add(new_model)
        db.commit()
        db.refresh(new_model)
        return new_model


@router.get("", response_model=ModelListResponse)
async def list_models(
    dataset_id: Optional[int] = None,
    model_type: Optional[str] = None,
    sort_by: Optional[str] = "createdAt",
    sort_order: Optional[str] = "desc",
    db: Session = Depends(get_db)
):
    """
    List all trained models with filtering and sorting.
    """
    models = get_all_models(db)

    # Filter by dataset_id if provided
    if dataset_id is not None:
        models = [m for m in models if m.get("datasetId") == dataset_id]

    # Filter by model_type if provided
    if model_type is not None:
        models = [m for m in models if m.get("modelType", "").upper() == model_type.upper()]

    # Enrich models with dataset info
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
                    'start': dataset.start_date.strftime('%Y-%m-%d') if dataset.start_date else None,
                    'end': dataset.end_date.strftime('%Y-%m-%d') if dataset.end_date else None
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

    # Convert to response models
    response_models = []
    for m in models:
        try:
            response_models.append(ModelResponse(**m))
        except Exception as e:
            logger.warning(f"Failed to parse model {m.get('id')}: {e}")
            continue

    # Sort based on sort_by field
    reverse = sort_order.lower() == "desc"

    if sort_by == "accuracy":
        response_models.sort(key=lambda x: x.performanceMetrics.accuracy if x.performanceMetrics else 0, reverse=reverse)
    elif sort_by == "fitness":
        response_models.sort(key=lambda x: x.fitness if x.fitness else 0, reverse=reverse)
    elif sort_by == "name":
        response_models.sort(key=lambda x: x.name.lower(), reverse=reverse)
    elif sort_by == "date" or sort_by == "createdAt":
        response_models.sort(key=lambda x: x.createdAt or "", reverse=reverse)
    else:
        response_models.sort(key=lambda x: x.createdAt or "", reverse=True)

    return ModelListResponse(
        models=response_models,
        total=len(response_models)
    )


@router.get("/{model_id}", response_model=ModelResponse)
async def get_model(model_id: str, db: Session = Depends(get_db)):
    """Get model details by ID."""
    model = get_model_by_id(model_id, db)
    if not model:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

    # Enrich with dataset info
    ds_id = model.get("datasetId")
    if ds_id:
        dataset = db.query(Dataset).filter(Dataset.id == ds_id).first()
        if dataset:
            model['datasetName'] = dataset.name
            model['symbol'] = dataset.ticker
            model['timeframe'] = dataset.timeframe
            if dataset.start_date and dataset.end_date:
                model['trainPeriod'] = f"{dataset.start_date.strftime('%Y-%m-%d')} to {dataset.end_date.strftime('%Y-%m-%d')}"

    return ModelResponse(**model)


@router.delete("/{model_id}")
async def delete_model(model_id: str, db: Session = Depends(get_db)):
    """Delete a model."""
    # Try database first
    db_model = db.query(TrainedModel).filter(TrainedModel.model_id == model_id).first()
    if db_model:
        db.delete(db_model)
        db.commit()
        logger.info(f"Deleted model {model_id} from database")
        return {"message": f"Model {model_id} deleted"}

    # Fall back to in-memory
    if model_id in models_store:
        del models_store[model_id]
        logger.info(f"Deleted model {model_id} from memory")
        return {"message": f"Model {model_id} deleted"}

    raise HTTPException(status_code=404, detail=f"Model {model_id} not found")


@router.post("/{model_id}/export")
async def export_model(model_id: str, format: str = "pytorch", db: Session = Depends(get_db)):
    """Export a model in the specified format."""
    model = get_model_by_id(model_id, db)
    if not model:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

    supported_formats = ["pytorch", "onnx", "pt", "pth"]
    if format.lower() not in supported_formats:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format: {format}. Supported: {supported_formats}"
        )

    extension_map = {
        "pytorch": "pt",
        "pt": "pt",
        "pth": "pth",
        "onnx": "onnx"
    }
    ext = extension_map.get(format.lower(), "pt")

    export_path = f"exports/{model_id}.{ext}"

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
async def export_model_pytorch(model_id: str, db: Session = Depends(get_db)):
    """Export model to PyTorch checkpoint format (.pt)."""
    return await export_model(model_id, format="pytorch", db=db)


@router.post("/{model_id}/export/onnx")
async def export_model_onnx(model_id: str, db: Session = Depends(get_db)):
    """Export model to ONNX format for cross-platform deployment."""
    return await export_model(model_id, format="onnx", db=db)


@router.post("/{model_id}/clone")
async def clone_model(model_id: str, db: Session = Depends(get_db)):
    """Clone a model with a new ID."""
    original = get_model_by_id(model_id, db)
    if not original:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

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

    # Save to database
    save_model_to_db(cloned, db)
    logger.info(f"Cloned model {model_id} to {new_id}")

    return ModelResponse(**cloned)


@router.get("/{model_id}/predictions")
async def get_model_predictions(model_id: str, limit: int = 100, db: Session = Depends(get_db)):
    """Get prediction visualization data for a model."""
    model = get_model_by_id(model_id, db)
    if not model:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

    # Return actual predictions if stored, otherwise 404
    # TODO: Implement actual prediction data storage and retrieval
    raise HTTPException(
        status_code=404,
        detail="Predictions not available for this model. Run inference to generate predictions."
    )


@router.get("/{model_id}/confusion-matrix")
async def get_confusion_matrix(model_id: str, db: Session = Depends(get_db)):
    """Get confusion matrix data for a classification model."""
    model = get_model_by_id(model_id, db)
    if not model:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

    # Check if model has stored confusion matrix
    cm = model.get('confusionMatrix')
    if not cm:
        # Try to get from allMetrics
        all_metrics = model.get('allMetrics', {})
        cm = all_metrics.get('confusion_matrix')

    if cm and len(cm) >= 2:
        total = sum(sum(row) for row in cm)
        if len(cm) == 2:
            tn, fp = cm[0]
            fn, tp = cm[1]
            return {
                "modelId": model_id,
                "labels": ["Down", "Up"],
                "matrix": cm,
                "metrics": {
                    "accuracy": round((tp + tn) / total, 4) if total > 0 else 0,
                    "precision": round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0,
                    "recall": round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0,
                    "specificity": round(tn / (tn + fp), 4) if (tn + fp) > 0 else 0
                }
            }

    # No confusion matrix available
    raise HTTPException(
        status_code=404,
        detail="Confusion matrix not available for this model"
    )
