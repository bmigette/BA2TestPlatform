"""
Model model for storing trained model metadata
"""

from sqlalchemy import Column, Integer, String, DateTime, JSON, ForeignKey, Float, Text
from sqlalchemy.sql import func
from .database import Base


class Model(Base):
    """Trained Model model"""

    __tablename__ = "models"

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(Integer, ForeignKey("optimization_jobs.id"), nullable=False)
    model_type = Column(String(100), nullable=False)  # LSTM, N-BEATS, RNN, etc.

    # Architecture and hyperparameters
    architecture = Column(JSON, nullable=False)
    hyperparameters = Column(JSON, nullable=False)

    # Training metrics
    training_metrics = Column(JSON, nullable=True)
    accuracy = Column(Float, nullable=True)
    loss = Column(Float, nullable=True)
    val_accuracy = Column(Float, nullable=True)
    val_loss = Column(Float, nullable=True)

    # Prediction targets configuration
    # Stores the full target configs used during training (type, params, order)
    prediction_targets = Column(JSON, nullable=True)

    # Model storage
    file_path = Column(String(500), nullable=False)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    def __repr__(self):
        return f"<Model(id={self.id}, type='{self.model_type}', accuracy={self.accuracy})>"
