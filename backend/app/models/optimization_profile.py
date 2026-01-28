"""
Optimization Profile model for storing reusable optimization configurations
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, Text
from sqlalchemy.sql import func
from .database import Base


class OptimizationProfile(Base):
    """Optimization Profile model"""

    __tablename__ = "optimization_profiles"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)

    # Model configuration
    model_types = Column(JSON, nullable=False)  # List of selected model types
    parameter_ranges = Column(JSON, nullable=False)  # Hyperparameter ranges
    prediction_targets = Column(JSON, nullable=False)  # Target configurations

    # Training configuration
    train_test_split = Column(Float, default=80.0)  # Train/test split percentage
    genetic_config = Column(JSON, nullable=True)  # Genetic algorithm settings
    metrics_config = Column(JSON, nullable=True)  # Optimization metric settings

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    def __repr__(self):
        return f"<OptimizationProfile(id={self.id}, name='{self.name}')>"
