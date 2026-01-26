"""
Optimization Profile model for storing reusable optimization configurations
"""

from sqlalchemy import Column, Integer, String, DateTime, JSON, Text
from sqlalchemy.sql import func
from .database import Base


class OptimizationProfile(Base):
    """Optimization Profile model"""

    __tablename__ = "optimization_profiles"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)

    # Configuration
    model_types = Column(JSON, nullable=False)
    parameter_ranges = Column(JSON, nullable=False)
    prediction_targets = Column(JSON, nullable=False)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    def __repr__(self):
        return f"<OptimizationProfile(id={self.id}, name='{self.name}')>"
