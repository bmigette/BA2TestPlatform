"""
Dataset model for storing dataset metadata
"""

from sqlalchemy import Column, Integer, String, DateTime, JSON, Text, Float
from sqlalchemy.sql import func
from .database import Base


class Dataset(Base):
    """Dataset model"""

    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    ticker = Column(String(50), nullable=False)
    timeframe = Column(String(20), nullable=False)
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    rows_count = Column(Integer, nullable=False, default=0)

    # JSON fields for configuration
    technical_indicators = Column(JSON, nullable=True)
    fundamentals_config = Column(JSON, nullable=True)
    sentiment_config = Column(JSON, nullable=True)

    # Complete generation config for regeneration
    # Stores: data_provider, original_start_date, original_end_date,
    # indicator_collection_id, and all parameters used during creation
    generation_config = Column(JSON, nullable=True)

    # Normalization buffer percentage for live data headroom (default 35%)
    normalization_buffer_pct = Column(Float, default=0.35, nullable=False)

    # File storage
    file_path = Column(String(500), nullable=False)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    def __repr__(self):
        return f"<Dataset(id={self.id}, name='{self.name}', ticker='{self.ticker}')>"
