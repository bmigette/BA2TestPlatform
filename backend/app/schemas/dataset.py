"""
Pydantic schemas for dataset API requests and responses
"""

from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, Dict, Any


class DatasetCreate(BaseModel):
    """Schema for creating a new dataset"""
    ticker: str = Field(..., description="Stock ticker symbol (e.g., AAPL, MSFT)")
    timeframe: str = Field(..., description="Timeframe for data (e.g., 1d, 1h, 4h)")
    start_date: Optional[str] = Field(None, description="Start date for data (YYYY-MM-DD)")
    end_date: Optional[str] = Field(None, description="End date for data (YYYY-MM-DD)")
    name: Optional[str] = Field(None, description="Custom name for dataset")
    technical_indicators: Optional[Dict[str, Any]] = Field(None, description="Technical indicators configuration")
    fundamentals_config: Optional[Dict[str, Any]] = Field(None, description="Fundamentals configuration")
    sentiment_config: Optional[Dict[str, Any]] = Field(None, description="Sentiment analysis configuration")


class DatasetResponse(BaseModel):
    """Schema for dataset response"""
    id: int
    name: str
    ticker: str
    timeframe: str
    start_date: datetime
    end_date: datetime
    rows_count: int
    technical_indicators: Optional[Dict[str, Any]]
    fundamentals_config: Optional[Dict[str, Any]]
    sentiment_config: Optional[Dict[str, Any]]
    file_path: str
    created_at: datetime
    updated_at: Optional[datetime]

    class Config:
        from_attributes = True


class DatasetListResponse(BaseModel):
    """Schema for dataset list response"""
    datasets: list[DatasetResponse]
    total: int
