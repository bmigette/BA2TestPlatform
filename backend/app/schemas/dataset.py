"""
Pydantic schemas for dataset API requests and responses
"""

from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, Dict, Any, List


class DatasetCreate(BaseModel):
    """Schema for creating a new dataset"""
    ticker: str = Field(..., description="Stock ticker symbol (e.g., AAPL, MSFT)")
    timeframe: str = Field(..., description="Timeframe for data (e.g., 1d, 1h, 4h)")
    start_date: Optional[str] = Field(None, description="Start date for data (YYYY-MM-DD)")
    end_date: Optional[str] = Field(None, description="End date for data (YYYY-MM-DD)")
    name: Optional[str] = Field(None, description="Custom name for dataset")
    data_provider: Optional[str] = Field("yfinance", description="Data provider to use")
    technical_indicators: Optional[List[Dict[str, Any]]] = Field(None, description="Technical indicators configuration")
    fundamentals_config: Optional[Dict[str, Any]] = Field(None, description="Fundamentals configuration")
    sentiment_config: Optional[Dict[str, Any]] = Field(None, description="Sentiment analysis configuration")
    indicator_collection_id: Optional[int] = Field(None, description="ID of indicator collection to use")
    normalization_buffer_pct: float = Field(0.35, description="Buffer percentage for normalization (default 35%)")


class DatasetUpdate(BaseModel):
    """Schema for updating an existing dataset"""
    name: Optional[str] = Field(None, description="New name for dataset")
    ticker: Optional[str] = Field(None, description="New ticker symbol (will regenerate data)")
    timeframe: Optional[str] = Field(None, description="New timeframe (will regenerate data)")
    start_date: Optional[str] = Field(None, description="New start date (will regenerate data)")
    end_date: Optional[str] = Field(None, description="New end date (will regenerate data)")
    data_provider: Optional[str] = Field(None, description="Data provider (yfinance, fmp)")
    technical_indicators: Optional[List[Dict[str, Any]]] = Field(None, description="New indicators (will regenerate data)")
    normalization_buffer_pct: Optional[float] = Field(None, description="New buffer percentage")
    sentiment_config: Optional[Dict[str, Any]] = Field(None, description="Sentiment analysis configuration")
    fundamentals_config: Optional[Dict[str, Any]] = Field(None, description="Fundamentals configuration")


class DatasetDuplicate(BaseModel):
    """Schema for duplicating a dataset"""
    new_ticker: Optional[str] = Field(None, description="New ticker symbol for duplicate")
    new_name: Optional[str] = Field(None, description="New name for duplicate")


class DatasetRegenerate(BaseModel):
    """Schema for partial dataset regeneration"""
    regenerate_ohlcv: bool = Field(True, description="Re-fetch OHLCV data from provider")
    regenerate_technical: bool = Field(True, description="Recalculate technical indicators")
    regenerate_sentiment: bool = Field(True, description="Re-fetch and recalculate sentiment/news data")
    regenerate_fundamentals: bool = Field(True, description="Re-fetch fundamentals data")
    regenerate_macro: bool = Field(True, description="Re-fetch macro economic data")


class DatasetResponse(BaseModel):
    """Schema for dataset response"""
    id: int
    name: str
    ticker: str
    timeframe: str
    start_date: datetime
    end_date: datetime
    rows_count: int
    status: str = "ready"
    error_message: Optional[str] = None
    progress_message: Optional[str] = None
    task_id: Optional[str] = None
    technical_indicators: Optional[List[Dict[str, Any]]]
    fundamentals_config: Optional[Dict[str, Any]]
    sentiment_config: Optional[Dict[str, Any]]
    generation_config: Optional[Dict[str, Any]]
    normalization_buffer_pct: float
    file_path: str
    created_at: datetime
    updated_at: Optional[datetime]

    class Config:
        from_attributes = True


class DatasetListResponse(BaseModel):
    """Schema for dataset list response"""
    datasets: list[DatasetResponse]
    total: int
