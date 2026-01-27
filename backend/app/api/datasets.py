"""
Dataset API endpoints
Updated for preview endpoint and Parquet export
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from typing import List
import logging
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path

from app.models.database import get_db, SessionLocal
from app.models.dataset import Dataset, DatasetStatus
from app.schemas.dataset import DatasetCreate, DatasetResponse, DatasetListResponse, DatasetUpdate, DatasetDuplicate
from app.indicators import TechnicalIndicators
from app.services.fundamentals import FundamentalsService
from app.services.macro import MacroService
from app.services.sentiment import SentimentService
from dataproviders.ohlcv.YFinanceDataProvider import YFinanceDataProvider

logger = logging.getLogger(__name__)

# Timeframe configuration for multi-timeframe indicators
SUPPORTED_TIMEFRAMES = ["15m", "1h", "4h", "1d"]
from typing import Dict, Any, Optional
TIMEFRAME_INTERVAL_MAP = {
    "15m": "15m",
    "1h": "1h",
    "4h": "4h",
    "1d": "1d",
    "D1": "1d"
}

# Default indicators configuration for each timeframe
DEFAULT_INDICATORS = {
    "sma_20": {"type": "sma", "period": 20},
    "sma_50": {"type": "sma", "period": 50},
    "ema_12": {"type": "ema", "period": 12},
    "ema_26": {"type": "ema", "period": 26},
    "rsi_14": {"type": "rsi", "period": 14},
    "macd": {"type": "macd", "fast": 12, "slow": 26, "signal": 9},
    "bbands": {"type": "bollinger", "period": 20, "std_dev": 2.0},
    "atr_14": {"type": "atr", "period": 14},
    "stoch": {"type": "stochastic", "k_period": 14, "d_period": 3, "smooth_k": 3}
}

router = APIRouter()


@router.post("", response_model=DatasetResponse, status_code=status.HTTP_201_CREATED)
async def create_dataset(
    dataset_create: DatasetCreate,
    db: Session = Depends(get_db)
):
    """
    Create a new dataset by fetching data from a provider and saving it.

    Args:
        dataset_create: Dataset creation parameters
        db: Database session

    Returns:
        Created dataset with metadata
    """
    db_dataset = None
    try:
        logger.info(f"Creating dataset for {dataset_create.ticker} with timeframe {dataset_create.timeframe}")

        # Debug log all dataset generation options
        logger.debug("=" * 60)
        logger.debug("DATASET GENERATION OPTIONS:")
        logger.debug(f"  Ticker: {dataset_create.ticker}")
        logger.debug(f"  Timeframe: {dataset_create.timeframe}")
        logger.debug(f"  Start Date: {dataset_create.start_date}")
        logger.debug(f"  End Date: {dataset_create.end_date}")
        logger.debug(f"  Name: {dataset_create.name}")
        logger.debug(f"  Data Provider: {dataset_create.data_provider or 'yfinance'}")
        logger.debug(f"  Normalization Buffer: {dataset_create.normalization_buffer_pct}%")
        logger.debug(f"  Indicator Collection ID: {dataset_create.indicator_collection_id}")
        logger.debug(f"  Technical Indicators: {dataset_create.technical_indicators}")
        logger.debug(f"  Fundamentals Config: {dataset_create.fundamentals_config}")
        logger.debug(f"  Sentiment Config: {dataset_create.sentiment_config}")
        logger.debug("=" * 60)

        # Generate dataset name if not provided
        if not dataset_create.name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dataset_create.name = f"{dataset_create.ticker}_{dataset_create.timeframe}_{timestamp}"

        # Calculate date range if not provided (default to 1 year)
        if not dataset_create.end_date:
            end_date = datetime.now()
        else:
            end_date = datetime.strptime(dataset_create.end_date, "%Y-%m-%d")

        if not dataset_create.start_date:
            start_date = end_date - timedelta(days=365)
        else:
            start_date = datetime.strptime(dataset_create.start_date, "%Y-%m-%d")

        # Build generation config for regeneration capability
        generation_config = {
            "data_provider": dataset_create.data_provider or "yfinance",
            "original_start_date": dataset_create.start_date,
            "original_end_date": dataset_create.end_date,
            "indicator_collection_id": dataset_create.indicator_collection_id,
            "created_at": datetime.now().isoformat()
        }

        # Create datasets directory if it doesn't exist
        datasets_dir = Path("datasets")
        datasets_dir.mkdir(exist_ok=True)
        file_path = datasets_dir / f"{dataset_create.name}.csv"

        # Create database record in PENDING status first
        db_dataset = Dataset(
            name=dataset_create.name,
            ticker=dataset_create.ticker,
            timeframe=dataset_create.timeframe,
            start_date=start_date,
            end_date=end_date,
            rows_count=0,
            status=DatasetStatus.BUILDING.value,
            technical_indicators=dataset_create.technical_indicators,
            fundamentals_config=dataset_create.fundamentals_config,
            sentiment_config=dataset_create.sentiment_config,
            generation_config=generation_config,
            normalization_buffer_pct=dataset_create.normalization_buffer_pct,
            file_path=str(file_path)
        )

        db.add(db_dataset)
        db.commit()
        db.refresh(db_dataset)
        logger.info(f"Created dataset record with ID {db_dataset.id} in BUILDING status")

        # Now fetch data - if this fails, dataset will remain in BUILDING or ERROR status
        provider = YFinanceDataProvider()
        logger.info(f"Fetching data from {start_date.date()} to {end_date.date()}")

        # Convert timeframe to YFinance interval format
        interval_map = {
            "1m": "1m",
            "5m": "5m",
            "15m": "15m",
            "30m": "30m",
            "1h": "1h",
            "4h": "4h",
            "1d": "1d",
            "1w": "1wk",
            "1mo": "1mo"
        }
        interval = interval_map.get(dataset_create.timeframe, "1d")

        data_points = provider.get_data(
            symbol=dataset_create.ticker,
            start_date=start_date,
            end_date=end_date,
            interval=interval
        )

        if not data_points:
            db_dataset.status = DatasetStatus.ERROR.value
            db_dataset.error_message = f"No data available for {dataset_create.ticker}"
            db.commit()
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"No data available for {dataset_create.ticker}"
            )

        # Convert data points to DataFrame
        df = pd.DataFrame([{
            'Date': dp.timestamp,
            'Open': dp.open,
            'High': dp.high,
            'Low': dp.low,
            'Close': dp.close,
            'Volume': dp.volume
        } for dp in data_points])

        df = df.sort_values('Date').reset_index(drop=True)
        logger.info(f"Fetched {len(df)} OHLC data points")

        # Apply technical indicators if configured
        if dataset_create.technical_indicators:
            logger.info(f"Applying {len(dataset_create.technical_indicators)} technical indicators...")
            try:
                # Convert list format to dict format for add_indicators_to_dataframe
                indicators_dict = {}
                for indicator in dataset_create.technical_indicators:
                    indicator_type = indicator.get('type', indicator.get('name', 'unknown'))
                    indicator_name = indicator.get('name', f"{indicator_type}_{indicator.get('period', '')}")
                    indicators_dict[indicator_name] = indicator

                df = TechnicalIndicators.add_indicators_to_dataframe(df, indicators_dict)
                logger.info(f"Added technical indicators. DataFrame now has {len(df.columns)} columns")
            except Exception as e:
                logger.error(f"Error applying technical indicators: {e}")
                # Continue without indicators rather than failing the whole dataset

        # Fetch and add sentiment features if configured
        if dataset_create.sentiment_config and dataset_create.sentiment_config.get('enabled'):
            logger.info("Fetching sentiment data...")
            logger.debug(f"Sentiment config: {dataset_create.sentiment_config}")
            try:
                sentiment_service = SentimentService()

                # Get news sources from config (supports multiple providers)
                news_sources = dataset_create.sentiment_config.get('news_sources', [])
                if not news_sources:
                    # Fallback to legacy 'provider' field
                    legacy_provider = dataset_create.sentiment_config.get('provider', 'fmp')
                    news_sources = [legacy_provider]

                logger.info(f"Fetching news from {len(news_sources)} source(s): {news_sources}")

                # Fetch from all configured news sources
                all_articles = []
                for source in news_sources:
                    # Convert source name to provider name (e.g., 'fmp_news' -> 'fmp')
                    provider = source.replace('_news', '').replace('_company', '').replace('_global', '')
                    try:
                        logger.debug(f"Fetching news from provider: {provider}")
                        articles = sentiment_service.fetch_news_for_ticker(
                            ticker=dataset_create.ticker,
                            start_date=df['Date'].min() if hasattr(df['Date'].min(), 'to_pydatetime') else start_date,
                            end_date=df['Date'].max() if hasattr(df['Date'].max(), 'to_pydatetime') else end_date,
                            provider=provider,
                            enrich_content=dataset_create.sentiment_config.get('enrich_content', True)
                        )
                        if articles:
                            logger.info(f"Fetched {len(articles)} articles from {provider}")
                            all_articles.extend(articles)
                        else:
                            logger.warning(f"No articles from {provider}")
                    except Exception as e:
                        logger.warning(f"Error fetching from {provider}: {e}")

                if all_articles:
                    # Analyze sentiment and add features
                    logger.info(f"Total articles from all sources: {len(all_articles)}")
                    df = sentiment_service.create_sentiment_features(df, all_articles)
                    logger.info(f"Added sentiment features from {len(all_articles)} articles")
                    # Store articles count in sentiment_config
                    dataset_create.sentiment_config['articles_count'] = len(all_articles)
                else:
                    logger.warning("No news articles found for sentiment analysis from any source")
                    dataset_create.sentiment_config['articles_count'] = 0

            except Exception as e:
                logger.error(f"Error fetching sentiment: {e}")
                # Continue without sentiment rather than failing the whole dataset

        # Fetch and add fundamentals if configured
        if dataset_create.fundamentals_config and dataset_create.fundamentals_config.get('enabled'):
            logger.info("Fetching fundamentals data...")
            logger.debug(f"Fundamentals config: {dataset_create.fundamentals_config}")
            try:
                fundamentals_config = dataset_create.fundamentals_config

                # Check if using new statement-based config
                statement_types = fundamentals_config.get('statement_types')
                if statement_types:
                    # Use new statement-based features with lookback
                    lookback_statements = fundamentals_config.get('lookback_statements', 2)
                    providers = fundamentals_config.get('fundamentals_providers', ['yfinance'])

                    logger.info(f"Creating statement features: {statement_types} with {lookback_statements} periods")
                    df = FundamentalsService.create_statement_features(
                        df=df,
                        ticker=dataset_create.ticker,
                        statement_types=statement_types,
                        lookback_statements=lookback_statements,
                        providers=providers,
                        frequency='quarterly'
                    )
                    new_cols = [c for c in df.columns if c.startswith(('bs_', 'is_', 'cf_', 'earn_'))]
                    logger.info(f"Added {len(new_cols)} statement feature columns")
                else:
                    # Legacy mode: add current fundamentals as constant columns
                    fundamentals = FundamentalsService.get_fundamental_data(dataset_create.ticker)

                    if fundamentals and fundamentals.get('current'):
                        current = fundamentals['current']
                        added_fundamentals = []
                        for key, value in current.items():
                            if value is not None:
                                df[f'fundamental_{key}'] = value
                                added_fundamentals.append(key)
                        logger.info(f"Added {len(added_fundamentals)} fundamental columns: {added_fundamentals}")
                    else:
                        logger.warning("No fundamentals data available")

                # Fetch macro indicators if configured
                macro_indicators = fundamentals_config.get('macro_indicators', [])
                if macro_indicators:
                    logger.info(f"Fetching macro indicators: {macro_indicators}")
                    try:
                        macro_service = MacroService()
                        df = macro_service.integrate_macro_with_ohlc(df, macro_indicators)
                        # Rename columns to have macro_ prefix
                        for indicator in macro_indicators:
                            if indicator in df.columns:
                                df = df.rename(columns={indicator: f'macro_{indicator}'})
                                if f'{indicator}_yoy_change' in df.columns:
                                    df = df.rename(columns={f'{indicator}_yoy_change': f'macro_{indicator}_yoy_change'})
                        logger.info(f"Added macro columns for: {macro_indicators}")
                    except Exception as e:
                        logger.warning(f"Error fetching macro data: {e}")

            except Exception as e:
                logger.error(f"Error fetching fundamentals: {e}")
                # Continue without fundamentals rather than failing the whole dataset

        # Save dataset to file
        df.to_csv(file_path, index=False)
        logger.info(f"Saved dataset to {file_path} with {len(df.columns)} columns")

        # Update dataset with actual data and set to READY
        db_dataset.start_date = df['Date'].min()
        db_dataset.end_date = df['Date'].max()
        db_dataset.rows_count = len(df)
        db_dataset.status = DatasetStatus.READY.value
        db_dataset.error_message = None
        db.commit()
        db.refresh(db_dataset)

        logger.info(f"Dataset {db_dataset.id} is now READY with {len(df)} rows and {len(df.columns)} columns")

        return db_dataset

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating dataset: {e}", exc_info=True)
        # If we have a dataset record, mark it as error
        if db_dataset and db_dataset.id:
            try:
                db_dataset.status = DatasetStatus.ERROR.value
                db_dataset.error_message = str(e)
                db.commit()
            except Exception:
                db.rollback()
        else:
            db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create dataset: {str(e)}"
        )


@router.get("", response_model=DatasetListResponse)
async def list_datasets(db: Session = Depends(get_db)):
    """
    List all datasets

    Args:
        db: Database session

    Returns:
        List of all datasets
    """
    try:
        datasets = db.query(Dataset).order_by(Dataset.created_at.desc()).all()
        return {
            "datasets": datasets,
            "total": len(datasets)
        }
    except Exception as e:
        logger.error(f"Error listing datasets: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list datasets: {str(e)}"
        )


@router.get("/{dataset_id}/preview")
async def get_dataset_preview(dataset_id: int, db: Session = Depends(get_db)):
    """
    Get dataset preview data for charting

    Args:
        dataset_id: Dataset ID
        db: Database session

    Returns:
        Dataset preview data (OHLC values)
    """
    try:
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()

        if not dataset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dataset with ID {dataset_id} not found"
            )

        # Load CSV file
        file_path = Path(dataset.file_path)
        if not file_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dataset file not found: {file_path}"
            )

        df = pd.read_csv(file_path)

        # Use pandas to_json with proper NaN handling, then parse back
        # This is the most reliable way to handle NaN/inf for JSON
        import json
        json_str = df.to_json(orient='records', date_format='iso')
        data = json.loads(json_str)

        return {
            "dataset_id": dataset_id,
            "rows": len(data),
            "data": data
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting dataset preview: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get dataset preview: {str(e)}"
        )


@router.get("/{dataset_id}/stats")
async def get_dataset_stats(dataset_id: int, db: Session = Depends(get_db)):
    """
    Get comprehensive statistics for a dataset

    Args:
        dataset_id: Dataset ID
        db: Database session

    Returns:
        Dataset statistics including row count, column count, date range,
        missing data percentages, and basic statistics for numeric columns
    """
    try:
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()

        if not dataset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dataset with ID {dataset_id} not found"
            )

        # Load CSV file
        file_path = Path(dataset.file_path)
        if not file_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dataset file not found: {file_path}"
            )

        df = pd.read_csv(file_path)

        # Basic counts
        total_rows = len(df)
        total_columns = len(df.columns)

        # Date range
        date_column = 'Date' if 'Date' in df.columns else df.columns[0]
        if date_column in df.columns:
            df[date_column] = pd.to_datetime(df[date_column])
            date_range = {
                "start": str(df[date_column].min()),
                "end": str(df[date_column].max())
            }
        else:
            date_range = None

        # Missing data percentage per column
        missing_data = {}
        for col in df.columns:
            missing_count = df[col].isna().sum()
            missing_pct = (missing_count / total_rows * 100) if total_rows > 0 else 0
            missing_data[col] = {
                "count": int(missing_count),
                "percentage": round(missing_pct, 2)
            }

        # Basic statistics for numeric columns
        numeric_stats = {}
        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns

        for col in numeric_columns:
            col_data = df[col].dropna()
            if len(col_data) > 0:
                numeric_stats[col] = {
                    "count": int(len(col_data)),
                    "mean": round(float(col_data.mean()), 4),
                    "std": round(float(col_data.std()), 4),
                    "min": round(float(col_data.min()), 4),
                    "max": round(float(col_data.max()), 4),
                    "median": round(float(col_data.median()), 4)
                }

        # Column types
        column_types = {col: str(dtype) for col, dtype in df.dtypes.items()}

        return {
            "dataset_id": dataset_id,
            "total_rows": total_rows,
            "total_columns": total_columns,
            "date_range": date_range,
            "columns": list(df.columns),
            "column_types": column_types,
            "missing_data": missing_data,
            "numeric_statistics": numeric_stats
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error calculating dataset statistics: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to calculate dataset statistics: {str(e)}"
        )


@router.get("/{dataset_id}/columns")
async def get_dataset_columns(dataset_id: int, db: Session = Depends(get_db)):
    """
    Get all columns in a dataset, categorized by type.

    Categories:
    - price: OHLCV data (Date, Open, High, Low, Close, Volume)
    - technical: Technical indicators (SMA, EMA, RSI, MACD, etc.)
    - fundamental: Fundamental data (P/E, EPS, FCF, etc.)
    - sentiment: Sentiment features (news counts, scores)
    - macro: Macro economic indicators (interest rates, GDP, etc.)
    - target: Prediction targets (price_up_*, price_down_*)
    - other: Unclassified columns

    Args:
        dataset_id: Dataset ID
        db: Database session

    Returns:
        Categorized column information with data types
    """
    try:
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()

        if not dataset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dataset with ID {dataset_id} not found"
            )

        file_path = Path(dataset.file_path)
        if not file_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dataset file not found: {file_path}"
            )

        # Load dataset to get columns
        df = pd.read_csv(file_path, nrows=5)  # Just need headers and dtypes

        # Categorize columns
        price_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
        technical_patterns = ['SMA', 'EMA', 'RSI', 'MACD', 'BB_', 'ATR', 'ADX', 'CCI', 'MFI',
                             'OBV', 'VWAP', 'Stoch', 'Williams', 'ROC', 'MOM', 'TRIX',
                             'DX', 'PLUS_DI', 'MINUS_DI', 'Aroon', 'CMO', 'PPO', 'UO']
        fundamental_patterns = ['fundamental_', 'PE', 'EPS', 'FCF', 'Revenue', 'Debt', 'ROE', 'ROA',
                               'BookValue', 'Dividend', 'MarketCap', 'PB', 'PS',
                               'days_to', 'last_', 'next_']
        sentiment_patterns = ['news_', 'sentiment_', 'positive', 'negative', 'neutral']
        macro_patterns = ['macro_', 'interest_rate', 'gdp', 'inflation', 'unemployment', 'cpi',
                         'fed_', 'treasury', 'yield_', 'vix']
        target_patterns = ['price_up_', 'price_down_', 'target_', 'label_']

        def categorize_column(col_name: str) -> str:
            col_lower = col_name.lower()
            col_upper = col_name.upper()

            if col_name in price_cols:
                return 'price'
            if any(p in col_upper for p in technical_patterns):
                return 'technical'
            if any(p in col_lower for p in fundamental_patterns):
                return 'fundamental'
            if any(p in col_lower for p in sentiment_patterns):
                return 'sentiment'
            if any(p in col_lower for p in macro_patterns):
                return 'macro'
            if any(p in col_lower for p in target_patterns):
                return 'target'
            return 'other'

        columns = {}
        for col in df.columns:
            category = categorize_column(col)
            if category not in columns:
                columns[category] = []
            columns[category].append({
                'name': col,
                'dtype': str(df[col].dtype),
                'category': category
            })

        # Count by category
        category_counts = {cat: len(cols) for cat, cols in columns.items()}

        return {
            'dataset_id': dataset_id,
            'total_columns': len(df.columns),
            'category_counts': category_counts,
            'columns': columns,
            'all_columns': list(df.columns)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting dataset columns: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get dataset columns: {str(e)}"
        )


@router.get("/{dataset_id}", response_model=DatasetResponse)
async def get_dataset(dataset_id: int, db: Session = Depends(get_db)):
    """
    Get dataset details by ID

    Args:
        dataset_id: Dataset ID
        db: Database session

    Returns:
        Dataset details
    """
    try:
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()

        if not dataset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dataset with ID {dataset_id} not found"
            )

        return dataset

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting dataset: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get dataset: {str(e)}"
        )


@router.get("/{dataset_id}/export")
async def export_dataset(dataset_id: int, db: Session = Depends(get_db)):
    """
    Export dataset as CSV file for download

    Args:
        dataset_id: Dataset ID
        db: Database session

    Returns:
        CSV file download
    """
    try:
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()

        if not dataset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dataset with ID {dataset_id} not found"
            )

        # Check if file exists
        file_path = Path(dataset.file_path)
        if not file_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dataset file not found: {file_path}"
            )

        logger.info(f"Exporting dataset {dataset_id}: {file_path}")

        # Return the CSV file as a download
        return FileResponse(
            path=str(file_path),
            media_type="text/csv",
            filename=f"{dataset.name}.csv",
            headers={"Content-Disposition": f"attachment; filename={dataset.name}.csv"}
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting dataset: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to export dataset: {str(e)}"
        )


@router.get("/{dataset_id}/export/parquet")
async def export_dataset_parquet(dataset_id: int, db: Session = Depends(get_db)):
    """
    Export dataset as Parquet file for download

    Args:
        dataset_id: Dataset ID
        db: Database session

    Returns:
        Parquet file download
    """
    try:
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()

        if not dataset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dataset with ID {dataset_id} not found"
            )

        # Check if CSV file exists
        csv_path = Path(dataset.file_path)
        if not csv_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dataset file not found: {csv_path}"
            )

        logger.info(f"Exporting dataset {dataset_id} to Parquet: {csv_path}")

        # Read CSV and convert to Parquet
        df = pd.read_csv(csv_path)

        # Create temporary Parquet file
        parquet_path = csv_path.with_suffix('.parquet')
        df.to_parquet(parquet_path, engine='pyarrow', compression='snappy', index=False)

        logger.info(f"Created Parquet file: {parquet_path}")

        # Return the Parquet file as a download
        return FileResponse(
            path=str(parquet_path),
            media_type="application/octet-stream",
            filename=f"{dataset.name}.parquet",
            headers={"Content-Disposition": f"attachment; filename={dataset.name}.parquet"}
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting dataset to Parquet: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to export dataset to Parquet: {str(e)}"
        )


@router.delete("/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_dataset(dataset_id: int, db: Session = Depends(get_db)):
    """
    Delete a dataset

    Args:
        dataset_id: Dataset ID
        db: Database session
    """
    try:
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()

        if not dataset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dataset with ID {dataset_id} not found"
            )

        # Delete file from disk
        file_path = Path(dataset.file_path)
        if file_path.exists():
            file_path.unlink()
            logger.info(f"Deleted file: {file_path}")

        # Delete from database
        db.delete(dataset)
        db.commit()

        logger.info(f"Deleted dataset with ID {dataset_id}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting dataset: {e}", exc_info=True)
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete dataset: {str(e)}"
        )


@router.post("/{dataset_id}/regenerate", response_model=DatasetResponse)
async def regenerate_dataset(
    dataset_id: int,
    db: Session = Depends(get_db)
):
    """
    Regenerate a dataset by re-fetching data from the provider.

    This endpoint can be used to:
    - Retry a failed dataset generation
    - Refresh data for an existing dataset
    - Reset a dataset that's stuck in BUILDING status

    Args:
        dataset_id: Dataset ID to regenerate
        db: Database session

    Returns:
        Regenerated dataset
    """
    # =========================================================================
    # PHASE 1: Read config from DB and set status to BUILDING
    # =========================================================================
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset with ID {dataset_id} not found"
        )

    logger.info(f"Regenerating dataset {dataset_id} ({dataset.name})")

    # Extract all config into local variables (so we can close DB session)
    dataset_name = dataset.name
    ticker = dataset.ticker
    timeframe = dataset.timeframe
    file_path = dataset.file_path
    technical_indicators = dataset.technical_indicators.copy() if dataset.technical_indicators else None
    fundamentals_config = dataset.fundamentals_config.copy() if dataset.fundamentals_config else None
    sentiment_config = dataset.sentiment_config.copy() if dataset.sentiment_config else None
    gen_config = (dataset.generation_config or {}).copy()

    # Parse dates
    if gen_config.get("original_start_date"):
        start_date = datetime.strptime(gen_config["original_start_date"], "%Y-%m-%d")
    else:
        start_date = dataset.start_date

    if gen_config.get("original_end_date"):
        end_date = datetime.strptime(gen_config["original_end_date"], "%Y-%m-%d")
    else:
        end_date = dataset.end_date

    # Set status to BUILDING and commit immediately
    dataset.status = DatasetStatus.BUILDING.value
    dataset.error_message = None
    db.commit()

    # =========================================================================
    # PHASE 2: Long-running operations WITHOUT holding DB session
    # =========================================================================
    try:
        # Fetch OHLC data
        provider = YFinanceDataProvider()
        logger.info(f"Fetching data from {start_date.date()} to {end_date.date()}")

        interval_map = {
            "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
            "1h": "1h", "4h": "4h", "1d": "1d", "1w": "1wk", "1mo": "1mo"
        }
        interval = interval_map.get(timeframe, "1d")

        data_points = provider.get_data(
            symbol=ticker,
            start_date=start_date,
            end_date=end_date,
            interval=interval
        )

        if not data_points:
            # Update status to ERROR using fresh session
            with SessionLocal() as error_db:
                error_dataset = error_db.query(Dataset).filter(Dataset.id == dataset_id).first()
                if error_dataset:
                    error_dataset.status = DatasetStatus.ERROR.value
                    error_dataset.error_message = f"No data available for {ticker}"
                    error_db.commit()
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"No data available for {ticker}"
            )

        # Convert to DataFrame
        df = pd.DataFrame([{
            'Date': dp.timestamp,
            'Open': dp.open,
            'High': dp.high,
            'Low': dp.low,
            'Close': dp.close,
            'Volume': dp.volume
        } for dp in data_points])
        df = df.sort_values('Date').reset_index(drop=True)
        logger.info(f"Fetched {len(df)} OHLC data points")

        # Apply technical indicators if configured
        if technical_indicators:
            logger.info(f"Applying {len(technical_indicators)} technical indicators...")
            try:
                indicators_dict = {}
                for indicator in technical_indicators:
                    indicator_type = indicator.get('type', indicator.get('name', 'unknown'))
                    indicator_name = indicator.get('name', f"{indicator_type}_{indicator.get('period', '')}")
                    indicators_dict[indicator_name] = indicator

                df = TechnicalIndicators.add_indicators_to_dataframe(df, indicators_dict)
                logger.info(f"Added technical indicators. DataFrame now has {len(df.columns)} columns")
            except Exception as e:
                logger.error(f"Error applying technical indicators: {e}")

        # Fetch and add sentiment features if configured
        if sentiment_config and sentiment_config.get('enabled'):
            logger.info("Fetching sentiment data...")
            try:
                sentiment_service = SentimentService()

                # Get news sources from config (supports multiple providers)
                news_sources = sentiment_config.get('news_sources', [])
                if not news_sources:
                    legacy_provider = sentiment_config.get('provider', 'fmp')
                    news_sources = [legacy_provider]

                logger.info(f"Fetching news from {len(news_sources)} source(s): {news_sources}")

                # Fetch from all configured news sources
                all_articles = []
                for source in news_sources:
                    news_provider = source.replace('_news', '').replace('_company', '').replace('_global', '')
                    try:
                        articles = sentiment_service.fetch_news_for_ticker(
                            ticker=ticker,
                            start_date=df['Date'].min() if hasattr(df['Date'].min(), 'to_pydatetime') else start_date,
                            end_date=df['Date'].max() if hasattr(df['Date'].max(), 'to_pydatetime') else end_date,
                            provider=news_provider,
                            enrich_content=sentiment_config.get('enrich_content', True)
                        )
                        if articles:
                            logger.info(f"Fetched {len(articles)} articles from {news_provider}")
                            all_articles.extend(articles)
                        else:
                            logger.warning(f"No articles from {news_provider}")
                    except Exception as e:
                        logger.warning(f"Error fetching from {news_provider}: {e}")

                if all_articles:
                    logger.info(f"Total articles from all sources: {len(all_articles)}")
                    df = sentiment_service.create_sentiment_features(df, all_articles)
                    logger.info(f"Added sentiment features from {len(all_articles)} articles")
                    # Store articles count in sentiment_config
                    sentiment_config['articles_count'] = len(all_articles)
                else:
                    logger.warning("No news articles found for sentiment analysis")
                    sentiment_config['articles_count'] = 0

            except Exception as e:
                logger.error(f"Error fetching sentiment: {e}")

        # Fetch and add fundamentals if configured
        if fundamentals_config and fundamentals_config.get('enabled'):
            logger.info("Fetching fundamentals data...")
            try:
                statement_types = fundamentals_config.get('statement_types')
                if statement_types:
                    lookback_statements = fundamentals_config.get('lookback_statements', 2)
                    providers = fundamentals_config.get('fundamentals_providers', ['yfinance'])

                    logger.info(f"Creating statement features: {statement_types} with {lookback_statements} periods")
                    df = FundamentalsService.create_statement_features(
                        df=df,
                        ticker=ticker,
                        statement_types=statement_types,
                        lookback_statements=lookback_statements,
                        providers=providers,
                        frequency='quarterly'
                    )
                    new_cols = [c for c in df.columns if c.startswith(('bs_', 'is_', 'cf_', 'earn_'))]
                    logger.info(f"Added {len(new_cols)} statement feature columns")
                else:
                    # Legacy mode
                    fundamentals = FundamentalsService.get_fundamental_data(ticker)
                    if fundamentals and fundamentals.get('current'):
                        current = fundamentals['current']
                        for key, value in current.items():
                            if value is not None:
                                df[f'fundamental_{key}'] = value

                # Fetch macro indicators if configured
                macro_indicators = fundamentals_config.get('macro_indicators', [])
                if macro_indicators:
                    logger.info(f"Fetching macro indicators: {macro_indicators}")
                    try:
                        macro_service = MacroService()
                        df = macro_service.integrate_macro_with_ohlc(df, macro_indicators)
                        for indicator in macro_indicators:
                            if indicator in df.columns:
                                df = df.rename(columns={indicator: f'macro_{indicator}'})
                                if f'{indicator}_yoy_change' in df.columns:
                                    df = df.rename(columns={f'{indicator}_yoy_change': f'macro_{indicator}_yoy_change'})
                        logger.info(f"Added macro columns for: {macro_indicators}")
                    except Exception as e:
                        logger.warning(f"Error fetching macro data: {e}")

            except Exception as e:
                logger.error(f"Error fetching fundamentals: {e}")

        # Save to file
        save_path = Path(file_path)
        save_path.parent.mkdir(exist_ok=True)
        df.to_csv(save_path, index=False)
        logger.info(f"Saved regenerated dataset to {save_path} with {len(df.columns)} columns")

        # =========================================================================
        # PHASE 3: Update DB with fresh session
        # =========================================================================
        with SessionLocal() as final_db:
            final_dataset = final_db.query(Dataset).filter(Dataset.id == dataset_id).first()
            if final_dataset:
                final_dataset.start_date = df['Date'].min()
                final_dataset.end_date = df['Date'].max()
                final_dataset.rows_count = len(df)
                final_dataset.status = DatasetStatus.READY.value
                final_dataset.error_message = None

                gen_config["regenerated_at"] = datetime.now().isoformat()
                final_dataset.generation_config = gen_config

                final_db.commit()
                final_db.refresh(final_dataset)

                logger.info(f"Dataset {dataset_id} regenerated successfully with {len(df)} rows and {len(df.columns)} columns")

                return final_dataset

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error regenerating dataset: {e}", exc_info=True)
        # Mark as error using fresh session
        try:
            with SessionLocal() as error_db:
                error_dataset = error_db.query(Dataset).filter(Dataset.id == dataset_id).first()
                if error_dataset:
                    error_dataset.status = DatasetStatus.ERROR.value
                    error_dataset.error_message = str(e)
                    error_db.commit()
        except Exception:
            pass
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to regenerate dataset: {str(e)}"
        )


@router.post("/{dataset_id}/duplicate", response_model=DatasetResponse, status_code=status.HTTP_201_CREATED)
async def duplicate_dataset(
    dataset_id: int,
    duplicate_request: DatasetDuplicate,
    db: Session = Depends(get_db)
):
    """
    Duplicate a dataset, optionally with a different ticker.

    Re-fetches data using the stored generation_config with optional new ticker.

    Args:
        dataset_id: ID of dataset to duplicate
        duplicate_request: Optional new ticker and name
        db: Database session

    Returns:
        Newly created dataset
    """
    try:
        # Get original dataset
        original = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not original:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dataset with ID {dataset_id} not found"
            )

        # Determine new ticker and name
        new_ticker = duplicate_request.new_ticker or original.ticker
        if duplicate_request.new_name:
            new_name = duplicate_request.new_name
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            new_name = f"{new_ticker}_{original.timeframe}_{timestamp}"

        logger.info(f"Duplicating dataset {dataset_id} to {new_name} with ticker {new_ticker}")

        # Fetch data using the original generation config
        provider = YFinanceDataProvider()

        # Use original dates from generation_config or dataset
        gen_config = original.generation_config or {}
        start_date = datetime.strptime(gen_config.get("original_start_date"), "%Y-%m-%d") if gen_config.get("original_start_date") else original.start_date
        end_date = datetime.strptime(gen_config.get("original_end_date"), "%Y-%m-%d") if gen_config.get("original_end_date") else original.end_date

        # Convert timeframe to interval
        interval_map = {
            "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
            "1h": "1h", "4h": "4h", "1d": "1d", "1w": "1wk", "1mo": "1mo"
        }
        interval = interval_map.get(original.timeframe, "1d")

        data_points = provider.get_data(
            symbol=new_ticker,
            start_date=start_date,
            end_date=end_date,
            interval=interval
        )

        if not data_points:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"No data available for {new_ticker}"
            )

        # Convert to DataFrame
        df = pd.DataFrame([{
            'Date': dp.timestamp,
            'Open': dp.open,
            'High': dp.high,
            'Low': dp.low,
            'Close': dp.close,
            'Volume': dp.volume
        } for dp in data_points])
        df = df.sort_values('Date').reset_index(drop=True)

        # Save to new file
        datasets_dir = Path("datasets")
        datasets_dir.mkdir(exist_ok=True)
        file_path = datasets_dir / f"{new_name}.csv"
        df.to_csv(file_path, index=False)

        # Build new generation config
        new_gen_config = {
            "data_provider": gen_config.get("data_provider", "yfinance"),
            "original_start_date": gen_config.get("original_start_date"),
            "original_end_date": gen_config.get("original_end_date"),
            "indicator_collection_id": gen_config.get("indicator_collection_id"),
            "duplicated_from": dataset_id,
            "created_at": datetime.now().isoformat()
        }

        # Create new database record
        new_dataset = Dataset(
            name=new_name,
            ticker=new_ticker,
            timeframe=original.timeframe,
            start_date=df['Date'].min(),
            end_date=df['Date'].max(),
            rows_count=len(df),
            technical_indicators=original.technical_indicators,
            fundamentals_config=original.fundamentals_config,
            sentiment_config=original.sentiment_config,
            generation_config=new_gen_config,
            normalization_buffer_pct=original.normalization_buffer_pct,
            file_path=str(file_path)
        )

        db.add(new_dataset)
        db.commit()
        db.refresh(new_dataset)

        logger.info(f"Created duplicate dataset with ID {new_dataset.id}")
        return new_dataset

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error duplicating dataset: {e}", exc_info=True)
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to duplicate dataset: {str(e)}"
        )


@router.put("/{dataset_id}", response_model=DatasetResponse)
async def update_dataset(
    dataset_id: int,
    dataset_update: DatasetUpdate,
    db: Session = Depends(get_db)
):
    """
    Update dataset properties and regenerate the dataset.

    Always regenerates the dataset with the updated configuration to ensure
    all indicators, sentiment, and fundamentals are applied.

    Args:
        dataset_id: Dataset ID to update
        dataset_update: Fields to update
        db: Database session

    Returns:
        Updated dataset
    """
    try:
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not dataset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dataset with ID {dataset_id} not found"
            )

        logger.info(f"Updating and regenerating dataset {dataset_id}")
        logger.info(f"Update request: start_date={dataset_update.start_date}, end_date={dataset_update.end_date}")
        logger.info(f"Current dataset: start_date={dataset.start_date}, end_date={dataset.end_date}")

        # Update simple fields first
        if dataset_update.name:
            dataset.name = dataset_update.name

        if dataset_update.normalization_buffer_pct is not None:
            dataset.normalization_buffer_pct = dataset_update.normalization_buffer_pct

        if dataset_update.technical_indicators is not None:
            dataset.technical_indicators = dataset_update.technical_indicators

        if dataset_update.sentiment_config is not None:
            dataset.sentiment_config = dataset_update.sentiment_config

        if dataset_update.fundamentals_config is not None:
            dataset.fundamentals_config = dataset_update.fundamentals_config

        # Update ticker/timeframe if provided
        new_ticker = dataset_update.ticker or dataset.ticker
        new_timeframe = dataset_update.timeframe or dataset.timeframe
        dataset.ticker = new_ticker
        dataset.timeframe = new_timeframe

        # Parse dates - copy dict to ensure SQLAlchemy detects changes
        gen_config = dict(dataset.generation_config) if dataset.generation_config else {}
        if dataset_update.start_date:
            start_date = datetime.strptime(dataset_update.start_date, "%Y-%m-%d")
            gen_config["original_start_date"] = dataset_update.start_date
        elif gen_config.get("original_start_date"):
            start_date = datetime.strptime(gen_config["original_start_date"], "%Y-%m-%d")
        else:
            start_date = dataset.start_date

        if dataset_update.end_date:
            end_date = datetime.strptime(dataset_update.end_date, "%Y-%m-%d")
            gen_config["original_end_date"] = dataset_update.end_date
        elif gen_config.get("original_end_date"):
            end_date = datetime.strptime(gen_config["original_end_date"], "%Y-%m-%d")
        else:
            end_date = dataset.end_date

        # Save updated generation_config with new dates
        dataset.generation_config = gen_config

        # Set status to BUILDING
        dataset.status = DatasetStatus.BUILDING.value
        dataset.error_message = None
        db.commit()

        # Fetch OHLC data
        provider = YFinanceDataProvider()
        interval_map = {
            "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
            "1h": "1h", "4h": "4h", "1d": "1d", "1w": "1wk", "1mo": "1mo"
        }
        interval = interval_map.get(new_timeframe, "1d")

        logger.info(f"Fetching data for {new_ticker} from {start_date.date()} to {end_date.date()}")

        data_points = provider.get_data(
            symbol=new_ticker,
            start_date=start_date,
            end_date=end_date,
            interval=interval
        )

        if not data_points:
            dataset.status = DatasetStatus.ERROR.value
            dataset.error_message = f"No data available for {new_ticker}"
            db.commit()
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"No data available for {new_ticker}"
            )

        # Convert to DataFrame
        df = pd.DataFrame([{
            'Date': dp.timestamp,
            'Open': dp.open,
            'High': dp.high,
            'Low': dp.low,
            'Close': dp.close,
            'Volume': dp.volume
        } for dp in data_points])
        df = df.sort_values('Date').reset_index(drop=True)
        logger.info(f"Fetched {len(df)} OHLC data points")

        # Apply technical indicators if configured
        if dataset.technical_indicators:
            logger.info(f"Applying {len(dataset.technical_indicators)} technical indicators...")
            try:
                indicators_dict = {}
                for indicator in dataset.technical_indicators:
                    indicator_type = indicator.get('type', indicator.get('name', 'unknown'))
                    indicator_name = indicator.get('name', f"{indicator_type}_{indicator.get('period', '')}")
                    indicators_dict[indicator_name] = indicator

                df = TechnicalIndicators.add_indicators_to_dataframe(df, indicators_dict)
                logger.info(f"Added technical indicators. DataFrame now has {len(df.columns)} columns")
            except Exception as e:
                logger.error(f"Error applying technical indicators: {e}")

        # Fetch and add sentiment features if configured
        if dataset.sentiment_config and dataset.sentiment_config.get('enabled'):
            logger.info("Fetching sentiment data...")
            logger.debug(f"Sentiment config: {dataset.sentiment_config}")
            try:
                sentiment_service = SentimentService()

                # Get news sources from config (supports multiple providers)
                news_sources = dataset.sentiment_config.get('news_sources', [])
                if not news_sources:
                    legacy_provider = dataset.sentiment_config.get('provider', 'fmp')
                    news_sources = [legacy_provider]

                logger.info(f"Fetching news from {len(news_sources)} source(s): {news_sources}")

                all_articles = []
                for source in news_sources:
                    provider = source.replace('_news', '').replace('_company', '').replace('_global', '')
                    try:
                        logger.debug(f"Fetching news from provider: {provider}")
                        articles = sentiment_service.fetch_news_for_ticker(
                            ticker=new_ticker,
                            start_date=df['Date'].min() if hasattr(df['Date'].min(), 'to_pydatetime') else start_date,
                            end_date=df['Date'].max() if hasattr(df['Date'].max(), 'to_pydatetime') else end_date,
                            provider=provider,
                            enrich_content=dataset.sentiment_config.get('enrich_content', True)
                        )
                        if articles:
                            logger.info(f"Fetched {len(articles)} articles from {provider}")
                            all_articles.extend(articles)
                        else:
                            logger.warning(f"No articles from {provider}")
                    except Exception as e:
                        logger.warning(f"Error fetching from {provider}: {e}")

                if all_articles:
                    logger.info(f"Total articles from all sources: {len(all_articles)}")
                    df = sentiment_service.create_sentiment_features(df, all_articles)
                    logger.info(f"Added sentiment features from {len(all_articles)} articles")
                    # Store articles count - copy dict to ensure SQLAlchemy detects change
                    updated_sentiment_config = dict(dataset.sentiment_config) if dataset.sentiment_config else {}
                    updated_sentiment_config['articles_count'] = len(all_articles)
                    dataset.sentiment_config = updated_sentiment_config
                else:
                    logger.warning("No news articles found for sentiment analysis from any source")
                    updated_sentiment_config = dict(dataset.sentiment_config) if dataset.sentiment_config else {}
                    updated_sentiment_config['articles_count'] = 0
                    dataset.sentiment_config = updated_sentiment_config

            except Exception as e:
                logger.error(f"Error fetching sentiment: {e}")

        # Fetch and add fundamentals if configured
        if dataset.fundamentals_config and dataset.fundamentals_config.get('enabled'):
            logger.info("Fetching fundamentals data...")
            logger.debug(f"Fundamentals config: {dataset.fundamentals_config}")
            try:
                fundamentals_config = dataset.fundamentals_config

                # Check if using new statement-based config
                statement_types = fundamentals_config.get('statement_types')
                if statement_types:
                    # Use new statement-based features with lookback
                    lookback_statements = fundamentals_config.get('lookback_statements', 2)
                    providers = fundamentals_config.get('fundamentals_providers', ['yfinance'])

                    logger.info(f"Creating statement features: types={statement_types}, lookback={lookback_statements}, providers={providers}")

                    df = FundamentalsService.create_statement_features(
                        df=df,
                        ticker=new_ticker,
                        statement_types=statement_types,
                        lookback_statements=lookback_statements,
                        providers=providers,
                        frequency='quarterly'
                    )

                    # Count how many statement columns were added
                    statement_cols = [c for c in df.columns if any(c.startswith(p + '_q') for p in ['bs', 'is', 'cf', 'earn'])]
                    logger.info(f"Added {len(statement_cols)} statement feature columns")
                else:
                    # Legacy mode: add current fundamentals as constant columns
                    fundamentals = FundamentalsService.get_fundamental_data(new_ticker)

                    if fundamentals and fundamentals.get('current'):
                        current = fundamentals['current']
                        added_fundamentals = []
                        for key, value in current.items():
                            if value is not None:
                                df[f'fundamental_{key}'] = value
                                added_fundamentals.append(key)
                        logger.info(f"Added {len(added_fundamentals)} fundamental columns: {added_fundamentals}")
                    else:
                        logger.warning("No fundamentals data available")

                # Fetch macro indicators if configured
                macro_indicators = dataset.fundamentals_config.get('macro_indicators', [])
                if macro_indicators:
                    logger.info(f"Fetching macro indicators: {macro_indicators}")
                    try:
                        macro_service = MacroService()
                        df = macro_service.integrate_macro_with_ohlc(df, macro_indicators)
                        # Rename columns to have macro_ prefix
                        for indicator in macro_indicators:
                            if indicator in df.columns:
                                df = df.rename(columns={indicator: f'macro_{indicator}'})
                                if f'{indicator}_yoy_change' in df.columns:
                                    df = df.rename(columns={f'{indicator}_yoy_change': f'macro_{indicator}_yoy_change'})
                        logger.info(f"Added macro columns for: {macro_indicators}")
                    except Exception as e:
                        logger.warning(f"Error fetching macro data: {e}")

            except Exception as e:
                logger.error(f"Error fetching fundamentals: {e}")

        # Save to file
        file_path = Path(dataset.file_path)
        file_path.parent.mkdir(exist_ok=True)
        df.to_csv(file_path, index=False)
        logger.info(f"Saved updated dataset to {file_path} with {len(df.columns)} columns")

        # Update dataset record
        dataset.start_date = df['Date'].min()
        dataset.end_date = df['Date'].max()
        dataset.rows_count = len(df)
        dataset.status = DatasetStatus.READY.value
        dataset.error_message = None

        # Update generation config
        gen_config["updated_at"] = datetime.now().isoformat()
        dataset.generation_config = gen_config

        db.commit()
        db.refresh(dataset)

        logger.info(f"Dataset {dataset_id} updated and regenerated successfully with {len(df)} rows and {len(df.columns)} columns")
        return dataset

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating dataset: {e}", exc_info=True)
        # Mark as error
        try:
            dataset.status = DatasetStatus.ERROR.value
            dataset.error_message = str(e)
            db.commit()
        except Exception:
            db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update dataset: {str(e)}"
        )


@router.post("/{dataset_id}/calculate-indicators")
async def calculate_multi_timeframe_indicators(
    dataset_id: int,
    timeframes: List[str] = None,
    indicators: dict = None,
    db: Session = Depends(get_db)
):
    """
    Calculate multi-timeframe technical indicators for a dataset.

    This endpoint fetches data at multiple timeframes (15m, 1h, 4h, D1) and calculates
    technical indicators for each, then merges them into the dataset.

    Args:
        dataset_id: Dataset ID to add indicators to
        timeframes: List of timeframes to calculate (default: ["15m", "1h", "4h", "1d"])
        indicators: Custom indicator configuration (default: uses DEFAULT_INDICATORS)
        db: Database session

    Returns:
        Updated dataset with indicator columns
    """
    try:
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()

        if not dataset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dataset with ID {dataset_id} not found"
            )

        # Use default timeframes if not specified
        if timeframes is None:
            timeframes = SUPPORTED_TIMEFRAMES

        # Validate timeframes
        invalid_timeframes = [tf for tf in timeframes if tf not in TIMEFRAME_INTERVAL_MAP]
        if invalid_timeframes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid timeframes: {invalid_timeframes}. Supported: {list(TIMEFRAME_INTERVAL_MAP.keys())}"
            )

        # Use default indicators if not specified
        if indicators is None:
            indicators = DEFAULT_INDICATORS

        # Load the dataset
        file_path = Path(dataset.file_path)
        if not file_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dataset file not found: {file_path}"
            )

        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'])

        logger.info(f"Calculating multi-timeframe indicators for dataset {dataset_id}")
        logger.info(f"Timeframes: {timeframes}, Indicators: {list(indicators.keys())}")

        # Get base timeframe data range
        start_date = df['Date'].min()
        end_date = df['Date'].max()

        # Initialize provider
        provider = YFinanceDataProvider()

        # Calculate indicators for each timeframe
        all_indicators = {}

        for tf in timeframes:
            logger.info(f"Fetching data for timeframe {tf}")

            # Fetch data at this timeframe
            interval = TIMEFRAME_INTERVAL_MAP[tf]
            try:
                data_points = provider.get_data(
                    symbol=dataset.ticker,
                    start_date=start_date,
                    end_date=end_date,
                    interval=interval
                )

                if not data_points:
                    logger.warning(f"No data available for timeframe {tf}")
                    continue

                # Convert to DataFrame
                tf_df = pd.DataFrame([{
                    'Date': dp.timestamp,
                    'Open': dp.open,
                    'High': dp.high,
                    'Low': dp.low,
                    'Close': dp.close,
                    'Volume': dp.volume
                } for dp in data_points])

                tf_df = tf_df.sort_values('Date').reset_index(drop=True)

                # Prepare indicators config with timeframe prefix
                tf_indicators = {}
                for ind_name, ind_config in indicators.items():
                    prefixed_name = f"{ind_name}_{tf}"
                    tf_indicators[prefixed_name] = ind_config

                # Calculate indicators for this timeframe
                tf_df_with_indicators = TechnicalIndicators.add_indicators_to_dataframe(
                    tf_df, tf_indicators
                )

                # Extract indicator columns (exclude OHLCV)
                ohlcv_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
                indicator_cols = [col for col in tf_df_with_indicators.columns if col not in ohlcv_cols]

                # Store indicator data with dates
                for col in indicator_cols:
                    all_indicators[col] = tf_df_with_indicators[['Date', col]].copy()

                logger.info(f"Calculated {len(indicator_cols)} indicators for timeframe {tf}")

            except Exception as e:
                logger.warning(f"Failed to calculate indicators for timeframe {tf}: {e}")
                continue

        # Merge indicators into main dataset
        result_df = df.copy()

        for ind_name, ind_df in all_indicators.items():
            # Merge on Date using forward fill for different timeframe resolutions
            ind_df = ind_df.rename(columns={ind_name: ind_name})
            ind_df = ind_df.set_index('Date')

            # Resample to match main dataset timeframe and forward fill
            result_df = result_df.set_index('Date') if 'Date' not in result_df.index.names else result_df

            # Align indicator data to main dataset dates
            aligned_indicator = ind_df.reindex(result_df.index, method='ffill')
            result_df[ind_name] = aligned_indicator[ind_name]
            result_df = result_df.reset_index()

        # Save updated dataset
        result_df.to_csv(file_path, index=False)

        # Update dataset metadata
        indicator_config = dataset.technical_indicators or {}
        indicator_config['multi_timeframe'] = {
            'timeframes': timeframes,
            'indicators': list(indicators.keys()),
            'calculated_at': datetime.now().isoformat()
        }
        dataset.technical_indicators = indicator_config
        db.commit()
        db.refresh(dataset)

        logger.info(f"Successfully calculated multi-timeframe indicators for dataset {dataset_id}")

        return {
            "dataset_id": dataset_id,
            "timeframes": timeframes,
            "indicators_added": list(all_indicators.keys()),
            "total_columns": len(result_df.columns),
            "rows": len(result_df),
            "message": f"Successfully calculated {len(all_indicators)} indicator columns across {len(timeframes)} timeframes"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error calculating multi-timeframe indicators: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to calculate indicators: {str(e)}"
        )


@router.get("/supported-indicators")
async def get_supported_indicators():
    """
    Get list of supported technical indicators and timeframes.

    Returns:
        Dictionary with supported timeframes and indicators
    """
    return {
        "timeframes": SUPPORTED_TIMEFRAMES,
        "indicators": DEFAULT_INDICATORS,
        "description": "Multi-timeframe technical indicators for financial datasets"
    }


@router.post("/{dataset_id}/calculate-fundamentals")
async def calculate_fundamental_features(
    dataset_id: int,
    metrics: List[str] = None,
    db: Session = Depends(get_db)
):
    """
    Calculate fundamental-derived features for a dataset.

    Creates features for each fundamental metric:
    - days_to_last_{metric}: Days since last reported value
    - last_{metric}: Most recent value
    - last_{metric}_percent: Percent change from previous period
    - days_to_next_{metric}: Estimated days to next report
    - next_{metric}_forecast: Simple forecast based on trend

    Args:
        dataset_id: Dataset ID
        metrics: List of metrics (default: ['fcf', 'pe', 'eps', 'revenue'])
        db: Database session

    Returns:
        Updated dataset with fundamental features
    """
    try:
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()

        if not dataset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dataset with ID {dataset_id} not found"
            )

        # Default metrics
        if metrics is None:
            metrics = ['fcf', 'pe', 'eps', 'revenue', 'de', 'roe']

        # Load dataset
        file_path = Path(dataset.file_path)
        if not file_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dataset file not found: {file_path}"
            )

        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'])

        logger.info(f"Calculating fundamental features for dataset {dataset_id}, ticker: {dataset.ticker}")

        # Calculate fundamental features
        result_df = FundamentalsService.create_fundamental_features(
            df, dataset.ticker, metrics
        )

        # Save updated dataset
        result_df.to_csv(file_path, index=False)

        # Count added columns
        added_columns = [col for col in result_df.columns if col not in df.columns]

        # Update dataset metadata
        fundamentals_config = dataset.fundamentals_config or {}
        fundamentals_config['calculated_metrics'] = metrics
        fundamentals_config['calculated_at'] = datetime.now().isoformat()
        fundamentals_config['feature_columns'] = added_columns
        dataset.fundamentals_config = fundamentals_config
        db.commit()
        db.refresh(dataset)

        logger.info(f"Successfully calculated {len(added_columns)} fundamental features for dataset {dataset_id}")

        return {
            "dataset_id": dataset_id,
            "ticker": dataset.ticker,
            "metrics": metrics,
            "features_added": added_columns,
            "total_columns": len(result_df.columns),
            "rows": len(result_df),
            "message": f"Successfully calculated {len(added_columns)} fundamental features"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error calculating fundamental features: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to calculate fundamental features: {str(e)}"
        )


@router.get("/{dataset_id}/fundamentals")
async def get_fundamental_data(dataset_id: int, db: Session = Depends(get_db)):
    """
    Get fundamental data for the ticker in a dataset.

    Args:
        dataset_id: Dataset ID
        db: Database session

    Returns:
        Fundamental data for the ticker
    """
    try:
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()

        if not dataset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dataset with ID {dataset_id} not found"
            )

        fund_data = FundamentalsService.get_fundamental_data(dataset.ticker)

        return {
            "dataset_id": dataset_id,
            "ticker": dataset.ticker,
            "fundamentals": fund_data
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting fundamental data: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get fundamental data: {str(e)}"
        )


@router.post("/{dataset_id}/calculate-macro")
async def calculate_macro_features(
    dataset_id: int,
    indicators: List[str] = None,
    include_yield_curve: bool = True,
    db: Session = Depends(get_db)
):
    """
    Integrate macro economic data with OHLC dataset using forward-fill.

    Fetches macroeconomic indicators (interest rates, GDP, inflation, etc.)
    and aligns them with the dataset's time series using forward-fill.

    Args:
        dataset_id: Dataset ID
        indicators: List of indicators (default: all available)
        include_yield_curve: Include yield curve features (default: True)
        db: Database session

    Returns:
        Updated dataset with macro features
    """
    try:
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()

        if not dataset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dataset with ID {dataset_id} not found"
            )

        # Load dataset
        file_path = Path(dataset.file_path)
        if not file_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dataset file not found: {file_path}"
            )

        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'])

        logger.info(f"Calculating macro features for dataset {dataset_id}")

        # Initialize macro service
        macro_service = MacroService()

        # Integrate macro data
        result_df = macro_service.integrate_macro_with_ohlc(df, indicators)

        # Add yield curve features if requested
        if include_yield_curve:
            result_df = macro_service.create_yield_curve_features(result_df)

        # Save updated dataset
        result_df.to_csv(file_path, index=False)

        # Count added columns
        added_columns = [col for col in result_df.columns if col not in df.columns]

        logger.info(f"Successfully calculated {len(added_columns)} macro features for dataset {dataset_id}")

        return {
            "dataset_id": dataset_id,
            "indicators": indicators or list(MacroService.MACRO_INDICATORS.keys()),
            "include_yield_curve": include_yield_curve,
            "features_added": added_columns,
            "total_columns": len(result_df.columns),
            "rows": len(result_df),
            "message": f"Successfully calculated {len(added_columns)} macro features"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error calculating macro features: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to calculate macro features: {str(e)}"
        )


@router.get("/supported-macro-indicators")
async def get_supported_macro_indicators():
    """
    Get list of supported macroeconomic indicators.

    Returns:
        Dictionary of available macro indicators with metadata
    """
    return {
        "indicators": MacroService.get_supported_indicators(),
        "description": "Macroeconomic indicators from FRED (Federal Reserve Economic Data)"
    }


@router.post("/{dataset_id}/calculate-sentiment")
async def calculate_sentiment_features(
    dataset_id: int,
    db: Session = Depends(get_db)
):
    """
    Calculate sentiment features for a dataset.

    Fetches news articles for the ticker, runs sentiment analysis using
    Transformers (FinBERT), and creates aggregated sentiment features:
    - news_1d_positive_short, news_1d_positive_medium, news_1d_positive_long
    - news_1w_negative_short, news_1w_negative_medium, news_1w_negative_long
    - And combinations for 1d, 1w, 1m, 6m periods

    Args:
        dataset_id: Dataset ID
        db: Database session

    Returns:
        Updated dataset with sentiment features
    """
    try:
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()

        if not dataset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dataset with ID {dataset_id} not found"
            )

        # Load dataset
        file_path = Path(dataset.file_path)
        if not file_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dataset file not found: {file_path}"
            )

        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'])

        logger.info(f"Calculating sentiment features for dataset {dataset_id}, ticker: {dataset.ticker}")

        # Initialize sentiment service
        sentiment_service = SentimentService()

        # Get date range
        start_date = df['Date'].min()
        end_date = df['Date'].max()

        # Fetch news articles
        news_articles = sentiment_service.fetch_news_for_ticker(
            dataset.ticker, start_date, end_date
        )

        # Analyze sentiment
        analyzed_articles = sentiment_service.analyze_news_articles(news_articles)

        # Create sentiment features
        result_df = sentiment_service.create_sentiment_features(df, analyzed_articles)

        # Save updated dataset
        result_df.to_csv(file_path, index=False)

        # Count added columns
        added_columns = [col for col in result_df.columns if col not in df.columns]

        # Update dataset metadata
        sentiment_config = dataset.sentiment_config or {}
        sentiment_config['calculated_at'] = datetime.now().isoformat()
        sentiment_config['articles_analyzed'] = len(analyzed_articles)
        sentiment_config['feature_columns'] = added_columns
        dataset.sentiment_config = sentiment_config
        db.commit()
        db.refresh(dataset)

        logger.info(f"Successfully calculated {len(added_columns)} sentiment features for dataset {dataset_id}")

        return {
            "dataset_id": dataset_id,
            "ticker": dataset.ticker,
            "articles_analyzed": len(analyzed_articles),
            "features_added": added_columns,
            "total_columns": len(result_df.columns),
            "rows": len(result_df),
            "message": f"Successfully calculated {len(added_columns)} sentiment features from {len(analyzed_articles)} articles"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error calculating sentiment features: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to calculate sentiment features: {str(e)}"
        )


@router.get("/sentiment-feature-descriptions")
async def get_sentiment_feature_descriptions():
    """
    Get descriptions for all sentiment features.

    Returns:
        Dictionary mapping feature names to descriptions
    """
    return {
        "features": SentimentService.get_feature_descriptions(),
        "lookback_periods": SentimentService.LOOKBACK_PERIODS,
        "sentiment_categories": SentimentService.SENTIMENT_CATEGORIES,
        "impact_timeframes": SentimentService.IMPACT_TIMEFRAMES,
        "description": "Aggregated news sentiment features for ML model training"
    }


@router.post("/{dataset_id}/analyze-news")
async def analyze_news_for_dataset(
    dataset_id: int,
    db: Session = Depends(get_db)
):
    """
    Fetch and analyze news articles for a dataset's ticker.

    Args:
        dataset_id: Dataset ID
        db: Database session

    Returns:
        Analyzed news articles with sentiment
    """
    try:
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()

        if not dataset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dataset with ID {dataset_id} not found"
            )

        # Load dataset to get date range
        file_path = Path(dataset.file_path)
        if not file_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dataset file not found: {file_path}"
            )

        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'])

        start_date = df['Date'].min()
        end_date = df['Date'].max()

        # Fetch and analyze news
        sentiment_service = SentimentService()
        news_articles = sentiment_service.fetch_news_for_ticker(
            dataset.ticker, start_date, end_date
        )
        analyzed_articles = sentiment_service.analyze_news_articles(news_articles)

        # Convert dates to strings for JSON serialization
        for article in analyzed_articles:
            if isinstance(article.get('date'), datetime):
                article['date'] = article['date'].isoformat()

        return {
            "dataset_id": dataset_id,
            "ticker": dataset.ticker,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "article_count": len(analyzed_articles),
            "articles": analyzed_articles
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing news: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to analyze news: {str(e)}"
        )


# ============= Multi-Dataset Endpoints =============

@router.post("/validate-compatibility")
async def validate_dataset_compatibility(
    dataset_ids: List[int],
    db: Session = Depends(get_db)
):
    """
    Validate compatibility of multiple datasets for combined training.

    Checks:
    - All datasets exist
    - Timeframes match
    - Compatible date ranges
    - Compatible feature columns

    Args:
        dataset_ids: List of dataset IDs to validate
        db: Database session

    Returns:
        Compatibility report with warnings and combined statistics
    """
    if len(dataset_ids) < 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least 2 datasets required for compatibility check"
        )

    datasets = []
    for dataset_id in dataset_ids:
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not dataset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dataset with ID {dataset_id} not found"
            )
        datasets.append(dataset)

    # Check timeframe compatibility
    timeframes = set(d.timeframe for d in datasets)
    timeframe_compatible = len(timeframes) == 1

    # Check tickers
    tickers = list(set(d.ticker for d in datasets))

    # Load and analyze features
    feature_sets = []
    date_ranges = []
    total_rows = 0

    for dataset in datasets:
        file_path = Path(dataset.file_path)
        if file_path.exists():
            df = pd.read_csv(file_path)
            feature_sets.append(set(df.columns.tolist()))
            total_rows += len(df)

            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                date_ranges.append({
                    'dataset_id': dataset.id,
                    'ticker': dataset.ticker,
                    'start': df['Date'].min().isoformat(),
                    'end': df['Date'].max().isoformat(),
                    'rows': len(df)
                })

    # Find common features
    if feature_sets:
        common_features = set.intersection(*feature_sets)
        all_features = set.union(*feature_sets)
        missing_features = {ds.id: list(all_features - fs) for ds, fs in zip(datasets, feature_sets)}
    else:
        common_features = set()
        all_features = set()
        missing_features = {}

    # Build warnings
    warnings = []
    if not timeframe_compatible:
        warnings.append(f"Timeframes do not match: {', '.join(timeframes)}")
    if len(tickers) > 1:
        warnings.append(f"Multiple tickers: {', '.join(tickers)} - training will handle each separately")
    if len(common_features) < len(all_features):
        warnings.append(f"{len(all_features) - len(common_features)} features not present in all datasets")

    return {
        "compatible": timeframe_compatible and len(common_features) > 5,
        "dataset_count": len(datasets),
        "tickers": tickers,
        "timeframe_match": timeframe_compatible,
        "timeframe": list(timeframes)[0] if timeframe_compatible else None,
        "common_features": len(common_features),
        "total_features": len(all_features),
        "total_rows": total_rows,
        "date_ranges": date_ranges,
        "missing_features_by_dataset": missing_features,
        "warnings": warnings
    }


@router.post("/combine-preview")
async def combine_datasets_preview(
    dataset_ids: List[int],
    db: Session = Depends(get_db)
):
    """
    Preview combined statistics for multiple datasets.

    Args:
        dataset_ids: List of dataset IDs to combine
        db: Database session

    Returns:
        Combined statistics and chronological overview
    """
    if len(dataset_ids) < 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least 2 datasets required"
        )

    datasets = []
    for dataset_id in dataset_ids:
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not dataset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dataset with ID {dataset_id} not found"
            )
        datasets.append(dataset)

    # Load and combine data
    combined_stats = []
    all_dates = []
    total_rows = 0

    for dataset in datasets:
        file_path = Path(dataset.file_path)
        if file_path.exists():
            df = pd.read_csv(file_path)
            df['Date'] = pd.to_datetime(df['Date'])

            stats = {
                'dataset_id': dataset.id,
                'name': dataset.name,
                'ticker': dataset.ticker,
                'timeframe': dataset.timeframe,
                'rows': len(df),
                'start_date': df['Date'].min().isoformat(),
                'end_date': df['Date'].max().isoformat(),
                'columns': len(df.columns)
            }

            # Add numeric column stats
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            if 'Close' in df.columns:
                stats['price_range'] = {
                    'min': float(df['Close'].min()),
                    'max': float(df['Close'].max()),
                    'mean': float(df['Close'].mean())
                }

            combined_stats.append(stats)
            all_dates.extend(df['Date'].tolist())
            total_rows += len(df)

    # Calculate combined timeline
    if all_dates:
        all_dates = sorted(set(all_dates))
        timeline = {
            'start': all_dates[0].isoformat(),
            'end': all_dates[-1].isoformat(),
            'unique_dates': len(all_dates)
        }
    else:
        timeline = None

    return {
        "datasets": combined_stats,
        "combined": {
            "total_datasets": len(datasets),
            "total_rows": total_rows,
            "timeline": timeline,
            "unique_tickers": list(set(d.ticker for d in datasets))
        }
    }


@router.post("/check-timeframe-match")
async def check_timeframe_match(
    dataset_ids: List[int],
    db: Session = Depends(get_db)
):
    """
    Check if datasets have matching timeframes.

    Args:
        dataset_ids: List of dataset IDs to check
        db: Database session

    Returns:
        Timeframe compatibility status
    """
    if len(dataset_ids) < 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least 2 datasets required"
        )

    timeframe_info = []
    for dataset_id in dataset_ids:
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not dataset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dataset with ID {dataset_id} not found"
            )
        timeframe_info.append({
            'dataset_id': dataset.id,
            'name': dataset.name,
            'ticker': dataset.ticker,
            'timeframe': dataset.timeframe
        })

    timeframes = set(d['timeframe'] for d in timeframe_info)

    return {
        "match": len(timeframes) == 1,
        "common_timeframe": list(timeframes)[0] if len(timeframes) == 1 else None,
        "timeframes_found": list(timeframes),
        "datasets": timeframe_info,
        "message": (
            f"All datasets use {list(timeframes)[0]} timeframe"
            if len(timeframes) == 1
            else f"Timeframe mismatch: {', '.join(timeframes)}"
        )
    }


@router.get("/{dataset_id}/sentiment")
async def get_dataset_sentiment(
    dataset_id: int,
    provider: str = Query("fmp", description="News provider (fmp, alphavantage, google, finnhub, alpaca)"),
    db: Session = Depends(get_db)
):
    """
    Get sentiment markers for a dataset.

    Fetches real news articles for the dataset's ticker and date range,
    analyzes sentiment, and returns markers for chart visualization.

    Args:
        dataset_id: Dataset ID
        provider: News provider to use (fmp, alphavantage, google, finnhub, alpaca)
        db: Database session

    Returns:
        List of sentiment markers with date, sentiment, score, headline, source
    """
    try:
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not dataset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dataset with ID {dataset_id} not found"
            )

        logger.info(f"Fetching sentiment for dataset {dataset_id} ({dataset.ticker}) using {provider}")

        sentiment_service = SentimentService()

        # Fetch news articles
        articles = sentiment_service.fetch_news_for_ticker(
            ticker=dataset.ticker,
            start_date=dataset.start_date,
            end_date=dataset.end_date,
            provider=provider
        )

        # Analyze sentiment for each article
        analyzed = sentiment_service.analyze_news_articles(articles)

        # Convert to markers for chart
        markers = []
        for article in analyzed:
            # Handle date conversion
            article_date = article.get('date', '')
            if isinstance(article_date, datetime):
                date_str = article_date.isoformat()
            else:
                date_str = str(article_date)

            markers.append({
                "date": date_str,
                "sentiment": article.get('sentiment', 'neutral'),
                "score": article.get('sentiment_score', 0.5),
                "headline": article.get('title', ''),
                "source": article.get('source', 'Unknown'),
                "impact_timeframe": article.get('impact_timeframe', 'medium')
            })

        logger.info(f"Returning {len(markers)} sentiment markers for dataset {dataset_id}")

        return {
            "dataset_id": dataset_id,
            "ticker": dataset.ticker,
            "provider": provider,
            "markers": markers,
            "total_count": len(markers),
            "is_mock": any(m['source'].startswith('Mock') for m in markers)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching sentiment for dataset {dataset_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch sentiment: {str(e)}"
        )
