from enum import Enum
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


class TimeInterval(str, Enum):
    """
    Standard timeframe intervals for market data.
    
    Maps user-friendly names to market data provider API formats.
    """
    # Minutes
    M1 = "1m"   # 1 minute
    M5 = "5m"   # 5 minutes
    M15 = "15m" # 15 minutes
    M30 = "30m" # 30 minutes
    
    # Hours
    H1 = "1h"   # 1 hour
    H4 = "4h"   # 4 hours
    
    # Days/Weeks/Months
    D1 = "1d"   # 1 day (daily)
    W1 = "1wk"  # 1 week (weekly)
    MO1 = "1mo" # 1 month (monthly)
    
    @classmethod
    def to_yfinance_interval(cls, interval: str) -> str:
        """
        Convert interval to yfinance-compatible format.
        
        Args:
            interval: TimeInterval value or string
        
        Returns:
            yfinance-compatible interval string
        
        Note: Most intervals are passed through as-is. 
        If a specific provider doesn't support an interval, 
        the provider implementation should handle the conversion.
        """
        # Return interval as-is - let provider handle any necessary conversions
        return interval
    
    
    @classmethod
    def get_all_intervals(cls) -> list:
        """Get list of all supported intervals."""
        return [member.value for member in cls]


@dataclass
class MarketDataPoint:
    """
    Represents a single market data point with OHLC data.
    
    Attributes:
        symbol: The ticker symbol (e.g., 'AAPL', 'MSFT')
        timestamp: The datetime of the data point
        open: Opening price
        high: Highest price
        low: Lowest price
        close: Closing price
        volume: Trading volume
        interval: The timeframe interval (e.g., '1d', '1h', '5m')
    """
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    interval: str = '1d'
    
    def __repr__(self):
        return (f"MarketDataPoint(symbol={self.symbol}, "
                f"timestamp={self.timestamp.strftime('%Y-%m-%d %H:%M')}, "
                f"O={self.open:.2f}, H={self.high:.2f}, L={self.low:.2f}, "
                f"C={self.close:.2f}, V={self.volume:.0f})")



class InstrumentType(str, Enum):
    STOCK = "stock"
    CRYPTO = "crypto"


class TimeHorizon(str, Enum):
    SHORT_TERM = "SHORT_TERM"
    MEDIUM_TERM = "MEDIUM_TERM"
    LONG_TERM = "LONG_TERM"

class ExpertEventRuleType(str, Enum):
    TRADING_RECOMMENDATION_RULE = "trading_recommendation_rule"

class AnalysisUseCase(str, Enum):
    ENTER_MARKET = "enter_market"
    OPEN_POSITIONS = "open_positions"
    