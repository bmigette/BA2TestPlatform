"""
Strategy model for storing trading strategies with conditions
"""

from sqlalchemy import Column, Integer, String, DateTime, JSON, Float, Boolean, Text
from sqlalchemy.sql import func
from .database import Base


class Strategy(Base):
    """Trading strategy with entry/exit conditions"""

    __tablename__ = "strategies"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)

    # Auto-computed from conditions for compatibility matching
    required_fields = Column(JSON, nullable=True)

    # Condition trees (JSON)
    # Deprecated - kept for backwards compatibility
    entry_conditions = Column(JSON, nullable=True)
    # New separate buy/sell conditions
    buy_entry_conditions = Column(JSON, nullable=True)
    sell_entry_conditions = Column(JSON, nullable=True)
    exit_conditions = Column(JSON, nullable=True)

    # Initial TP/SL with optimization ranges
    initial_tp_percent = Column(Float, default=5.0)
    initial_tp_optimize = Column(Boolean, default=False)
    initial_tp_min = Column(Float, nullable=True)
    initial_tp_max = Column(Float, nullable=True)
    initial_tp_step = Column(Float, nullable=True)

    initial_sl_percent = Column(Float, default=2.0)
    initial_sl_optimize = Column(Boolean, default=False)
    initial_sl_min = Column(Float, nullable=True)
    initial_sl_max = Column(Float, nullable=True)
    initial_sl_step = Column(Float, nullable=True)

    # Classic Risk Manager params with optimization ranges (Phase 4 joint optimizer).
    # Each param mirrors the TP/SL pattern: a baseline value column + *_optimize flag
    # + *_min/_max/_step bounds. The handler builds rm_cfg from these for
    # strategy_param_space._collect_rm; the four percent/multiplier params are Float,
    # max_concurrent_positions is Integer.
    rm_risk_per_trade_pct = Column(Float, default=1.0)
    rm_risk_per_trade_pct_optimize = Column(Boolean, default=False)
    rm_risk_per_trade_pct_min = Column(Float, nullable=True)
    rm_risk_per_trade_pct_max = Column(Float, nullable=True)
    rm_risk_per_trade_pct_step = Column(Float, nullable=True)

    rm_per_instrument_cap_pct = Column(Float, default=20.0)
    rm_per_instrument_cap_pct_optimize = Column(Boolean, default=False)
    rm_per_instrument_cap_pct_min = Column(Float, nullable=True)
    rm_per_instrument_cap_pct_max = Column(Float, nullable=True)
    rm_per_instrument_cap_pct_step = Column(Float, nullable=True)

    rm_min_stop_pct = Column(Float, default=2.0)
    rm_min_stop_pct_optimize = Column(Boolean, default=False)
    rm_min_stop_pct_min = Column(Float, nullable=True)
    rm_min_stop_pct_max = Column(Float, nullable=True)
    rm_min_stop_pct_step = Column(Float, nullable=True)

    rm_atr_stop_mult = Column(Float, default=2.0)
    rm_atr_stop_mult_optimize = Column(Boolean, default=False)
    rm_atr_stop_mult_min = Column(Float, nullable=True)
    rm_atr_stop_mult_max = Column(Float, nullable=True)
    rm_atr_stop_mult_step = Column(Float, nullable=True)

    rm_max_concurrent_positions = Column(Integer, default=5)
    rm_max_concurrent_positions_optimize = Column(Boolean, default=False)
    rm_max_concurrent_positions_min = Column(Integer, nullable=True)
    rm_max_concurrent_positions_max = Column(Integer, nullable=True)
    rm_max_concurrent_positions_step = Column(Integer, nullable=True)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    def __repr__(self):
        return f"<Strategy(id={self.id}, name='{self.name}')>"

    def to_dict(self):
        """Convert to dictionary for API response"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "requiredFields": self.required_fields or [],
            # Include both old and new fields for backwards compatibility
            "entryConditions": self.entry_conditions,
            "buyEntryConditions": self.buy_entry_conditions,
            "sellEntryConditions": self.sell_entry_conditions,
            "exitConditions": self.exit_conditions or [],
            "initialTpPercent": self.initial_tp_percent,
            "initialTpOptimize": self.initial_tp_optimize,
            "initialTpMin": self.initial_tp_min,
            "initialTpMax": self.initial_tp_max,
            "initialTpStep": self.initial_tp_step,
            "initialSlPercent": self.initial_sl_percent,
            "initialSlOptimize": self.initial_sl_optimize,
            "initialSlMin": self.initial_sl_min,
            "initialSlMax": self.initial_sl_max,
            "initialSlStep": self.initial_sl_step,
            # Classic Risk Manager params with optimization ranges (Phase 4)
            "rmRiskPerTradePct": self.rm_risk_per_trade_pct,
            "rmRiskPerTradePctOptimize": self.rm_risk_per_trade_pct_optimize,
            "rmRiskPerTradePctMin": self.rm_risk_per_trade_pct_min,
            "rmRiskPerTradePctMax": self.rm_risk_per_trade_pct_max,
            "rmRiskPerTradePctStep": self.rm_risk_per_trade_pct_step,
            "rmPerInstrumentCapPct": self.rm_per_instrument_cap_pct,
            "rmPerInstrumentCapPctOptimize": self.rm_per_instrument_cap_pct_optimize,
            "rmPerInstrumentCapPctMin": self.rm_per_instrument_cap_pct_min,
            "rmPerInstrumentCapPctMax": self.rm_per_instrument_cap_pct_max,
            "rmPerInstrumentCapPctStep": self.rm_per_instrument_cap_pct_step,
            "rmMinStopPct": self.rm_min_stop_pct,
            "rmMinStopPctOptimize": self.rm_min_stop_pct_optimize,
            "rmMinStopPctMin": self.rm_min_stop_pct_min,
            "rmMinStopPctMax": self.rm_min_stop_pct_max,
            "rmMinStopPctStep": self.rm_min_stop_pct_step,
            "rmAtrStopMult": self.rm_atr_stop_mult,
            "rmAtrStopMultOptimize": self.rm_atr_stop_mult_optimize,
            "rmAtrStopMultMin": self.rm_atr_stop_mult_min,
            "rmAtrStopMultMax": self.rm_atr_stop_mult_max,
            "rmAtrStopMultStep": self.rm_atr_stop_mult_step,
            "rmMaxConcurrentPositions": self.rm_max_concurrent_positions,
            "rmMaxConcurrentPositionsOptimize": self.rm_max_concurrent_positions_optimize,
            "rmMaxConcurrentPositionsMin": self.rm_max_concurrent_positions_min,
            "rmMaxConcurrentPositionsMax": self.rm_max_concurrent_positions_max,
            "rmMaxConcurrentPositionsStep": self.rm_max_concurrent_positions_step,
            "createdAt": self.created_at.isoformat() if self.created_at else None,
            "updatedAt": self.updated_at.isoformat() if self.updated_at else None,
        }
