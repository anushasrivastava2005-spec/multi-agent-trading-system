"""
SQLAlchemy ORM models for the QuantAgent database.
Indexed for fast lookups on ticker and time-series queries.
"""

from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Text,
    Index, ForeignKey, UniqueConstraint
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class Company(Base):
    """Registered company / stock ticker."""
    __tablename__ = "companies"

    id        = Column(Integer, primary_key=True, autoincrement=True)
    name      = Column(String(120), nullable=False)
    ticker    = Column(String(50), unique=True, nullable=False)
    sector    = Column(String(100), default="Unknown")
    file_path = Column(String(500))

    predictions = relationship("Prediction", back_populates="company",
                               cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_company_ticker", "ticker"),
    )

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "ticker": self.ticker,
            "sector": self.sector,
        }


class Prediction(Base):
    """Stored prediction from the multi-agent pipeline."""
    __tablename__ = "predictions"

    id               = Column(Integer, primary_key=True, autoincrement=True)
    company_id       = Column(Integer, ForeignKey("companies.id"), nullable=False)
    timestamp        = Column(DateTime, default=datetime.utcnow)
    decision         = Column(String(10), nullable=False)          # LONG / SHORT
    confidence       = Column(Float, nullable=False)
    reasoning        = Column(Text)
    risk_reward      = Column(Float)
    stop_loss        = Column(Float)
    take_profit      = Column(Float)
    entry_price      = Column(Float)
    indicator_output = Column(Text)    # JSON blob
    pattern_output   = Column(Text)    # JSON blob
    trend_output     = Column(Text)    # JSON blob
    risk_output      = Column(Text)    # JSON blob
    decision_output  = Column(Text)    # JSON blob

    company = relationship("Company", back_populates="predictions")

    __table_args__ = (
        Index("idx_pred_company_time", "company_id", "timestamp"),
        Index("idx_pred_timestamp", "timestamp"),
    )

    def to_dict(self):
        import json
        def _safe_json(val):
            if val is None:
                return None
            try:
                return json.loads(val)
            except (json.JSONDecodeError, TypeError):
                return val

        return {
            "id": self.id,
            "company_id": self.company_id,
            "ticker": self.company.ticker if self.company else None,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "decision": self.decision,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "risk_reward": self.risk_reward,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "entry_price": self.entry_price,
            "indicator_output": _safe_json(self.indicator_output),
            "pattern_output": _safe_json(self.pattern_output),
            "trend_output": _safe_json(self.trend_output),
            "risk_output": _safe_json(self.risk_output),
            "decision_output": _safe_json(self.decision_output),
        }
