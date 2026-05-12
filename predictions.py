"""
Prediction and backtesting endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from database.connection import get_db
from database.models import Company, Prediction
from services.orchestrator import run_pipeline
from services.backtest_service import run_backtest

router = APIRouter(prefix="/api", tags=["Predictions"])


@router.post("/predict/{ticker}")
def predict(ticker: str, db: Session = Depends(get_db)):
    """
    Run the full LangGraph multi-agent pipeline for a stock and return the prediction.
    This calls IndicatorAgent → PatternAgent → TrendAgent → DecisionAgent via LangGraph.
    Returns rich results including markdown reports and base64 chart images.
    """
    company = db.query(Company).filter(Company.ticker == ticker.upper()).first()
    if not company:
        raise HTTPException(status_code=404, detail=f"Company {ticker} not found")

    try:
        result = run_pipeline(ticker.upper(), company.id, db)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")


@router.get("/predictions")
def list_predictions(
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """List recent predictions across all companies."""
    preds = (
        db.query(Prediction)
        .order_by(Prediction.timestamp.desc())
        .limit(limit)
        .all()
    )
    return {
        "count": len(preds),
        "predictions": [p.to_dict() for p in preds],
    }


@router.get("/predictions/{ticker}")
def get_predictions_for_ticker(
    ticker: str,
    limit: int = Query(10, ge=1, le=50),
    db: Session = Depends(get_db),
):
    """Get prediction history for a specific ticker."""
    company = db.query(Company).filter(Company.ticker == ticker.upper()).first()
    if not company:
        raise HTTPException(status_code=404, detail=f"Company {ticker} not found")

    preds = (
        db.query(Prediction)
        .filter(Prediction.company_id == company.id)
        .order_by(Prediction.timestamp.desc())
        .limit(limit)
        .all()
    )
    return {
        "ticker": ticker.upper(),
        "count": len(preds),
        "predictions": [p.to_dict() for p in preds],
    }


@router.post("/backtest/{ticker}")
def backtest(ticker: str):
    """
    Run a rule-based backtest for a ticker on daily data.
    No LLM calls — fully deterministic and fast.
    """
    try:
        result = run_backtest(ticker.upper())
        return result
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"No data for {ticker}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backtest error: {str(e)}")
