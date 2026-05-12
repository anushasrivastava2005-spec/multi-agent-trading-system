"""
Market data and indicator endpoints.
"""

from fastapi import APIRouter, HTTPException, Query

from services.data_service import (
    get_hourly,
    get_daily,
    get_indicators,
    find_support_resistance,
    compute_pivot_points,
)

router = APIRouter(prefix="/api/data", tags=["Market Data"])


@router.get("/{ticker}")
def get_market_data(
    ticker: str,
    timeframe: str = Query("hourly", regex="^(hourly|daily)$"),
    limit: int = Query(100, ge=10, le=500),
):
    """
    Get OHLCV candle data for a ticker.
    timeframe: 'hourly' or 'daily'
    """
    try:
        if timeframe == "hourly":
            df = get_hourly(ticker.upper(), window=limit)
        else:
            df = get_daily(ticker.upper(), window=limit)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"No data for {ticker}")

    records = df.to_dict(orient="records")
    # Convert datetime to string for JSON
    for r in records:
        if "datetime" in r:
            r["datetime"] = str(r["datetime"])

    return {
        "ticker": ticker.upper(),
        "timeframe": timeframe,
        "count": len(records),
        "data": records,
    }


@router.get("/{ticker}/indicators")
def get_indicator_data(
    ticker: str,
    timeframe: str = Query("hourly", regex="^(hourly|daily)$"),
):
    """Get OHLCV data with all technical indicators computed."""
    try:
        df = get_indicators(ticker.upper(), timeframe=timeframe)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"No data for {ticker}")

    records = df.to_dict(orient="records")
    for r in records:
        if "datetime" in r:
            r["datetime"] = str(r["datetime"])
        # Replace NaN with None for JSON
        for key, val in r.items():
            if isinstance(val, float) and (val != val):  # NaN check
                r[key] = None

    return {
        "ticker": ticker.upper(),
        "timeframe": timeframe,
        "count": len(records),
        "data": records,
    }


@router.get("/{ticker}/levels")
def get_levels(ticker: str):
    """Get support/resistance levels and pivot points."""
    try:
        df = get_indicators(ticker.upper(), timeframe="hourly")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"No data for {ticker}")

    supports, resistances = find_support_resistance(df)
    pivots = compute_pivot_points(df)

    return {
        "ticker": ticker.upper(),
        "current_price": float(df["close"].iloc[-1]),
        "supports": [round(s, 2) for s in supports],
        "resistances": [round(r, 2) for r in resistances],
        "pivot_points": pivots,
    }
