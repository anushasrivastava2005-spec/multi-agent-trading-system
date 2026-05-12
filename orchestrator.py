"""
Orchestrator — runs the LangGraph multi-agent pipeline.

Pipeline (LangGraph):
  1. Fetch market data (via yfinance or CSV)
  2. Generate chart images (pattern + trend)
  3. Run LangGraph: Indicator Agent → Pattern Agent → Trend Agent → Decision Agent
  4. Store prediction in DB
"""

import json
import logging
import time
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf
from sqlalchemy.orm import Session

from agents.trading_graph import TradingGraph, DEFAULT_CONFIG
from agents.static_util import generate_kline_image, generate_trend_image
from database.models import Prediction
from config import (
    GROQ_API_KEY, GROQ_MODEL, LLM_TEMPERATURE,
    YFINANCE_SYMBOLS, ASSET_DISPLAY_NAMES,
)

logger = logging.getLogger(__name__)

# ── Build config from environment ──────────────────────────────────────────
# Uses FREE Groq Llama models by default
_config = DEFAULT_CONFIG.copy()
_config["groq_api_key"] = GROQ_API_KEY
_config["agent_llm_provider"] = "groq"
_config["graph_llm_provider"] = "groq"
_config["agent_llm_model"] = "llama-3.3-70b-versatile"       # tool-calling
_config["graph_llm_model"] = "meta-llama/llama-4-scout-17b-16e-instruct"  # vision analysis
_config["agent_llm_temperature"] = LLM_TEMPERATURE
_config["graph_llm_temperature"] = LLM_TEMPERATURE

# ── Lazy-init the TradingGraph (avoid import-time crash if no API key) ─────
_trading_graph = None


def _get_trading_graph():
    global _trading_graph
    if _trading_graph is None:
        _trading_graph = TradingGraph(config=_config)
    return _trading_graph


def fetch_live_data(ticker: str, interval: str = "1h", days: int = 10) -> pd.DataFrame:
    """Fetch live OHLCV data from Yahoo Finance. Falls back to local CSV if unavailable."""
    # For known symbols, use the mapping. For NSE stocks, append .NS
    yf_symbol = YFINANCE_SYMBOLS.get(ticker, f"{ticker}.NS")
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=days)

    try:
        df = yf.download(
            tickers=yf_symbol,
            start=start_dt,
            end=end_dt,
            interval=interval,
            auto_adjust=True,
            prepost=False,
            progress=False,
        )

        if df is not None and not df.empty:
            if isinstance(df, pd.Series):
                df = df.to_frame()

            df = df.reset_index()

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # Rename columns
            column_mapping = {
                "Date": "Datetime",
                "Open": "Open",
                "High": "High",
                "Low": "Low",
                "Close": "Close",
                "Volume": "Volume",
            }
            existing_columns = {
                old: new for old, new in column_mapping.items() if old in df.columns
            }
            df = df.rename(columns=existing_columns)

            required_columns = ["Datetime", "Open", "High", "Low", "Close"]
            if all(col in df.columns for col in required_columns):
                df = df[required_columns]
                df["Datetime"] = pd.to_datetime(df["Datetime"])
                logger.info(f"Fetched {len(df)} candles from Yahoo Finance for {yf_symbol}")
                return df
            else:
                logger.warning(f"Missing columns from YF. Available: {list(df.columns)}")
    except Exception as e:
        logger.warning(f"Yahoo Finance fetch failed for {yf_symbol}: {e}")

    # ── Fallback: use local CSV data ───────────────────────────────────
    logger.info(f"Falling back to local CSV data for {ticker}")
    try:
        from services.data_service import get_hourly
        df_local = get_hourly(ticker, window=60)
        if len(df_local) >= 20:
            df_local = df_local.rename(columns={
                "datetime": "Datetime",
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
            })
            required = ["Datetime", "Open", "High", "Low", "Close"]
            if all(col in df_local.columns for col in required):
                df_local = df_local[required]
                df_local["Datetime"] = pd.to_datetime(df_local["Datetime"])
                logger.info(f"Loaded {len(df_local)} candles from local CSV for {ticker}")
                return df_local
    except Exception as e2:
        logger.warning(f"CSV fallback also failed for {ticker}: {e2}")

    return pd.DataFrame()


def run_pipeline(ticker: str, company_id: int, db: Session) -> dict:
    """
    Execute the full LangGraph multi-agent analysis pipeline for a ticker.
    Returns the complete result dict and stores it in the DB.
    """
    start = time.time()
    logger.info(f"Starting LangGraph pipeline for {ticker}")

    trading_graph = _get_trading_graph()

    # ── Step 1: Fetch live data from Yahoo Finance ─────────────────────
    df = fetch_live_data(ticker, interval="1h", days=10)
    if df.empty:
        raise ValueError(f"No live data available for {ticker}")

    # ── Step 2: Prepare OHLCV slice ────────────────────────────────────
    df_slice = df.tail(45).reset_index(drop=True)

    required_columns = ["Datetime", "Open", "High", "Low", "Close"]
    kline_data = {}
    for col in required_columns:
        if col == "Datetime":
            kline_data[col] = df_slice[col].dt.strftime("%Y-%m-%d %H:%M:%S").tolist()
        else:
            kline_data[col] = df_slice[col].tolist()

    # ── Step 3: Pre-compute chart images ───────────────────────────────
    try:
        p_image = generate_kline_image(kline_data)
        t_image = generate_trend_image(kline_data)
    except Exception as e:
        logger.warning(f"Chart generation failed: {e}")
        p_image = {"pattern_image": ""}
        t_image = {"trend_image": ""}

    # ── Step 4: Build initial state and run LangGraph ──────────────────
    display_name = ASSET_DISPLAY_NAMES.get(ticker, ticker)

    initial_state = {
        "kline_data": kline_data,
        "analysis_results": None,
        "messages": [],
        "time_frame": "1hour",
        "stock_name": display_name,
        "pattern_image": p_image.get("pattern_image", ""),
        "trend_image": t_image.get("trend_image", ""),
    }

    final_state = trading_graph.graph.invoke(initial_state)

    elapsed = round(time.time() - start, 2)
    logger.info(f"LangGraph pipeline for {ticker} completed in {elapsed}s")

    # ── Step 5: Extract results ────────────────────────────────────────
    indicator_report = final_state.get("indicator_report", "")
    pattern_report = final_state.get("pattern_report", "")
    trend_report = final_state.get("trend_report", "")
    final_decision_raw = final_state.get("final_trade_decision", "")

    # Parse the decision JSON
    decision = "LONG"
    confidence = 0.5
    reasoning = final_decision_raw
    risk_reward_ratio = "1.5"
    forecast_horizon = "N/A"

    if final_decision_raw:
        try:
            start_idx = final_decision_raw.find("{")
            end_idx = final_decision_raw.rfind("}") + 1
            if start_idx != -1 and end_idx > 0:
                json_str = final_decision_raw[start_idx:end_idx]
                decision_data = json.loads(json_str)
                decision = decision_data.get("decision", "LONG")
                reasoning = decision_data.get("justification", final_decision_raw)
                risk_reward_ratio = str(decision_data.get("risk_reward_ratio", "1.5"))
                forecast_horizon = decision_data.get("forecast_horizon", "N/A")
        except (json.JSONDecodeError, Exception):
            pass

    # ── Step 6: Get current price for SL/TP estimation ─────────────────
    current_price = kline_data["Close"][-1] if kline_data["Close"] else 0
    atr_estimate = abs(kline_data["High"][-1] - kline_data["Low"][-1]) if kline_data["High"] else current_price * 0.02

    if decision == "LONG":
        stop_loss = current_price - (atr_estimate * 2.0)
        take_profit = current_price + (atr_estimate * 3.0)
    else:
        stop_loss = current_price + (atr_estimate * 2.0)
        take_profit = current_price - (atr_estimate * 3.0)

    risk = abs(current_price - stop_loss)
    reward = abs(take_profit - current_price)
    rr_numeric = round(reward / risk, 2) if risk > 0 else 1.5

    # ── Step 7: Store in DB ────────────────────────────────────────────
    prediction = Prediction(
        company_id=company_id,
        timestamp=datetime.utcnow(),
        decision=decision,
        confidence=confidence,
        reasoning=reasoning,
        risk_reward=rr_numeric,
        stop_loss=round(stop_loss, 2),
        take_profit=round(take_profit, 2),
        entry_price=round(current_price, 2),
        indicator_output=indicator_report[:2000] if indicator_report else "",
        pattern_output=pattern_report[:2000] if pattern_report else "",
        trend_output=trend_report[:2000] if trend_report else "",
        risk_output=json.dumps({"risk_reward": rr_numeric, "stop_loss": round(stop_loss, 2), "take_profit": round(take_profit, 2)}),
        decision_output=final_decision_raw[:2000] if final_decision_raw else "",
    )
    db.add(prediction)
    db.commit()
    db.refresh(prediction)

    # ── Step 8: Assemble response ──────────────────────────────────────
    return {
        "ticker": ticker,
        "prediction_id": prediction.id,
        "timestamp": prediction.timestamp.isoformat(),
        "decision": decision,
        "confidence": confidence,
        "reasoning": reasoning,
        "risk_reward": rr_numeric,
        "stop_loss": round(stop_loss, 2),
        "take_profit": round(take_profit, 2),
        "entry_price": round(current_price, 2),
        "forecast_horizon": forecast_horizon,
        # Rich agent reports
        "indicator_report": indicator_report,
        "pattern_report": pattern_report,
        "trend_report": trend_report,
        "final_decision_raw": final_decision_raw,
        # Chart images as base64
        "pattern_chart": p_image.get("pattern_image", ""),
        "trend_chart": t_image.get("trend_image", ""),
        "pipeline_time_seconds": elapsed,
    }
