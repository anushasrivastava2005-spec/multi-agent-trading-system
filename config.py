"""
Central configuration for the QuantAgent backend.
All settings are loaded from environment variables with sensible defaults.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
PROJECT_DIR = BASE_DIR.parent
DATA_DIR = PROJECT_DIR / "data" / "cleaneddata" / "cleaneddata"
DB_PATH = BASE_DIR / "quantagent.db"
DATABASE_URL = f"sqlite:///{DB_PATH}"

# ── LLM Provider Configuration ────────────────────────────────────────────
# OpenAI (default — used for LangGraph vision agents)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_AGENT_MODEL = os.getenv("OPENAI_AGENT_MODEL", "gpt-4o-mini")
OPENAI_GRAPH_MODEL = os.getenv("OPENAI_GRAPH_MODEL", "gpt-4o")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))

# Groq (legacy, still available for fallback)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_TEMPERATURE = float(os.getenv("GROQ_TEMPERATURE", "0.1"))
GROQ_MAX_TOKENS = int(os.getenv("GROQ_MAX_TOKENS", "2048"))

# ── Data Pipeline ──────────────────────────────────────────────────────────
CANDLE_WINDOW_HOURLY = 100          # number of 1-hour candles fed to agents
CANDLE_WINDOW_DAILY = 60            # daily candles for context
PREDICTION_HORIZON = 5              # predict next N candles ahead

# ── Agent Confidence Weights ───────────────────────────────────────────────
CONFIDENCE_WEIGHTS = {
    "indicator_strength": 0.30,
    "trend_strength": 0.25,
    "pattern_confidence": 0.20,
    "agent_agreement": 0.25,
}

# ── Risk Parameters ────────────────────────────────────────────────────────
RISK_ATR_MULTIPLIER_SL = 2.0       # ATR multiplier for stop-loss
RISK_ATR_MULTIPLIER_TP = 3.0       # ATR multiplier for take-profit
RISK_REWARD_MIN = 1.2
RISK_REWARD_MAX = 1.8

# ── Yahoo Finance Symbol Mapping ──────────────────────────────────────────
YFINANCE_SYMBOLS = {
    "SPX": "^GSPC",
    "BTC": "BTC-USD",
    "GC": "GC=F",
    "NQ": "NQ=F",
    "CL": "CL=F",
    "ES": "ES=F",
    "DJI": "^DJI",
    "QQQ": "QQQ",
    "VIX": "^VIX",
    "DXY": "DX-Y.NYB",
    "AAPL": "AAPL",
    "TSLA": "TSLA",
    "MSFT": "MSFT",
    "GOOGL": "GOOGL",
    "AMZN": "AMZN",
    "META": "META",
    "NVDA": "NVDA",
}

ASSET_DISPLAY_NAMES = {
    "SPX": "S&P 500",
    "BTC": "Bitcoin",
    "GC": "Gold Futures",
    "NQ": "Nasdaq Futures",
    "CL": "Crude Oil",
    "ES": "E-mini S&P 500",
    "DJI": "Dow Jones",
    "QQQ": "Invesco QQQ Trust",
    "VIX": "Volatility Index",
    "DXY": "US Dollar Index",
    "AAPL": "Apple Inc.",
    "TSLA": "Tesla Inc.",
    "MSFT": "Microsoft Corp.",
    "GOOGL": "Alphabet Inc.",
    "AMZN": "Amazon.com Inc.",
    "META": "Meta Platforms Inc.",
    "NVDA": "NVIDIA Corp.",
}
