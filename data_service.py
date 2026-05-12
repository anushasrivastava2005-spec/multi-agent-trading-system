"""
Data service: loads CSV market data, resamples to hourly/daily,
computes all technical indicators using the `ta` library, and caches results.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import ta
from cachetools import TTLCache

from config import DATA_DIR, CANDLE_WINDOW_HOURLY, CANDLE_WINDOW_DAILY

logger = logging.getLogger(__name__)

# ── Caches (TTL 30 min, up to 60 tickers in memory) ───────────────────────
_raw_cache: dict[str, pd.DataFrame] = {}
_hourly_cache: TTLCache = TTLCache(maxsize=60, ttl=1800)
_daily_cache: TTLCache = TTLCache(maxsize=60, ttl=1800)
_indicator_cache: TTLCache = TTLCache(maxsize=60, ttl=1800)


# ═══════════════════════════════════════════════════════════════════════════
#  CSV LOADING
# ═══════════════════════════════════════════════════════════════════════════

def _csv_path_for_ticker(ticker: str) -> Path:
    """Resolve the CSV file path for a ticker symbol."""
    return DATA_DIR / f"cleaned_NSE_{ticker}-EQ.csv"


def load_raw(ticker: str) -> pd.DataFrame:
    """Load raw minute-level OHLCV from CSV. Returns cached if available."""
    if ticker in _raw_cache:
        return _raw_cache[ticker]

    path = _csv_path_for_ticker(ticker)
    if not path.exists():
        raise FileNotFoundError(f"No data file for ticker: {ticker}")

    df = pd.read_csv(
        path,
        usecols=["open", "high", "low", "close", "volume", "time"],
        parse_dates=["time"],
    )
    df.rename(columns={"time": "datetime"}, inplace=True)
    df.sort_values("datetime", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Drop rows with missing OHLC
    df.dropna(subset=["open", "high", "low", "close"], inplace=True)

    _raw_cache[ticker] = df
    logger.info(f"Loaded {len(df)} minute candles for {ticker}")
    return df


# ═══════════════════════════════════════════════════════════════════════════
#  RESAMPLING
# ═══════════════════════════════════════════════════════════════════════════

def _resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample minute data to a given frequency (e.g. '1h', '1D')."""
    tmp = df.set_index("datetime")
    resampled = tmp.resample(rule).agg({
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
    }).dropna()
    resampled.reset_index(inplace=True)
    return resampled


def get_hourly(ticker: str, window: Optional[int] = None) -> pd.DataFrame:
    """Get hourly candles. Uses cache."""
    if ticker not in _hourly_cache:
        raw = load_raw(ticker)
        _hourly_cache[ticker] = _resample(raw, "1h")
    df = _hourly_cache[ticker]
    if window:
        return df.tail(window).reset_index(drop=True)
    return df


def get_daily(ticker: str, window: Optional[int] = None) -> pd.DataFrame:
    """Get daily candles. Uses cache."""
    if ticker not in _daily_cache:
        raw = load_raw(ticker)
        _daily_cache[ticker] = _resample(raw, "1D")
    df = _daily_cache[ticker]
    if window:
        return df.tail(window).reset_index(drop=True)
    return df


# ═══════════════════════════════════════════════════════════════════════════
#  TECHNICAL INDICATORS
# ═══════════════════════════════════════════════════════════════════════════

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a full suite of technical indicators on an OHLCV DataFrame.
    Adds columns in-place and returns the DataFrame.
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]

    # ── Momentum ───────────────────────────────────────────────────────
    df["rsi"] = ta.momentum.RSIIndicator(close, window=14).rsi()

    macd = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
    df["macd"]        = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"]   = macd.macd_diff()

    df["roc"] = ta.momentum.ROCIndicator(close, window=10).roc()

    stoch = ta.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3)
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()

    df["williams_r"] = ta.momentum.WilliamsRIndicator(high, low, close, lbp=14).williams_r()

    # ── Volatility ─────────────────────────────────────────────────────
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df["bb_upper"]  = bb.bollinger_hband()
    df["bb_lower"]  = bb.bollinger_lband()
    df["bb_middle"] = bb.bollinger_mavg()

    df["atr"] = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()

    # ── Moving Averages ────────────────────────────────────────────────
    df["ema_9"]  = ta.trend.EMAIndicator(close, window=9).ema_indicator()
    df["ema_21"] = ta.trend.EMAIndicator(close, window=21).ema_indicator()
    df["sma_50"] = ta.trend.SMAIndicator(close, window=50).sma_indicator()

    # ── ADX ────────────────────────────────────────────────────────────
    adx_ind = ta.trend.ADXIndicator(high, low, close, window=14)
    df["adx"] = adx_ind.adx()

    return df


def get_indicators(ticker: str, timeframe: str = "hourly") -> pd.DataFrame:
    """
    Return candles with all indicators pre-computed. Cached.
    timeframe: 'hourly' or 'daily'.
    """
    cache_key = f"{ticker}_{timeframe}"
    if cache_key in _indicator_cache:
        return _indicator_cache[cache_key]

    window = CANDLE_WINDOW_HOURLY if timeframe == "hourly" else CANDLE_WINDOW_DAILY
    # Fetch extra candles so indicator warm-up doesn't cause NaN at the tail
    extra = 60
    if timeframe == "hourly":
        df = get_hourly(ticker, window=window + extra)
    else:
        df = get_daily(ticker, window=window + extra)

    df = compute_indicators(df.copy())
    df = df.tail(window).reset_index(drop=True)

    _indicator_cache[cache_key] = df
    return df


# ═══════════════════════════════════════════════════════════════════════════
#  SUPPORT / RESISTANCE
# ═══════════════════════════════════════════════════════════════════════════

def find_swing_points(df: pd.DataFrame, order: int = 5):
    """
    Detect swing highs and swing lows using rolling window comparison.
    Returns lists of (index, price) tuples.
    """
    highs = df["high"].values
    lows = df["low"].values
    swing_highs = []
    swing_lows = []

    for i in range(order, len(df) - order):
        if all(highs[i] >= highs[i - j] for j in range(1, order + 1)) and \
           all(highs[i] >= highs[i + j] for j in range(1, order + 1)):
            swing_highs.append({"index": i, "price": float(highs[i]),
                                "datetime": str(df["datetime"].iloc[i])})

        if all(lows[i] <= lows[i - j] for j in range(1, order + 1)) and \
           all(lows[i] <= lows[i + j] for j in range(1, order + 1)):
            swing_lows.append({"index": i, "price": float(lows[i]),
                               "datetime": str(df["datetime"].iloc[i])})

    return swing_highs, swing_lows


def find_support_resistance(df: pd.DataFrame, order: int = 5):
    """
    Identify key support and resistance levels from swing points.
    Returns (list_of_support_levels, list_of_resistance_levels).
    """
    swing_highs, swing_lows = find_swing_points(df, order)

    resistance_levels = [sh["price"] for sh in swing_highs[-5:]] if swing_highs else []
    support_levels = [sl["price"] for sl in swing_lows[-5:]] if swing_lows else []

    return support_levels, resistance_levels


def compute_regression_slope(series: pd.Series, window: int = 20) -> float:
    """Compute linear regression slope of the last `window` values, normalized."""
    y = series.tail(window).values
    if len(y) < 2:
        return 0.0
    x = np.arange(len(y))
    coeffs = np.polyfit(x, y, 1)
    slope = coeffs[0]
    # Normalize by mean price to make comparable across stocks
    mean_price = np.mean(y)
    if mean_price == 0:
        return 0.0
    return float(slope / mean_price)


def compute_pivot_points(df: pd.DataFrame):
    """Compute classic pivot points from the most recent complete candle."""
    last = df.iloc[-2] if len(df) > 1 else df.iloc[-1]
    h, l, c = float(last["high"]), float(last["low"]), float(last["close"])
    pivot = (h + l + c) / 3
    return {
        "pivot": round(pivot, 2),
        "r1": round(2 * pivot - l, 2),
        "r2": round(pivot + (h - l), 2),
        "s1": round(2 * pivot - h, 2),
        "s2": round(pivot - (h - l), 2),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  FORMATTING FOR LLM PROMPTS
# ═══════════════════════════════════════════════════════════════════════════

def format_ohlc_for_prompt(df: pd.DataFrame, n: int = 20) -> str:
    """Format last N candles as a compact table for LLM consumption."""
    subset = df.tail(n)[["datetime", "open", "high", "low", "close", "volume"]].copy()
    subset["datetime"] = subset["datetime"].astype(str)
    lines = ["datetime | open | high | low | close | volume"]
    for _, row in subset.iterrows():
        lines.append(
            f"{row['datetime']} | {row['open']:.2f} | {row['high']:.2f} | "
            f"{row['low']:.2f} | {row['close']:.2f} | {int(row['volume'])}"
        )
    return "\n".join(lines)


def get_ticker_list() -> list[str]:
    """Return all valid ticker symbols from the data directory."""
    tickers = []
    for p in sorted(DATA_DIR.glob("cleaned_NSE_*-EQ.csv")):
        if p.stat().st_size > 100:
            match = p.stem.replace("cleaned_NSE_", "").replace("-EQ", "")
            tickers.append(match)
    return tickers
