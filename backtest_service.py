"""
Backtesting engine — RULE-BASED (no LLM calls).

Simulates trading decisions using indicator-based signals,
tracks entries, stop-loss, take-profit, and computes performance metrics.
"""

import logging
import math
import numpy as np
import pandas as pd
from services.data_service import get_daily, compute_indicators
from config import RISK_ATR_MULTIPLIER_SL, RISK_ATR_MULTIPLIER_TP

logger = logging.getLogger(__name__)


def _generate_signal(row, prev_row) -> int:
    """
    Generate a trading signal from indicators.
    Returns: 1 (LONG), -1 (SHORT), 0 (no action)
    """
    score = 0

    rsi = row.get("rsi", 50)
    macd_hist = row.get("macd_hist", 0)
    prev_macd_hist = prev_row.get("macd_hist", 0)
    stoch_k = row.get("stoch_k", 50)
    stoch_d = row.get("stoch_d", 50)
    ema_9 = row.get("ema_9", 0)
    ema_21 = row.get("ema_21", 0)
    williams_r = row.get("williams_r", -50)

    # MACD crossover (strongest signal)
    if macd_hist > 0 and prev_macd_hist <= 0:
        score += 3
    elif macd_hist < 0 and prev_macd_hist >= 0:
        score -= 3

    # RSI
    if rsi < 30:
        score += 2
    elif rsi > 70:
        score -= 2
    elif rsi < 45:
        score += 0.5
    elif rsi > 55:
        score -= 0.5

    # Stochastic
    if stoch_k < 20:
        score += 1
    elif stoch_k > 80:
        score -= 1

    # EMA alignment
    if ema_9 > ema_21:
        score += 1
    else:
        score -= 1

    # Williams %R
    if williams_r < -80:
        score += 0.5
    elif williams_r > -20:
        score -= 0.5

    if score >= 3:
        return 1
    elif score <= -3:
        return -1
    return 0


def run_backtest(ticker: str, initial_capital: float = 100000.0) -> dict:
    """
    Walk-forward backtest using rule-based signals on daily candles.

    Returns performance metrics and trade log.
    """
    # Load daily data with extra history for indicator warm-up
    df = get_daily(ticker, window=None)
    if df is None or len(df) < 100:
        return {"error": f"Insufficient data for {ticker}"}

    df = compute_indicators(df.copy())
    df.dropna(subset=["rsi", "macd_hist", "atr"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    if len(df) < 50:
        return {"error": "Insufficient indicator data after warm-up"}

    capital = initial_capital
    position = 0         # 1 = long, -1 = short, 0 = flat
    entry_price = 0.0
    stop_loss = 0.0
    take_profit = 0.0
    trades = []
    equity_curve = []
    peak_equity = initial_capital
    max_drawdown = 0.0

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i - 1]
        current_price = float(row["close"])
        atr = float(row.get("atr", current_price * 0.02))
        dt = str(row["datetime"])

        # ── Check open position exits ──────────────────────────────
        if position != 0:
            hit_sl = False
            hit_tp = False

            if position == 1:  # Long
                hit_sl = current_price <= stop_loss
                hit_tp = current_price >= take_profit
            elif position == -1:  # Short
                hit_sl = current_price >= stop_loss
                hit_tp = current_price <= take_profit

            if hit_sl or hit_tp:
                if position == 1:
                    pnl = current_price - entry_price
                else:
                    pnl = entry_price - current_price

                pnl_pct = pnl / entry_price * 100
                capital += pnl * (capital / entry_price * 0.1)  # 10% position size

                trades.append({
                    "exit_date": dt,
                    "exit_price": current_price,
                    "pnl": round(pnl, 2),
                    "pnl_pct": round(pnl_pct, 2),
                    "exit_reason": "stop_loss" if hit_sl else "take_profit",
                    "direction": "LONG" if position == 1 else "SHORT",
                    "entry_price": entry_price,
                })
                position = 0

        # ── Generate new signal ────────────────────────────────────
        if position == 0:
            signal = _generate_signal(row, prev_row)

            if signal != 0:
                position = signal
                entry_price = current_price

                if signal == 1:  # Long
                    stop_loss = current_price - (atr * RISK_ATR_MULTIPLIER_SL)
                    take_profit = current_price + (atr * RISK_ATR_MULTIPLIER_TP)
                else:  # Short
                    stop_loss = current_price + (atr * RISK_ATR_MULTIPLIER_SL)
                    take_profit = current_price - (atr * RISK_ATR_MULTIPLIER_TP)

                if trades or not trades:
                    # Add entry info to be completed on exit
                    pass

        # Track equity
        equity_curve.append({
            "datetime": dt,
            "equity": round(capital, 2),
        })

        # Max drawdown
        peak_equity = max(peak_equity, capital)
        dd = (peak_equity - capital) / peak_equity * 100
        max_drawdown = max(max_drawdown, dd)

    # ── Compute metrics ────────────────────────────────────────────────
    total_trades = len(trades)
    if total_trades == 0:
        return {
            "ticker": ticker,
            "total_trades": 0,
            "message": "No trades generated",
            "equity_curve": equity_curve[-60:],
        }

    winning = [t for t in trades if t["pnl"] > 0]
    losing = [t for t in trades if t["pnl"] <= 0]
    win_rate = len(winning) / total_trades * 100

    # Directional accuracy (did price move in predicted direction?)
    correct_direction = sum(1 for t in trades if t["pnl"] > 0)
    directional_accuracy = correct_direction / total_trades * 100

    # Sharpe ratio (annualized from daily equity returns)
    equity_values = [e["equity"] for e in equity_curve]
    equity_series = pd.Series(equity_values)
    daily_returns = equity_series.pct_change().dropna()

    if len(daily_returns) > 1 and daily_returns.std() > 0:
        sharpe = float((daily_returns.mean() / daily_returns.std()) * math.sqrt(252))
    else:
        sharpe = 0.0

    # Total return
    total_return = (capital - initial_capital) / initial_capital * 100

    # Average win vs average loss
    avg_win = np.mean([t["pnl_pct"] for t in winning]) if winning else 0
    avg_loss = np.mean([t["pnl_pct"] for t in losing]) if losing else 0

    return {
        "ticker": ticker,
        "initial_capital": initial_capital,
        "final_capital": round(capital, 2),
        "total_return_pct": round(total_return, 2),
        "total_trades": total_trades,
        "winning_trades": len(winning),
        "losing_trades": len(losing),
        "win_rate_pct": round(win_rate, 2),
        "directional_accuracy_pct": round(directional_accuracy, 2),
        "avg_win_pct": round(avg_win, 2),
        "avg_loss_pct": round(avg_loss, 2),
        "max_drawdown_pct": round(max_drawdown, 2),
        "sharpe_ratio": round(sharpe, 4),
        "trades": trades[-20:],  # Last 20 trades for display
        "equity_curve": equity_curve[-120:],  # Last 120 days
    }
