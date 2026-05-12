"""
RiskAgent — FULLY DETERMINISTIC (no LLM).

Calculates stop-loss and take-profit using ATR-based volatility,
and ensures the risk-reward ratio stays within acceptable bounds.
"""

import logging
import pandas as pd
from agents.base_agent import BaseAgent
from config import RISK_ATR_MULTIPLIER_SL, RISK_ATR_MULTIPLIER_TP

logger = logging.getLogger(__name__)


class RiskAgent(BaseAgent):
    name = "RiskAgent"
    description = "Calculates stop-loss, take-profit, and risk-reward ratio"

    def analyze(self, df: pd.DataFrame, bias: str = "bullish", **kwargs) -> dict:
        """
        Expects DataFrame with 'close', 'atr', 'high', 'low' columns.
        bias: 'bullish' or 'bearish' — determines SL/TP direction.
        """
        latest = df.iloc[-1]
        current_price = float(latest["close"])
        atr = float(latest.get("atr", current_price * 0.02))  # fallback 2%

        # ── Volatility metrics ─────────────────────────────────────────
        returns = df["close"].pct_change().dropna()
        volatility = float(returns.std()) if len(returns) > 1 else 0.02
        avg_volume = float(df["volume"].tail(20).mean()) if "volume" in df else 0

        # ── Adaptive ATR multiplier ────────────────────────────────────
        # Higher volatility → wider stops
        vol_factor = 1.0
        if volatility > 0.03:
            vol_factor = 1.3
        elif volatility > 0.02:
            vol_factor = 1.15

        sl_mult = RISK_ATR_MULTIPLIER_SL * vol_factor
        tp_mult = RISK_ATR_MULTIPLIER_TP * vol_factor

        # ── Calculate SL & TP ──────────────────────────────────────────
        if bias == "bullish":
            stop_loss   = current_price - (atr * sl_mult)
            take_profit = current_price + (atr * tp_mult)
        else:
            stop_loss   = current_price + (atr * sl_mult)
            take_profit = current_price - (atr * tp_mult)

        # ── Risk-reward ratio ──────────────────────────────────────────
        risk = abs(current_price - stop_loss)
        reward = abs(take_profit - current_price)
        risk_reward = round(reward / risk, 2) if risk > 0 else 0.0

        # ── Position risk percentage ───────────────────────────────────
        position_risk_pct = round((risk / current_price) * 100, 2)

        return {
            "stop_loss": round(stop_loss, 2),
            "take_profit": round(take_profit, 2),
            "risk_reward": risk_reward,
            "entry_price": current_price,
            "atr": round(atr, 4),
            "volatility": round(volatility, 6),
            "position_risk_pct": position_risk_pct,
            "sl_distance": round(risk, 2),
            "tp_distance": round(reward, 2),
            "vol_factor": vol_factor,
        }
