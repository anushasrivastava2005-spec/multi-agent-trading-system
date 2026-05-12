"""
Technical tools and trendline utility functions.
Replaces TA-Lib with the pure-Python `ta` library to avoid native build issues on Windows.
"""

import base64
import io
from typing import Annotated

import matplotlib
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd
import ta as ta_lib
from langchain_core.tools import tool

from agents.color_style import my_color_style

matplotlib.use("Agg")


# ═══════════════════════════════════════════════════════════════════════════
#  TRENDLINE HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def check_trend_line(support: bool, pivot: int, slope: float, y: np.array):
    """Compute sum of differences between line and prices."""
    intercept = -slope * pivot + y.iloc[pivot]
    line_vals = slope * np.arange(len(y)) + intercept
    diffs = line_vals - y

    if support and diffs.max() > 1e-5:
        return -1.0
    elif not support and diffs.min() < -1e-5:
        return -1.0

    err = (diffs ** 2.0).sum()
    return err


def optimize_slope(support: bool, pivot: int, init_slope: float, y: np.array):
    """Optimize trendline slope via numerical differentiation."""
    slope_unit = (y.max() - y.min()) / len(y)
    opt_step = 1.0
    min_step = 0.0001
    curr_step = opt_step

    best_slope = init_slope
    best_err = check_trend_line(support, pivot, init_slope, y)
    assert best_err >= 0.0

    get_derivative = True
    derivative = None
    while curr_step > min_step:
        if get_derivative:
            slope_change = best_slope + slope_unit * min_step
            test_err = check_trend_line(support, pivot, slope_change, y)
            derivative = test_err - best_err

            if test_err < 0.0:
                slope_change = best_slope - slope_unit * min_step
                test_err = check_trend_line(support, pivot, slope_change, y)
                derivative = best_err - test_err

            if test_err < 0.0:
                raise Exception("Derivative failed. Check your data.")

            get_derivative = False

        if derivative > 0.0:
            test_slope = best_slope - slope_unit * curr_step
        else:
            test_slope = best_slope + slope_unit * curr_step

        test_err = check_trend_line(support, pivot, test_slope, y)
        if test_err < 0 or test_err >= best_err:
            curr_step *= 0.5
        else:
            best_err = test_err
            best_slope = test_slope
            get_derivative = True

    return (best_slope, -best_slope * pivot + y.iloc[pivot])


def fit_trendlines_single(data: np.array):
    """Fit support and resistance trendlines on single-column data."""
    x = np.arange(len(data))
    coefs = np.polyfit(x, data, 1)
    line_points = coefs[0] * x + coefs[1]

    upper_pivot = (data - line_points).argmax()
    lower_pivot = (data - line_points).argmin()

    support_coefs = optimize_slope(True, lower_pivot, coefs[0], data)
    resist_coefs = optimize_slope(False, upper_pivot, coefs[0], data)

    return (support_coefs, resist_coefs)


def fit_trendlines_high_low(high: np.array, low: np.array, close: np.array):
    """Fit support and resistance trendlines using high/low/close."""
    x = np.arange(len(close))
    coefs = np.polyfit(x, close, 1)
    line_points = coefs[0] * x + coefs[1]
    upper_pivot = (high - line_points).argmax()
    lower_pivot = (low - line_points).argmin()

    support_coefs = optimize_slope(True, lower_pivot, coefs[0], low)
    resist_coefs = optimize_slope(False, upper_pivot, coefs[0], high)

    return (support_coefs, resist_coefs)


def get_line_points(candles, line_points):
    """Place line points in tuples for mplfinance."""
    idx = candles.index
    line_i = len(candles) - len(line_points)
    assert line_i >= 0
    points = []
    for i in range(line_i, len(candles)):
        points.append((idx[i], line_points[i - line_i]))
    return points


def split_line_into_segments(line_points):
    """Convert line points into segment pairs."""
    return [[line_points[i], line_points[i + 1]] for i in range(len(line_points) - 1)]


# ═══════════════════════════════════════════════════════════════════════════
#  TECHNICAL TOOLS  (uses pure-python `ta` library, NOT talib)
# ═══════════════════════════════════════════════════════════════════════════

class TechnicalTools:

    @staticmethod
    @tool
    def generate_trend_image(
        kline_data: Annotated[
            dict,
            "Dictionary containing OHLCV data with keys 'Datetime', 'Open', 'High', 'Low', 'Close'.",
        ]
    ) -> dict:
        """
        Generate a candlestick chart with trendlines from OHLCV data,
        save it locally and return a base64-encoded image.
        """
        data = pd.DataFrame(kline_data)
        candles = data.iloc[-50:].copy()

        candles["Datetime"] = pd.to_datetime(candles["Datetime"])
        candles.set_index("Datetime", inplace=True)

        support_coefs_c, resist_coefs_c = fit_trendlines_single(candles["Close"])
        support_coefs, resist_coefs = fit_trendlines_high_low(
            candles["High"], candles["Low"], candles["Close"]
        )

        support_line_c = support_coefs_c[0] * np.arange(len(candles)) + support_coefs_c[1]
        resist_line_c = resist_coefs_c[0] * np.arange(len(candles)) + resist_coefs_c[1]
        support_line = support_coefs[0] * np.arange(len(candles)) + support_coefs[1]
        resist_line = resist_coefs[0] * np.arange(len(candles)) + resist_coefs[1]

        s_seq = get_line_points(candles, support_line)
        r_seq = get_line_points(candles, resist_line)
        s_seq2 = get_line_points(candles, support_line_c)
        r_seq2 = get_line_points(candles, resist_line_c)

        s_segments = split_line_into_segments(s_seq)
        r_segments = split_line_into_segments(r_seq)
        s2_segments = split_line_into_segments(s_seq2)
        r2_segments = split_line_into_segments(r_seq2)

        all_segments = s_segments + r_segments + s2_segments + r2_segments
        colors = (
            ["white"] * len(s_segments)
            + ["white"] * len(r_segments)
            + ["blue"] * len(s2_segments)
            + ["red"] * len(r2_segments)
        )

        apds = [
            mpf.make_addplot(support_line_c, color="blue", width=1, label="Close Support"),
            mpf.make_addplot(resist_line_c, color="red", width=1, label="Close Resistance"),
        ]

        fig, axlist = mpf.plot(
            candles,
            type="candle",
            style=my_color_style,
            addplot=apds,
            alines=dict(alines=all_segments, colors=colors, linewidths=1),
            returnfig=True,
            figsize=(12, 6),
            block=False,
        )

        axlist[0].set_ylabel("Price", fontweight="normal")
        axlist[0].set_xlabel("Datetime", fontweight="normal")

        # Save to base64
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches="tight", pad_inches=0.1)
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)

        return {
            "trend_image": img_b64,
            "trend_image_description": "Trend-enhanced candlestick chart with support/resistance lines.",
        }

    @staticmethod
    @tool
    def generate_kline_image(
        kline_data: Annotated[
            dict,
            "Dictionary containing OHLCV data with keys 'Datetime', 'Open', 'High', 'Low', 'Close'.",
        ],
    ) -> dict:
        """
        Generate a candlestick (K-line) chart from OHLCV data and return a base64-encoded image.
        """
        df = pd.DataFrame(kline_data)
        df = df.tail(40)

        try:
            df.index = pd.to_datetime(df["Datetime"], format="%Y-%m-%d %H:%M:%S")
        except ValueError:
            df.index = pd.to_datetime(df["Datetime"])

        fig, axlist = mpf.plot(
            df[["Open", "High", "Low", "Close"]],
            type="candle",
            style=my_color_style,
            figsize=(12, 6),
            returnfig=True,
            block=False,
        )
        axlist[0].set_ylabel("Price", fontweight="normal")
        axlist[0].set_xlabel("Datetime", fontweight="normal")

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)

        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode("utf-8")

        return {
            "pattern_image": img_b64,
            "pattern_image_description": "Candlestick chart returned as base64 string.",
        }

    @staticmethod
    @tool
    def compute_rsi(
        kline_data: Annotated[
            dict,
            "Dictionary with a 'Close' key containing a list of float closing prices.",
        ],
        period: Annotated[
            int, "Lookback period for RSI calculation (default is 14)"
        ] = 14,
    ) -> dict:
        """
        Compute the Relative Strength Index (RSI) using the pure-python ta library.
        """
        df = pd.DataFrame(kline_data)
        rsi = ta_lib.momentum.RSIIndicator(df["Close"], window=period).rsi()
        return {"rsi": rsi.fillna(0).round(2).tolist()[-28:]}

    @staticmethod
    @tool
    def compute_macd(
        kline_data: Annotated[
            dict,
            "Dictionary with a 'Close' key containing a list of float closing prices.",
        ],
        fastperiod: Annotated[int, "Fast EMA period"] = 12,
        slowperiod: Annotated[int, "Slow EMA period"] = 26,
        signalperiod: Annotated[int, "Signal line EMA period"] = 9,
    ) -> dict:
        """
        Compute the Moving Average Convergence Divergence (MACD).
        """
        df = pd.DataFrame(kline_data)
        macd_ind = ta_lib.trend.MACD(
            df["Close"],
            window_slow=slowperiod,
            window_fast=fastperiod,
            window_sign=signalperiod,
        )
        macd_vals = macd_ind.macd().fillna(0).round(2).tolist()
        signal_vals = macd_ind.macd_signal().fillna(0).round(2).tolist()[-28:]
        hist_vals = macd_ind.macd_diff().fillna(0).round(2).tolist()[-28:]
        return {
            "macd": macd_vals,
            "macd_signal": signal_vals,
            "macd_hist": hist_vals,
        }

    @staticmethod
    @tool
    def compute_stoch(
        kline_data: Annotated[
            dict,
            "Dictionary with 'High', 'Low', and 'Close' keys, each mapping to lists of float values.",
        ]
    ) -> dict:
        """
        Compute the Stochastic Oscillator %K and %D.
        """
        df = pd.DataFrame(kline_data)
        stoch = ta_lib.momentum.StochasticOscillator(
            df["High"], df["Low"], df["Close"],
            window=14, smooth_window=3,
        )
        return {
            "stoch_k": stoch.stoch().fillna(0).round(2).tolist()[-28:],
            "stoch_d": stoch.stoch_signal().fillna(0).round(2).tolist()[-28:],
        }

    @staticmethod
    @tool
    def compute_roc(
        kline_data: Annotated[
            dict,
            "Dictionary with a 'Close' key containing a list of float closing prices.",
        ],
        period: Annotated[
            int, "Number of periods over which to calculate ROC (default is 10)"
        ] = 10,
    ) -> dict:
        """
        Compute the Rate of Change (ROC) indicator.
        """
        df = pd.DataFrame(kline_data)
        roc = ta_lib.momentum.ROCIndicator(df["Close"], window=period).roc()
        return {"roc": roc.fillna(0).round(2).tolist()[-28:]}

    @staticmethod
    @tool
    def compute_willr(
        kline_data: Annotated[
            dict,
            "Dictionary with 'High', 'Low', and 'Close' keys containing float lists.",
        ],
        period: Annotated[int, "Lookback period for Williams %R"] = 14,
    ) -> dict:
        """
        Compute the Williams %R indicator.
        """
        df = pd.DataFrame(kline_data)
        willr = ta_lib.momentum.WilliamsRIndicator(
            df["High"], df["Low"], df["Close"], lbp=period
        ).williams_r()
        return {"willr": willr.fillna(0).round(2).tolist()[-28:]}
