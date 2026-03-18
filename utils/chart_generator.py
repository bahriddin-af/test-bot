"""
Chart generation module.
Produces candlestick charts with overlaid indicators and equity curves.
"""

from __future__ import annotations

import io
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend – must be set before pyplot import

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

try:
    import mplfinance as mpf  # type: ignore
    HAS_MPF = True
except ImportError:
    HAS_MPF = False

from utils.logger import get_logger

logger = get_logger(__name__)


class ChartGenerator:
    """
    Generates PNG charts as bytes, suitable for sending via Telegram.
    """

    def __init__(self) -> None:
        self._style = mpf.make_mpf_style(
            base_mpf_style="charles",
            marketcolors=mpf.make_marketcolors(
                up="#00c896",
                down="#ff4560",
                edge="inherit",
                wick="inherit",
                volume="in",
            ),
            facecolor="#1a1a2e",
            edgecolor="#3a3a5c",
            figcolor="#1a1a2e",
            gridcolor="#2a2a4c",
            gridstyle="--",
            rc={"axes.labelcolor": "#ccccdd", "xtick.color": "#aaaacc", "ytick.color": "#aaaacc"},
        ) if HAS_MPF else None

    # ------------------------------------------------------------------
    # Signal chart
    # ------------------------------------------------------------------

    def generate_signal_chart(self, df: pd.DataFrame, signal: Any) -> bytes:
        """
        Generate a candlestick chart with EMAs, entry/SL/TP lines,
        volume, RSI and MACD subplots.

        Args:
            df:     OHLCV DataFrame with indicator columns already added.
            signal: Signal dataclass with entry_price, sl_price, tp1–3, direction.

        Returns:
            PNG image as bytes.
        """
        # Work with the most recent 100 candles
        chart_df = df.tail(100).copy()

        # --- Ensure index is a DatetimeIndex (required by mplfinance) ---
        if not isinstance(chart_df.index, pd.DatetimeIndex):
            chart_df.index = pd.to_datetime(chart_df.index)

        chart_df.index.name = "Date"
        chart_df.columns = [c.lower() for c in chart_df.columns]

        # Required OHLCV column renaming for mplfinance
        rename_map = {
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
        chart_df.rename(columns=rename_map, inplace=True)

        if not HAS_MPF:
            return self._fallback_chart(chart_df, signal)

        try:
            return self._mpf_chart(chart_df, signal)
        except Exception as exc:
            logger.exception("mplfinance chart failed: %s – using fallback", exc)
            return self._fallback_chart(chart_df, signal)

    def _mpf_chart(self, chart_df: pd.DataFrame, signal: Any) -> bytes:
        """Render chart using mplfinance with add_plot overlays."""
        addplots = []
        colors_ema = {"ema_20": "#00c8ff", "ema_50": "#ffaa00", "ema_200": "#ff4560"}
        for col, color in colors_ema.items():
            if col in chart_df.columns:
                addplots.append(
                    mpf.make_addplot(chart_df[col], color=color, width=1.2, panel=0)
                )

        # RSI panel
        if "rsi" in chart_df.columns:
            addplots.append(
                mpf.make_addplot(
                    chart_df["rsi"],
                    panel=2,
                    color="#c084fc",
                    ylabel="RSI",
                    width=1.0,
                )
            )

        # MACD panel
        if all(c in chart_df.columns for c in ["macd", "macd_signal", "macd_hist"]):
            addplots.append(
                mpf.make_addplot(chart_df["macd"], panel=3, color="#00c8ff", ylabel="MACD")
            )
            addplots.append(
                mpf.make_addplot(chart_df["macd_signal"], panel=3, color="#ffaa00")
            )
            addplots.append(
                mpf.make_addplot(
                    chart_df["macd_hist"],
                    panel=3,
                    type="bar",
                    color="#888888",
                    alpha=0.6,
                )
            )

        # Horizontal signal lines
        hlines_prices = [signal.entry_price, signal.sl_price, signal.tp1, signal.tp2, signal.tp3]
        hlines_colors = ["#ffffff", "#ff4560", "#00c896", "#00e5b0", "#00f0c0"]

        buf = io.BytesIO()
        fig, axes = mpf.plot(
            chart_df,
            type="candle",
            style=self._style,
            addplot=addplots if addplots else None,
            volume=True,
            panel_ratios=(4, 1, 1, 1),
            title=f"\nXAU/USD  {signal.direction} Signal  |  Strength: {signal.strength.name}",
            hlines=dict(hlines=hlines_prices, colors=hlines_colors, linestyle="--", linewidths=1.5),
            returnfig=True,
            figsize=(14, 10),
        )

        # Annotate horizontal lines
        ax_main = axes[0]
        labels = ["Entry", "SL", "TP1", "TP2", "TP3"]
        for price, color, label in zip(hlines_prices, hlines_colors, labels):
            ax_main.annotate(
                f"{label}: {price:.2f}",
                xy=(len(chart_df) - 1, price),
                xycoords=("data", "data"),
                fontsize=7,
                color=color,
                ha="right",
            )

        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="#1a1a2e")
        plt.close(fig)
        return buf.getvalue()

    def _fallback_chart(self, chart_df: pd.DataFrame, signal: Any) -> bytes:
        """Simple matplotlib candlestick-style line chart when mplfinance unavailable."""
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), facecolor="#1a1a2e")
        fig.subplots_adjust(hspace=0.1)

        ax_price, ax_rsi, ax_macd = axes
        for ax in axes:
            ax.set_facecolor("#0f0f1e")
            ax.tick_params(colors="#aaaacc")

        close = chart_df["Close"] if "Close" in chart_df.columns else chart_df["close"]
        x = range(len(close))

        ax_price.plot(x, close, color="#00c8ff", linewidth=1.5, label="Close")
        ax_price.axhline(signal.entry_price, color="#ffffff", linestyle="--", linewidth=1, label="Entry")
        ax_price.axhline(signal.sl_price, color="#ff4560", linestyle="--", linewidth=1, label="SL")
        ax_price.axhline(signal.tp1, color="#00c896", linestyle="--", linewidth=1, label="TP1")
        ax_price.axhline(signal.tp2, color="#00e5b0", linestyle="--", linewidth=1, label="TP2")
        ax_price.axhline(signal.tp3, color="#00f0c0", linestyle="--", linewidth=1, label="TP3")
        ax_price.legend(fontsize=7, facecolor="#1a1a2e", labelcolor="#ccccdd")
        ax_price.set_title(
            f"XAU/USD  {signal.direction} Signal  |  Strength: {signal.strength.name}",
            color="#ccccdd",
        )

        if "rsi" in chart_df.columns:
            ax_rsi.plot(x, chart_df["rsi"], color="#c084fc", linewidth=1)
            ax_rsi.axhline(70, color="#ff4560", linestyle=":", linewidth=0.8)
            ax_rsi.axhline(30, color="#00c896", linestyle=":", linewidth=0.8)
            ax_rsi.set_ylabel("RSI", color="#aaaacc")

        if "macd" in chart_df.columns and "macd_signal" in chart_df.columns:
            ax_macd.plot(x, chart_df["macd"], color="#00c8ff", linewidth=1, label="MACD")
            ax_macd.plot(x, chart_df["macd_signal"], color="#ffaa00", linewidth=1, label="Signal")
            ax_macd.set_ylabel("MACD", color="#aaaacc")
            ax_macd.legend(fontsize=7, facecolor="#1a1a2e", labelcolor="#ccccdd")

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor="#1a1a2e")
        plt.close(fig)
        return buf.getvalue()

    # ------------------------------------------------------------------
    # Equity curve
    # ------------------------------------------------------------------

    def generate_equity_curve(self, equity_data: List[Dict]) -> bytes:
        """
        Generate an equity-curve chart from trade history data.

        Args:
            equity_data: List of {'time': str, 'cumulative_pnl': float} dicts,
                         as returned by TradeHistory.get_equity_curve().

        Returns:
            PNG image as bytes.
        """
        if not equity_data:
            return self._empty_chart("No trade data available for equity curve")

        times = [d["time"] for d in equity_data]
        equity = [d["cumulative_pnl"] for d in equity_data]
        equity_arr = np.array(equity, dtype=float)

        # Compute drawdown
        peak = np.maximum.accumulate(equity_arr)
        drawdown = peak - equity_arr

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(12, 7), facecolor="#1a1a2e",
            gridspec_kw={"height_ratios": [3, 1]},
        )

        for ax in (ax1, ax2):
            ax.set_facecolor("#0f0f1e")
            ax.tick_params(colors="#aaaacc")
            ax.spines["bottom"].set_color("#3a3a5c")
            ax.spines["left"].set_color("#3a3a5c")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        x = range(len(equity))

        # Equity line
        ax1.plot(x, equity_arr, color="#00c8ff", linewidth=1.8, label="Equity Curve")
        ax1.fill_between(x, equity_arr, 0, alpha=0.15, color="#00c8ff")

        # Drawdown shading in red
        dd_mask = drawdown > 0
        if dd_mask.any():
            ax1.fill_between(
                x,
                equity_arr,
                peak,
                where=dd_mask,
                alpha=0.3,
                color="#ff4560",
                label="Drawdown",
            )

        final_pnl = equity_arr[-1]
        pnl_color = "#00c896" if final_pnl >= 0 else "#ff4560"
        ax1.set_title(
            f"Equity Curve  |  Total P&L: ${final_pnl:+.2f}",
            color="#ccccdd",
            fontsize=12,
        )
        ax1.set_ylabel("Cumulative P&L ($)", color="#aaaacc")
        ax1.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="#ccccdd")
        ax1.axhline(0, color="#555577", linewidth=0.8, linestyle="--")

        # Drawdown subplot
        ax2.fill_between(x, -drawdown, 0, alpha=0.7, color="#ff4560")
        ax2.set_ylabel("Drawdown ($)", color="#aaaacc")
        ax2.set_xlabel("Trade #", color="#aaaacc")

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor="#1a1a2e")
        plt.close(fig)
        return buf.getvalue()

    @staticmethod
    def _empty_chart(message: str) -> bytes:
        """Return a simple placeholder chart when no data is available."""
        fig, ax = plt.subplots(figsize=(8, 4), facecolor="#1a1a2e")
        ax.set_facecolor("#0f0f1e")
        ax.text(
            0.5, 0.5, message,
            ha="center", va="center",
            fontsize=12, color="#aaaacc",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=80, bbox_inches="tight", facecolor="#1a1a2e")
        plt.close(fig)
        return buf.getvalue()
