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
        # MT4/MT5 default theme: white background, thin candles – this is
        # the chart traders actually see in their broker app, so it reads
        # as familiar instead of a stylised "crypto app" dark theme.
        self._style = mpf.make_mpf_style(
            base_mpf_style="charles",
            marketcolors=mpf.make_marketcolors(
                up="#2e9e6b",
                down="#e0483e",
                edge="inherit",
                wick="inherit",
                volume="in",
            ),
            facecolor="#ffffff",
            edgecolor="#c6c9d0",
            figcolor="#ffffff",
            gridcolor="#e3e5ea",
            gridstyle="--",
            rc={"axes.labelcolor": "#1a1a1a", "xtick.color": "#5c6066", "ytick.color": "#5c6066"},
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
        # Work with the most recent candles only – fewer candles means each
        # one is wider and the chart reads like a real trading platform
        # instead of a dense, hard-to-follow wall of candles.
        chart_df = df.tail(45).copy()

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
        """
        Render a single-panel candlestick chart matching a real trading
        app's order screen (MT4/MT5 style): just price + the Entry/SL/TP
        lines with a price tag on the right axis. No RSI/MACD/volume
        sub-panels – those add clutter without helping someone read
        "where do I enter, where's my stop, where's my target" at a glance.
        """
        labels = ["Entry", "SL", "TP1", "TP2", "TP3"]
        hlines_prices = [signal.entry_price, signal.sl_price, signal.tp1, signal.tp2, signal.tp3]
        hlines_colors = ["#2962ff", "#e0483e", "#1a9e7a", "#148067", "#0e6654"]

        buf = io.BytesIO()
        fig, axes = mpf.plot(
            chart_df,
            type="candle",
            style=self._style,
            volume=False,
            title=f"\nXAU/USD  {signal.direction} Signal  |  Strength: {signal.strength.name}",
            hlines=dict(hlines=hlines_prices, colors=hlines_colors, linestyle=":", linewidths=1.3),
            returnfig=True,
            figsize=(9, 8),
        )

        ax_main = axes[0]
        self._draw_price_labels(ax_main, chart_df, hlines_prices, hlines_colors, labels)

        fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor="#ffffff")
        plt.close(fig)
        return buf.getvalue()

    @staticmethod
    def _draw_price_labels(
        ax_main: Any,
        chart_df: pd.DataFrame,
        prices: List[float],
        colors: List[str],
        labels: List[str],
    ) -> None:
        """
        Draw Entry/SL/TP price tags as readable boxes on the right edge,
        like a real trading platform's price axis. When two levels sit too
        close together (common when TP1/TP2/TP3 are near each other), the
        tags are pushed apart vertically and connected back to their real
        price with a thin leader line, so nothing overlaps or becomes
        unreadable.
        """
        x_right = len(chart_df) - 1
        ax_main.set_xlim(ax_main.get_xlim()[0], x_right + 9)

        # Base the minimum gap on the axis's actual rendered range (not the
        # raw candle high/low), since mplfinance adds its own padding –
        # using the real ylim keeps label boxes from touching at any zoom.
        y_lo, y_hi = ax_main.get_ylim()
        min_gap = max((y_hi - y_lo) * 0.075, 1e-6)

        # Resolve overlaps by walking levels low-to-high and pushing
        # anything too close to its neighbour further up.
        order = sorted(range(len(prices)), key=lambda i: prices[i])
        label_y = {}
        last_y = None
        for i in order:
            y = prices[i]
            if last_y is not None and y - last_y < min_gap:
                y = last_y + min_gap
            label_y[i] = y
            last_y = y

        for i, (price, color, label) in enumerate(zip(prices, colors, labels)):
            y = label_y[i]
            ax_main.annotate(
                f"{label}  {price:.2f}",
                xy=(x_right, price),
                xycoords="data",
                xytext=(x_right + 1.5, y),
                textcoords="data",
                fontsize=8,
                fontweight="bold",
                color="#ffffff",
                va="center",
                ha="left",
                bbox=dict(boxstyle="round,pad=0.3", fc=color, ec="none", alpha=0.95),
                arrowprops=dict(arrowstyle="-", color=color, lw=0.8, alpha=0.7),
            )

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
