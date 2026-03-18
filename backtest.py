"""
Backtesting engine for the Gold Trading Bot strategy.
Simulates signal generation and trade execution over historical data.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from config import get_config, Config
from bot.indicators import add_all_indicators
from bot.signal_engine import SignalEngine, SignalDirection
from utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Result data class
# ---------------------------------------------------------------------------


@dataclass
class BacktestResult:
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    profit_factor: float = 0.0
    equity_curve: List[float] = field(default_factory=list)
    trade_log: List[Dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Backtester
# ---------------------------------------------------------------------------


class Backtester:
    """
    Walk-forward backtesting engine.
    Uses the same signal logic as SignalEngine but operates on historical bars.
    """

    def __init__(self, config: Optional[Config] = None) -> None:
        self.config = config or get_config()
        self._signal_engine = SignalEngine(self.config)

    # ------------------------------------------------------------------
    # Data fetching
    # ------------------------------------------------------------------

    async def fetch_historical_data(
        self,
        symbol: Optional[str] = None,
        period: str = "1y",
        interval: str = "1h",
    ) -> pd.DataFrame:
        """
        Download historical OHLCV data from yfinance.

        Args:
            symbol:   yfinance ticker (defaults to config).
            period:   yfinance period string, e.g. '1y', '6mo'.
            interval: Bar interval, e.g. '1h', '1d'.

        Returns:
            OHLCV DataFrame with lowercase columns, sorted by time.
        """
        sym = symbol or self.config.yfinance_symbol
        loop = asyncio.get_event_loop()

        def _dl() -> pd.DataFrame:
            ticker = yf.Ticker(sym)
            df = ticker.history(period=period, interval=interval)
            if df.empty:
                return df
            df.columns = [c.lower() for c in df.columns]
            df = df[["open", "high", "low", "close", "volume"]].copy()
            df.dropna(inplace=True)
            df.sort_index(inplace=True)
            return df

        try:
            df = await loop.run_in_executor(None, _dl)
            logger.info("Downloaded %d bars (%s %s) for %s", len(df), period, interval, sym)
            return df
        except Exception as exc:
            logger.error("Failed to fetch historical data: %s", exc)
            return pd.DataFrame()

    # ------------------------------------------------------------------
    # Core backtest loop
    # ------------------------------------------------------------------

    async def run_backtest(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        initial_balance: float = 10_000.0,
    ) -> BacktestResult:
        """
        Run a walk-forward backtest over historical data.

        Args:
            start_date:      Slice start (inclusive). None = use all data.
            end_date:        Slice end (inclusive). None = use all data.
            initial_balance: Starting equity in USD.

        Returns:
            BacktestResult dataclass with all metrics.
        """
        logger.info("Starting backtest…")
        df_raw = await self.fetch_historical_data(period="2y", interval="1h")
        if df_raw.empty:
            logger.error("No data – aborting backtest")
            return BacktestResult()

        # Optional date slice
        if start_date:
            df_raw = df_raw[df_raw.index >= pd.Timestamp(start_date, tz=df_raw.index.tz)]
        if end_date:
            df_raw = df_raw[df_raw.index <= pd.Timestamp(end_date, tz=df_raw.index.tz)]

        if len(df_raw) < 250:
            logger.warning("Insufficient data for backtest (%d bars)", len(df_raw))
            return BacktestResult()

        result = BacktestResult()
        balance = initial_balance
        equity_curve = [balance]
        open_trade: Optional[Dict] = None

        # Warm-up window: need at least 200 bars for EMA 200
        WARMUP = 210

        for i in range(WARMUP, len(df_raw)):
            window = df_raw.iloc[:i].copy()
            bar = df_raw.iloc[i]
            current_price = float(bar["close"])

            # Manage open trade
            if open_trade:
                direction = open_trade["direction"]
                sl = open_trade["sl"]
                tp1 = open_trade["tp1"]
                tp2 = open_trade["tp2"]
                tp3 = open_trade["tp3"]
                entry = open_trade["entry"]
                high = float(bar["high"])
                low = float(bar["low"])

                exit_price = None
                reason = None

                if direction == "BUY":
                    if low <= sl:
                        exit_price, reason = sl, "SL"
                    elif high >= tp3:
                        exit_price, reason = tp3, "TP3"
                    elif high >= tp2:
                        exit_price, reason = tp2, "TP2"
                    elif high >= tp1:
                        exit_price, reason = tp1, "TP1"
                else:
                    if high >= sl:
                        exit_price, reason = sl, "SL"
                    elif low <= tp3:
                        exit_price, reason = tp3, "TP3"
                    elif low <= tp2:
                        exit_price, reason = tp2, "TP2"
                    elif low <= tp1:
                        exit_price, reason = tp1, "TP1"

                if exit_price is not None:
                    pips = (exit_price - entry) * (1 if direction == "BUY" else -1) * 100
                    pnl = pips * open_trade["lot_size"]
                    balance += pnl
                    equity_curve.append(balance)
                    result.trade_log.append(
                        {
                            "entry_time": open_trade["entry_time"],
                            "exit_time": str(bar.name),
                            "direction": direction,
                            "entry": entry,
                            "exit": exit_price,
                            "reason": reason,
                            "pnl": pnl,
                            "balance": balance,
                        }
                    )
                    if pnl > 0:
                        result.wins += 1
                    else:
                        result.losses += 1
                    result.total_pnl += pnl
                    result.total_trades += 1
                    open_trade = None
                continue  # Don't open new trade on same bar

            # Signal generation on window with indicators
            enriched = add_all_indicators(window)
            if enriched.empty or len(enriched) < 10:
                continue

            # Store as cached data for signal checks
            self._signal_engine._cached_data = {"1h": enriched, "1D": enriched}

            trend_dir, trend_confs, trend_reasons = self._signal_engine.check_trend_following(enriched)
            mom_dir, mom_confs, mom_reasons = self._signal_engine.check_momentum(enriched)
            vol_dir, vol_confs, vol_reasons = self._signal_engine.check_volatility(enriched)

            total_confs = trend_confs + mom_confs + vol_confs
            buy_votes = sum([
                trend_confs if trend_dir == "BUY" else 0,
                mom_confs if mom_dir == "BUY" else 0,
                vol_confs if vol_dir == "BUY" else 0,
            ])
            sell_votes = total_confs - buy_votes

            if total_confs < self.config.min_confirmations:
                continue

            if buy_votes > sell_votes:
                direction = "BUY"
            elif sell_votes > buy_votes:
                direction = "SELL"
            else:
                continue

            last = enriched.iloc[-1]
            entry_price = float(last["close"])
            atr = float(last.get("atr", entry_price * 0.005))
            sl_dist = atr * self.config.atr_multiplier_sl
            rr = self.config.rr_ratio_min

            if direction == "BUY":
                sl = entry_price - sl_dist
                tp1 = entry_price + sl_dist * rr
                tp2 = entry_price + sl_dist * rr * 1.5
                tp3 = entry_price + sl_dist * rr * 2.5
            else:
                sl = entry_price + sl_dist
                tp1 = entry_price - sl_dist * rr
                tp2 = entry_price - sl_dist * rr * 1.5
                tp3 = entry_price - sl_dist * rr * 2.5

            # Position sizing
            risk_amount = balance * (self.config.risk_percent / 100.0)
            sl_pips = abs(entry_price - sl) * 100
            lot_size = round(risk_amount / max(sl_pips, 1.0), 2)
            lot_size = max(0.01, min(10.0, lot_size))

            open_trade = {
                "entry_time": str(bar.name),
                "direction": direction,
                "entry": entry_price,
                "sl": sl,
                "tp1": tp1,
                "tp2": tp2,
                "tp3": tp3,
                "lot_size": lot_size,
            }

        # Final metrics
        result.win_rate = (result.wins / result.total_trades * 100.0) if result.total_trades > 0 else 0.0
        result.equity_curve = equity_curve
        result.max_drawdown = self.calculate_max_drawdown(equity_curve)
        if len(equity_curve) > 1:
            returns = np.diff(equity_curve) / np.array(equity_curve[:-1])
            result.sharpe_ratio = self.calculate_sharpe_ratio(returns.tolist())
        result.profit_factor = self._calculate_profit_factor(result.trade_log)

        logger.info(
            "Backtest complete: %d trades | WR=%.1f%% | PnL=$%.2f | MaxDD=$%.2f | Sharpe=%.2f",
            result.total_trades,
            result.win_rate,
            result.total_pnl,
            result.max_drawdown,
            result.sharpe_ratio,
        )
        return result

    # ------------------------------------------------------------------
    # Metrics helpers
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.0) -> float:
        """
        Annualised Sharpe ratio from a list of per-trade return fractions.

        Args:
            returns:        List of fractional returns (e.g. 0.01 = 1%).
            risk_free_rate: Annual risk-free rate (default 0).

        Returns:
            Sharpe ratio (0.0 if insufficient data).
        """
        if len(returns) < 2:
            return 0.0
        arr = np.array(returns, dtype=float)
        mean_r = float(np.mean(arr))
        std_r = float(np.std(arr, ddof=1))
        if std_r == 0:
            return 0.0
        # Assume ~252 trading days × 6.5 h = ~1638 hourly bars per year
        annualise = np.sqrt(1638)
        return round(float((mean_r - risk_free_rate / 1638) / std_r * annualise), 3)

    @staticmethod
    def calculate_max_drawdown(equity_curve: List[float]) -> float:
        """
        Maximum peak-to-trough dollar drawdown from the equity curve.

        Returns:
            Maximum drawdown in USD (positive value).
        """
        if not equity_curve:
            return 0.0
        arr = np.array(equity_curve, dtype=float)
        peak = np.maximum.accumulate(arr)
        dd = peak - arr
        return float(np.max(dd))

    @staticmethod
    def _calculate_profit_factor(trade_log: List[Dict]) -> float:
        """Gross profit / gross loss ratio."""
        gross_profit = sum(t["pnl"] for t in trade_log if t["pnl"] > 0)
        gross_loss = abs(sum(t["pnl"] for t in trade_log if t["pnl"] < 0))
        if gross_loss == 0:
            return gross_profit if gross_profit > 0 else 0.0
        return round(gross_profit / gross_loss, 3)

    # ------------------------------------------------------------------
    # Export & visualisation
    # ------------------------------------------------------------------

    def export_results(self, result: BacktestResult, filename: str = "backtest_results") -> None:
        """
        Export trade log to CSV and Excel.

        Args:
            result:   BacktestResult to export.
            filename: Base filename (no extension). Files saved to data/ directory.
        """
        os.makedirs("data", exist_ok=True)
        if not result.trade_log:
            logger.warning("No trades to export")
            return

        df = pd.DataFrame(result.trade_log)
        csv_path = f"data/{filename}.csv"
        xlsx_path = f"data/{filename}.xlsx"

        df.to_csv(csv_path, index=False)
        logger.info("Trade log exported to %s", csv_path)

        try:
            with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
                df.to_excel(writer, sheet_name="Trades", index=False)
                # Summary sheet
                summary = {
                    "Metric": [
                        "Total Trades", "Wins", "Losses", "Win Rate (%)",
                        "Total P&L ($)", "Max Drawdown ($)", "Sharpe Ratio", "Profit Factor",
                    ],
                    "Value": [
                        result.total_trades, result.wins, result.losses,
                        round(result.win_rate, 2), round(result.total_pnl, 2),
                        round(result.max_drawdown, 2), result.sharpe_ratio, result.profit_factor,
                    ],
                }
                pd.DataFrame(summary).to_excel(writer, sheet_name="Summary", index=False)
            logger.info("Results exported to %s", xlsx_path)
        except Exception as exc:
            logger.warning("Excel export failed: %s (CSV still available)", exc)

    def plot_equity_curve(self, result: BacktestResult) -> None:
        """
        Save an equity-curve chart as PNG in the data/ directory.

        Args:
            result: BacktestResult with equity_curve populated.
        """
        if not result.equity_curve:
            logger.warning("No equity curve data to plot")
            return
        os.makedirs("data", exist_ok=True)

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(12, 7), facecolor="#1a1a2e",
            gridspec_kw={"height_ratios": [3, 1]},
        )

        eq = np.array(result.equity_curve, dtype=float)
        x = range(len(eq))
        peak = np.maximum.accumulate(eq)
        dd = peak - eq

        for ax in (ax1, ax2):
            ax.set_facecolor("#0f0f1e")
            ax.tick_params(colors="#aaaacc")

        ax1.plot(x, eq, color="#00c8ff", linewidth=1.5)
        ax1.fill_between(x, eq, eq[0], where=(eq >= eq[0]), alpha=0.15, color="#00c896")
        ax1.fill_between(x, eq, eq[0], where=(eq < eq[0]), alpha=0.15, color="#ff4560")
        ax1.fill_between(x, eq, peak, where=(dd > 0), alpha=0.3, color="#ff4560", label="Drawdown")
        ax1.set_title(
            f"Equity Curve  |  Trades={result.total_trades}  "
            f"WR={result.win_rate:.1f}%  P&L=${result.total_pnl:+.2f}  "
            f"Sharpe={result.sharpe_ratio:.2f}",
            color="#ccccdd",
        )
        ax1.set_ylabel("Balance ($)", color="#aaaacc")
        ax1.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="#ccccdd")

        ax2.fill_between(x, -dd, 0, alpha=0.7, color="#ff4560")
        ax2.set_ylabel("Drawdown ($)", color="#aaaacc")
        ax2.set_xlabel("Bar #", color="#aaaacc")

        path = "data/backtest_equity_curve.png"
        fig.savefig(path, dpi=110, bbox_inches="tight", facecolor="#1a1a2e")
        plt.close(fig)
        logger.info("Equity curve chart saved to %s", path)

    def print_report(self, result: BacktestResult) -> None:
        """Print a formatted summary report to stdout."""
        print("\n" + "=" * 58)
        print("  GOLD TRADING BOT – BACKTEST REPORT")
        print("=" * 58)
        print(f"  Total Trades   : {result.total_trades}")
        print(f"  Wins           : {result.wins}")
        print(f"  Losses         : {result.losses}")
        print(f"  Win Rate       : {result.win_rate:.1f}%")
        print(f"  Total P&L      : ${result.total_pnl:+.2f}")
        print(f"  Max Drawdown   : ${result.max_drawdown:.2f}")
        print(f"  Sharpe Ratio   : {result.sharpe_ratio:.3f}")
        print(f"  Profit Factor  : {result.profit_factor:.3f}")
        if result.equity_curve:
            peak_balance = max(result.equity_curve)
            final_balance = result.equity_curve[-1]
            print(f"  Peak Balance   : ${peak_balance:.2f}")
            print(f"  Final Balance  : ${final_balance:.2f}")
        print("=" * 58 + "\n")

    # ------------------------------------------------------------------
    # Parameter optimisation
    # ------------------------------------------------------------------

    async def optimize_parameters(
        self,
        param_grid: Optional[Dict[str, List[Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Grid-search over key strategy parameters and return the set that
        produces the best Sharpe ratio.

        Args:
            param_grid: Dict mapping config attribute name to list of values to try.
                        Defaults to a small built-in grid.

        Returns:
            Dict of best parameter values found.
        """
        if param_grid is None:
            param_grid = {
                "risk_percent": [1.0, 1.5, 2.0],
                "atr_multiplier_sl": [1.0, 1.5, 2.0],
                "rr_ratio_min": [1.5, 2.0, 2.5],
                "min_confirmations": [2, 3, 4],
            }

        best_sharpe = -float("inf")
        best_params: Dict[str, Any] = {}
        iteration = 0

        # Build all combinations
        import itertools
        keys = list(param_grid.keys())
        value_lists = [param_grid[k] for k in keys]
        combos = list(itertools.product(*value_lists))
        total = len(combos)
        logger.info("Parameter optimisation: %d combinations to test", total)

        for combo in combos:
            iteration += 1
            params = dict(zip(keys, combo))
            # Patch config
            for k, v in params.items():
                try:
                    object.__setattr__(self.config, k, v)
                except Exception:
                    self.config.__dict__[k] = v

            try:
                res = await self.run_backtest()
            except Exception as exc:
                logger.debug("Optimisation combo %d failed: %s", iteration, exc)
                continue

            if res.total_trades < 5:
                continue

            if res.sharpe_ratio > best_sharpe:
                best_sharpe = res.sharpe_ratio
                best_params = params.copy()
                logger.info(
                    "New best [%d/%d]: sharpe=%.3f params=%s",
                    iteration, total, best_sharpe, best_params,
                )

        logger.info("Optimisation complete. Best params: %s (Sharpe=%.3f)", best_params, best_sharpe)
        # Restore best params to config
        for k, v in best_params.items():
            try:
                object.__setattr__(self.config, k, v)
            except Exception:
                self.config.__dict__[k] = v
        return best_params


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


async def _run() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Gold Trading Bot Backtester")
    parser.add_argument("--start", default=None, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD")
    parser.add_argument("--balance", type=float, default=10_000.0, help="Starting balance")
    parser.add_argument("--export", action="store_true", help="Export results to CSV/Excel")
    parser.add_argument("--plot", action="store_true", help="Save equity curve PNG")
    parser.add_argument("--optimize", action="store_true", help="Run parameter optimisation")
    args = parser.parse_args()

    start_dt = datetime.strptime(args.start, "%Y-%m-%d") if args.start else None
    end_dt = datetime.strptime(args.end, "%Y-%m-%d") if args.end else None

    bt = Backtester()

    if args.optimize:
        best = await bt.optimize_parameters()
        print("Best parameters found:", best)
    else:
        result = await bt.run_backtest(start_dt, end_dt, args.balance)
        bt.print_report(result)

        if args.export:
            bt.export_results(result)
        if args.plot:
            bt.plot_equity_curve(result)


if __name__ == "__main__":
    asyncio.run(_run())
