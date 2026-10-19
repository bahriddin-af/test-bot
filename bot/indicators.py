"""
Technical indicators module.
Uses pandas_ta (pure Python) for all calculations.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import pandas_ta as ta
except ImportError:
    ta = None  # type: ignore

from utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Individual indicator functions
# ---------------------------------------------------------------------------


def calculate_ema(df: pd.DataFrame, period: int, column: str = "close") -> pd.Series:
    """
    Calculate Exponential Moving Average.

    Args:
        df:     OHLCV DataFrame.
        period: EMA look-back period.
        column: Column to calculate EMA on.

    Returns:
        pd.Series of EMA values.
    """
    return df[column].ewm(span=period, adjust=False).mean()


def calculate_all_emas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ema_20, ema_50, ema_200 columns to *df* (in-place).

    Returns:
        The same DataFrame with new columns appended.
    """
    df["ema_20"] = calculate_ema(df, 20)
    df["ema_50"] = calculate_ema(df, 50)
    df["ema_200"] = calculate_ema(df, 200)
    return df


def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Add *rsi* column using Wilder's smoothed RSI.

    Returns:
        DataFrame with *rsi* column added.
    """
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    # avg_loss == 0 means no down candles in the window: RSI is 100 if there
    # was any gain, or 50 (flat/no data yet) if there was none either.
    no_loss_rsi = np.where(avg_gain > 0, 100.0, 50.0)
    df["rsi"] = rsi.where(avg_loss != 0, no_loss_rsi).fillna(50.0)
    return df


def calculate_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal_period: int = 9,
) -> pd.DataFrame:
    """
    Add *macd*, *macd_signal*, *macd_hist* columns.

    Returns:
        DataFrame with MACD columns added.
    """
    ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
    df["macd"] = ema_fast - ema_slow
    df["macd_signal"] = df["macd"].ewm(span=signal_period, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    return df


def calculate_stoch_rsi(
    df: pd.DataFrame,
    rsi_period: int = 14,
    stoch_period: int = 14,
    smooth_k: int = 3,
    smooth_d: int = 3,
) -> pd.DataFrame:
    """
    Add *stoch_rsi_k* and *stoch_rsi_d* columns.

    Returns:
        DataFrame with Stochastic RSI columns added.
    """
    # Ensure RSI exists
    if "rsi" not in df.columns:
        calculate_rsi(df, rsi_period)

    rsi = df["rsi"]
    rsi_min = rsi.rolling(stoch_period).min()
    rsi_max = rsi.rolling(stoch_period).max()
    rsi_range = rsi_max - rsi_min
    raw_k = ((rsi - rsi_min) / rsi_range.replace(0, np.nan)) * 100.0
    df["stoch_rsi_k"] = raw_k.rolling(smooth_k).mean().fillna(50.0)
    df["stoch_rsi_d"] = df["stoch_rsi_k"].rolling(smooth_d).mean().fillna(50.0)
    return df


def calculate_bbands(
    df: pd.DataFrame,
    period: int = 20,
    std: float = 2.0,
) -> pd.DataFrame:
    """
    Add *bb_upper*, *bb_middle*, *bb_lower*, *bb_width* columns.

    Returns:
        DataFrame with Bollinger Band columns added.
    """
    rolling = df["close"].rolling(period)
    df["bb_middle"] = rolling.mean()
    std_dev = rolling.std(ddof=0)
    df["bb_upper"] = df["bb_middle"] + std * std_dev
    df["bb_lower"] = df["bb_middle"] - std * std_dev
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"].replace(
        0, np.nan
    )
    return df


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Add *atr* column using True Range.

    Returns:
        DataFrame with *atr* column added.
    """
    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / period, adjust=False).mean()
    return df


def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Add *adx*, *adx_pos* (+DI), *adx_neg* (-DI) columns.

    Returns:
        DataFrame with ADX columns added.
    """
    high = df["high"]
    low = df["low"]
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = df["close"].shift(1)

    plus_dm = high - prev_high
    minus_dm = prev_low - low
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100.0 * (plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr.replace(0, np.nan))
    minus_di = 100.0 * (minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr.replace(0, np.nan))

    di_sum = plus_di + minus_di
    dx = 100.0 * ((plus_di - minus_di).abs() / di_sum.replace(0, np.nan))
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()

    df["adx"] = adx.fillna(0.0)
    df["adx_pos"] = plus_di.fillna(0.0)
    df["adx_neg"] = minus_di.fillna(0.0)
    return df


def calculate_supertrend(
    df: pd.DataFrame,
    period: int = 10,
    multiplier: float = 3.0,
) -> pd.DataFrame:
    """
    Add *supertrend* (price level) and *supertrend_direction* (1=up, -1=down) columns.

    Returns:
        DataFrame with Supertrend columns added.
    """
    if "atr" not in df.columns:
        calculate_atr(df, period)

    hl2 = (df["high"] + df["low"]) / 2.0
    upper_band = hl2 + multiplier * df["atr"]
    lower_band = hl2 - multiplier * df["atr"]

    supertrend = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=int)

    supertrend.iloc[0] = upper_band.iloc[0]
    direction.iloc[0] = 1

    for i in range(1, len(df)):
        prev_st = supertrend.iloc[i - 1]
        prev_dir = direction.iloc[i - 1]
        curr_close = df["close"].iloc[i]
        curr_upper = upper_band.iloc[i]
        curr_lower = lower_band.iloc[i]

        if prev_dir == 1:
            curr_st = min(curr_upper, max(prev_st, curr_lower))
            if curr_close < curr_st:
                curr_dir = -1
                curr_st = curr_upper
            else:
                curr_dir = 1
        else:
            curr_st = max(curr_lower, min(prev_st, curr_upper))
            if curr_close > curr_st:
                curr_dir = 1
                curr_st = curr_lower
            else:
                curr_dir = -1

        supertrend.iloc[i] = curr_st
        direction.iloc[i] = curr_dir

    df["supertrend"] = supertrend
    df["supertrend_direction"] = direction
    return df


def calculate_obv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add *obv* (On-Balance Volume) column.

    Returns:
        DataFrame with *obv* column added.
    """
    close = df["close"]
    volume = df["volume"]
    sign = np.sign(close.diff()).fillna(0)
    df["obv"] = (sign * volume).cumsum()
    return df


def calculate_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add *vwap* (Volume-Weighted Average Price) column.
    Resets at the start of each trading session (day).

    Returns:
        DataFrame with *vwap* column added.
    """
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    vol = df["volume"]
    cum_tp_vol = (tp * vol).cumsum()
    cum_vol = vol.cumsum()
    df["vwap"] = cum_tp_vol / cum_vol.replace(0, np.nan)
    return df


def calculate_pivot_points(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate classic pivot points based on the previous period's H/L/C.

    Args:
        df: OHLCV DataFrame (uses the last completed candle).

    Returns:
        Dict with keys: pivot, r1, r2, r3, s1, s2, s3.
    """
    if len(df) < 2:
        return {}
    prev = df.iloc[-2]
    high = float(prev["high"])
    low = float(prev["low"])
    close = float(prev["close"])
    pivot = (high + low + close) / 3.0
    r1 = 2.0 * pivot - low
    s1 = 2.0 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    r3 = high + 2.0 * (pivot - low)
    s3 = low - 2.0 * (high - pivot)
    return {
        "pivot": round(pivot, 3),
        "r1": round(r1, 3),
        "r2": round(r2, 3),
        "r3": round(r3, 3),
        "s1": round(s1, 3),
        "s2": round(s2, 3),
        "s3": round(s3, 3),
    }


def calculate_fibonacci_levels(high: float, low: float) -> Dict[str, float]:
    """
    Calculate Fibonacci retracement and extension levels.

    Args:
        high: Swing high price.
        low:  Swing low price.

    Returns:
        Dict with Fibonacci level names as keys and prices as values.
    """
    diff = high - low
    levels = {
        "0.0": round(high, 3),
        "0.236": round(high - 0.236 * diff, 3),
        "0.382": round(high - 0.382 * diff, 3),
        "0.500": round(high - 0.500 * diff, 3),
        "0.618": round(high - 0.618 * diff, 3),
        "0.786": round(high - 0.786 * diff, 3),
        "1.0": round(low, 3),
        "1.272": round(low - 0.272 * diff, 3),
        "1.618": round(low - 0.618 * diff, 3),
    }
    return levels


def detect_rsi_divergence(df: pd.DataFrame, lookback: int = 14) -> Optional[str]:
    """
    Detect bullish or bearish RSI divergence over the last *lookback* candles.

    Returns:
        'bullish'  – price makes lower low but RSI makes higher low
        'bearish'  – price makes higher high but RSI makes lower high
        None       – no divergence detected
    """
    if "rsi" not in df.columns:
        calculate_rsi(df)
    if len(df) < lookback + 2:
        return None

    window = df.tail(lookback)
    price_high_idx = window["close"].idxmax()
    price_low_idx = window["close"].idxmin()
    rsi_at_price_high = window.loc[price_high_idx, "rsi"]
    rsi_at_price_low = window.loc[price_low_idx, "rsi"]
    latest_price = window["close"].iloc[-1]
    latest_rsi = window["rsi"].iloc[-1]
    first_price = window["close"].iloc[0]
    first_rsi = window["rsi"].iloc[0]

    # Bearish divergence: price higher but RSI lower
    if latest_price > first_price and latest_rsi < first_rsi:
        return "bearish"
    # Bullish divergence: price lower but RSI higher
    if latest_price < first_price and latest_rsi > first_rsi:
        return "bullish"
    return None


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply every indicator to *df* and return the enriched DataFrame.

    The caller should pass a copy if they want to preserve the original.
    Columns added:
        ema_20, ema_50, ema_200,
        rsi,
        macd, macd_signal, macd_hist,
        stoch_rsi_k, stoch_rsi_d,
        bb_upper, bb_middle, bb_lower, bb_width,
        atr,
        adx, adx_pos, adx_neg,
        supertrend, supertrend_direction,
        obv,
        vwap,
    """
    if df.empty or len(df) < 30:
        logger.warning("DataFrame too small for indicator calculation (%d rows)", len(df))
        return df

    # Normalise column names to lowercase
    df.columns = [c.lower() for c in df.columns]
    required = {"open", "high", "low", "close"}
    if not required.issubset(set(df.columns)):
        logger.error("DataFrame missing OHLC columns: %s", df.columns.tolist())
        return df

    # Ensure volume exists (some gold feeds omit it)
    if "volume" not in df.columns:
        df["volume"] = 0.0

    try:
        calculate_all_emas(df)
        calculate_rsi(df)
        calculate_macd(df)
        calculate_stoch_rsi(df)
        calculate_bbands(df)
        calculate_atr(df)
        calculate_adx(df)
        calculate_supertrend(df)
        calculate_obv(df)
        calculate_vwap(df)
    except Exception as exc:
        logger.exception("Error computing indicators: %s", exc)

    return df
