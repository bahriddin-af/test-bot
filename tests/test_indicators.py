"""
Unit tests for bot/indicators.py
Run with: pytest tests/test_indicators.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bot.indicators import (
    add_all_indicators,
    calculate_adx,
    calculate_all_emas,
    calculate_atr,
    calculate_bbands,
    calculate_ema,
    calculate_fibonacci_levels,
    calculate_macd,
    calculate_rsi,
    calculate_stoch_rsi,
    detect_rsi_divergence,
)
from bot.signal_engine import SignalStrength, SignalDirection


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """
    Generate a synthetic OHLCV DataFrame with 300 rows for testing.
    Uses a simple sine-wave price model to produce realistic OHLC data.
    """
    np.random.seed(42)
    n = 300
    t = np.linspace(0, 4 * np.pi, n)
    base_price = 2000.0

    # Close prices: trending upward with a sine wave component
    close = base_price + 50 * np.sin(t) + np.cumsum(np.random.randn(n) * 0.5)

    # OHLC derived from close
    noise = np.abs(np.random.randn(n)) * 2
    high = close + noise
    low = close - noise
    open_ = close + np.random.randn(n) * 1.0
    volume = np.random.randint(1000, 50000, n).astype(float)

    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=idx,
    )
    return df


@pytest.fixture
def minimal_df() -> pd.DataFrame:
    """50-row minimal DataFrame for edge-case tests."""
    np.random.seed(7)
    n = 50
    close = 1950.0 + np.cumsum(np.random.randn(n))
    noise = np.abs(np.random.randn(n)) * 1.5
    df = pd.DataFrame(
        {
            "open": close + np.random.randn(n),
            "high": close + noise,
            "low": close - noise,
            "close": close,
            "volume": np.random.randint(500, 10000, n).astype(float),
        },
        index=pd.date_range("2024-06-01", periods=n, freq="1h"),
    )
    return df


# ---------------------------------------------------------------------------
# EMA tests
# ---------------------------------------------------------------------------


def test_calculate_ema_length(sample_df: pd.DataFrame) -> None:
    """EMA series must have the same length as the input DataFrame."""
    ema = calculate_ema(sample_df, period=20)
    assert len(ema) == len(sample_df)


def test_calculate_ema_values_finite(sample_df: pd.DataFrame) -> None:
    """EMA should not contain NaN after warm-up period."""
    ema = calculate_ema(sample_df, period=20)
    # After the warm-up the EMA should be finite
    assert ema.iloc[20:].isna().sum() == 0


def test_calculate_ema(sample_df: pd.DataFrame) -> None:
    """EMA values should be close to a simple rolling mean for a flat series."""
    flat = sample_df.copy()
    flat["close"] = 2000.0
    ema = calculate_ema(flat, period=10)
    # For a constant series the EMA should converge to that constant
    assert abs(float(ema.iloc[-1]) - 2000.0) < 1e-6


def test_calculate_all_emas_columns(sample_df: pd.DataFrame) -> None:
    """calculate_all_emas should add ema_20, ema_50, ema_200 columns."""
    df = sample_df.copy()
    result = calculate_all_emas(df)
    for col in ("ema_20", "ema_50", "ema_200"):
        assert col in result.columns, f"Missing column: {col}"


# ---------------------------------------------------------------------------
# RSI tests
# ---------------------------------------------------------------------------


def test_calculate_rsi_range(sample_df: pd.DataFrame) -> None:
    """RSI values must be in [0, 100]."""
    df = sample_df.copy()
    calculate_rsi(df, period=14)
    assert "rsi" in df.columns
    assert df["rsi"].min() >= 0.0
    assert df["rsi"].max() <= 100.0


def test_calculate_rsi_no_nan(sample_df: pd.DataFrame) -> None:
    """RSI should not contain NaN (filled to 50 for initial bars)."""
    df = sample_df.copy()
    calculate_rsi(df, period=14)
    assert df["rsi"].isna().sum() == 0


def test_calculate_rsi_extreme(sample_df: pd.DataFrame) -> None:
    """Constant increasing prices should push RSI near 100."""
    df = sample_df.copy()
    df["close"] = [2000.0 + i for i in range(len(df))]
    calculate_rsi(df, period=14)
    assert float(df["rsi"].iloc[-1]) > 80.0


# ---------------------------------------------------------------------------
# MACD tests
# ---------------------------------------------------------------------------


def test_calculate_macd(sample_df: pd.DataFrame) -> None:
    """MACD should add macd, macd_signal and macd_hist columns."""
    df = sample_df.copy()
    calculate_macd(df)
    for col in ("macd", "macd_signal", "macd_hist"):
        assert col in df.columns, f"Missing column: {col}"


def test_macd_histogram_is_difference(sample_df: pd.DataFrame) -> None:
    """MACD histogram must equal MACD minus signal line."""
    df = sample_df.copy()
    calculate_macd(df)
    diff = (df["macd"] - df["macd_signal"] - df["macd_hist"]).abs()
    assert diff.max() < 1e-9


# ---------------------------------------------------------------------------
# Bollinger Bands tests
# ---------------------------------------------------------------------------


def test_calculate_bbands_width(sample_df: pd.DataFrame) -> None:
    """Upper band must always be above lower band."""
    df = sample_df.copy()
    calculate_bbands(df, period=20, std=2.0)
    for col in ("bb_upper", "bb_lower", "bb_middle", "bb_width"):
        assert col in df.columns
    # Upper > lower (after warm-up)
    valid = df.dropna(subset=["bb_upper", "bb_lower"])
    assert (valid["bb_upper"] >= valid["bb_lower"]).all()


def test_bbands_width_positive(sample_df: pd.DataFrame) -> None:
    """Bollinger Band width must be non-negative."""
    df = sample_df.copy()
    calculate_bbands(df, period=20, std=2.0)
    valid = df["bb_width"].dropna()
    assert (valid >= 0).all()


# ---------------------------------------------------------------------------
# ATR tests
# ---------------------------------------------------------------------------


def test_calculate_atr_positive(sample_df: pd.DataFrame) -> None:
    """ATR values must be strictly positive."""
    df = sample_df.copy()
    calculate_atr(df, period=14)
    assert "atr" in df.columns
    valid = df["atr"].dropna()
    assert (valid > 0).all()


def test_calculate_atr_length(sample_df: pd.DataFrame) -> None:
    """ATR series must match DataFrame length."""
    df = sample_df.copy()
    calculate_atr(df, period=14)
    assert len(df["atr"]) == len(df)


# ---------------------------------------------------------------------------
# ADX tests
# ---------------------------------------------------------------------------


def test_calculate_adx_range(sample_df: pd.DataFrame) -> None:
    """ADX must be in [0, 100]."""
    df = sample_df.copy()
    calculate_adx(df, period=14)
    for col in ("adx", "adx_pos", "adx_neg"):
        assert col in df.columns
    assert df["adx"].min() >= 0.0
    assert df["adx"].max() <= 100.0


def test_calculate_adx_no_nan(sample_df: pd.DataFrame) -> None:
    """ADX should not contain NaN (filled to 0)."""
    df = sample_df.copy()
    calculate_adx(df, period=14)
    assert df["adx"].isna().sum() == 0


# ---------------------------------------------------------------------------
# Signal strength enum tests
# ---------------------------------------------------------------------------


def test_signal_strength_enum() -> None:
    """SignalStrength values must be ordered correctly."""
    assert SignalStrength.WEAK < SignalStrength.MEDIUM
    assert SignalStrength.MEDIUM < SignalStrength.STRONG
    assert SignalStrength.STRONG < SignalStrength.VERY_STRONG
    assert int(SignalStrength.WEAK) == 1
    assert int(SignalStrength.VERY_STRONG) == 4


def test_signal_direction_constants() -> None:
    """SignalDirection constants must have expected string values."""
    assert SignalDirection.BUY == "BUY"
    assert SignalDirection.SELL == "SELL"
    assert SignalDirection.NEUTRAL == "NEUTRAL"


# ---------------------------------------------------------------------------
# Fibonacci tests
# ---------------------------------------------------------------------------


def test_fibonacci_levels(sample_df: pd.DataFrame) -> None:
    """Fibonacci levels should be ordered between high and low."""
    high = 2100.0
    low = 2000.0
    fibs = calculate_fibonacci_levels(high, low)

    required_keys = {"0.0", "0.236", "0.382", "0.500", "0.618", "0.786", "1.0"}
    assert required_keys.issubset(set(fibs.keys()))

    # All retracement levels between low and high
    for key in ("0.236", "0.382", "0.500", "0.618", "0.786"):
        assert low <= fibs[key] <= high, f"Fib level {key}={fibs[key]} outside [{low}, {high}]"


def test_fibonacci_zero_level(sample_df: pd.DataFrame) -> None:
    """Level 0.0 must equal the high, level 1.0 must equal the low."""
    fibs = calculate_fibonacci_levels(2100.0, 2000.0)
    assert fibs["0.0"] == 2100.0
    assert fibs["1.0"] == 2000.0


# ---------------------------------------------------------------------------
# add_all_indicators integration test
# ---------------------------------------------------------------------------


def test_add_all_indicators_columns(sample_df: pd.DataFrame) -> None:
    """add_all_indicators should populate the full set of expected columns."""
    df = sample_df.copy()
    enriched = add_all_indicators(df)
    expected_cols = [
        "ema_20", "ema_50", "ema_200",
        "rsi",
        "macd", "macd_signal", "macd_hist",
        "bb_upper", "bb_middle", "bb_lower",
        "atr",
        "adx", "adx_pos", "adx_neg",
        "supertrend", "supertrend_direction",
        "obv",
        "vwap",
    ]
    for col in expected_cols:
        assert col in enriched.columns, f"Missing indicator column: {col}"


def test_add_all_indicators_no_extra_nan(sample_df: pd.DataFrame) -> None:
    """After warm-up, key indicator columns should not contain NaN."""
    df = sample_df.copy()
    enriched = add_all_indicators(df)
    tail = enriched.tail(50)
    for col in ("rsi", "macd", "ema_20", "atr", "adx"):
        nan_count = tail[col].isna().sum()
        assert nan_count == 0, f"Column {col} has {nan_count} NaN values in last 50 rows"


def test_add_all_indicators_small_df() -> None:
    """add_all_indicators should return without error on a small DataFrame."""
    small = pd.DataFrame(
        {
            "open": [2000.0, 2001.0],
            "high": [2005.0, 2006.0],
            "low": [1995.0, 1996.0],
            "close": [2002.0, 2003.0],
            "volume": [1000.0, 1100.0],
        }
    )
    result = add_all_indicators(small)
    # Should return without crashing
    assert result is not None


# ---------------------------------------------------------------------------
# RSI divergence test
# ---------------------------------------------------------------------------


def test_detect_rsi_divergence_no_crash(sample_df: pd.DataFrame) -> None:
    """detect_rsi_divergence should return a valid value or None without crashing."""
    df = sample_df.copy()
    calculate_rsi(df)
    result = detect_rsi_divergence(df, lookback=14)
    assert result in ("bullish", "bearish", None)


def test_detect_rsi_divergence_needs_rsi(sample_df: pd.DataFrame) -> None:
    """detect_rsi_divergence should calculate RSI itself if not present."""
    df = sample_df.copy()
    if "rsi" in df.columns:
        del df["rsi"]
    # Should not raise
    result = detect_rsi_divergence(df, lookback=14)
    assert result in ("bullish", "bearish", None)
