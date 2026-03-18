"""
Configuration module using Pydantic Settings.
Loads all settings from environment variables / .env file.
"""

from __future__ import annotations

import os
from typing import Optional

from pydantic import validator
from pydantic_settings import BaseSettings


class Config(BaseSettings):
    # Broker Settings
    broker_api_key: str = "demo_key"
    broker_api_secret: str = "demo_secret"
    broker_name: str = "oanda"

    # Telegram
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    # Trading
    symbol: str = "XAU/USD"
    timeframe: str = "1h"
    risk_percent: float = 1.5
    max_trades: int = 3
    demo_mode: bool = True

    # Risk Management
    atr_multiplier_sl: float = 1.5
    rr_ratio_min: float = 2.0
    max_daily_loss_percent: float = 5.0
    max_drawdown_percent: float = 10.0

    # Indicators
    ema_fast: int = 20
    ema_medium: int = 50
    ema_slow: int = 200
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    adx_period: int = 14
    adx_threshold: float = 25.0
    bb_period: int = 20
    bb_std: float = 2.0
    atr_period: int = 14

    # Signal
    min_confirmations: int = 3

    # News Filter
    news_filter_enabled: bool = True
    news_filter_hours_before: int = 2
    news_filter_hours_after: int = 1

    # Database
    db_path: str = "data/trades.db"

    # Logging
    log_level: str = "INFO"
    log_dir: str = "logs"

    # Paper trading initial balance
    initial_balance: float = 10000.0

    @validator("risk_percent")
    def validate_risk_percent(cls, v: float) -> float:
        if not (0.1 <= v <= 5.0):
            raise ValueError(
                f"risk_percent must be between 0.1 and 5.0, got {v}"
            )
        return v

    @validator("max_trades")
    def validate_max_trades(cls, v: int) -> int:
        if v < 1:
            raise ValueError("max_trades must be at least 1")
        return v

    @validator("rr_ratio_min")
    def validate_rr_ratio(cls, v: float) -> float:
        if v < 1.0:
            raise ValueError("rr_ratio_min must be at least 1.0")
        return v

    @property
    def display_symbol(self) -> str:
        """Return a human-readable symbol string."""
        return self.symbol.replace("/", "")

    @property
    def yfinance_symbol(self) -> str:
        """Return the yfinance-compatible ticker for gold."""
        mapping = {
            "XAU/USD": "GC=F",
            "XAUUSD": "GC=F",
            "GOLD": "GC=F",
        }
        return mapping.get(self.symbol.upper(), "GC=F")

    @property
    def is_paper_trading(self) -> bool:
        """Alias for demo_mode."""
        return self.demo_mode

    @property
    def formatted_symbol(self) -> str:
        """Return symbol in XAU/USD format."""
        sym = self.symbol.upper().replace("XAUUSD", "XAU/USD")
        return sym

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"


def get_config() -> Config:
    """Return a singleton Config instance."""
    return Config()
