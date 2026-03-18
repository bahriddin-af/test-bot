"""
Logging setup with rotating file handler and colorized console output.
"""

from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from typing import Dict, Optional

# Try to import colorama for Windows color support
try:
    import colorama
    colorama.init(autoreset=True)
    HAS_COLORAMA = True
except ImportError:
    HAS_COLORAMA = False

# ANSI color codes (work on Linux/macOS; colorama makes them work on Windows)
COLORS = {
    "DEBUG": "\033[36m",       # Cyan
    "INFO": "\033[32m",        # Green
    "WARNING": "\033[33m",     # Yellow
    "ERROR": "\033[31m",       # Red
    "CRITICAL": "\033[35m",    # Magenta
    "RESET": "\033[0m",
}

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "gold_bot.log")
MAX_BYTES = 10 * 1024 * 1024  # 10 MB
BACKUP_COUNT = 5

# Module-level log levels (override root level for specific noisy modules)
MODULE_LEVELS: Dict[str, int] = {
    "ccxt": logging.WARNING,
    "urllib3": logging.WARNING,
    "asyncio": logging.WARNING,
    "telegram": logging.INFO,
    "aiosqlite": logging.WARNING,
    "matplotlib": logging.WARNING,
    "yfinance": logging.WARNING,
    "peewee": logging.WARNING,
}

_loggers: Dict[str, logging.Logger] = {}


class ColorFormatter(logging.Formatter):
    """Formatter that adds ANSI color codes to log level names."""

    def format(self, record: logging.LogRecord) -> str:
        level = record.levelname
        color = COLORS.get(level, COLORS["RESET"])
        reset = COLORS["RESET"]
        record.levelname = f"{color}{level:8s}{reset}"
        formatted = super().format(record)
        record.levelname = level  # restore for other handlers
        return formatted


def _ensure_log_dir() -> None:
    """Create the logs directory if it doesn't exist."""
    os.makedirs(LOG_DIR, exist_ok=True)


def _build_file_handler() -> RotatingFileHandler:
    """Build and return a rotating file handler."""
    _ensure_log_dir()
    handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=MAX_BYTES,
        backupCount=BACKUP_COUNT,
        encoding="utf-8",
    )
    handler.setLevel(logging.DEBUG)
    file_fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(file_fmt)
    return handler


def _build_console_handler() -> logging.StreamHandler:
    """Build and return a colorized console (stdout) handler."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    console_fmt = ColorFormatter(
        fmt="%(asctime)s | %(levelname)s | %(name)-25s | %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(console_fmt)
    return handler


def _configure_root_logger() -> None:
    """Configure the root logger once."""
    root = logging.getLogger()
    if root.handlers:
        return  # already configured

    root.setLevel(logging.DEBUG)
    root.addHandler(_build_file_handler())
    root.addHandler(_build_console_handler())

    # Apply per-module level overrides
    for module_name, level in MODULE_LEVELS.items():
        logging.getLogger(module_name).setLevel(level)


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """
    Return (or create) a named logger.

    Args:
        name: Logger name, typically __name__ of the calling module.
        level: Optional override log level for this specific logger.

    Returns:
        Configured logging.Logger instance.
    """
    _configure_root_logger()

    if name in _loggers:
        return _loggers[name]

    logger = logging.getLogger(name)
    if level is not None:
        logger.setLevel(level)

    _loggers[name] = logger
    return logger
