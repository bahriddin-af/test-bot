"""Database package for Gold Trading Bot."""
from database.trade_history import TradeHistory, initialize_db

__all__ = ["TradeHistory", "initialize_db"]
