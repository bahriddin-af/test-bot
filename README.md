# Gold Trading Bot (XAU/USD)

An automated, production-ready algorithmic trading bot for Gold (XAU/USD) built with Python. Combines multi-timeframe technical analysis, robust risk management, news filtering, and Telegram notifications.

---

## Features

- **Multi-Timeframe Analysis** – Checks 1h, 4h and daily charts before generating a signal
- **12+ Technical Indicators** – EMA, RSI, MACD, Stochastic RSI, Bollinger Bands, ATR, ADX, Supertrend, OBV, VWAP, Pivot Points, Fibonacci
- **Signal Confirmation Engine** – Requires a configurable minimum number of confluent signals (default 3)
- **Risk Management** – ATR-based stop-loss, dynamic position sizing, daily loss limit, max drawdown circuit-breaker
- **Partial Profit Taking** – Automatically closes 33% at TP1, 50% of remainder at TP2, full close at TP3
- **Trailing Stop & Break-Even** – Protects profits after TP1 hit
- **News Filter** – Pauses trading around high-impact USD/Gold Forex Factory events
- **Paper Trading Mode** – Full simulation using real market data, zero risk
- **Telegram Alerts** – Signal notifications, trade opens/closes, daily reports with charts
- **Backtesting Engine** – Walk-forward backtest with Sharpe ratio, profit factor, parameter optimisation
- **Chart Generation** – Candlestick charts with indicators sent directly to Telegram
- **Docker Ready** – Single `docker-compose up` deployment
- **Fully Async** – Built on asyncio / aiosqlite for efficient I/O

---

## Project Structure

```
gold-trading-bot/
├── main.py               # Entry point & main loop
├── config.py             # Pydantic settings
├── backtest.py           # Backtesting engine
├── bot/
│   ├── indicators.py     # Technical indicators
│   ├── signal_engine.py  # Signal generation
│   ├── risk_manager.py   # Risk management
│   ├── order_manager.py  # Trade execution / paper trading
│   ├── news_filter.py    # Forex Factory news filter
│   └── telegram_bot.py   # Telegram notifications
├── database/
│   └── trade_history.py  # SQLite async ORM
├── utils/
│   ├── chart_generator.py # mplfinance charts
│   └── logger.py          # Rotating coloured logger
├── tests/
│   └── test_indicators.py # Pytest test suite
├── logs/                  # Rotating log files
├── data/                  # SQLite DB & exports
├── .env.example           # Configuration template
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

---

## Installation

### Prerequisites
- Python 3.11+
- pip

### Steps

```bash
# 1. Clone / unzip the project
cd gold-trading-bot

# 2. Create and activate a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy and edit the configuration
cp .env.example .env
# Open .env and fill in your settings (see Configuration section)
```

---

## Configuration

Edit `.env` (copied from `.env.example`):

| Variable | Description | Default |
|---|---|---|
| `DEMO_MODE` | Paper trading (no real orders) | `True` |
| `SYMBOL` | Trading symbol | `XAU/USD` |
| `TIMEFRAME` | Analysis timeframe | `1h` |
| `RISK_PERCENT` | Risk per trade (0.1–5.0%) | `1.5` |
| `MAX_TRADES` | Max concurrent open trades | `3` |
| `ATR_MULTIPLIER_SL` | ATR multiplier for stop-loss | `1.5` |
| `RR_RATIO_MIN` | Minimum risk-reward ratio | `2.0` |
| `MAX_DAILY_LOSS_PERCENT` | Daily loss circuit-breaker (%) | `5.0` |
| `MAX_DRAWDOWN_PERCENT` | Max drawdown circuit-breaker (%) | `10.0` |
| `MIN_CONFIRMATIONS` | Minimum signal confirmations | `3` |
| `TELEGRAM_BOT_TOKEN` | Telegram bot token | – |
| `TELEGRAM_CHAT_ID` | Telegram chat/channel ID | – |
| `NEWS_FILTER_ENABLED` | Enable news blackout | `True` |
| `NEWS_FILTER_HOURS_BEFORE` | Blackout hours before event | `2` |
| `NEWS_FILTER_HOURS_AFTER` | Blackout hours after event | `1` |

---

## Running the Bot

```bash
# Paper trading mode (safe, no real money)
python main.py

# The bot:
# - Runs a cycle every 5 minutes
# - Sends a heartbeat to Telegram every hour
# - Sends a daily report at 22:00 UTC
# - Logs everything to logs/gold_bot.log
```

---

## Backtesting

```bash
# Run a backtest over the last 2 years of hourly gold data
python backtest.py

# With options
python backtest.py --start 2023-01-01 --end 2024-01-01 --balance 10000 --export --plot

# Run parameter optimisation
python backtest.py --optimize
```

### Backtest output includes:
- Total trades, Win rate, P&L
- Max drawdown, Sharpe ratio, Profit factor
- CSV + Excel trade log (`data/backtest_results.*`)
- Equity curve PNG (`data/backtest_equity_curve.png`)

---

## Running Tests

```bash
pip install pytest pytest-asyncio
pytest tests/ -v
```

---

## Docker Deployment

```bash
# Build and start
docker-compose up -d

# View logs
docker-compose logs -f gold_bot

# Stop
docker-compose down
```

The container mounts `./logs` and `./data` volumes for persistence across restarts.

---

## Telegram Bot Setup

1. Open Telegram and message [@BotFather](https://t.me/BotFather)
2. Send `/newbot` and follow the prompts
3. Copy the API token to `TELEGRAM_BOT_TOKEN` in `.env`
4. Start a chat with your bot or add it to a channel
5. Get your chat ID:
   - For personal chat: message [@userinfobot](https://t.me/userinfobot)
   - For a channel: use `@username` (e.g. `@mychannel`) or the numeric ID
6. Set `TELEGRAM_CHAT_ID` in `.env`

---

## Strategy Explanation

### Signal Generation
The bot analyses XAU/USD across three timeframes (1h primary, 4h and daily for bias) using five independent check categories:

1. **Trend Following** – EMA alignment (20/50/200) and Supertrend direction
2. **Momentum** – RSI level & divergence, MACD crossover, Stochastic RSI
3. **Support & Resistance** – Classic pivot points and Fibonacci retracement levels
4. **Volatility** – Bollinger Band breakouts and ADX trend strength
5. **Volume** – OBV trend confirmation and relative volume spikes

A signal is generated only when at least `MIN_CONFIRMATIONS` (default 3) independent confirmations point in the same direction.

### Trade Management
- **Entry**: At the close of the trigger candle
- **Stop Loss**: ATR × `ATR_MULTIPLIER_SL` from entry
- **TP1**: `RR_RATIO_MIN` × SL distance (33% closed here, SL moved to break-even)
- **TP2**: 1.5 × TP1 distance (50% of remainder closed)
- **TP3**: 2.5 × TP1 distance (full close)
- **Trailing Stop**: Applied throughout the trade after entry
- **Time Exit**: Positions open > 48 hours are closed automatically

---

## Risk Warning

**Trading gold and foreign exchange involves substantial risk of loss. This software is provided for educational and research purposes only. Past performance of the backtester is not indicative of future results. Never risk money you cannot afford to lose. Always test thoroughly in paper/demo mode before considering live deployment. The authors accept no liability for financial losses incurred through the use of this software.**

---

## License

MIT License – see LICENSE file for details.
