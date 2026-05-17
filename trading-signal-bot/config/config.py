"""
Trading Signal Bot - Configuration
All sensitive credentials are loaded from .env file
Never hardcode credentials in this file
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# ADMIN SECURITY
# ============================================================
ADMIN_KEY = os.getenv("ADMIN_KEY")  # Your secret key to start the bot

# ============================================================
# TELEGRAM
# ============================================================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

# ============================================================
# BYBIT (Crypto Futures)
# ============================================================
BYBIT_API_KEY = os.getenv("BYBIT_API_KEY")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET")
BYBIT_TESTNET = os.getenv("BYBIT_TESTNET", "true").lower() == "true"

# ============================================================
# TRADELOCKER (Forex)
# ============================================================
TRADELOCKER_EMAIL = os.getenv("TRADELOCKER_EMAIL")
TRADELOCKER_PASSWORD = os.getenv("TRADELOCKER_PASSWORD")
TRADELOCKER_SERVER = os.getenv("TRADELOCKER_SERVER", "PRDTL")
TRADELOCKER_ENV = os.getenv("TRADELOCKER_ENV", "https://demo.tradelocker.com")

# ============================================================
# ALPACA (Stocks & Options)
# ============================================================
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

# ============================================================
# FINNHUB (News & Sentiment)
# ============================================================
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")

# ============================================================
# RISK MANAGEMENT
# ============================================================
MAX_RISK_PER_TRADE = float(os.getenv("MAX_RISK_PER_TRADE", "1.0"))       # % of account
MAX_DAILY_LOSS = float(os.getenv("MAX_DAILY_LOSS", "3.0"))                # % of account
DEFAULT_LEVERAGE = int(os.getenv("DEFAULT_LEVERAGE", "10"))
MIN_SIGNAL_CONFIDENCE = float(os.getenv("MIN_SIGNAL_CONFIDENCE", "75.0")) # % minimum to send signal

# ============================================================
# SIGNAL SETTINGS
# ============================================================
MAX_SIGNALS_PER_DAY = int(os.getenv("MAX_SIGNALS_PER_DAY", "10"))  # Per market
SIGNAL_COOLDOWN_MINUTES = int(os.getenv("SIGNAL_COOLDOWN_MINUTES", "30"))

# ============================================================
# CRYPTO PAIRS TO TRADE
# ============================================================
CRYPTO_PAIRS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT",
    "BNBUSDT", "XRPUSDT", "DOGEUSDT"
]

# ============================================================
# FOREX PAIRS TO TRADE
# ============================================================
FOREX_PAIRS = [
    "EURUSD", "GBPUSD", "USDJPY",
    "AUDUSD", "USDCAD", "XAUUSD"  # Gold included
]

# ============================================================
# STOCKS TO TRADE
# ============================================================
STOCK_SYMBOLS = [
    "AAPL", "TSLA", "NVDA", "MSFT",
    "AMZN", "META", "GOOGL", "SPY"
]

# ============================================================
# TIMEFRAMES
# ============================================================
CRYPTO_TIMEFRAMES = {
    "entry": "15",      # 15 minutes for entry signals
    "trend": "60",      # 1 hour for trend bias
    "structure": "240"  # 4 hours for market structure
}

FOREX_TIMEFRAMES = {
    "entry": "15",
    "trend": "240",
    "structure": "D"    # Daily for structure
}

STOCK_TIMEFRAMES = {
    "entry": "5",       # 5 min for intraday
    "trend": "60",
    "structure": "D"
}

# ============================================================
# MARKET HOURS (UTC)
# ============================================================
FOREX_SESSIONS = {
    "london_open": "07:00",
    "london_close": "16:00",
    "ny_open": "12:00",
    "ny_close": "21:00",
    "overlap_start": "12:00",  # Best time - London/NY overlap
    "overlap_end": "16:00"
}

STOCK_MARKET_HOURS = {
    "open": "13:30",   # 9:30 AM ET in UTC
    "close": "20:00"   # 4:00 PM ET in UTC
}

# ============================================================
# DATABASE
# ============================================================
DB_PATH = os.getenv("DB_PATH", "data/trades.db")

# ============================================================
# LOGGING
# ============================================================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", "logs/bot.log")
