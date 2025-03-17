from datetime import datetime, timedelta  # config.py

# ===========================
# General Configuration
# ===========================
STOCK_SYMBOL = "WMT"  # Default stock to trade

# using a fixed datetime for today so we can keep the output consistent
TODAY = datetime(2025, 3, 11, 5, 51, 45, 203360) # datetime.today()
# pulling 180 days (more than we are using to graph, this is for sma strategy to work, allows "warmup")
TRADE_WINDOW_START_DATE = (TODAY - timedelta(days=180)).strftime('%Y-%m-%d')
TRADE_WINDOW_END_DATE = TODAY.strftime('%Y-%m-%d')

# ===========================
# Trading Parameters
# ===========================
INITIAL_CAPITAL = 32000  # Starting capital for trading
MIN_PROFIT_THRESHOLD = 0.005  # 0.5% profit target for selling
MIN_HOLD_DAYS = 1  # Minimum days to hold before selling
STOP_LOSS_THRESHOLD = -0.03  # -3% stop-loss limit

# ===========================
# AI Settings
# ===========================
USE_AI = True  # Toggle AI model usage

# ===========================
# Strategy Parameters
# ===========================
STRATEGY_NAME = "SMA"  # Default strategy
SMA_SHORT_WINDOW = 20  # Short-term Simple Moving Average window
SMA_LONG_WINDOW = 50  # Long-term Simple Moving Average window

DEBUG=False
