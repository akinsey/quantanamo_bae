# config.py
# Existing imports and definitions...
from datetime import datetime, timedelta

# Stock Parameters (already present)
STOCK_SYMBOL = "WMT"
TRADE_WINDOW_START_DATE = (datetime.today() - timedelta(days=180)).strftime('%Y-%m-%d')
TRADE_WINDOW_END_DATE = datetime.today().strftime('%Y-%m-%d')

# Trade Parameters (already present)
INITIAL_CAPITAL = 32000
MIN_PROFIT_THRESHOLD = 0.005  # 0.5% profit
MIN_HOLD_DAYS = 1
STOP_LOSS_THRESHOLD = -0.03  # -3% stop-loss

# AI Toggle (already present)
USE_AI = True

# =========================================
#  NEW: Strategy Window Parameters
# =========================================
SMA_SHORT_WINDOW = 20
SMA_LONG_WINDOW = 50
