from datetime import datetime, timedelta

# Stock Parameters
STOCK_SYMBOL = "WMT"
TRADE_WINDOW_START_DATE = (datetime.today() - timedelta(days=180)).strftime('%Y-%m-%d')
TRADE_WINDOW_END_DATE = datetime.today().strftime('%Y-%m-%d')

# Trade Parameters
INITIAL_CAPITAL = 32000
MIN_PROFIT_THRESHOLD = 0.005  # 0.5% profit before selling
MIN_HOLD_DAYS = 1
STOP_LOSS_THRESHOLD = -0.03  # 3% loss before triggering stop-loss sell

# AI Toggle
USE_AI = True
