from datetime import datetime, timedelta

# Stock Parameters
# Stock Parameters
STOCK_SYMBOL = "WMT"

# Strategy Parameters
STRATEGY_PARAMS = {
    "sma_crossover": {"short_window": 20, "long_window": 50, "warmup_days": 200},
    "rsi": {"period": 14, "overbought": 70, "oversold": 30, "warmup_days": 100},
    "macd": {"short_window": 12, "long_window": 26, "signal_window": 9, "warmup_days": 150}
}

# Selected Strategy (Modify This)
SELECTED_STRATEGY = "sma_crossover"

# Calculate the required data range based on the selected strategy
WARMUP_DAYS = STRATEGY_PARAMS[SELECTED_STRATEGY].get("warmup_days", 180)
TRADE_WINDOW_START_DATE = (datetime.today() - timedelta(days=WARMUP_DAYS)).strftime('%Y-%m-%d')
TRADE_WINDOW_END_DATE = datetime.today().strftime('%Y-%m-%d')

# Trade Parameters
INITIAL_CAPITAL = 32000
MIN_PROFIT_THRESHOLD = 0.005  # 0.5% profit
MIN_HOLD_DAYS = 1
STOP_LOSS_THRESHOLD = -0.03  # -3% stop-loss

# AI Toggle
USE_AI = True

