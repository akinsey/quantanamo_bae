import pandas as pd
from config import STOCK_SYMBOL

def flatten_columns(data):
    """Flattens multi-index column names from Yahoo Finance."""
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in data.columns]
    return data

def get_close_column(data):
    """Finds the correct 'Close' column dynamically, accounting for ticker suffixes."""
    data = flatten_columns(data)

    expected_col = f"Close_{STOCK_SYMBOL}"
    if expected_col in data.columns:
        return expected_col

    close_candidates = [col for col in data.columns if "Close" in col]
    if not close_candidates:
        raise ValueError(f"Could not find 'Close' column in data. Available columns: {list(data.columns)}")

    return close_candidates[0]

def ensure_numeric(data, col_name):
    """Convert column to numeric safely."""
    data[col_name] = pd.to_numeric(data[col_name], errors='coerce')

def sma_crossover_strategy(data, short_window=20, long_window=50):
    """SMA Crossover Strategy: Ensures enough data is used for proper rolling calculations."""
    close_col = get_close_column(data)
    ensure_numeric(data, close_col)

    # Ensure we have enough historical data before applying SMA
    if len(data) < long_window + 10:
        print(f"WARNING: Not enough historical data for SMA calculation (Required: {long_window + 10}, Available: {len(data)})")
        return data

    data["SMA_short"] = data[close_col].rolling(window=short_window).mean()
    data["SMA_long"] = data[close_col].rolling(window=long_window).mean()
    data["Signal"] = 0

    # Ensure SMAs are correctly calculated before applying conditions
    valid_sma = data["SMA_short"].notna() & data["SMA_long"].notna()

    # Restore Exact Trend Change Logic
    data.loc[valid_sma & (data["SMA_short"] > data["SMA_long"]), "Signal"] = 1  # Buy Signal
    data.loc[valid_sma & (data["SMA_short"] < data["SMA_long"]), "Signal"] = -1  # Sell Signal

    return data

def rsi_strategy(data, period=14, overbought=70, oversold=30):
    """RSI Strategy: Generates trading signals based on Relative Strength Index (RSI)."""
    close_col = get_close_column(data)
    ensure_numeric(data, close_col)

    delta = data[close_col].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    data["RSI"] = 100 - (100 / (1 + rs))
    data["Signal"] = 0
    data.loc[data["RSI"] > overbought, "Signal"] = -1  # Sell signal
    data.loc[data["RSI"] < oversold, "Signal"] = 1  # Buy signal

    return data

def macd_strategy(data, short_window=12, long_window=26, signal_window=9):
    """MACD Strategy: Generates trading signals based on MACD crossover."""
    close_col = get_close_column(data)
    ensure_numeric(data, close_col)

    data["MACD"] = data[close_col].ewm(span=short_window, adjust=False).mean() - data[close_col].ewm(span=long_window, adjust=False).mean()
    data["Signal_Line"] = data["MACD"].ewm(span=signal_window, adjust=False).mean()
    data["Signal"] = 0
    data.loc[data["MACD"] > data["Signal_Line"], "Signal"] = 1  # Buy signal
    data.loc[data["MACD"] < data["Signal_Line"], "Signal"] = -1  # Sell signal

    return data

STRATEGIES = {
    "sma_crossover": sma_crossover_strategy,
    "rsi": rsi_strategy,
    "macd": macd_strategy
}

def apply_strategy(data, strategy_name):
    """Applies the selected strategy to the data."""
    if strategy_name not in STRATEGIES:
        raise ValueError(f"Strategy '{strategy_name}' not found! Available strategies: {list(STRATEGIES.keys())}")

    return STRATEGIES[strategy_name](data)

