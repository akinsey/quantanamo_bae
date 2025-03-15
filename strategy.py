import logging # strategy.py
logger = logging.getLogger(__name__)

import numpy as np
from config import SMA_SHORT_WINDOW, SMA_LONG_WINDOW

def generate_trading_signals(data):
    """
    Generate trading signals based on rolling averages.
    Uses SMA_SHORT_WINDOW and SMA_LONG_WINDOW from config.py.
    """
    logger.info("Generating trading signals using SMA crossover...")

    # Validate we have enough data for window
    min_required_days = max(SMA_SHORT_WINDOW, SMA_LONG_WINDOW)
    if len(data) < min_required_days:
        logger.warning(f"Data length ({len(data)}) is less than {min_required_days} days required for SMA calculations.")
        raise ValueError("Not enough data for SMA window")

    # Validate that the 'Close' column exists
    if 'Close' not in data.columns:
        logger.error("Missing 'Close' column in data. Cannot generate signals.")
        raise ValueError("Missing 'Close' column in data")

    # Param approach
    data['SMA_short'] = data['Close'].rolling(window=SMA_SHORT_WINDOW).mean()
    data['SMA_long']  = data['Close'].rolling(window=SMA_LONG_WINDOW).mean()

    # Check if these new columns exist or contain NaNs
    if data['SMA_short'].isna().all() or data['SMA_long'].isna().all():
        logger.warning("Insufficient data for rolling window. All SMA values are NaN.")
        raise ValueError("SMA calculations resulted in all NaN values.")

    data['Signal'] = np.where(data['SMA_short'] > data['SMA_long'], 1, -1)

    buys = (data['Signal'] == 1).sum()
    sells = (data['Signal'] == -1).sum()
    logger.info(f"Trading signals generated. Buy signals: {buys}, Sell signals: {sells}")

    return data
