# strategy.py

import logging
logger = logging.getLogger(__name__)

import numpy as np
from config import SMA_SHORT_WINDOW, SMA_LONG_WINDOW

def generate_trading_signals(data):
    """
    Generate trading signals based on rolling averages.
    Uses SMA_SHORT_WINDOW and SMA_LONG_WINDOW from config.py.
    """
    logger.info("Generating trading signals using SMA crossover...")

    data['SMA_short'] = data['Close'].rolling(window=SMA_SHORT_WINDOW).mean()
    data['SMA_long'] = data['Close'].rolling(window=SMA_LONG_WINDOW).mean()

    data['Signal'] = np.where(data['SMA_short'] > data['SMA_long'], 1, -1)

    buys = (data['Signal'] == 1).sum()
    sells = (data['Signal'] == -1).sum()
    logger.info(f"Trading signals generated. Buy signals: {buys}, Sell signals: {sells}")

    return data
