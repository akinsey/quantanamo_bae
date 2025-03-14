import logging
logger = logging.getLogger(__name__)

import numpy as np

def generate_trading_signals(data):
    logger.info("Generating trading signals using SMA crossover...")
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['Signal'] = np.where(data['SMA_20'] > data['SMA_50'], 1, -1)
    buys = (data['Signal'] == 1).sum()
    sells = (data['Signal'] == -1).sum()
    logger.info(f"Trading signals generated. Buy signals: {buys}, Sell signals: {sells}")
    return data
