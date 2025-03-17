import logging
import numpy as np
from strategies.strategy_base import Strategy
from utils import extract_close_column

logger = logging.getLogger(__name__)

class MACD(Strategy):
    def __init__(self, data, fast_period=12, slow_period=26, signal_period=9):
        super().__init__(data)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period

    def get_name(self): return "MACD"

    def get_feature_column_names(self): return ['MACD', 'MACD_signal']

    def generate_signals(self):
        """Generate MACD trading signals robustly."""
        logger.info("Generating MACD trade signals...")

        # Robust selection of Close column
        close_col = extract_close_column(self.data)

        close_prices = self.data[close_col[0]].astype(float)

        self.data['EMA_fast'] = self.ema(close_prices, self.fast_period)
        self.data['EMA_slow'] = self.ema(close_prices, self.slow_period)

        self.data['MACD'] = self.data['EMA_fast'] - self.data['EMA_slow']

        self.data['MACD_signal'] = self.ema(self.data['MACD'], self.signal_period)
        self.data['MACD_histogram'] = self.data['MACD'] - self.data['MACD_signal']

        # Generate signals based on MACD crossover
        self.data['Signal'] = 0
        self.data.loc[self.data['MACD'] > self.data['MACD_signal'], 'Signal'] = 1
        self.data.loc[self.data['MACD'] < self.data['MACD_signal'], 'Signal'] = -1

        buys = (self.data['Signal'] == 1).sum()
        sells = (self.data['Signal'] == -1).sum()
        logger.info(f"Buy signals: {buys}, Sell signals: {sells}")

        return self.data['Signal']

    def ema(self, prices, period):
        """Compute Exponential Moving Average."""
        return prices.ewm(span=period, adjust=False).mean()
