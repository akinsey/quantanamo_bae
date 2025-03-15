import logging
import numpy as np
from strategies.strategy_base import Strategy

logger = logging.getLogger(__name__)

class RSI(Strategy):
    def __init__(self, data, period=14, overbought=70, oversold=30):
        super().__init__(data)
        self.period = period
        self.overbought = overbought
        self.oversold = oversold

    def generate_signals(self):
        """Generate RSI signals using Wilder's smoothing method robustly."""
        logger.info("Generating RSI trading signals...")

        # Robust selection of Close column
        close_col = [col for col in self.data.columns if 'Close' in col]
        if not close_col:
            logger.error("Could not find any column containing 'Close'")
            raise ValueError("Missing 'Close' column in data")

        close_prices = self.data[close_col[0]].values.astype(float)
        logger.info(f"Using close column: {close_col[0]} with {len(close_prices)} rows")

        if len(close_prices) <= self.period:
            logger.error(f"Insufficient data ({len(close_prices)}) for RSI calculation, requires at least {self.period + 1}")
            raise ValueError("Insufficient data for RSI calculation.")

        delta = np.diff(close_prices)
        gain = np.maximum(delta, 0)
        loss = np.abs(np.minimum(delta, 0))

        initial_gain = gain[:self.period]
        initial_loss = loss[:self.period]

        if len(initial_gain) < self.period or len(initial_loss) < self.period:
            logger.error("Not enough data to calculate initial average gain/loss.")
            raise ValueError("Insufficient data for initial RSI averages.")

        avg_gain = np.zeros(len(close_prices))
        avg_loss = np.zeros(len(close_prices))

        avg_gain[self.period] = np.mean(initial_gain)
        avg_loss[self.period] = np.mean(initial_loss)

        if np.isnan(avg_gain[self.period]) or np.isnan(avg_loss[self.period]):
            logger.error("Initial avg_gain or avg_loss resulted in NaN.")
            raise ValueError("NaN encountered during RSI initialization.")

        for i in range(self.period + 1, len(close_prices)):
            avg_gain[i] = (avg_gain[i - 1] * (self.period - 1) + gain[i - 1]) / self.period
            avg_loss[i] = (avg_loss[i - 1] * (self.period - 1) + loss[i - 1]) / self.period

        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))

        self.data['RSI'] = rsi

        # Generate signals based on RSI
        self.data['Signal'] = 0
        self.data.loc[rsi < self.oversold, 'Signal'] = 1
        self.data.loc[rsi > self.overbought, 'Signal'] = -1

        buys = (self.data['Signal'] == 1).sum()
        sells = (self.data['Signal'] == -1).sum()
        logger.info(f"RSI signals generated. Buy signals: {buys}, Sell signals: {sells}")

        return self.data['Signal']

    def get_ai_features(self):
        return ['RSI']
