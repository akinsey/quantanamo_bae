import logging # strategies/sma_strategy.py
import numpy as np
from config import SMA_SHORT_WINDOW, SMA_LONG_WINDOW
from strategies.strategy_base import Strategy

logger = logging.getLogger(__name__)

class SMA(Strategy):
    def __init__(self, data, short_window=SMA_SHORT_WINDOW, long_window=SMA_LONG_WINDOW):
        super().__init__(data)
        self.short_window = short_window
        self.long_window = long_window

    def get_name(self): return "SMA"

    def get_feature_column_names(self): return ["SMA_short", "SMA_long"]

    def generate_signals(self):
        """Generate buy/sell signals based on SMA crossover."""
        logger.info("Generating SMA trade signals...")

        # Validate sufficient data for SMA calculations
        min_required_days = max(SMA_SHORT_WINDOW, SMA_LONG_WINDOW)
        if len(self.data) < min_required_days:
            logger.warning(f"Data length ({len(self.data)}) is insufficient for SMA calculations.")
            raise ValueError("Not enough data for SMA window")

        if 'Close' not in self.data.columns:
            logger.error("Missing 'Close' column in data. Cannot generate signals.")
            raise ValueError("Missing 'Close' column in data")

        # Calculate SMAs
        self.data['SMA_short'] = self.data['Close'].rolling(window=SMA_SHORT_WINDOW).mean()
        self.data['SMA_long'] = self.data['Close'].rolling(window=SMA_LONG_WINDOW).mean()

        # Ensure SMA calculations are valid
        if self.data['SMA_short'].isna().all() or self.data['SMA_long'].isna().all():
            logger.warning("Insufficient data for rolling window. All SMA values are NaN.")
            raise ValueError("SMA calculations resulted in all NaN values.")

        # Generate trading signals (1 = Buy, -1 = Sell)
        self.data['Signal'] = np.where(self.data['SMA_short'] > self.data['SMA_long'], 1, -1)

        # Logging trade signal counts
        buys = (self.data['Signal'] == 1).sum()
        sells = (self.data['Signal'] == -1).sum()
        logger.info(f"Buy signals: {buys}, Sell signals: {sells}")

        return self.data["Signal"]

