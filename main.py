import logging
import argparse
from config import *
from data_loader import fetch_historical_data
from ai_model import AIModel
from backtester import Backtester

# Add your new strategies here
from strategies.sma import SMA
from strategies.rsi import RSI
from strategies.macd import MACD

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)

class QuantanamoBae:
    def __init__(self, stock_symbol, strategy_name='SMA', use_ai=USE_AI):
        self.stock_symbol = stock_symbol
        self.strategy_name = strategy_name
        self.use_ai = use_ai
        self.data = None
        self.strategy = None
        self.model = None
        self.scaler = None
        self.logger = logging.getLogger(__name__)

    def select_strategy(self):
        """
        Select the trading strategy dynamically.
        """
        strategies = {
            "SMA": SMA,
            "RSI": RSI,
            "MACD": MACD
        }

        if self.strategy_name not in strategies:
            self.logger.error(f"Invalid strategy name '{self.strategy_name}'. Available: {list(strategies.keys())}")
            exit(1)

        self.logger.info(f"Using {self.strategy_name} strategy.")
        return strategies[self.strategy_name]

    def prepare_data(self):
        """
        Fetch historical stock data and generate trading signals.
        """
        self.logger.info(f"Fetching data for {self.stock_symbol}...")
        self.data = fetch_historical_data(self.stock_symbol, TRADE_WINDOW_START_DATE, TRADE_WINDOW_END_DATE)

        if self.data.empty:
            self.logger.error("No data retrieved. Exiting.")
            exit(1)

        # Dynamically initialize the selected strategy and generate signals
        StrategyClass = self.select_strategy()
        self.strategy = StrategyClass(self.data)
        self.data["Signal"] = self.strategy.generate_signals()

    def train_and_backtest(self):
        """
        Train AI model if enabled and run backtest.
        """
        if self.use_ai:
            ai_model = AIModel(self.strategy)
            self.model, self.scaler = ai_model.train(self.data)

        backtester = Backtester(self.data, INITIAL_CAPITAL, self.use_ai, self.model, self.scaler)
        backtester.run()

    def run(self):
        """
        Execute the entire pipeline: data loading, strategy selection, AI training, and backtesting.
        """
        try:
            self.logger.info("Welcome to Quantanamo Bae")
            self.prepare_data()
            self.train_and_backtest()
            self.logger.info("We hope you enjoyed your stay.")
        except Exception as e:
            self.logger.critical(f"Fatal error: {e}", exc_info=True)
            exit(1)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Quantanamo Bae with selected trading strategy.")
    parser.add_argument('--strategy', type=str, choices=['SMA', 'RSI', 'MACD'], default='SMA', help="Trading strategy to use (default: SMA)")
    args = parser.parse_args()

    quantanamo_bae = QuantanamoBae(STOCK_SYMBOL, args.strategy, USE_AI)
    quantanamo_bae.run()
