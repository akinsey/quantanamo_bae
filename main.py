import logging
from config import *
from data_loader import fetch_historical_data
from strategy import generate_trading_signals
from ai_model import AIModel
from backtester import Backtester  # Import the class-based backtester

# Configure logging globally before the class is instantiated
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)

class QuantanamoBae:
    def __init__(self, stock_symbol, use_ai):
        """
        Initializes the trading system.
        """
        self.stock_symbol = stock_symbol
        self.use_ai = use_ai
        self.data = None
        self.model = None
        self.scaler = None
        self.logger = logging.getLogger(__name__)  # Now picks up global config

    def prepare_data(self):
        """
        Fetch historical stock data and generate trading signals.
        """
        self.logger.info(f"Fetching data for {self.stock_symbol}...")
        self.data = fetch_historical_data(self.stock_symbol, TRADE_WINDOW_START_DATE, TRADE_WINDOW_END_DATE)

        if self.data.empty:
            self.logger.error("No data retrieved. Exiting.")
            exit(1)

        self.data = generate_trading_signals(self.data)  # Generate trading signals after fetching data

    def train_and_backtest(self):
        """
        Train AI model if enabled and run backtest.
        """
        if self.use_ai:
            ai_model = AIModel()
            self.model, self.scaler = ai_model.train(self.data)

        backtester = Backtester(self.data, INITIAL_CAPITAL, self.use_ai, self.model, self.scaler)
        backtester.run()  # Run the backtest

    def run(self):
        """
        Execute the entire pipeline: data loading, signal generation, AI training, and backtesting.
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
    quantanamo_bae = QuantanamoBae(STOCK_SYMBOL, USE_AI)
    quantanamo_bae.run()
