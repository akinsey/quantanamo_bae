import logging
from config import *
from data_loader import fetch_historical_data
from strategy import generate_trading_signals
from ai_model import AIModel
from backtester import backtest_trading_strategy

# Configure logging globally before the class is instantiated
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)

class TradingSystem:
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

    def load_data(self):
        self.logger.info(f"Fetching data for {self.stock_symbol}...")
        self.data = fetch_historical_data(self.stock_symbol, TRADE_WINDOW_START_DATE, TRADE_WINDOW_END_DATE)
        if self.data.empty:
            self.logger.error("No data retrieved. Exiting.")
            exit(1)

    def generate_signals(self):
        self.data = generate_trading_signals(self.data)

    def train_ai(self):
        if self.use_ai:
            ai_model = AIModel()
            self.model, self.scaler = ai_model.train(self.data)

    def backtest(self):
        backtest_trading_strategy(self.data, INITIAL_CAPITAL, self.use_ai, self.model, self.scaler)

    def run(self):
        try:
            self.logger.info("Welcome to Quantanamo Bae")
            self.load_data()
            self.generate_signals()
            self.train_ai()
            self.backtest()
            self.logger.info("We hope you enjoyed your stay.")
        except Exception as e:
            self.logger.critical(f"Fatal error: {e}", exc_info=True)
            exit(1)

if __name__ == "__main__":
    trading_system = TradingSystem(STOCK_SYMBOL, USE_AI)
    trading_system.run()
