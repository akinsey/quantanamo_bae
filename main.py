import logging
from config import *
from data_loader import fetch_historical_data
from strategies.sma import SMA  # Modular strategy
from ai_model import AIModel
from backtester import Backtester  # Class-based backtester

# Configure logging globally before the class is instantiated
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)

class QuantanamoBae:
    def __init__(self, stock_symbol, strategy_name, use_ai):
        """
        Initializes the trading system.
        """
        self.stock_symbol = stock_symbol
        self.strategy_name = strategy_name if strategy_name else 'SMA'  # New: User-defined strategy
        self.use_ai = use_ai
        self.data = None
        self.strategy = None  # New: Store the selected strategy instance
        self.model = None
        self.scaler = None
        self.logger = logging.getLogger(__name__)  # Now picks up global config

    def select_strategy(self):
        """
        Select the trading strategy dynamically.
        """
        strategies = {
            "SMA": SMA
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
        self.strategy = StrategyClass(self.data)  # Instantiate strategy
        self.data["Signal"] = self.strategy.generate_signals()

    def train_and_backtest(self):
        """
        Train AI model if enabled and run backtest.
        """
        if self.use_ai:
            ai_model = AIModel(self.strategy)  # Pass strategy to AIModel
            self.model, self.scaler = ai_model.train(self.data)

        backtester = Backtester(self.data, INITIAL_CAPITAL, self.use_ai, self.model, self.scaler)
        backtester.run()  # Run the backtest

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
    quantanamo_bae = QuantanamoBae(STOCK_SYMBOL, STRATEGY_NAME, USE_AI)
    quantanamo_bae.run()
