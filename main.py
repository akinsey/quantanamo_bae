import logging # main.py
from config import *
from data_loader import fetch_historical_data
from strategy import apply_strategy
from ai_model import train_ai_model
from backtester import backtest_trading_strategy

# Configure the logging system
logging.basicConfig(
    level=logging.INFO,  # You can set DEBUG, INFO, WARNING, etc.
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    try:
        logger.info("Welcome to Quantanamo Bae")

        # Fetch historical stock data
        data = fetch_historical_data(STOCK_SYMBOL, TRADE_WINDOW_START_DATE, TRADE_WINDOW_END_DATE)

        if data.empty:
            logger.error("No data retrieved. Exiting program.")
            exit(1)

        # Apply selected trading strategy
        data = apply_strategy(data, "sma_crossover")  # Can change dynamically in config.py

        # Train AI Model if enabled
        model, scaler = None, None
        if USE_AI:
            model, scaler = train_ai_model(data)

        # Run Backtest
        backtest_trading_strategy(data, INITIAL_CAPITAL, USE_AI, model, scaler)

        logger.info("We hope you enjoyed your stay.")

    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        exit(1)
