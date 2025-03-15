import logging  # main.py
from config import *
from data_loader import fetch_historical_data
from strategy import generate_trading_signals
from ai_model import train_ai_model
from backtester import backtest_trading_strategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """
    Main execution function for running the trading strategy.
    Loads data, generates signals, trains AI model (if enabled), and backtests the strategy.
    """
    try:
        logger.info("Welcome to Quantanamo Bae")

        # Load historical data
        data = fetch_historical_data(STOCK_SYMBOL, TRADE_WINDOW_START_DATE, TRADE_WINDOW_END_DATE)
        if data.empty:
            logger.error("No data retrieved. Exiting program.")
            exit(1)

        # Generate trading signals
        data = generate_trading_signals(data)

        # Train AI model if enabled
        model, scaler = None, None
        if USE_AI:
            model, scaler = train_ai_model(data)

        # Run backtest
        backtest_trading_strategy(data, INITIAL_CAPITAL, USE_AI, model, scaler)

        logger.info("We hope you enjoyed your stay")
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        exit(1)

if __name__ == "__main__":
    main()
