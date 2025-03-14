# main.py
import logging
from config import *
from data_loader import fetch_historical_data
from strategy import generate_trading_signals
from ai_model import train_ai_model
from backtester import backtest_trading_strategy

# Configure the logging system
logging.basicConfig(
    level=logging.INFO,  # You can set DEBUG, INFO, WARNING, etc.
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Welcome to Quantanamo Bae")

    data = fetch_historical_data(STOCK_SYMBOL, TRADE_WINDOW_START_DATE, TRADE_WINDOW_END_DATE)
    data = generate_trading_signals(data)

    model, scaler = None, None
    if USE_AI:
        model, scaler = train_ai_model(data)

    backtest_trading_strategy(data, INITIAL_CAPITAL, USE_AI, model, scaler)

    logger.info("We hope you enjoyed your stay")
