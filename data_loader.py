import logging
logger = logging.getLogger(__name__)

import time
import yfinance as yf
import pandas as pd

def fetch_historical_data(stock_symbol, start_date, end_date):
    logger.info(f"Fetching historical data for {stock_symbol} from {start_date} to {end_date}")
    time.sleep(1)
    data = yf.download(stock_symbol, start=start_date, end=end_date)
    logger.info("Data fetched successfully.")
    return data
