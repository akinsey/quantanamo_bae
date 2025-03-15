import logging # data_loader.py
logger = logging.getLogger(__name__)

import time
import yfinance as yf
import pandas as pd

def fetch_historical_data(stock_symbol, start_date, end_date):
    logger.info(f"Fetching historical data for {stock_symbol} from {start_date} to {end_date}")

    # Wrap the data fetch in a try/except
    try:
        time.sleep(1)  # Simulate delay
        data = yf.download(stock_symbol, start=start_date, end=end_date)
    except Exception as e:
        logger.error(f"Error fetching data from yfinance: {e}")
        # Return empty DataFrame or raise an error
        return pd.DataFrame()

    # Now check if data is empty
    if data.empty:
        logger.warning(f"No data returned for {stock_symbol} in range {start_date} to {end_date}.")
        # Return the empty DataFrame or handle it differently
        return data

    logger.info("Data fetched successfully.")
    return data
