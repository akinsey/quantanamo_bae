import logging  # data_loader.py
logger = logging.getLogger(__name__)

import time
import yfinance as yf
import pandas as pd

def fetch_historical_data(stock_symbol, start_date, end_date):
    """
    Fetch historical stock market data using Yahoo Finance API.

    :param stock_symbol: Stock ticker symbol (e.g., "AAPL")
    :param start_date: Start date for data retrieval (YYYY-MM-DD)
    :param end_date: End date for data retrieval (YYYY-MM-DD)
    :return: Pandas DataFrame containing historical stock data
    """
    logger.info(f"Fetching historical data for {stock_symbol} from {start_date} to {end_date}")

    try:
        data = yf.download(stock_symbol, start=start_date, end=end_date, auto_adjust=True)
    except Exception as e:
        logger.error(f"Error fetching data from yfinance: {e}")
        return pd.DataFrame()  # Return an empty DataFrame on failure

    if data.empty:
        logger.warning(f"No data returned for {stock_symbol} in range {start_date} to {end_date}.")
        return data

    logger.info("Data fetched successfully.")
    return data
