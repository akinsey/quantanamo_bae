import logging
import time
import yfinance as yf
import pandas as pd
from config import STOCK_SYMBOL, TRADE_WINDOW_START_DATE, TRADE_WINDOW_END_DATE

logger = logging.getLogger(__name__)

def fetch_historical_data(stock_symbol, start_date, end_date, extra_days=100):
    """Fetch historical data with additional days to ensure SMA calculations work properly."""
    logger.info(f"Fetching historical data for {stock_symbol} from {start_date} to {end_date}")

    # Extend data request to include extra days for SMA calculation
    adjusted_start_date = pd.to_datetime(start_date) - pd.Timedelta(days=extra_days)
    adjusted_start_date = adjusted_start_date.strftime('%Y-%m-%d')

    try:
        time.sleep(1)  # Simulate delay for API request
        data = yf.download(stock_symbol, start=adjusted_start_date, end=end_date)
    except Exception as e:
        logger.error(f"Error fetching data from yfinance: {e}")
        return pd.DataFrame()

    if data.empty:
        logger.warning(f"No data returned for {stock_symbol} in range {adjusted_start_date} to {end_date}.")
        return data

    logger.info(f"Fetched {len(data)} rows of data (including extra {extra_days} days).")

    return data
