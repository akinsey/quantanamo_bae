import time
import yfinance as yf
import pandas as pd

def fetch_historical_data(stock_symbol, start_date, end_date):
    """Fetch historical stock data from Yahoo Finance."""
    print("Fetching historical data...")
    time.sleep(1)
    data = yf.download(stock_symbol, start=start_date, end=end_date)
    print("Data fetched successfully.")
    return data
