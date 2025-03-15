import matplotlib.pyplot as plt # plotter.py
import logging

logger = logging.getLogger(__name__)

def plot_trading_strategy(data, stock_symbol, buy_signals, sell_signals):
    """Plot trading signals on a stock price chart."""
    logger.info(f"Generating plot for {stock_symbol}...")

    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['Close'], label='Close Price', color='steelblue')

    for date, price, _ in buy_signals:
        plt.scatter(date, price, marker='^', color='green', label='Buy Signal', alpha=1)

    for date, price, _, _ in sell_signals:
        plt.scatter(date, price, marker='v', color='red', label='Sell Signal', alpha=1)

    plt.title(f"Trading Strategy for {stock_symbol}")
    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.show()

    logger.info("Plot generated successfully.")

