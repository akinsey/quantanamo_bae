import matplotlib.pyplot as plt # plotter.py
import logging
import os
import time

logger = logging.getLogger(__name__)

def plot_trading_strategy(data, stock_symbol, buy_signals, sell_signals, output_dir="plots"):
    """Plot trading signals on a stock price chart and save as an image."""
    logger.info(f"Generating plot for {stock_symbol}...")

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Create the figure
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

    # Save the figure instead of displaying it
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    image_path = os.path.join(output_dir, f"{stock_symbol}_strategy.png")
    plt.savefig(image_path, dpi=300, bbox_inches="tight")
    time.sleep(1)  # Allow time for filesystem sync before test runs
    plt.close()  # Close the figure to free up memory

    logger.info(f"Plot saved to {image_path}")
