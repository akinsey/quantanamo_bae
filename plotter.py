import matplotlib.pyplot as plt
import logging
import os

logger = logging.getLogger(__name__)

def plot_trading_strategy(data, stock_symbol, buy_signals, sell_signals, output_dir="plots"):
    """
    Plot trading signals on a stock price chart and save as an image.

    :param data: Pandas DataFrame containing stock price data
    :param stock_symbol: Stock ticker symbol
    :param buy_signals: List of buy signals (date, price, shares)
    :param sell_signals: List of sell signals (date, price, shares, profit percentage)
    :param output_dir: Directory to save the plot
    """
    logger.info(f"Generating plot for {stock_symbol}...")

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(data.index, data['Close'], label='Close Price', color='steelblue', linewidth=1.5)

    # Plot Buy signals
    if buy_signals:
        buy_dates, buy_prices, _ = zip(*buy_signals)
        ax.scatter(buy_dates, buy_prices, marker='^', color='green', label='Buy Signal', alpha=1, edgecolors='k', zorder=5)

    # Plot Sell signals
    if sell_signals:
        sell_dates, sell_prices, shares, profit_perc = zip(*sell_signals)
        ax.scatter(sell_dates, sell_prices, marker='v', color='red', label='Sell Signal', alpha=1, edgecolors='k', zorder=5)

        # Draw lines connecting each buy to its corresponding sell
        first_trade_path = True
    for (buy_date, buy_price, _), (sell_date, sell_price, shares, profit) in zip(buy_signals, sell_signals):
            ax.plot([buy_date, sell_date], [buy_price, sell_price], linestyle='dashed', color='black', alpha=0.9, label='Trade Path' if first_trade_path else "")
            first_trade_path = False

            # Annotate with trade details
            ax.annotate(f'Profit: {profit:.2f}%',
                        xy=(sell_date, sell_price),
                        xytext=(0, 15),
                        textcoords='offset points',
                        ha='center',
                        fontsize=6,
                        bbox=dict(boxstyle='round,pad=0.2', edgecolor='gray', facecolor='white', alpha=0.7))

    # Overlay Moving Averages (TODO: verify sma overlay data is correct)
    # if 'SMA_short' in data.columns:
    #     ax.plot(data.index.intersection(data['SMA_short'].dropna().index), data['SMA_short'].dropna(), label='50-Day SMA', color='orange', linestyle='--', linewidth=1)

    # if 'SMA_long' in data.columns:
    #     ax.plot(data.index.intersection(data['SMA_long'].dropna().index), data['SMA_long'].dropna(), label='200-Day SMA', color='purple', linestyle='--', linewidth=1)


    # Configure plot labels and title
    ax.set_title(f"Trading Strategy for {stock_symbol}", fontsize=14)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Stock Price", fontsize=12)
    ax.legend(loc='best', fontsize='small', frameon=True, markerscale=1.2)
    ax.grid(True, linestyle='--', alpha=0.6)

    # Rotate date labels for better readability
    plt.xticks(rotation=45)

    # Save the figure
    image_path = os.path.join(output_dir, f"{stock_symbol}_strategy.png")
    plt.tight_layout()
    plt.savefig(image_path, dpi=300)
    plt.close(fig)  # Close the figure to free up memory

    logger.info(f"Plot saved to {image_path}")
