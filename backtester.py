import logging # backtester.py
import numpy as np
from config import MIN_PROFIT_THRESHOLD, STOP_LOSS_THRESHOLD, STOCK_SYMBOL
from plotter import plot_trading_strategy  # Import the new plotter module

logger = logging.getLogger(__name__)

def backtest_trading_strategy(data, initial_capital, use_ai, model=None, scaler=None):
    """Backtest the trading strategy using historical data."""
    logger.info("Starting backtest...")

    capital = initial_capital
    position = 0
    trade_log = []
    last_buy_price = 0
    buy_date = None
    trade_start_date = None
    trade_end_date = None
    buy_signals = []
    sell_signals = []

    for i in range(len(data) - 1):
        current_price = data.iloc[i]['Close'].item()
        current_date = data.index[i]

        if trade_start_date is None:
            trade_start_date = current_date

        # Determine trade signal
        if use_ai and model is not None and scaler is not None:
            features = np.array([[data.iloc[i]['SMA_short'], data.iloc[i]['SMA_long']]], dtype=np.float64)
            features_scaled = scaler.transform(features.reshape(1, -1))
            ai_signal = model.predict(features_scaled)[0]
            trade_signal = int(ai_signal)
        else:
            trade_signal = int(data.iloc[i]['Signal'].item())

        # Execute Buy Order
        if trade_signal == 1 and capital > 0:
            shares = capital / current_price
            position = shares
            last_buy_price = current_price
            buy_date = current_date
            buy_signals.append((current_date, current_price, shares))
            trade_log.append(f"BUY {current_date.date()} @ {current_price:.2f}")
            capital = 0

        # Execute Sell Order
        elif position > 0:
            price_change = (current_price - last_buy_price) / last_buy_price
            days_held = (current_date - buy_date).days if buy_date else 0

            if (trade_signal == -1
                or price_change >= MIN_PROFIT_THRESHOLD
                or price_change <= STOP_LOSS_THRESHOLD):
                capital = position * current_price
                sell_signals.append((current_date, current_price, position, price_change * 100))
                trade_log.append(
                    f"SELL {current_date.date()} @ {current_price:.2f} | "
                    f"Profit: {price_change * 100:.2f}%, Held: {days_held} days"
                )
                position = 0

        trade_end_date = current_date

    # Final portfolio valuation
    final_value = capital + (position * data.iloc[-1]['Close'].item())
    total_return = ((final_value - initial_capital) / initial_capital) * 100
    total_days = (trade_end_date - trade_start_date).days if trade_start_date and trade_end_date else 0

    logger.info("Trade Summary:")
    logger.info(" | ".join(trade_log[-10:]))  # Show last 10 trades
    logger.info(f"Final Portfolio Value: ${final_value:.2f}")
    logger.info(f"Total Return: {total_return:.2f}% over {total_days} days")

    # Call the new plot function
    plot_trading_strategy(data, STOCK_SYMBOL, buy_signals, sell_signals)

    logger.info("Backtest complete.")
