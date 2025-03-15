import logging
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from config import MIN_PROFIT_THRESHOLD, STOP_LOSS_THRESHOLD, STOCK_SYMBOL
from plotter import plot_trading_strategy  # Import the new plotter module

logger = logging.getLogger(__name__)
console = Console()

def calculate_trade_statistics(trades, initial_capital, final_value, total_days):
    """Compute key performance metrics from backtest results."""
    trades_df = pd.DataFrame(trades, columns=["Date", "Action", "Price", "Shares", "Profit %", "Days Held"])

    if trades_df.empty:
        console.print("[bold red]⚠ No trades executed. Skipping performance calculations.[/bold red]")
        logger.warning("No trades executed. Skipping performance calculations.")
        return {}

    wins = trades_df[trades_df["Profit %"] > 0]
    losses = trades_df[trades_df["Profit %"] <= 0]

    total_trades = len(trades_df)
    win_rate = (len(wins) / total_trades * 100) if total_trades > 0 else 0
    max_drawdown = (trades_df["Price"].cummax() - trades_df["Price"]).max()
    sharpe_ratio = trades_df["Profit %"].mean() / trades_df["Profit %"].std() if trades_df["Profit %"].std() > 0 else 0
    profit_factor = abs(wins["Profit %"].sum() / losses["Profit %"].sum()) if not losses.empty else float("inf")
    cagr = ((final_value / initial_capital) ** (365 / total_days) - 1) * 100 if total_days > 0 else 0

    stats = {
        "Total Trades": total_trades,
        "Win Rate (%)": win_rate,
        "Max Drawdown": max_drawdown,
        "Sharpe Ratio": sharpe_ratio,
        "Profit Factor": profit_factor,
        "CAGR (%)": cagr
    }

    # Log performance metrics
    logger.info(f"Performance Metrics: {stats}")

    return stats

def backtest_trading_strategy(data, initial_capital, use_ai, model=None, scaler=None):
    """Backtest the trading strategy using historical data."""
    logger.info("Starting backtest...")

    capital = initial_capital
    position = 0
    last_buy_price = 0
    buy_date = None
    trade_start_date = None
    trade_end_date = None
    buy_signals = []
    sell_signals = []
    trades = []

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
            trades.append([current_date, "BUY", current_price, shares, None, None])
            capital = 0

        # Execute Sell Order
        elif position > 0:
            price_change = (current_price - last_buy_price) / last_buy_price
            days_held = (current_date - buy_date).days if buy_date else 0

            if trade_signal == -1 or price_change >= MIN_PROFIT_THRESHOLD or price_change <= STOP_LOSS_THRESHOLD:
                capital = position * current_price
                sell_signals.append((current_date, current_price, position, price_change * 100))
                trades.append([current_date, "SELL", current_price, position, price_change * 100, days_held])
                position = 0

        trade_end_date = current_date

    # Final portfolio valuation
    final_value = capital + (position * data.iloc[-1]['Close'].item())
    total_return = ((final_value - initial_capital) / initial_capital) * 100
    total_days = (trade_end_date - trade_start_date).days if trade_start_date and trade_end_date else 0

    # Compute statistics
    stats = calculate_trade_statistics(trades, initial_capital, final_value, total_days)

    # Log final portfolio summary
    logger.info(f"Final Portfolio Value: ${final_value:.2f}")
    logger.info(f"Total Return: {total_return:.2f}% over {total_days} days")

    # Portfolio Summary Table
    console.print("\n[bold cyan]💰 Final Portfolio Value:[/bold cyan]", f"[bold yellow]${final_value:.2f}[/bold yellow]")
    console.print("[bold cyan]📈 Total Return:[/bold cyan]", f"[bold green]{total_return:.2f}%[/bold green]")
    console.print("[bold cyan]📆 Backtest Duration:[/bold cyan]", f"[bold magenta]{total_days} days[/bold magenta]\n")

    # Performance Metrics Table
    console.print("[bold blue]📊 Performance Metrics:[/bold blue]")
    metrics_table = Table(title="", header_style="bold cyan")
    metrics_table.add_column("Metric", justify="left")
    metrics_table.add_column("Value", justify="right")

    for key, value in stats.items():
        metrics_table.add_row(key, f"{value:.2f}")

    console.print(metrics_table)

    # Call the new plot function
    plot_trading_strategy(data, STOCK_SYMBOL, buy_signals, sell_signals)

    logger.info("Backtest complete.")
