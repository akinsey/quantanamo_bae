import logging
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from config import MIN_PROFIT_THRESHOLD, STOP_LOSS_THRESHOLD, STOCK_SYMBOL
from plotter import plot_trading_strategy  # Import the plotter for visualization

class Backtester:
    def __init__(self, data, initial_capital, use_ai, model=None, scaler=None):
        """
        Initializes the Backtester with trading parameters.
        """
        self.logger = logging.getLogger(__name__)
        self.console = Console()

        self.data = data
        self.initial_capital = initial_capital
        self.use_ai = use_ai
        self.model = model
        self.scaler = scaler

        self.capital = initial_capital
        self.position = 0
        self.last_buy_price = 0
        self.buy_date = None
        self.trade_start_date = None
        self.trade_end_date = None
        self.buy_signals = []
        self.sell_signals = []
        self.trades = []

    def run(self):
        """
        Executes the backtest on the provided data.
        """
        self.logger.info("Starting backtest...")

        for i in range(len(self.data) - 1):
            row = self.data.iloc[i]
            current_price = row['Close'].item()
            current_date = self.data.index[i]

            if self.trade_start_date is None:
                self.trade_start_date = current_date

            # Determine trade signal
            trade_signal = self.determine_trade_signal(row)

            # Execute Buy or Sell Order
            self.execute_trade(trade_signal, current_price, current_date)
            self.trade_end_date = current_date

        # Final portfolio valuation
        final_value, total_return, total_days = self.calculate_final_portfolio()

        # Compute statistics and display results
        self.process_results(final_value, total_return, total_days)

        self.logger.info("Backtest complete.")

    def determine_trade_signal(self, row):
        """Determines whether to buy, sell, or hold based on AI model or traditional signal."""
        if self.use_ai and self.model and self.scaler:
            features = np.array([[row['SMA_short'], row['SMA_long']]], dtype=np.float64)
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            return int(self.model.predict(features_scaled)[0])
        return int(row['Signal'].item())

    def execute_trade(self, trade_signal, current_price, current_date):
        """Handles buy and sell trade execution."""
        # Execute Buy Order
        if trade_signal == 1 and self.capital > 0:
            self.position = self.capital / current_price
            self.last_buy_price = current_price
            self.buy_date = current_date
            self.buy_signals.append((current_date, current_price, self.position))
            self.trades.append([current_date, "BUY", current_price, self.position, None, None])
            self.capital = 0

        # Execute Sell Order
        elif self.position > 0:
            price_change = (current_price - self.last_buy_price) / self.last_buy_price
            days_held = (current_date - self.buy_date).days if self.buy_date else 0

            if trade_signal == -1 or price_change >= MIN_PROFIT_THRESHOLD or price_change <= STOP_LOSS_THRESHOLD:
                self.capital = self.position * current_price
                self.sell_signals.append((current_date, current_price, self.position, price_change * 100))
                self.trades.append([current_date, "SELL", current_price, self.position, price_change * 100, days_held])
                self.position = 0

    def calculate_final_portfolio(self):
        """Calculates the final portfolio value and return statistics."""
        final_value = self.capital + (self.position * self.data.iloc[-1]['Close'].item())
        total_return = ((final_value - self.initial_capital) / self.initial_capital) * 100
        total_days = (self.trade_end_date - self.trade_start_date).days if self.trade_start_date and self.trade_end_date else 0
        return final_value, total_return, total_days

    def calculate_trade_statistics(self, final_value, total_days):
        """
        Compute key performance metrics from backtest results.
        """
        trades_df = pd.DataFrame(self.trades, columns=["Date", "Action", "Price", "Shares", "Profit %", "Days Held"])

        if trades_df.empty:
            self.logger.warning("No trades executed. Skipping performance calculations.")
            return {}

        wins = trades_df[trades_df["Profit %"] > 0]
        losses = trades_df[trades_df["Profit %"] <= 0]

        total_trades = len(trades_df)
        win_rate = (len(wins) / total_trades * 100) if total_trades > 0 else 0
        max_drawdown = (trades_df["Price"].cummax() - trades_df["Price"]).max()
        sharpe_ratio = trades_df["Profit %"].mean() / trades_df["Profit %"].std() if trades_df["Profit %"].std() > 0 else 0
        profit_factor = abs(wins["Profit %"].sum() / losses["Profit %"].sum()) if not losses.empty else float("inf")
        cagr = ((final_value / self.initial_capital) ** (365 / total_days) - 1) * 100 if total_days > 0 else 0

        stats = {
            "Total Trades": total_trades,
            "Win Rate (%)": win_rate,
            "Max Drawdown": max_drawdown,
            "Sharpe Ratio": sharpe_ratio,
            "Profit Factor": profit_factor,
            "CAGR (%)": cagr
        }

        self.logger.info(f"Performance Metrics: {stats}")
        return stats

    def process_results(self, final_value, total_return, total_days):
        """Logs and displays the final portfolio statistics and results."""
        stats = self.calculate_trade_statistics(final_value, total_days)
        self.logger.info(f"Final Portfolio Value: ${final_value:.2f}")
        self.logger.info(f"Total Return: {total_return:.2f}% over {total_days} days")
        self.display_results(final_value, total_return, total_days, stats)
        plot_trading_strategy(self.data, STOCK_SYMBOL, self.buy_signals, self.sell_signals)

    def display_results(self, final_value, total_return, total_days, stats):
        """
        Display backtest results in the console.
        """
        self.console.print("\n[bold cyan]💰 Final Portfolio Value:[/bold cyan]", f"[bold yellow]${final_value:.2f}[/bold yellow]")
        self.console.print("[bold cyan]📈 Total Return:[/bold cyan]", f"[bold green]{total_return:.2f}%[/bold green]")
        self.console.print("[bold cyan]📆 Backtest Duration:[/bold cyan]", f"[bold magenta]{total_days} days[/bold magenta]\n")

        # Performance Metrics Table
        self.console.print("[bold blue]📊 Performance Metrics:[/bold blue]")
        metrics_table = Table(title="", header_style="bold cyan")
        metrics_table.add_column("Metric", justify="left")
        metrics_table.add_column("Value", justify="right")

        for key, value in stats.items():
            metrics_table.add_row(key, f"{value:.2f}")

        self.console.print(metrics_table)
