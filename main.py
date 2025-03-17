import logging
import argparse
import sys

from config import *
from data_loader import fetch_stock_data
from ai_model import AIModel
from backtester import Backtester
from plotter import plot_trading_strategy
from strategies.sma import SMA
from strategies.rsi import RSI
from strategies.macd import MACD
from rich.console import Console
from rich.table import Table

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)


class QuantanamoBae:
    """Main class for running trading strategies."""

    def __init__(self, stock_symbol, strategy_name=STRATEGY_NAME, use_ai=USE_AI, plot_results=False):
        self.stock_symbol = stock_symbol
        self.strategy_name = strategy_name
        self.use_ai = use_ai
        self.plot_results = plot_results
        self.data = None
        self.strategy = None
        self.model = None
        self.scaler = None
        self.console = Console()
        self.logger = logging.getLogger(__name__)

    def select_strategy(self):
        """Select trading strategy based on provided name."""
        strategies = {"SMA": SMA, "RSI": RSI, "MACD": MACD}

        if self.strategy_name not in strategies:
            self.logger.error(
                f"Invalid strategy '{self.strategy_name}'. Valid options: {list(strategies.keys())}"
            )
            sys.exit(1)

        return strategies[self.strategy_name]

    def prepare_data(self):
        """Fetch and prepare historical stock data."""
        self.data = fetch_stock_data(
            self.stock_symbol, TRADE_WINDOW_START_DATE, TRADE_WINDOW_END_DATE
        )

        if self.data.empty:
            self.logger.error("No data retrieved. Exiting.")
            sys.exit(1)

        if DEBUG:
            self.logger.info(f"Data retrieved with {self.data.isna().sum().sum()} missing values.")
            self.logger.info(f"Data preview:\n{self.data.head()}")

        StrategyClass = self.select_strategy()
        self.strategy = StrategyClass(self.data)
        self.data["Signal"] = self.strategy.generate_signals()

    def train_and_backtest(self):
        """Train AI model (if enabled) and perform backtesting."""
        if self.use_ai:
            ai_model = AIModel(self.strategy)
            self.model, self.scaler = ai_model.train(self.data)

        backtester = Backtester(
            self.data, INITIAL_CAPITAL, self.use_ai, self.strategy, self.model, self.scaler
        )
        stats = backtester.run()
        self.display_results(stats)

        if self.plot_results:
            buy_signals = backtester.get_buy_signals()
            sell_signals = backtester.get_sell_signals()
            plot_trading_strategy(self.data, self.stock_symbol, buy_signals, sell_signals)

    def display_results(self, stats):
        """
        Display backtest results in the console.
        """

        # Performance Stats Table
        self.console.print("[bold blue]📊 Performance Statistics:[/bold blue]")
        metrics_table = Table(title="", header_style="bold cyan")
        metrics_table.add_column("Metric", justify="left")
        metrics_table.add_column("Value", justify="right")
        trunc = lambda amount: f"{amount:.2f}" if isinstance(amount, float) else f"{amount}"

        for key, value in stats.items():
            metrics_table.add_row(key, trunc(value))

        self.console.print(metrics_table)

    def run(self):
        """Execute the entire pipeline."""
        try:
            self.logger.info("Welcome to Quantanamo Bae")
            if DEBUG:
                valid_flags = {"strategy_name", "stock_symbol", "use_ai", "plot_results"}
                config = {k: v for k, v in vars(self).items() if k in valid_flags}
                self.logger.info(f"Configuration: \n{config}")
            self.prepare_data()
            self.train_and_backtest()
            self.logger.info("We hope you enjoyed your stay!")
        except Exception as e:
            self.logger.critical(f"Fatal error occurred: {e}", exc_info=True)
            sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantanamo Bae Trading Strategy Executor.")
    parser.add_argument(
        '--strategy', type=str, choices=['SMA', 'RSI', 'MACD'], default='SMA', help="Trading strategy."
    )
    parser.add_argument(
        '--stock', type=str, default=STOCK_SYMBOL, help="Stock symbol to trade."
    )
    parser.add_argument(
        '--disable-ai', action='store_false', dest='use_ai', help="Disable AI model for predictions (AI enabled by default)."
    )
    parser.add_argument(
        '--plot', action='store_true', help="Enable plotting of results."
    )

    args = parser.parse_args()

    quantanamo_bae = QuantanamoBae(args.stock, args.strategy, args.use_ai, args.plot)
    quantanamo_bae.run()
