import logging
import numpy as np
import pandas as pd
from config import MIN_PROFIT_THRESHOLD, STOP_LOSS_THRESHOLD, STOCK_SYMBOL

class Backtester:
    def __init__(self, data, initial_capital, use_ai, strategy, model=None, scaler=None):
        """
        Initializes the Backtester with trading parameters.
        """
        self.logger = logging.getLogger(__name__)

        self.data = data
        self.initial_capital = initial_capital
        self.use_ai = use_ai
        self.strategy = strategy
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

    def get_buy_signals(self): return self.buy_signals

    def get_sell_signals(self): return self.sell_signals

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

        self.logger.info("Backtest complete.")
        return self.calculate_trade_statistics()

    def determine_trade_signal(self, row):
        """Determines whether to buy, sell, or hold based on AI model or traditional signal."""
        if self.use_ai and self.model and self.scaler:
            feature_columns_names = self.strategy.get_feature_column_names()

            # Dynamically extract the correct feature columns from the row
            actual_features = []
            for feature in feature_columns_names:
                matched_cols = [col for col in row.index if feature in col]
                if not matched_cols:
                    self.logger.error(f"Could not find feature column '{feature}' in row.")
                    return 0  # Default to hold signal if features missing
                actual_features.append(row[matched_cols[0]])

            # TODO: check if we can grab the predictions from the ai model itself instead
            # of regenerating them here, is there a reason this is duplicated, is it even duplicated
            #
            # Reshape correctly for scaler and model (1 sample, N features)
            features = np.array([actual_features], dtype=np.float64)
            # features = np.array([[row['SMA_short'], row['SMA_long']]], dtype=np.float64)
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

    def calculate_trade_statistics(self):
        """
        Compute key performance metrics from backtest results.
        """
        trades_df = pd.DataFrame(self.trades, columns=["Date", "Action", "Price", "Shares", "Profit %", "Days Held"])

        if trades_df.empty:
            self.logger.warning("No trades executed. Skipping performance calculations.")
            return {}

        trading_days = (self.trade_end_date - self.trade_start_date).days if self.trade_start_date and self.trade_end_date else 0

        wins = trades_df[trades_df["Profit %"] > 0]
        losses = trades_df[trades_df["Profit %"] <= 0]

        # Ensure Avg Trade Duration is calculated correctly
        if "Days Held" not in trades_df.columns or trades_df["Days Held"].isnull().all():
            avg_trade_duration = trading_days / max(1, len(trades_df))  # Fallback method
        else:
            avg_trade_duration = trades_df["Days Held"].mean()

        # Correct Max Drawdown Calculation using Portfolio Equity
        portfolio_values = self.initial_capital + (trades_df["Profit %"].cumsum() / 100 * self.initial_capital)
        rolling_max = portfolio_values.cummax()
        drawdown = (portfolio_values - rolling_max) / rolling_max
        max_drawdown = float(drawdown.min())  # Ensure correct drawdown value as scalar

        # Basic Metrics
        total_trades = len(trades_df)
        win_rate = (len(wins) / total_trades * 100) if total_trades > 0 else 0
        sharpe_ratio = trades_df["Profit %"].mean() / trades_df["Profit %"].std() if trades_df["Profit %"].std() > 0 else 0
        total_profit = wins["Profit %"].sum()
        total_loss = abs(losses["Profit %"].sum())
        profit_factor = total_profit / total_loss if total_loss > 0 else float("inf")
        cagr = ((self.capital / self.initial_capital) ** (365 / trading_days) - 1) * 100 if trading_days > 0 else 0

        avg_profit_per_trade = trades_df["Profit %"].mean() if not trades_df.empty else 0
        max_trade_gain = trades_df["Profit %"].max() if not trades_df.empty else 0
        max_trade_loss = trades_df["Profit %"].min() if not trades_df.empty else 0
        risk_reward_ratio = abs(max_trade_gain / max_trade_loss) if max_trade_loss != 0 else float("inf")
        volatility = trades_df["Profit %"].std() if not trades_df.empty else 0

        # Fix Streak Calculation
        trades_df["Win"] = trades_df["Profit %"] > 0
        trades_df["Streak"] = trades_df["Win"].groupby((trades_df["Win"] != trades_df["Win"].shift()).cumsum()).cumcount() + 1

        longest_win_streak = trades_df[trades_df["Win"] == True]["Streak"].max() if not trades_df.empty else 0
        longest_loss_streak = trades_df[trades_df["Win"] == False]["Streak"].max() if not trades_df.empty else 0

        # Debug Prints for Validation
        print(f"Buy Signals: {len(self.buy_signals)}, Sell Signals: {len(self.sell_signals)}, Total Trades: {total_trades}")
        print(f"Avg Trade Duration: {avg_trade_duration}, Computed from {'trading_days' if 'Days Held' not in trades_df.columns else 'Days Held'}")
        print(f"Max Drawdown: {max_drawdown}")

        # Color coding for profit/loss
        profit_color = "green" if self.capital > self.initial_capital else "red" if self.capital < self.initial_capital else "white"

        stats = {
            "Strategy": self.strategy.get_name(),
            "Total Trades": total_trades,
            "Win Rate (%)": win_rate,
            "Max Drawdown": max_drawdown,
            "Sharpe Ratio": sharpe_ratio,
            "Profit Factor": profit_factor,
            "CAGR (%)": cagr,
            "Avg Profit per Trade (%)": avg_profit_per_trade,
            "Avg Trade Duration (days)": avg_trade_duration,
            "Max Trade Gain (%)": max_trade_gain,
            "Max Trade Loss (%)": max_trade_loss,
            "Risk-Reward Ratio": risk_reward_ratio,
            "Volatility (Std Dev)": volatility,
            "Longest Win Streak": longest_win_streak,
            "Longest Loss Streak": longest_loss_streak,
            "Trading Days": trading_days,
            "Initial Portfolio Value": f"${self.initial_capital}",
            "Current Portfolio Value": f"[{profit_color}]${self.capital:.2f}[/{profit_color}]",
            "Percent Return": f"[{profit_color}]{((self.capital - self.initial_capital) / self.initial_capital) * 100:.2f}%[/{profit_color}]"
        }

        self.logger.info(f"Updated Statistics: {stats}")
        return stats
