from abc import ABC, abstractmethod  # strategy.py
from utils import extract_close_column, extract_feature_columns

class Strategy:
    def __init__(self, data):
        self.data = data  # Historical market data

    @abstractmethod
    def get_name(self):
        """Each strategy must implement this method to specify the name of the columns to use
        as features when AI training."""
        pass

    @abstractmethod
    def get_feature_column_names(self):
        """Each strategy must implement this method to specify the name of the columns to use
        as features when AI training."""
        pass

    @abstractmethod
    def generate_signals(self):
        """Each strategy must implement this method to generate trade signals."""
        pass

    def generate_signals_and_clean(self):
        """Generates trade signals and cleans the data of NaN values for the indicator columns and closing price column."""
        self.generate_signals()
        self.clean_signal_data()

    def clean_signal_data(self):
        """Cleans the data of NaN values for the indicator columns and closing price column."""
        indicator_strategy_column_names = self.get_feature_column_names()
        indicator_columns = extract_feature_columns(self.data, indicator_strategy_column_names)
        closing_price_column = extract_close_column(self.data)
        self.data = self.data.dropna(subset=indicator_columns + [closing_price_column])
