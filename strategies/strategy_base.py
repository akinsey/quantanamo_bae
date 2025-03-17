from abc import ABC, abstractmethod  # strategy.py

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
