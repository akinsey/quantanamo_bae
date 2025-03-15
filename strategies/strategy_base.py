from abc import ABC, abstractmethod  # strategy.py

class Strategy:
    def __init__(self, data):
        self.data = data  # Historical market data

    @abstractmethod
    def generate_signals(self):
        """Each strategy must implement this method to generate trade signals."""
        pass

    @abstractmethod
    def get_ai_features(self):
        """Each strategy must implement this method to specify features used for AI training."""
        pass
