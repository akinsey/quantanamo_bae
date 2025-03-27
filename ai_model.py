import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from utils import juice

class RFC_MLModel:
    def __init__(self, strategy):
        """Initialize the Random Forest Classifier ML model with logging, model, and scaler."""
        self.logger = logging.getLogger(__name__)  # Logger for tracking model activities
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)  # Random forest classifier with fixed randomness
        # 100 trees in the forest and randomness seeded by integer 42
        self.scaler = StandardScaler()  # StandardScaler for normalizing data
        self.trained = False  # Flag to track if the model has been trained
        self.indicator_strategy = strategy  # Strategy instance (e.g., SMAStrategy or RSIStrategy)
        self.accuracy = 0

    def train(self, data: pd.DataFrame):
        """Train the model using stock market data to predict market movements."""
        self.logger.info("Training AI model...")

        if self.trained:
            self.logger.info("Model is already trained. Skipping training.")
            return self.model, self.scaler  # Return existing trained model and scaler

        # get closing prices and indicator data,
        # removing any rows from data for which
        # closing price and indicator are not populated
        (data, closing_prices, indicator_data) = juice(data, self.indicator_strategy)

        if data.empty:
            self.logger.error("Data is empty after dropping NaN values.")
            return None, None

        # Target labels (y): The model should predict whether the stock price will go up (1) or down (0).
        # If tomorrow's closing price is higher than today's, assign 1; otherwise, assign 0.
        # e.g. We use the next days closing price
        y = closing_prices.shift(-1) > closing_prices  # Creates a Boolean series (True/False)
        y = y.astype(int)  # Converts Boolean values to integers (1 for up, 0 for down)
        y = y.values.reshape(-1,)  # Ensures y is formatted as a 1D NumPy array for sklearn

        # Split the dataset into training and testing sets
        # 80% of the data is used for training, and 20% is reserved for testing
        X_train, X_test, y_train, y_test = train_test_split(indicator_data, y, test_size=0.2, random_state=42)

        # Standardize (normalize) feature values to bring them to a similar scale
        # NOTES: standardizing (or normalizing) the feature values means making sure
        # all numbers are on a similar scale. For example, if one feature (SMA_short) has
        # values around 10 and another feature (SMA_long) has values around 1000, the
        # model might think the bigger numbers are more important just because they’re bigger.
        X_train_scaled = self.scaler.fit_transform(X_train)  # Fit and transform training data
        X_test_scaled = self.scaler.transform(X_test)  # Transform test data using the same scaler

        # Train the model using the training dataset
        self.model.fit(X_train_scaled, y_train)

        # Evaluate the trained model using the test set
        predictions = self.model.predict(X_test_scaled)  # Predict stock movement
        self.accuracy = accuracy_score(y_test, predictions)

        self.trained = True  # Set flag to indicate that the model has been trained

        # TODO: maybe don't return these:  just save the class instance and use it later
        return self.model, self.scaler  # Return trained model and scaler

    def log_accuracy(self):
        self.logger.info(f"Model trained with accuracy: {self.accuracy * 100:.2f}%")
