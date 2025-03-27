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

        # get daily gain and indicator data,
        # removing any rows from data for which
        # closing price and indicator are not populated
        (data, daily_gain_binary_data, indicator_data) = juice(data, self.indicator_strategy)

        if data.empty:
            self.logger.error("Data is empty after dropping NaN values.")
            return None, None

        ## Split the dataset into training and testing sets
        # 80% of the data is used for training, and 20% is reserved for testing
        verification_reserve_percent = 20
        training_indicator_data, testing_indicator_data, training_daily_gain_binary_data, testing_daily_gain_binary_data = train_test_split(indicator_data, daily_gain_binary_data, test_size=verification_reserve_percent / 100, random_state=42)

        ## Normalize indicator data
        # NOTES: Since we are working with multiple columns of indicator data,
        # standardizing (or normalizing) the feature values means making sure
        # all values are on a similar scale.
        #
        # For example, indicator (SMA_short) has values around 10 and
        # another indicator (SMA_long) has values around 1000...
        # The scaler will transform both columns to fit within a range from 0 to 1 (or -1 to +1)
        # to ensure they are compared with equal weight
        #
        # Normalizing can also ensure that more recent data is treated as more important than
        # older data

        # Fit and transform training data
        training_indicator_data = self.scaler.fit_transform(training_indicator_data)
        # Transform test data using the same scaler
        testing_indicator_data = self.scaler.transform(testing_indicator_data)

        # Populate/train the ML model by fitting the indicator data to the daily gain/loss data
        # using the data reserved for training
        self.model.fit(training_indicator_data, training_daily_gain_binary_data)

        ## Evaluate the accuracy the model
        # Run the trained ML model using the data reserved for testing to predict gain/loss
        predictions = self.model.predict(testing_indicator_data)
        # Calculate accuracy of the predictions against the actual gain/loss data
        self.accuracy = accuracy_score(testing_daily_gain_binary_data, predictions)

        # Set flag to indicate that the model has been trained
        self.trained = True

        # TODO: maybe don't return these:  just save the class instance and use it later
        return self.model, self.scaler  # Return trained model and scaler

    def log_accuracy(self):
        self.logger.info(f"Model trained with accuracy: {self.accuracy * 100:.2f}%")
