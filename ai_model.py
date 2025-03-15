import logging
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pandas as pd

class AIModel:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.trained = False

    def train(self, data: pd.DataFrame):
        """
        Train an AI model using stock market data to predict market movements.
        """
        self.logger.info("Training AI model...")

        if 'SMA_short' not in data.columns or 'SMA_long' not in data.columns:
            self.logger.error("Missing required SMA columns in data for AI model.")
            return None, None

        data = data.dropna()
        if data.empty:
            self.logger.error("Data is empty after dropping NaN values.")
            return None, None

        # Feature selection: Using SMA values to predict price direction
        X = data[['SMA_short', 'SMA_long']]
        y = np.where(data['Close'].shift(-1) > data['Close'], 1, 0).flatten()

        # Train/test split (80% training, 20% testing)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Normalize features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train the model
        start_time = time.time()
        self.model.fit(X_train_scaled, y_train)
        end_time = time.time()
        self.logger.info(f"Model training completed in {end_time - start_time:.2f} seconds")

        # Evaluate accuracy
        predictions = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, predictions)
        self.logger.info(f"Model accuracy: {accuracy:.2%}")

        # Trained bool
        self.trained = True
        return self.model, self.scaler

    def predict(self, X_new: pd.DataFrame):
        """
        Make predictions using the trained AI model.
        """
        if not self.trained:
            self.logger.error("Model must be trained before making predictions.")
            return None

        X_scaled = self.scaler.transform(X_new)
        return self.model.predict(X_scaled)
