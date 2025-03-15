import logging  # ai_model.py
logger = logging.getLogger(__name__)

import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def train_ai_model(data):
    """
    Train an AI model using stock market data to predict market movements.
    Uses a RandomForestClassifier on SMA indicators.

    :param data: Pandas DataFrame containing stock data with SMA features.
    :return: Trained model and scaler for feature normalization.
    """
    logger.info("Training AI model...")

    # Validate required columns
    if 'SMA_short' not in data.columns or 'SMA_long' not in data.columns:
        logger.error("Missing required SMA columns in data for AI model.")
        return None, None  # Could raise an exception instead

    data = data.dropna()  # Remove NaN values

    if data.empty:
        logger.error("Data is empty after dropna(). Cannot train AI model.")
        return None, None

    # Feature selection: Using SMA values to predict price direction
    X = data[['SMA_short', 'SMA_long']]
    y = np.where(data['Close'].shift(-1) > data['Close'], 1, 0).flatten()  # 1 if price increases, 0 otherwise

    # Train/test split (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train the model
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Evaluate accuracy
    predictions = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, predictions)
    logger.info(f"AI Model Training Accuracy: {accuracy:.2f}")

    return model, scaler
