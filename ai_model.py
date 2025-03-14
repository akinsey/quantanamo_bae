import logging
logger = logging.getLogger(__name__)

import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def train_ai_model(data):
    logger.info("Training AI model...")
    time.sleep(1)

    data = data.dropna()
    X = data[['SMA_20', 'SMA_50']]
    y = np.where(data['Close'].shift(-1) > data['Close'], 1, 0).flatten()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train_scaled, y_train)

    predictions = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, predictions)
    logger.info(f"AI Model Training Accuracy: {accuracy:.2f}")

    return model, scaler
