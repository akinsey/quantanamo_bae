import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from config import SELECTED_STRATEGY
from strategy import apply_strategy

logger = logging.getLogger(__name__)

def train_ai_model(data):
    """Train AI model using the correct SMA features and ensure sufficient data."""
    logger.info("Training AI model...")

    data = apply_strategy(data, SELECTED_STRATEGY)

    if "Signal" not in data.columns:
        raise ValueError(f"Strategy '{SELECTED_STRATEGY}' did not generate a 'Signal' column.")

    # Drop NaNs AFTER applying strategy (Ensures AI doesn't get bad data)
    data.dropna(subset=["SMA_short", "SMA_long", "Signal"], inplace=True)

    # Ensure we actually have trade signals before training AI
    if data["Signal"].nunique() == 1 and data["Signal"].unique()[0] == 0:
        print(f"[ERROR] AI Training Aborted: No Buy/Sell Signals Generated.")
        return None, None

    # Remove neutral (Signal=0) rows to focus AI on buy/sell decisions
    data = data[data["Signal"] != 0]

    # Ensure AI has enough samples to train
    if len(data) < 10:
        print(f"[ERROR] AI Training Aborted: Not enough trade signals for AI to learn.")
        return None, None

    print(f"DEBUG: AI Training Data Before Model Training:\n{data[['SMA_short', 'SMA_long', 'Signal']].head(10)}")

    X = data[['SMA_short', 'SMA_long']]
    y = data["Signal"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train_scaled, y_train)

    return model, scaler
