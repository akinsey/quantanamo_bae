import numpy as np

def generate_trading_signals(data):
    """Generate trading signals based on moving averages."""
    print("Generating trading signals...")
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['Signal'] = np.where(data['SMA_20'] > data['SMA_50'], 1, -1)
    print(f"Trading signals generated. Buy signals: {(data['Signal'] == 1).sum()}, Sell signals: {(data['Signal'] == -1).sum()}")
    return data
