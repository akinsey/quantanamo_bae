import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock
from backtester import Backtester

# Sample DataFrame for basic tests
@pytest.fixture
def sample_data():
    data = {
        'Close': [100, 102, 101, 105, 95, 110],
        'Signal': [0, 1, 0, -1, 0, 1],
        'SMA_short': [100, 101, 102, 103, 104, 105],  # Mocked AI features
        'SMA_long': [99, 100, 101, 102, 103, 104]     # Mocked AI features
    }
    return pd.DataFrame(data, index=pd.date_range('2025-01-01', periods=6))

@pytest.fixture
def mock_strategy():
    strategy = MagicMock()
    strategy.get_feature_column_names.return_value = ['SMA_short', 'SMA_long']
    strategy.get_name.return_value = 'Mock Strategy'
    return strategy

# Test Initialization
def test_init(sample_data, mock_strategy):
    backtester = Backtester(sample_data, 1000, False, mock_strategy)
    assert backtester.initial_capital == 1000
    assert backtester.position == 0
    assert backtester.capital == 1000
    assert len(backtester.buy_signals) == 0
    assert len(backtester.sell_signals) == 0

# Test Buy and Sell Signals Retrieval
def test_get_signals(sample_data, mock_strategy):
    backtester = Backtester(sample_data, 1000, False, mock_strategy)
    assert backtester.get_buy_signals() == []
    assert backtester.get_sell_signals() == []

# Test Trade Signal Determination - AI Model Mocked
def test_determine_trade_signal_ai(sample_data, mock_strategy):
    model = MagicMock()
    scaler = MagicMock()
    scaler.transform.return_value = np.array([[0.5, 0.8]])
    model.predict.return_value = [1]

    backtester = Backtester(sample_data, 1000, True, mock_strategy, model, scaler)
    row = sample_data.iloc[1]
    print(row)
    signal = backtester.determine_trade_signal(row)
    assert signal == 1

# Test Trade Signal Determination - Without AI
def test_determine_trade_signal_no_ai(sample_data, mock_strategy):
    backtester = Backtester(sample_data, 1000, False, mock_strategy)
    row = sample_data.iloc[1]
    signal = backtester.determine_trade_signal(row)
    assert signal == 1

# Test Execute Buy Trade
def test_execute_buy_trade(sample_data, mock_strategy):
    backtester = Backtester(sample_data, 1000, False, mock_strategy)
    backtester.execute_trade(1, 100, pd.Timestamp('2025-01-01'))
    assert backtester.position == 10
    assert backtester.capital == 0

# Test Execute Sell Trade
def test_execute_sell_trade(sample_data, mock_strategy):
    backtester = Backtester(sample_data, 1000, False, mock_strategy)
    backtester.execute_trade(1, 100, pd.Timestamp('2025-01-01'))
    backtester.execute_trade(-1, 105, pd.Timestamp('2025-01-02'))
    assert backtester.position == 0
    assert backtester.capital == 1050

# Test Calculate Trade Statistics
def test_calculate_trade_statistics(sample_data, mock_strategy):
    backtester = Backtester(sample_data, 1000, False, mock_strategy)
    backtester.execute_trade(1, 100, pd.Timestamp('2025-01-01'))
    backtester.execute_trade(-1, 105, pd.Timestamp('2025-01-02'))
    stats = backtester.calculate_trade_statistics()
    assert stats['Total Trades'] == 2
    assert stats['Win Rate (%)'] == 50
    assert 'Max Drawdown' in stats
    assert stats['Profit Factor'] > 1

# Test Run Full Backtest
def test_run(sample_data, mock_strategy):
    backtester = Backtester(sample_data, 1000, False, mock_strategy)
    results = backtester.run()
    assert isinstance(results, dict)
    assert results['Total Trades'] > 0

# Edge Case: Empty Data
def test_empty_data(mock_strategy):
    empty_data = pd.DataFrame(columns=['Close', 'Signal'])
    backtester = Backtester(empty_data, 1000, False, mock_strategy)
    results = backtester.run()
    assert results == {}
