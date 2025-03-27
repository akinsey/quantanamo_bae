import logging

def extract_close_column(data, column_name="Close"):
    """
    Robustly selects the 'Close' column from a DataFrame.

    :param data: Pandas DataFrame containing stock data.
    :param column_name: The name or partial match string to identify the Close column.
    :return: The matching column name.
    :raises ValueError: If no column is found.
    """
    close_col = [col for col in data.columns if column_name in col]
    
    if not close_col:
        logging.error(f"Could not find any column containing '{column_name}'")
        raise ValueError(f"Missing '{column_name}' column in data")

    return close_col[0]  # Return the first match

def extract_feature_columns(data, feature_column_names):
    """
    Dynamically find exact column names based on strategy feature keys.

    :param data: Pandas DataFrame containing stock data.
    :param feature_column_names: List of feature column names (partial match allowed).
    :param logger: Logger instance for error reporting.
    :return: List of matched feature column names.
    :raises ValueError: If any feature column is not found.
    """
    feature_columns = []
    for feature in feature_column_names:
        matched_cols = [col for col in data.columns if feature in col]
        if not matched_cols:
            logging.error(f"Could not find column containing '{feature}' in data.")
            raise ValueError(f"Missing feature column: {feature}")
        feature_columns.append(matched_cols[0])

    return feature_columns

# get daily gain and indicator data,
# removing any rows from data for which
# closing price and indicator are not populated
def juice(data, strategy):
    # Get indicator column names based on strategy feature keys
    # e.g. ['RSI'], ['SMA_short', 'SMA_long'], ['MACD', 'MACD_signal']
    indicator_strategy_column_names = strategy.get_feature_column_names()
    # get indicator data columns from `data` corresponding to the selected indicator strategy's column names
    indicator_columns = extract_feature_columns(data, indicator_strategy_column_names)
    # get closing price column from `data`
    closing_price_column = extract_close_column(data)

    # remove rows in `data` that don't have corresponding values
    # in indicator data or closing price columns
    data.dropna(subset=indicator_columns + [closing_price_column])

    ## get data from `data` based on processing columns
    # computed daily indicator data for the given strategy on the selected ticker
    indicator_data = data[indicator_columns]
    # daily closing prices for the selected ticker
    closing_prices = data[closing_price_column]

    ## Compute whether the stock price went up (1) or down (0).
    ## Used by the ML model during training to classify gain or loss based on indicator values
    # shift prices by a day to check if the current day's closing price
    # is greater than the previous day's closing price (gain)
    daily_gain_binary_data = closing_prices.shift(-1) > closing_prices
    # Converts Boolean values to integers (1 for gain, 0 for no gain)
    # and reformats as a 1D NumPy array for sklearn
    daily_gain_binary_data = daily_gain_binary_data.astype(int).values.reshape(-1,)

    return (data, daily_gain_binary_data, indicator_data)
