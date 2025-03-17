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
