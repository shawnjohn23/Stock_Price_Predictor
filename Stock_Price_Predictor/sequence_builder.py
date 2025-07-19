# sequence_builder.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def create_sequences(data: pd.DataFrame, sequence_length: int = 60, target_col: str = 'close'):
    """
    Create sequences from historical data for LSTM/GRU training.

    Args:
        data: DataFrame with features (should be scaled)
        sequence_length: how many days to look back
        target_col: column to predict (default: 'close')

    Returns:
        Tuple of np.arrays: (X, y)
    """
    X, y = [], []
    for i in range(sequence_length, len(data)):
        X.append(data.iloc[i - sequence_length:i].values)
        y.append(data.iloc[i][target_col])
    return np.array(X), np.array(y)


def scale_data(df: pd.DataFrame):
    """
    Scale all features between 0 and 1 using MinMaxScaler.
    Returns scaled DataFrame and scaler object (for inverse transform).
    """
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)
    scaled_df = pd.DataFrame(scaled, index=df.index, columns=df.columns)
    return scaled_df, scaler

def train_test_split_time_series(X, y, train_ratio=0.8):
    """
    Splits X and y while maintaining time order (no shuffling!).
    """
    train_size = int(len(X) * train_ratio)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    return X_train, X_test, y_train, y_test

# sequence_builder.py (append this)

def inverse_scale(scaler, y_scaled, column_index=3):
    """
    Inverse scales a 1D array of predictions using the original scaler.
    
    Args:
        scaler: fitted MinMaxScaler
        y_scaled: array of shape (n_samples,) – predicted or actual close values (scaled)
        column_index: index of 'close' in the original feature set (default: 3)
        
    Returns:
        Unscaled values in original price range.
    """
    dummy = np.zeros((len(y_scaled), scaler.n_features_in_))
    dummy[:, column_index] = y_scaled
    inv = scaler.inverse_transform(dummy)
    return inv[:, column_index]

