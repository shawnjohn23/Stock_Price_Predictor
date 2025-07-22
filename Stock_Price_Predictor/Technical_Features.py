#Technical Indicators 
import pandas as pd
import numpy as np

def add_technical_indicators(df):
    """
    Add common technical indicators to a DataFrame with columns: open, high, low, close, volume.
    Returns a new DataFrame with extra columns.
    """
    df = df.copy()

    # Simple Moving Averages
    df['sma_5'] = df['close'].rolling(window=5).mean()
    df['sma_10'] = df['close'].rolling(window=10).mean()
    df['sma_20'] = df['close'].rolling(window=20).mean()

    # Exponential Moving Averages
    df['ema_5'] = df['close'].ewm(span=5, adjust=False).mean()
    df['ema_10'] = df['close'].ewm(span=10, adjust=False).mean()

    # Relative Strength Index (RSI)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))

    # MACD (Moving Average Convergence Divergence)
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    df['bb_std'] = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']

    # Stochastic Oscillator %K and %D
    low_14 = df['low'].rolling(window=14).min()
    high_14 = df['high'].rolling(window=14).max()
    df['stoch_k'] = 100 * (df['close'] - low_14) / (high_14 - low_14)
    df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()

    # Average True Range (ATR)
    high_low = df['high'] - df['low']
    high_close_prev = np.abs(df['high'] - df['close'].shift())
    low_close_prev = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    df['atr_14'] = tr.rolling(window=14).mean()

    # On-Balance Volume (OBV)
    obv = [0]
    for i in range(1, len(df)):
        if df['close'].iloc[i] > df['close'].iloc[i-1]:
            obv.append(obv[-1] + df['volume'].iloc[i])
        elif df['close'].iloc[i] < df['close'].iloc[i-1]:
            obv.append(obv[-1] - df['volume'].iloc[i])
        else:
            obv.append(obv[-1])
    df['obv'] = obv

    # Daily Returns
    #df['return_1d'] = df['close'].pct_change(1)
    #df['return_5d'] = df['close'].pct_change(5)
	
    # Price Ratios
    #df['hl_ratio'] = df['high'] / df['low']
    #df['co_ratio'] = df['close'] / df['open']

    # Fill NA values (optional, depending on model needs)
    df = df.fillna(0)

    return df