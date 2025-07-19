# stock_data_loader.py

import yfinance as yf
import pandas as pd
import os
from datetime import datetime

def download_stock_data(ticker, start='2015-01-01', save_path='data/raw'):
    """
    Downloads clean OHLCV data from Yahoo Finance.
    """
    end = datetime.today().strftime('%Y-%m-%d')

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    filepath = os.path.join(save_path, f"{ticker}.csv")

    try:
        # Explicitly disable auto-adjust to avoid returns/dividends messing up columns
        df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)

        if df.empty:
            raise ValueError(f"No data found for {ticker}")

        # Only select needed columns
        cols_to_keep = ['Open', 'High', 'Low', 'Close', 'Volume']
        df = df[cols_to_keep]
        df.columns = [col.lower() for col in cols_to_keep]  # lowercase explicitly

        df = df.round(2)
        df.to_csv(filepath)
        print(f"✅ Saved cleaned data for {ticker} to {filepath}")
        return df

    except Exception as e:
        print(f"❌ Error downloading {ticker}: {e}")
        return None


def load_data(ticker, save_path='data/raw'):
    filepath = os.path.join(save_path, f"{ticker}.csv")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No saved data for {ticker}")
    return pd.read_csv(filepath, index_col='Date', parse_dates=True)
