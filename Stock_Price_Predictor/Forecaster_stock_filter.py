#Forecaster stock organizer
import numpy as np
from stock_data_loader import load_data
from sequence_builder import scale_data, inverse_scale
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

STOCKS = ['AAPL', 'GOOG', 'MSFT']  # Change as needed
SEQUENCE_LENGTH = 60
FORECAST_DAYS = 252 * 2  # 2 years
CLOSE_INDEX = 3  # Column index for 'close' in your data

model = load_model('trained_model.h5')

results = []

for ticker in STOCKS:
    # 1. Load and preprocess data
    df = load_data(ticker)
    df = df.dropna()
    scaled_df, scaler = scale_data(df)
    n_features = scaled_df.shape[1]

    # 2. Prepare the latest sequence for prediction
    last_sequence = scaled_df[-SEQUENCE_LENGTH:].values
    input_seq = last_sequence.reshape((1, SEQUENCE_LENGTH, n_features))

    preds_scaled = []

    for _ in range(FORECAST_DAYS):
        # Predict next OHLCV row (scaled)
        pred_scaled = model.predict(input_seq, verbose=0)[0]  # shape (n_features,)
        preds_scaled.append(pred_scaled)

        # Prepare next input sequence: roll forward the entire predicted row
        next_seq = np.vstack([input_seq[0][1:], pred_scaled])
        input_seq = next_seq.reshape((1, SEQUENCE_LENGTH, n_features))

    # After prediction loop
    preds_scaled = np.array(preds_scaled)  # shape (FORECAST_DAYS, n_features)
    preds_unscaled = inverse_scale(scaler, preds_scaled)  # shape (FORECAST_DAYS, n_features)
    diff = preds_unscaled[-1, CLOSE_INDEX] - preds_unscaled[0, CLOSE_INDEX]
    results.append((ticker, diff, preds_unscaled[:, CLOSE_INDEX]))

# Sort by difference descending
results_sorted = sorted(results, key=lambda x: x[1], reverse=True)

# Print results
for ticker, diff, preds_close in results_sorted:
    print(f"{ticker}: Forecasted 2-year close price change: {diff:.2f}")
    print(f"  First predicted close: {preds_close[0]:.2f}, Last predicted close: {preds_close[-1]:.2f}")

import pandas as pd

# Find the stock with the least difference (least change)
best_ticker, diff, best_preds = min(results, key=lambda x: abs(x[1]))
print(f"Best performing stock: {best_ticker} (difference {diff:.2f})")

# Load historical data for best stock
df_hist = load_data(best_ticker)
df_hist = df_hist.dropna()

# Get the last two years of history (504 trading days)
history_days = 252 * 2
df_hist_last2yrs = df_hist[-history_days:]
hist_dates = df_hist_last2yrs.index
hist_close = df_hist_last2yrs['close'].values

# Prepare forecast dates
forecast_days = len(best_preds)
last_hist_date = hist_dates[-1]
forecast_dates = pd.bdate_range(start=last_hist_date + pd.Timedelta(days=1), periods=forecast_days)

# Plot both history and forecast
plt.figure(figsize=(14,7))
plt.plot(hist_dates, hist_close, label='Actual Close (previous 2 years)', lw=2)
plt.plot(forecast_dates, best_preds, label='Predicted Close (next 2 years)', linestyle='--', lw=2)
plt.title(f'{best_ticker}: 2 Years History and 2 Years Forecast')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()