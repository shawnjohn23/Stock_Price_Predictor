# main.py

from stock_data_loader import download_stock_data
from stock_data_loader import load_data
from sequence_builder import scale_data, create_sequences

# Download data 
#download_stock_data('AAPL')

# Load data
df = load_data('AAPL')


# Scale it
scaled_df, scaler = scale_data(df)


# Create sequences
X, y = create_sequences(scaled_df, sequence_length=60)

print("✅ Input shape:", X.shape)  # e.g., (n_samples, 60, 5)
print("✅ Output shape:", y.shape)  # e.g., (n_samples,)

# main.py (continued)

from sequence_builder import train_test_split_time_series

# Split sequences
X_train, X_test, y_train, y_test = train_test_split_time_series(X, y)

print("🧪 Train shape:", X_train.shape, y_train.shape)
print("🧪 Test shape:", X_test.shape, y_test.shape)

# main.py (continued)

from model import build_lstm_gru_model

model = build_lstm_gru_model(X_train.shape[1:])



history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# Save model
model.save("trained_model.h5")
print("✅ Model saved to trained_model.h5")


# main.py (continued)

import matplotlib.pyplot as plt
from sequence_builder import inverse_scale
# Predict
y_pred_scaled = model.predict(X_test).flatten()

# Inverse scale
y_pred = inverse_scale(scaler, y_pred_scaled, column_index=3)
y_true = inverse_scale(scaler, y_test, column_index=3)
# main.py (improved plotting)

# Re-load full scaled data (for plotting full price history)
scaled_df, _ = scale_data(df)  # fresh, full set

# Get the index of the full data to align prediction points
full_index = df.index

# Compute date range for y_test
test_start = len(df) - len(y_test)
prediction_dates = full_index[test_start:]

# Inverse scale all actual closes (for full line plot)
actual_full_close = inverse_scale(scaler, scaled_df['close'].values, column_index=3)

# Plot
plt.figure(figsize=(12, 6))
plt.plot(full_index, actual_full_close, label='Actual Close Price (Full)', linewidth=2, alpha=0.4)
plt.plot(prediction_dates, y_true, label='Actual Close (Test)', linewidth=2)
plt.plot(prediction_dates, y_pred, label='Predicted Close (Test)', linestyle='--')

plt.title('LSTM-GRU Predicted vs Actual Stock Price')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("predicted_vs_actual_price.png", dpi=300)


plt.show()





