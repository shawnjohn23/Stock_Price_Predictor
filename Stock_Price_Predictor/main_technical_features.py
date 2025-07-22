#main_technical_features
from stock_data_loader import load_data
from Technical_Features import add_technical_indicators
from sequence_builder import scale_data, create_sequences, inverse_scale
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras import layers, regularizers, Sequential
import numpy as np
import matplotlib.pyplot as plt

# 1. Load and engineer features
df = load_data('AAPL')
#df = add_technical_indicators(df)
#df = df.dropna()  # Remove rows with missing indicator values

# 2. Scale features
scaled_df, scaler = scale_data(df)

# 3. Build sequences
X, y = create_sequences(scaled_df, sequence_length=60)

# 4. Time Series Cross-Validation
tscv = TimeSeriesSplit(n_splits=5)
all_val_scores = []
models = []
fold_indices = []

for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
"""
    def build_regularized_lstm(input_shape):
        model = Sequential()
        model.add(layers.LSTM(64, return_sequences=True, input_shape=input_shape,
                              kernel_regularizer=regularizers.l2(0.001)))
        model.add(layers.Dropout(0.3))
        model.add(layers.LSTM(32, kernel_regularizer=regularizers.l2(0.001)))
        model.add(layers.Dropout(0.3))
        model.add(layers.Dense(1))
        model.compile(optimizer='adam', loss='mse')
        return model
"""
from model import build_lstm_gru_model

model = build_lstm_gru_model(X_train.shape[1:])


#model = build_regularized_lstm(X_train.shape[1:])
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_val, y_val),
    verbose=1
)

val_loss = min(history.history['val_loss'])
all_val_scores.append(val_loss)
models.append(model)
fold_indices.append((train_idx, val_idx))
print(f"Fold {fold+1} best val_loss: {val_loss:.5f}")

print(f"Average best validation loss across folds: {np.mean(all_val_scores):.5f}")

# 5. Use the last fold's model and validation set for final evaluation
final_model = models[-1]
_, final_val_idx = fold_indices[-1]
X_test = X[final_val_idx]
y_test = y[final_val_idx]

# Save the model
final_model.save("trained_model.h5")
print("✅ Model saved to trained_model.h5")

# Predict and inverse scale
y_pred_scaled = final_model.predict(X_test).flatten()
y_pred = inverse_scale(scaler, y_pred_scaled)  # Adjust column index for 'close' in scaler
y_true = inverse_scale(scaler, y_test)

# Plot actual vs predicted (with time index)
full_index = df.index
test_start = len(df) - len(y_test)
prediction_dates = full_index[test_start:]

plt.figure(figsize=(12, 6))
plt.plot(prediction_dates, y_true, label='Actual Close (Test)', linewidth=2)
plt.plot(prediction_dates, y_pred, label='Predicted Close (Test)', linestyle='--')
plt.title('Actual vs Predicted Stock Price (Test Set)')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("predicted_vs_actual_price.png", dpi=300)
plt.show()
# You can then retrain on the full data or use the best model as needed.