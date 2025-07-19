# predictor_main.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from stock_data_loader import load_data
from sequence_builder import scale_data, inverse_scale
from predictor import predict_next_close


def main():
    # Configuration
    ticker = "AAPL"
    model_path = "trained_model.h5"
    sequence_length = 60
    forecast_days = 30

    # 1. Load historical data
    df = load_data(ticker)

    # 2. Scale full dataset
    df_scaled, scaler = scale_data(df)

    # 3. Load trained model
    model = load_model(model_path)

    # 4. Generate future predictions (scaled)
    future_preds_scaled = []
    df_scaled_copy = df_scaled.copy()

    for _ in range(forecast_days):
        pred_scaled = predict_next_close(model, df_scaled_copy, sequence_length)
        future_preds_scaled.append(pred_scaled)

        # Prepare next row by copying last and updating scaled close
        last_row = df_scaled_copy.iloc[-1].copy()
        last_row['close'] = pred_scaled
        df_scaled_copy = pd.concat([
            df_scaled_copy,
            pd.DataFrame([last_row])
        ], ignore_index=True)

    # 5. Build date index for forecast horizon (trading days)
    last_date = df.index[-1]
    forecast_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=forecast_days,
        freq='B'
    )

    # 6. Inverse scale predictions to actual price
    future_preds = inverse_scale(
        scaler,
        np.array(future_preds_scaled),
        column_index=list(df.columns).index('close')
    )

    # 7. Save predictions to CSV
    predictions_df = pd.DataFrame({
        'date': forecast_dates,
        'predicted_close': future_preds
    })
    predictions_df.to_csv('predictions.csv', index=False)
    print("✅ Predictions saved to predictions.csv")

    # 8. Plot only the forecasted predictions
    plt.figure(figsize=(12, 6))
    plt.plot(forecast_dates, future_preds, linestyle='--', marker='o', label='Forecast Close')

    plt.title(f'{ticker} 30-Day Close Price Forecast')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
