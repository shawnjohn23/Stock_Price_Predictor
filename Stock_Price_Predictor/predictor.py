# predictor.py

import numpy as np

def predict_next_close(model, df_scaled, sequence_length=60):
    """
    Uses the last `sequence_length` days to predict the next day's close price.
    """
    last_seq = df_scaled[-sequence_length:].values  # shape: (60, 5)
    X_input = np.expand_dims(last_seq, axis=0)      # shape: (1, 60, 5)
    
    y_pred_scaled = model.predict(X_input).flatten()[0]
    return y_pred_scaled
