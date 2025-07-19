# model.py

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout

def build_lstm_gru_model(input_shape):
    model = Sequential()
    
    model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
    model.add(GRU(64, return_sequences=False))
    model.add(Dropout(0.2))
    
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))  # output layer for regression (predicting next close price)

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model
