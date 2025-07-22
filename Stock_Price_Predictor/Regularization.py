#Regularization
from tensorflow.keras import layers, regularizers, Sequential

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

# Usage:
# model = build_regularized_lstm(X_train.shape[1:])