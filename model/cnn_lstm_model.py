from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout

def build_model(input_shape):
    model = Sequential()

    model.add(Conv1D(64, kernel_size=3, activation="relu", input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))

    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.3))

    model.add(Dense(1))

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )

    return model
