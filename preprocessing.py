import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess(file_path, window_size=100):
    data = pd.read_csv(file_path)
    signal = data.values.flatten()

    scaler = MinMaxScaler()
    signal = scaler.fit_transform(signal.reshape(-1, 1)).flatten()

    X, y = [], []
    for i in range(len(signal) - window_size):
        X.append(signal[i:i + window_size])
        y.append(signal[i + window_size])

    X = np.array(X).reshape(-1, window_size, 1)
    y = np.array(y)

    return X, y
