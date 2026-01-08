import numpy as np
from tensorflow.keras.models import load_model
from preprocessing import load_and_preprocess
from sklearn.metrics import mean_squared_error

model = load_model("earthquake_model.h5")
X, y = load_and_preprocess("data/seismic.csv")

preds = model.predict(X)
rmse = np.sqrt(mean_squared_error(y, preds))

print("RMSE:", rmse)
