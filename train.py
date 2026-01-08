from preprocessing import load_and_preprocess
from model.cnn_lstm_model import build_model
from sklearn.model_selection import train_test_split

X, y = load_and_preprocess("data/seismic.csv")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

model = build_model(X_train.shape[1:])

history = model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test, y_test)
)

model.save("earthquake_model.h5")
