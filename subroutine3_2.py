import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def run_lstm_model():
    data = pd.read_csv('data/endata.csv')
    data['time'] = pd.to_datetime(data['time'])

    data_encoded = pd.get_dummies(data, columns=['City'])

    data_encoded['year'] = data_encoded['time'].dt.year
    data_encoded['month'] = data_encoded['time'].dt.month
    data_encoded['day'] = data_encoded['time'].dt.day
    data_encoded['hour'] = data_encoded['time'].dt.hour
    data_encoded['weekday'] = data_encoded['time'].dt.weekday

    data_encoded = data_encoded.drop('time', axis=1)

    X = data_encoded.drop(columns=['PM2_5'])
    y = data_encoded['PM2_5']

    scaler_X = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler_X.fit_transform(X)

    scaler_y = MinMaxScaler(feature_range=(0, 1))
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

    def create_dataset(X, y, time_step=1):
        X_data, y_data = [], []
        for i in range(len(X) - time_step):
            X_data.append(X[i:(i + time_step), :])
            y_data.append(y[i + time_step, 0])
        return np.array(X_data), np.array(y_data)

    time_step = 30
    X_data, y_data = create_dataset(X_scaled, y_scaled, time_step)

    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, shuffle=False)

    validation_size = 8
    X_val = X_test[-validation_size:]
    y_val = y_test[-validation_size:]

    X_train_val = X_test
    y_train_val = y_test

    model = Sequential()
    model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))
    model.add(LSTM(units=50, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(units=1))
    # model.add(Dropout(0.1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(X_train_val, y_train_val, epochs=20, batch_size=32, validation_data=(X_val, y_val))
    model.summary()

    predictions = model.predict(X_val)
    predictions = scaler_y.inverse_transform(predictions)

    rmse = np.sqrt(mean_squared_error(scaler_y.inverse_transform(y_val.reshape(-1, 1)), predictions))
    print(f"验证集RMSE: {rmse}")

    test_loss = model.evaluate(X_test, y_test)

if __name__ == "__main__":
    run_lstm_model()
