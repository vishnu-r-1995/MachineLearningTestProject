import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import plotly.express as px
import yfinance as yf
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Flatten

def get_dataframe_for_ticker(ticker, period_of_time_series):
    dataframe = yf.Ticker(ticker).history(period = period_of_time_series).reset_index()
    return dataframe

def get_data_for_training(dataframe, lookback, future, Scale):
    dates_for_training = pd.to_datetime(dataframe['Date'])
    dataframe_for_training = dataframe[['Open','High','Low','Close','Volume','Dividends','Stock Splits']]
    dataframe_for_training = dataframe_for_training.astype(float)
    dataframe_for_training_scaled = Scaler.fit_transform(dataframe_for_training)
    
    x,y = [], []
    for i in range(lookback, len(dataframe_for_training) - future + 1):
        x.append(dataframe_for_training_scaled[i - lookback : i, 0 : dataframe_for_training.shape[1]])
        y.append(dataframe_for_training_scaled[i + future - 1 : i + future, 0])
    #np.array(x) gives a 3D array & np.array(y) gives 2D array
    return np.array(x), np.array(y), dataframe_for_training, dates_for_training
    
def Lstm_fallback(x, y):
    model = Sequential()
    model.add(LSTM(64, activation = 'relu', input_shape = (x.shape[1], x.shape[2]), return_sequences = True))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(LSTM(32, activation='relu', return_sequences=False))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(y.shape[1], activation='relu'))
    
    adam_optimizer = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)  
    model.compile(
            loss='mse',
            optimizer=adam_optimizer,
        )
    es = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=15, restore_best_weights=True)
    model.fit(x, y, epochs=100, verbose=1, callbacks=[es], validation_split=0.1, batch_size=16)
    return model

def Lstm_model(x, y):
    regressor = Sequential()
    regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (x.shape[1], x.shape[2])))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units = 50, return_sequences = True))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units = 50, return_sequences = True))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units = 50))
    regressor.add(Dropout(0.2))
    regressor.add(Dense(units = 1))
    regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
    
    es = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=15, restore_best_weights=True)
    regressor.fit(x, y, epochs = 100, validation_split=0.1, batch_size = 64, verbose=1, callbacks=[es])
    return regressor

Scaler = StandardScaler()
dataframe = get_dataframe_for_ticker('ENV', '5y')
Lstm_x, Lstm_y, dataframe_for_training, dates_for_training = get_data_for_training(dataframe, 30, 1, Scaler)

