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
from datetime import datetime, timedelta

def get_dataframe_for_ticker(ticker, time_period):
    dataframe = yf.Ticker(ticker).history(period = time_period).reset_index()
    return dataframe

def prep_data(dataframe, lookback, future, scaler, mode):
    dataframe_lstm = dataframe[['Open','High','Low','Close','Volume','Dividends','Stock Splits']]
    dataframe_lstm = dataframe_lstm.astype(float)
    dataframe_lstm_scaled = scaler.fit_transform(dataframe_lstm)
    
    x,y,dates_index = [], [], []
    for i in range(lookback, len(dataframe_lstm) - future + 1):
        x.append(dataframe_lstm_scaled[i - lookback : i, 0 : dataframe_lstm.shape[1]])
        dates_index.append(i + future - 1)
        if (mode == 'test'):
            y.append(dataframe_lstm.values[i + future - 1 : i + future, 0])
        else:
            y.append(dataframe_lstm_scaled[i + future - 1 : i + future, 0])

    dates_lstm = pd.to_datetime(dataframe['Date'].iloc[dates_index[0] : dates_index[len(dates_index) - 1] + 1])
    #np.array(x) gives a 3D array & np.array(y) gives 2D array
    return np.array(x), np.array(y), dataframe_lstm, dates_lstm
    
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
    es = tf.keras.callbacks.EarlyStopping(monitor = 'loss', patience = 15, restore_best_weights = True)
    model.fit(x, y, epochs = 50, verbose = 1, callbacks = [es], validation_split = 0.1, batch_size = 16)
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
    
    es = tf.keras.callbacks.EarlyStopping(monitor = 'loss', patience = 15, restore_best_weights = True)
    regressor.fit(x, y, epochs = 40, validation_split = 0.1, batch_size = 64, verbose = 1, callbacks = [es])
    return regressor

def predict_open(model, dates_for_testing, x_test, dataframe_for_testing, scaler):
    forecasting_dates = dates_for_testing.tolist()
    predicted = model.predict(x_test)
    predicted1 = np.repeat(predicted, dataframe_for_testing.shape[1], axis = -1)
    predicted_descaled = scaler.inverse_transform(predicted1)[:, 0]
    return predicted_descaled, forecasting_dates

def predict_future(model, dataframe, scaler):
    dataframe_lstm = dataframe[['Open','High','Low','Close','Volume','Dividends','Stock Splits']]
    x = []
    dataframe_lstm = dataframe_lstm.astype(float)
    dataframe_scaled = scaler.fit_transform(dataframe_lstm)
    x.append(dataframe_scaled[:, :])
    predicted = model.predict(np.array(x))
    predicted1 = np.repeat(predicted, dataframe_lstm.shape[1], axis = -1)
    predicted_descaled = scaler.inverse_transform(predicted1)[:, 0]
    
    last_date = dataframe['Date'].astype(str).iloc[-1]
    last_date_datetime = datetime.strptime(last_date.split(' ')[0], '%Y-%m-%d')
    next_date = last_date_datetime + timedelta(1)
    print('The Open Price in USD for date', next_date, 'is', predicted_descaled[-1])
    
def print_rmse(y_test, predicted_descaled):
    rmse = np.sqrt(np.mean(y_test - predicted_descaled)**2)
    print('rmse = ', rmse)

def plot_predicted_vs_real(predicted, real, x_axis, ticker):
    dataframe_for_plotting = pd.DataFrame({'Predicted':predicted, 'Real':real, 'Date':x_axis})
    dataframe_for_plotting.set_index('Date', inplace=True)
    plt.figure(figsize=(16,8))
    plt.title(ticker)
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Open Price USD', fontsize=18)
    plt.plot(dataframe_for_plotting[['Predicted','Real']])
    plt.legend(['Predicted','Real'])
    plt.show()

def get_output_dataframe(forecasting_dates, predicted_descaled):
    dates = []
    for i in forecasting_dates:
        dates.append(i.date())
    df_final = pd.DataFrame(columns=['Date','Open'])
    df_final['Date'] = pd.to_datetime(dates)
    df_final['Open'] = predicted_descaled
    return df_final

def results(dataframe_test, dataframe_train, dataframe_predict, lookback, future, scaler, ticker):
    x_train, y_train, dataframe_for_training, dates_for_training = prep_data(dataframe_train, lookback, future, scaler, 'train')
    x_test, y_test, dataframe_for_testing, dates_for_testing = prep_data(dataframe_test, lookback, future, scaler, 'test')
    
    model = Lstm_model(x_train, y_train)
    loss = pd.DataFrame(model.history.history)
    loss.plot()
    
    future = 30
    predicted_descaled, forecasting_dates = predict_open(model, dates_for_testing, x_test, dataframe_for_testing, scaler)
    
    predict_future(model, dataframe_predict, scaler)
    print_rmse(y_test.flatten(), np.array(predicted_descaled))
    plot_predicted_vs_real(np.array(predicted_descaled), y_test.flatten(), forecasting_dates, ticker)
    results = get_output_dataframe(forecasting_dates, predicted_descaled)   
    
    plt.show()
    fig = px.area(results, x = "Date", y = "Open", title = ticker)
    fig.update_yaxes(range = [results.Open.min() - 10, results.Open.max() + 10])
    fig.show()


scaler = StandardScaler()
ticker = 'GC=F'
time_period = '10y'
dataframe = get_dataframe_for_ticker(ticker, time_period)
dataframe_train = dataframe.iloc[0 : len(dataframe) - 300, :]
dataframe_test = dataframe.iloc[len(dataframe) - 300 : len(dataframe), :]
dataframe_predict = dataframe.iloc[len(dataframe) - 30 : len(dataframe), :]
results(dataframe_test, dataframe_train, dataframe_predict, 30, 1, scaler, ticker)
