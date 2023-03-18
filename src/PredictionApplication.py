import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import plotly.express as px
import yfinance as yf

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
    
Scaler = StandardScaler()
dataframe = get_dataframe_for_ticker('ENV', '5y')
Lstm_x, Lstm_y, dataframe_for_training, dates_for_training = get_data_for_training(dataframe, 30, 1, Scaler)