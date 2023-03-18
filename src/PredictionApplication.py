import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import plotly.express as px
import yfinance as yf

def get_dataframe_for_ticker(ticker, period_of_time_series):
    dataframe = yf.Ticker(ticker).history(period = period_of_time_series).reset_index()
    return dataframe
dataframe = get_dataframe_for_ticker('ENV', '5y')
