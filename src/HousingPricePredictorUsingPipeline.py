import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas_datareader import data
from datetime import datetime
import yfinance as yf
from pandas.plotting import scatter_matrix

def write_dataframe_to_csv(dataframe, file_path):
    dataframe.to_csv(file_path)
 
def get_csv_dataframe(file_path):
    return pd.read_csv(file_path)
   
def get_combined_dataframes_using_date(dataframe_a, dataframe_b):
    dataframe_a['DJI Adj Close'] = 0
    for i in range(len(dataframe_a)):
        datetime_str_a = dataframe_a.iloc[i].Date
        date_str_a = datetime_str_a.split(' ')[0]
        datetime_obj_a = datetime.strptime(date_str_a, '%Y-%m-%d')
        for j in range(len(dataframe_b)):
            date_str_b = dataframe_b.iloc[j].Date
            #datetime_obj_b = datetime.strptime(date_str_b, '%Y-%m-%d')
            datetime_obj_b = date_str_b
            if (datetime_obj_a.date() == datetime_obj_b.date()):
                dataframe_a.at[i, 'DJI Adj Close'] = dataframe_b.iloc[j]['Adj Close']
                break
    return dataframe_a

def get_dataframe_between_start_and_end_date(ticker, start_date, end_date):
    dataframe = data.DataReader(name=ticker, data_source='yahoo', start=start_date, end=end_date)
    return dataframe

def get_yfinance_dataframe_between_start_and_end_date(ticker, start_date, end_date):
    start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
    end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
    dataframe = yf.download(tickers = ticker, start = start_date, end = end_date, interval="1d")
    return dataframe.reset_index()

print('Starting Dataset Combine Application.....')
start_date = "2007-07-30"
end_date = "2023-04-07"    
dow_jones_dataframe = get_yfinance_dataframe_between_start_and_end_date("^DJI", start_date, end_date)
gold_inflation_crude_dataframe = get_csv_dataframe('C:\All\Dataset\gold_prediction_test.csv')
print('Starting Combine.....')
gold_inflation_crude_dji = get_combined_dataframes_using_date(gold_inflation_crude_dataframe, dow_jones_dataframe)
write_dataframe_to_csv(gold_inflation_crude_dji, 'C:\All\Dataset\gold_inflation_crude_dji.csv')

def get_correlation_matrix_for_dataframe(dataframe):
    correlation_matrix = dataframe.corr()
    return correlation_matrix

correlation_matrix = get_correlation_matrix_for_dataframe(gold_inflation_crude_dji)
print(correlation_matrix['Close'])

def display_scatter_matrix_for_dataframe(dataframe, list_of_columns):
    scatter_matrix(dataframe[list_of_columns], figsize = (12, 8))
    plt.show()

display_scatter_matrix_for_dataframe(gold_inflation_crude_dji, ['Close', 'Volume', 'Inflation', 'DJI Adj Close'])