import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas_datareader import data
from datetime import datetime
import yfinance as yf
from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer

def write_dataframe_to_csv(dataframe, file_path):
    dataframe.to_csv(file_path)
 
def get_csv_dataframe(file_path):
    return pd.read_csv(file_path)

def get_excel_dataframe(file_path):
    dfs = pd.read_excel(file_path)
    return dfs
    
def get_combined_dataframes_using_date(dataframe_a, dataframe_b):
    dataframe_a['Interest Rate'] = ''
    for i in range(len(dataframe_a)):
        datetime_str_a = dataframe_a.iloc[i].Date
        date_str_a = datetime_str_a.split(' ')[0]
        datetime_obj_a = datetime.strptime(date_str_a, '%Y-%m-%d')
        for j in range(len(dataframe_b)):
            date_str_b = dataframe_b.iloc[j]['Effective Date']
            datetime_obj_b = datetime.strptime(date_str_b, '%m/%d/%Y')
            #datetime_obj_b = date_str_b
            if (datetime_obj_a.date() == datetime_obj_b.date()):
                dataframe_a.at[i, 'Interest Rate'] = dataframe_b.iloc[j]['Rate (%)']
                break
    return dataframe_a

def get_combined_dataframes_using_month_and_year(dataframe_a, dataframe_b):
    dataframe_a['Consumer Sentiment'] = ''
    for i in range(len(dataframe_a)):
        datetime_str_a = dataframe_a.iloc[i].Date
        date_str_a = datetime_str_a.split(' ')[0]
        datetime_obj_a = datetime.strptime(date_str_a, '%Y-%m-%d')
        for j in range(len(dataframe_b)):
            date_str_b = dataframe_b.iloc[j]['DATE']
            datetime_obj_b = datetime.strptime(date_str_b, '%Y-%m-%d')
            #datetime_obj_b = date_str_b
            if (datetime_obj_a.year == datetime_obj_b.year and datetime_obj_a.month == datetime_obj_b.month):
                dataframe_a.at[i, 'Consumer Sentiment'] = dataframe_b.iloc[j]['UMCSENT']
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

def get_dataframe_with_missing_values_filled_using_imputer(dataframe, imputer):
    dataframe_with_numerical_columns = dataframe.select_dtypes(include = [np.number])
    imputer.fit(dataframe_with_numerical_columns)
    dataframe_array = imputer.transform(dataframe_with_numerical_columns)
    return pd.DataFrame(dataframe_array, 
                        columns = dataframe_with_numerical_columns.columns, 
                        index = dataframe_with_numerical_columns.index)

print('Starting Dataset Combine Application.....')
start_date = "2007-07-30"
end_date = "2023-04-07"
imputer = SimpleImputer(strategy = "median")    
gold_inflation_crude_dji_wmt_int_cs = get_csv_dataframe('C:\All\Dataset\gold_inflation_crude_dji_wmt_int_cs.csv')
print('Starting Combine.....')

gold_inflation_crude_dji_wmt_int_cs = get_dataframe_with_missing_values_filled_using_imputer(
    gold_inflation_crude_dji_wmt_int_cs, imputer)
gold_inflation_crude_dji_wmt_int_cs.plot(y = 'Consumer Sentiment')
plt.xticks(rotation = 45)
gold_inflation_crude_dji_wmt_int_cs.plot(y = 'Interest Rate')
plt.xticks(rotation = 45)
plt.show()


def get_correlation_matrix_for_dataframe(dataframe):
    correlation_matrix = dataframe.corr()
    return correlation_matrix

correlation_matrix = get_correlation_matrix_for_dataframe(gold_inflation_crude_dji_wmt_int_cs)
print(correlation_matrix['Close'])

def display_scatter_matrix_for_dataframe(dataframe, list_of_columns):
    scatter_matrix(dataframe[list_of_columns], figsize = (12, 8))
    plt.show()

display_scatter_matrix_for_dataframe(gold_inflation_crude_dji_wmt_int_cs, 
                                     ['Close', 'Volume', 'Inflation', 'DJI Adj Close', 
                                      'WMT Adj Close', 'Interest Rate', 'Consumer Sentiment'])