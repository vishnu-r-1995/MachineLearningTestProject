import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
from datetime import datetime
from sklearn.impute import SimpleImputer
from pandas.plotting import scatter_matrix
from sklearn.model_selection._split import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
import tensorflow as tf
from keras.optimizers.adam import Adam
from keras.wrappers.scikit_learn import KerasRegressor
from keras.callbacks import EarlyStopping

def get_dataframe_for_ticker(ticker, time_period):
    dataframe = yf.Ticker(ticker).history(period = time_period).reset_index()
    return dataframe

def get_csv_dataframe(file_path):
    return pd.read_csv(file_path)

def get_correlation_matrix_for_dataframe(dataframe):
    correlation_matrix = dataframe.corr()
    return correlation_matrix

def write_dataframe_to_csv(dataframe, file_path):
    dataframe.to_csv(file_path)

def get_combined_dataframes_using_date(dataframe_a, dataframe_b):
    dataframe_a['Inflation'] = 0
    for i in range(len(dataframe_a)):
        datetime_str_a = dataframe_a.iloc[i].Date
        date_str_a = datetime_str_a.split(' ')[0]
        datetime_obj_a = datetime.strptime(date_str_a, '%Y-%m-%d')
        for j in range(len(dataframe_b)):
            date_str_b = dataframe_b.iloc[j].DATE
            datetime_obj_b = datetime.strptime(date_str_b, '%Y-%m-%d')
            if (datetime_obj_a.date() == datetime_obj_b.date()):
                dataframe_a.at[i, 'Inflation'] = dataframe_b.iloc[j]['T10YIE']
                break
    return dataframe_a   
        
def get_dataframe_with_replaced_value(dataframe, column_name, from_value, to_value):
    dataframe[column_name].mask(dataframe[column_name] == from_value, to_value, inplace=True)
    return dataframe
  
ticker = 'GC=F'
time_period = '20y'
gold_inflation_dataframe = get_csv_dataframe('C:\All\Dataset\gold_prediction.csv');

def display_scatter_matrix_for_dataframe(dataframe, list_of_columns):
    scatter_matrix(dataframe[list_of_columns], figsize = (12, 8))
    plt.show()
    
correlation_matrix = get_correlation_matrix_for_dataframe(gold_inflation_dataframe)
train_set, test_set = train_test_split(gold_inflation_dataframe, test_size = 0.2, random_state = 42)
imputer = SimpleImputer(strategy = "median")
scaler = StandardScaler()
target_scaler = StandardScaler()

def get_dataframe_with_missing_values_filled_using_imputer(dataframe, imputer):
    dataframe_with_numerical_columns = dataframe.select_dtypes(include = [np.number])
    imputer.fit(dataframe_with_numerical_columns)
    dataframe_array = imputer.transform(dataframe_with_numerical_columns)
    return pd.DataFrame(dataframe_array, 
                        columns = dataframe_with_numerical_columns.columns, 
                        index = dataframe_with_numerical_columns.index)

def prep_data(dataframe, lookback, future, scaler, imputer, target_scaler):
    dataframe_lstm = dataframe[['Open','High','Low', 'Volume', 'Inflation']]
    dataframe_lstm = dataframe_lstm.astype(float)
    dataframe_lstm = get_dataframe_with_missing_values_filled_using_imputer(dataframe_lstm, imputer)
    dataframe_lstm_scaled = scaler.fit_transform(dataframe_lstm)
    
    dataframe_lstm_y = dataframe[['Close']].astype(float)
    dataframe_lstm_y_scaled = target_scaler.fit_transform(dataframe_lstm_y)
    
    x, y, dates_index = [], [], []
    for i in range(lookback, len(dataframe_lstm) - future + 1):
        x.append(dataframe_lstm_scaled[i - lookback : i, 0 : dataframe_lstm.shape[1]])
        dates_index.append(i + future - 1)
        y.append(dataframe_lstm_y_scaled[i + future - 1 : i + future, 0])

    dates_lstm = pd.to_datetime(dataframe['Date'].iloc[dates_index[0] : dates_index[len(dates_index) - 1] + 1])
    #np.array(x) gives a 3D array & np.array(y) gives 2D array
    return np.array(x), np.array(y), dataframe_lstm, dates_lstm

Lstm_x, Lstm_y, train_lstm_dataframe, train_dates = prep_data(dataframe = train_set, 
                                                              lookback = 30, 
                                                              future = 1, 
                                                              scaler = scaler, 
                                                              imputer = imputer,
                                                              target_scaler = target_scaler)

def Lstm_model():
    regressor = Sequential()
    regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (30, 5)))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units = 50, return_sequences = True))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units = 50, return_sequences = True))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units = 50))
    regressor.add(Dropout(0.2))
    regressor.add(Dense(units = 1))
    regressor.compile(optimizer = Adam(learning_rate = 0.001), loss = 'mean_squared_error', metrics = ['accuracy'])
    return regressor

    #es = tf.keras.callbacks.EarlyStopping(monitor = 'loss', patience = 15, restore_best_weights = True)
    #regressor.fit(x, y, epochs = 100, batch_size = 64, verbose = 1, callbacks = [es])
    #return regressor
    
estimator = KerasRegressor(build_fn = Lstm_model, epochs=100, batch_size=10, verbose=0)
early_stopping = EarlyStopping(monitor='loss', patience=1, verbose=1) 
history = estimator.fit(x = Lstm_x, 
                        y = Lstm_y, 
                        validation_split = 0.1, 
                        epochs = 5, 
                        batch_size = 10, 
                        callbacks = [early_stopping], 
                        verbose=1)

def visualize_learning_curve(history):
    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
visualize_learning_curve(history)

def prep_test_data(dataframe, lookback, future, scaler, imputer):
    dataframe_lstm = dataframe[['Open','High','Low','Volume', 'Inflation']]
    dataframe_lstm = dataframe_lstm.astype(float)
    dataframe_lstm = get_dataframe_with_missing_values_filled_using_imputer(dataframe_lstm, imputer)
    dataframe_lstm_scaled = scaler.transform(dataframe_lstm)
    
    dataframe_lstm_y = dataframe[['Close']].astype(float)
    dataframe_lstm_y = dataframe_lstm_y.values
    
    x, y, dates_index = [], [], []
    for i in range(lookback, len(dataframe_lstm) - future + 1):
        x.append(dataframe_lstm_scaled[i - lookback : i, 0 : dataframe_lstm.shape[1]])
        dates_index.append(i + future - 1)
        y.append(dataframe_lstm_y[i + future - 1 : i + future, 0])

    dates_lstm = pd.to_datetime(dataframe['Date'].iloc[dates_index[0] : dates_index[len(dates_index) - 1] + 1])
    #np.array(x) gives a 3D array & np.array(y) gives 2D array
    return np.array(x), np.array(y), dataframe_lstm, dates_lstm


Lstm_test_x, Lstm_test_y, test_lstm_dataframe, test_dates = prep_test_data(dataframe = test_set,
                                                              lookback = 30,
                                                              future = 1,
                                                              scaler = scaler,
                                                              imputer = imputer)

pred = estimator.predict(Lstm_test_x)
pred_reshaped = pred.reshape(-1, 1)
predicted_descaled = target_scaler.inverse_transform(pred.reshape(-1, 1))
print(predicted_descaled)
print(Lstm_test_y)
predicted_dataframe = pd.DataFrame({'y_pred': predicted_descaled[:,0]}) 
real_dataframe =  pd.DataFrame({'y_real': Lstm_test_y[:,0]})

def display_plot_comparing_dataframes(dataframe_real, dataframe_predicted):
    dataframe_combined = pd.concat([dataframe_real, dataframe_predicted], axis=1, join='inner')
    plt.plot(dataframe_combined)
    plt.legend(['real', 'predicted'])
    plt.show()

display_plot_comparing_dataframes(real_dataframe, predicted_dataframe)
