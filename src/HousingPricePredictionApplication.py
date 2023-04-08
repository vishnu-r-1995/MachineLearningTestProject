import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection._split import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

def get_csv_dataframe(file_path):
    return pd.read_csv(file_path)
    
housing_dataframe = get_csv_dataframe('C:\All\Dataset\housing.csv')

def print_dataframe_head_and_tail(dataframe):
    print(dataframe.head())
    print(dataframe.tail())

def print_dataframe_tail(dataframe):
    print(dataframe.tail())
#Create a column named index for the data frame
housing_dataframe_indexed = housing_dataframe.reset_index()

def add_one_hot_encoding_category_values_to_dataframe(dataframe, category):
    category_encoder = OneHotEncoder()
    category_encoded = category_encoder.fit_transform(dataframe[[category]])
    category_dataframe = pd.DataFrame(category_encoded.toarray(), 
                                      columns = category_encoder.get_feature_names_out(),
                                      index = dataframe.index)
    category_dataframe = category_dataframe.astype(float)
    return pd.concat([dataframe, category_dataframe], axis=1, join='inner')

housing_dataframe_indexed = add_one_hot_encoding_category_values_to_dataframe(
    housing_dataframe_indexed, "ocean_proximity")

def add_encoded_category_values_to_dataframe(dataframe, category, name_of_new_column):
    ordinal_encoder = OrdinalEncoder()
    category_encoded = ordinal_encoder.fit_transform(dataframe[[category]])
    dataframe[name_of_new_column] = category_encoded
    return dataframe

def get_dataframe_with_missing_values_filled_using_imputer(dataframe, imputer):
    dataframe_with_numerical_columns = dataframe.select_dtypes(include = [np.number])
    imputer.fit(dataframe_with_numerical_columns)
    dataframe_array = imputer.transform(dataframe_with_numerical_columns)
    return pd.DataFrame(dataframe_array, 
                        columns = dataframe_with_numerical_columns.columns, 
                        index = dataframe_with_numerical_columns.index)

def drop_column_from_dataframe(dataframe, column_name):
    dataframe.drop(column_name, axis = 1, inplace = True)
    return dataframe

housing_dataframe_indexed = drop_column_from_dataframe(housing_dataframe_indexed, "ocean_proximity")
housing_dataframe_indexed = housing_dataframe_indexed.astype(float)

imputer = SimpleImputer(strategy = "median")
housing_dataframe_indexed = get_dataframe_with_missing_values_filled_using_imputer(
    housing_dataframe_indexed, imputer)

#create categories based on the median income. eg: 0 - 1.5 will be in Category 1
housing_dataframe_indexed["income_category"] = pd.cut(housing_dataframe["median_income"],
                                              bins = [0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
                                              labels = [1, 2, 3, 4, 5])

#After pd.cut, the data type for income_category was category
housing_dataframe_indexed["income_category"] = housing_dataframe_indexed["income_category"].astype(float)

#stratified sets are dataframes
stratified_train_set, stratified_test_set = train_test_split(
    housing_dataframe_indexed, 
    test_size = 0.2, 
    stratify = housing_dataframe_indexed["income_category"], 
    random_state = 42)

for set in (stratified_train_set, stratified_test_set):
    set.drop("income_category", axis = 1, inplace = True)

def display_bar_chart_of_category_and_its_count(dataframe, category):
    #rot = 0 prevents rotation of x-axis values
    dataframe[category].value_counts().sort_index().plot.bar(rot = 0)
    plt.show()

def shuffle_and_split(dataframe, test_ratio):
    shuffled_indices = np.random.permutation(len(dataframe))
    test_set_size = int(len(dataframe) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return dataframe.iloc[train_indices], dataframe.iloc[test_indices]
    

def get_train_and_test_sets_split_using_stratified_sampling(
        dataframe, number_of_splits, stratified_sampling_category):
    splitter = StratifiedShuffleSplit(n_splits = number_of_splits, test_size = 0.2, random_state = 42)
    strat_splits = []
    for train_index, test_index in splitter.split(dataframe, housing_dataframe[stratified_sampling_category]):
        stratified_train_set_n = dataframe.iloc[train_index]
        stratified_test_set_n = dataframe.iloc[test_index]
        strat_splits.append([stratified_train_set_n, stratified_test_set_n])
    #For selecting one set --> stratified_train_set, stratified_test_set = strat_splits[0]
    return strat_splits


def display_scatter_plot_of_dataframe_using_longitude_and_latitude(dataframe):
    dataframe.plot(kind = "scatter", x = "longitude", y = "latitude", marker = "o", alpha = 0.2)
    plt.show()

def get_correlation_matrix_for_dataframe(dataframe):
    correlation_matrix = dataframe.corr()
    return correlation_matrix

housing_train_set = stratified_train_set.copy()
housing_train_set = housing_train_set.drop("median_house_value", axis = 1)
housing_train_set_labels = stratified_train_set[["median_house_value"]].copy()

housing_test_set = stratified_test_set.copy()
housing_test_set = housing_test_set.drop("median_house_value", axis = 1)
housing_test_set_labels = stratified_test_set[["median_house_value"]].copy()

#Scaling input data
std_scaler = StandardScaler()
housing_train_set_scaled = std_scaler.fit_transform(housing_train_set)
housing_test_set_scaled = std_scaler.transform(housing_test_set)

#Scaling output data, use to_frame() if Pandas Series is passed to fit_transform as 2D is expected
target_scaler = StandardScaler()
housing_train_set_labels_scaled = target_scaler.fit_transform(housing_train_set_labels)

def get_model_after_fitting_train_data_using_linear_regression(train_set_scaled, train_set_labels_scaled):
    #Linear Regression model
    regression_model = LinearRegression()
    return regression_model.fit(train_set_scaled, train_set_labels_scaled)

def get_predicted_values_as_dataframe_using_linear_regression(
        model, test_set_scaled, target_scaler, test_set_labels):
    scaled_predictions = model.predict(test_set_scaled)
    predictions = target_scaler.inverse_transform(scaled_predictions)
    return pd.DataFrame(predictions, 
                            columns = ["predicted"], 
                            index = test_set_labels.index)

regression_model = get_model_after_fitting_train_data_using_linear_regression(
    housing_train_set_scaled, housing_train_set_labels_scaled)

regression_predictions_dataframe = get_predicted_values_as_dataframe_using_linear_regression(
    regression_model, housing_test_set_scaled, target_scaler, housing_test_set_labels)

def get_model_after_fitting_train_data_using_decision_tree_regression(
        train_set_scaled, train_set_labels_scaled):
    decision_tree_model = DecisionTreeRegressor(random_state = 42)
    return decision_tree_model.fit(train_set_scaled, train_set_labels_scaled)

def get_predicted_values_as_dataframe_using_decision_tree_regression(
        model, test_set_scaled, target_scaler, test_set_labels):
    scaled_predictions = model.predict(test_set_scaled)
    predictions = target_scaler.inverse_transform(scaled_predictions.reshape(-1, 1))
    return pd.DataFrame(predictions, 
                            columns = ["predicted"], 
                            index = test_set_labels.index)

decision_tree_model = get_model_after_fitting_train_data_using_decision_tree_regression(
    housing_train_set_scaled, housing_train_set_labels_scaled)

decision_tree_predictions_dataframe = get_predicted_values_as_dataframe_using_decision_tree_regression(
    decision_tree_model, housing_test_set_scaled, target_scaler, housing_test_set_labels)

def get_model_after_fitting_train_data_using_random_forest_regression(
        train_set_scaled, train_set_labels_scaled):
    random_forest_model = RandomForestRegressor(random_state = 42)
    parameters = {'n_estimators':[1,10,30,50]}
    gd = GridSearchCV(random_forest_model, parameters)
    random_forest_model = gd.fit(train_set_scaled, train_set_labels_scaled)
    print(gd.best_params_)
    return random_forest_model

def get_predicted_values_as_dataframe_using_random_forest_regression(
        model, test_set_scaled, target_scaler, test_set_labels):
    scaled_predictions = model.predict(test_set_scaled)
    predictions = target_scaler.inverse_transform(scaled_predictions.reshape(-1, 1))
    return pd.DataFrame(predictions, 
                            columns = ["predicted"], 
                            index = test_set_labels.index)

random_forest_model = get_model_after_fitting_train_data_using_random_forest_regression(
    housing_train_set_scaled, housing_train_set_labels_scaled)

random_forest_predictions_dataframe = get_predicted_values_as_dataframe_using_random_forest_regression(
    random_forest_model, housing_test_set_scaled, target_scaler, housing_test_set_labels)
    
def display_plot_comparing_dataframes(dataframe_real, dataframe_predicted):
    dataframe_combined = pd.concat([dataframe_predicted, dataframe_predicted], axis=1, join='inner')
    plt.plot(dataframe_combined)
    plt.legend(['median_house_value', 'predicted'])
    plt.show()

def display_rmse(dataframe_real, dataframe_predicted):
    print('rmse =', np.sqrt((np.mean(dataframe_real.median_house_value) - np.mean(dataframe_predicted.predicted))**2))
    print('rmse =', mean_squared_error(dataframe_real.median_house_value, dataframe_predicted.predicted, squared = True))
    
#display_plot_comparing_dataframes(housing_test_set_labels, predictions_dataframe)
display_rmse(housing_test_set_labels, random_forest_predictions_dataframe)
#display_rmse(housing_test_set_labels, decision_tree_predictions_dataframe)
print_dataframe_tail(pd.concat([housing_test_set_labels, random_forest_predictions_dataframe], axis=1, join='inner'))
#print_dataframe_tail(pd.concat([housing_test_set_labels, decision_tree_predictions_dataframe], axis=1, join='inner'))