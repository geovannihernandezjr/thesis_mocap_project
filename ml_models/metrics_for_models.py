"""
Method to calcualte the metrics based on MSE, RMSE and MAE
"""
from keras.metrics import mean_squared_error, mean_absolute_error
from numpy import sqrt


def calculate_metrics(true_values, predictions):
    '''

    :param true_values: raw values or labels
    :param predictions: the prediction from the model
    :return: the list of metrics
    '''
    mse = mean_squared_error(true_values, predictions)
    rmse = sqrt(mse)
    mae = mean_absolute_error(true_values, predictions)
    return mse, rmse, mae
