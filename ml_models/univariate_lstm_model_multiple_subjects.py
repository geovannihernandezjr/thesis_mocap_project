
"""
@author Geovanni Hernandez
Created to design experiments for LSTM model for Univariate analysis of one single marker from MoCap data.
Using command line arguments different experiments can be ran with different size of timestep and batches size.
"""
import os
from argparse import ArgumentParser

"""Command Line Arguments For Experiments"""
parser = ArgumentParser(description="Begin experiments based on start and end values from cmd")
parser.add_argument("start", type=int, help="n integer for starter value of experiments")
# parser.add_argument("end", type=int, help="an integer for ending value of experiments")
parser.add_argument("marker", type=str, help="string for marker name for experiments")
parser.add_argument('timestep', type=int, help="an integer for value of timestep to use in model")
parser.add_argument("batch", type=int, help='an integer for batch size value of training of experiments')
parser.add_argument("-v", "--verbose", action="store_true", help="output verbosity")
parser.add_argument("--t", "--test", action='store_true', dest='test', help="use 250 values of data to run program")
"""Use Command Line Arguments to choice START & END"""
args = parser.parse_args()
START = args.start
# END = args.end
seed = 1
os.environ['PYTHONHASHSEED '] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # to ignore warnings from tensorflow
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import tensorflow.compat.v1 as tf_one
import tensorflow as tf
import random as rn
# The below is necessary for starting Numpy generated random numbers
# in a well-defined initial state.
np.random.seed(seed)
# The below is necessary for starting core Python generated random numbers
# in a well-defined state.
rn.seed(seed)
# The below tf.set_random_seed() will make random number generation
# in the TensorFlow backend have a well-defined initial state.
# For further details, see:
# https://www.tensorflow.org/api_docs/python/tf/set_random_seed
tf.compat.v1.random.set_random_seed(seed)
# from ml_models.set_random_seed import set_random_seed
from keras import backend as K
from data_preprocessing.file_retrive import *
from data_preprocessing.preprocessing import univariate_data_to_sequence
from ml_models.models import define_lstm
from ml_models.save_info import save_to_excel
from ml_models.metrics_for_models import calculate_metrics
from plotting.metric_plot import plot_metrics
from plotting.plot_data import plot_mocap_predictions, setup_data_to_plot
from sklearn.preprocessing import MinMaxScaler
from joblib import dump, load
from ml_models.set_random_seed import set_random_seed


"""HIPE SERVER DIRECTORY"""
# TRAINING_FILES = '/home/g_h62/conference_paper/Train/*.csv'
# TRAINING_PARENT_DIR = '/home/g_h62/conference_paper/Train'
# TESTING_FILES = '/home/g_h62/conference_paper/Test/*.csv'
# TESTING_PARENT_DIR = '/home/g_h62/conference_paper/Test'

"""LEAP SERVER DIRECTORY"""
TRAINING_PARENT_DIR = '/gpfs/home/g_h62/thesis_project/borg_predictions/Train/'
TRAINING_FILES = '/gpfs/home/g_h62/thesis_project/borg_predictions/Train/*.csv'
TESTING_PARENT_DIR = '/gpfs/home/g_h62/thesis_project/borg_predictions/Test/'
TESTING_FILES = '/gpfs/home/g_h62/thesis_project/borg_predictions/Test/*.csv'

"""Laptop directory for csv files which are merged with borg"""
# TRAINING_PARENT_DIR = 'C:/Users/geova/Desktop/Filled_Files/Train/'
# TRAINING_FILES = 'C:/Users/geova/Desktop/Filled_Files/Train/*.csv'
# TESTING_PARENT_DIR = 'C:/Users/geova/Desktop/Filled_Files/Test/'
# TESTING_FILES = 'C:/Users/geova/Desktop/Filled_Files/Test/*.csv'


"""Global Constants for Experiments"""
TIMESTEPS = args.timestep
NUM_FEATURES = 1 # only used one column for predicting
BATCH_SIZE = args.batch
STATEFUL = False
if args.test is True:
    INCREMENTS = 10
elif args.test is False:
    INCREMENTS = 1000
else:
    INCREMENTS = 0
"""Directory that will be created to save experiment results in"""
# parent_directory = 'C:/Users/geova/Desktop/Filled_Files/experiments/'
parent_directory = '/gpfs/home/g_h62/thesis_project/timesteps_gru/experiments/'
# parent_directory = '/home/g_h62/conference_paper/experiments/'
# parent_directory = 'D:/HIPE/Clean_qtm/Fill/experiments/'
# parent_directory = '/gpfs/home/g_h62/thesis_project/conference_paper/experiments'

""" Directory for experiment set """
experiment_set = f'LSTM-1L-SGD-MSE-Batch{BATCH_SIZE}_Experiments'
training_exp_directory = os.path.join(parent_directory, experiment_set)
if not os.path.exists(training_exp_directory):
    os.mkdir(training_exp_directory)

"""Directory for the type of markers being used in the experiment"""
marker_name = args.marker #"RElbowOut X" "All"
marker_exp_directory = os.path.join(training_exp_directory, marker_name)
if not os.path.exists(marker_exp_directory):
    os.mkdir(marker_exp_directory)

new_dir = f'LSTM_Model_T{TIMESTEPS}_B{BATCH_SIZE}'
file_path_to_save_results = os.path.join(marker_exp_directory, new_dir)
if os.path.exists(file_path_to_save_results):
    pass
else:
    os.mkdir(file_path_to_save_results)

def fit_stateless_lstm(model, batch_size, num_epoch=10, valid_data=None, v_split=None, valid_train=True,
                       train_loss_lst=None, train_mse_lst=None, train_rmse_lst=None, train_mae_lst=None,
                       val_loss_lst=None, val_mse_lst=None,
                       val_rmse_lst=None, val_mae_lst=None):
    """
        This is the implementation to train the model defined.
    :param model: the defined model created
    :param batch_size: the size of batch that will be used when training each sequence
    :param num_epoch: he number of iterations for training the model
    :param valid_data: the validation dataset
    :param v_split: the percentage of validation split that is going to be done on the provided training data
    :param valid_train: boolean to indicate if there is validation data being used
    :param train_loss_lst: list to contain all the values for loss curve
    :param train_mse_lst: list to contain all the mean square error values from training
    :param train_rmse_lst: list to contain all the root mse values from training
    :param train_mae_lst: list to contain all the mean absolute error from training
    :param val_loss_lst: list to contain the values for loss curve on based on validation data used
    :param val_mse_lst: list to contain the values for the mse values basedo no validation
    :param val_rmse_lst: list to contain the values for the rmse values based on validation
    :param val_mae_lst: list to contain the values for the mean absolute error values based on validation
    :return: the trained model
    """
    for epoch_i in range(num_epoch):
        for file_index, file in enumerate(training_file_list): # suffle order of names in list for randomization of training per file
            filename = tf.strings.split(file, os.sep)[-1].numpy().decode()
            """The LSTM network expects the input data (X) to be provided 
            with a specific array structure in the form of: [samples, time steps, features].
            Currently, our data is in the form: [samples, features] and we are framing the 
            problem as one time step for each sample. We can transform the prepared train and test input data into the 
            expected structure using numpy.reshape() as follows:"""
            trainX, trainY = fetch_train_data(filename, file_index)
            # trainX = trainX.reshape(trainX.shape[0], TIMESTEPS, trainX.shape[1])
            if valid_data is not None:
                validX, validY = valid_data[:, 0:-1], valid_data[:, -1]
                validX = validX.reshape(validX.shape[0], TIMESTEPS, validX.shape[1])  # samples, timesteps, features

            if True is valid_train:
                if v_split:
                    model.fit(trainX, trainY, epochs=1, batch_size=batch_size, validation_split=v_split, verbose=0,
                              shuffle=False)

                else:
                    model.fit(trainX, trainY, epochs=1, batch_size=batch_size, validation_data=(validX, validY),
                              verbose=0,
                              shuffle=False)
            else:
                model.fit(trainX, trainY, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)

        if valid_train:
            val_loss_lst.append(model.history.history['val_loss'])
            val_mse_lst.append(model.history.history['val_mse'])
            val_rmse_lst.append(model.history.history['val_rmse'])
            val_mae_lst.append(model.history.history['val_mae'])

        train_loss_lst.append(model.history.history['loss'])
        train_mse_lst.append(model.history.history['mse'])
        train_rmse_lst.append(model.history.history['rmse'])
        train_mae_lst.append(model.history.history['mae'])

def create_summary_of_experiment(file_path_to_save_results, png_name, validation_split, batch_size, neurons, stateful,
                                 each_epoch, optimizer, test_metrics):
    '''
        This is just to ensure and have some form of low memory backup of results and parameters used
    :param file_path_to_save_results: path to save summary output too
    :param png_name: the name of the png for each experiment
    :param validation_split: the percent validation split
    :param batch_size: the number for batch size used
    :param neurons: the number neurons
    :param stateful: if model is stateful or not
    :param each_epoch: if the model needed to be trainined one epoch at a time
    :param optimizer: the optmizer used for the experiment
    :param test_metrics: the metrics using testing
    :return:
    '''
    summary = '\t\t\t{}\n' \
              '\t\tvalidation_split: {}\n' \
              '\t\tbatch_size: {}\n' \
              '\t\tneurons: {}\n' \
              '\t\tStateful: {}\n' \
              '\t\tTrain per each epoch {}\n' \
              '\t\tOptimizer Used: {}\n' \
              '\t\tTest Metrics: {}\n'.format(png_name, validation_split, batch_size,
                                              neurons, stateful, each_epoch,
                                              optimizer, test_metrics)
    filename = "Summary_" + png_name + ".txt"
    file_path = os.path.join(file_path_to_save_results, filename)
    file = open(file_path, 'a+')
    file.write(summary)
    file.close()
"""Obtain training and testing file list and also the data"""
training_file_list, testing_file_list = get_file_list(TRAINING_FILES, TESTING_FILES)
training_ds_dict = get_training_data(training_file_list)
testing_ds_dict = get_testing_data(testing_file_list)
testing_file_names = np.array(list(testing_ds_dict.keys()))



def fetch_train_data(training_file_name, train_file_name_index):
    '''
    Obtaining the data from the file and doing scaling and splitting into X and Y components for the model.
    :param training_file_name: name of file that will be grabbed used for training
    :param train_file_name_index: the number of the file based on outer loop to account for which file it is on
    :return: the array of data for X and Y for the model
    '''
    mocap_data_from_file = training_ds_dict[training_file_name] # gather data from a file
    mocap_data = mocap_data_from_file[marker_name] # using marker name from args get data from column
    # convert dataframe to numpy array to get all values if testing the code then reduce number of values
    if args.test is True:
        mocap_data = mocap_data.values[:250]
    elif args.test is False:
        mocap_data = mocap_data.values
    if NUM_FEATURES == 1: # to avoid error between scaler transform and array shape
        mocap_data = mocap_data.reshape(len(mocap_data), 1)

    """using partial fit to scale data with the mocap data used
    since this will be the first file the index should be 0"""
    if train_file_name_index == 0:
        scaler = MinMaxScaler(feature_range=(-1,1)) # create scaler between -1,1
        scaler.partial_fit(mocap_data)
    else: # new data will also be scaled as it comes from the outer loop in fit
        scaler_filename = 'scaler_{}_T{}.gz'.format((train_file_name_index - 1), TIMESTEPS)
        scaler_file_saved = os.path.join(file_path_to_save_results, scaler_filename)
        scaler = load(scaler_file_saved) # load the scaler to then be used by the next file
        scaler.partial_fit(mocap_data)
    mocap_data_scaled = scaler.transform(mocap_data) # scale the data
    # only using one marker data so using the univariate method to create data to timeseries
    trainX, trainY = univariate_data_to_sequence(mocap_data_scaled, 0, None, TIMESTEPS, 0)
    scaler_filename = 'scaler_{}_T{}.gz'.format(train_file_name_index, TIMESTEPS)
    scaler_file_path = os.path.join(file_path_to_save_results, scaler_filename)
    dump(scaler, scaler_file_path) # save the new scaler that was created when data is fit to it
    return trainX, trainY
def fetch_test_data(test_file_name_index):
    '''
     Obtaining the data from the file and doing scaling and splitting into X and Y components to test the model.
    :param test_file_name_index: the index in the dictonary containing the testing file data
    :return: the raw mocap data, the X and Y componeents fro the model and the scaler to inverse tranform afterwards
    '''
    global saved_scaler
    testing_file_name = testing_file_names[test_file_name_index]
    mocap_data_from_file = testing_ds_dict[testing_file_name]
    mocap_data = mocap_data_from_file[marker_name]
    # convert dataframe to numpy array to get all values if testing the code then reduce number of values
    if args.test is True:
        mocap_data = mocap_data.values[:250]
    elif args.test is False:
        mocap_data = mocap_data.values
    if NUM_FEATURES == 1:
        mocap_data = mocap_data.reshape(len(mocap_data), 1)

    if test_file_name_index == 0:
        scaler_filename = 'scaler_{}_T{}.gz'.format((len(training_ds_dict) - 1), TIMESTEPS)
        saved_scaler = os.path.join(file_path_to_save_results, scaler_filename)

    scaler = load(saved_scaler) # load the scaler that was used in training to transform new values
    mocap_data_scaled = scaler.transform(mocap_data)
    # only using one marker data so using the univariate method to create data to timeseries
    testX, testY = univariate_data_to_sequence(mocap_data_scaled, 0, None, TIMESTEPS, 0)
    testY = scaler.inverse_transform(testY) # data predicted needs to be inverted back to original values
    return mocap_data, testX, testY, scaler



rows_keys = ['Name', 'Marker', 'loss function', 'optimizer', 'validation split', 'hidden_units',
             'timesteps','epochs', 'batch size',
             'test mse', 'test rmse', 'test mae']
LSTM_dict = {}

for row in rows_keys:
    LSTM_dict[row] = list()
def run_model(experiment_counter=0):
    # hyperparameters that will be used with model
    loss_function = 'mean_squared_error'
    opt = 'sgd'
    num_epochs = 10
    num_neurons = 20
    batch_size = BATCH_SIZE
    v_split = .15

    loss_lst = list()
    val_loss = list()
    rmse_lst = list()
    val_rmse_lst = list()
    mae_lst = list()
    val_mae_lst = list()
    mse_lst = list()
    val_mse_lst = list()

    # name that will be used for storing file names corresponding to the experiment ran
    png_name = f'LSTM_Model_T{TIMESTEPS}_B{BATCH_SIZE}_Experiment_{experiment_counter}_{marker_name}'
    # defining the model that will be used
    lstm_model = define_lstm(timesteps=TIMESTEPS, num_features=NUM_FEATURES, num_units=num_neurons,
                             stateful=STATEFUL)

    # flag for plotting
    repeat_each_epoch = True
    # training using model created
    fit_stateless_lstm(lstm_model, batch_size=batch_size, num_epoch=num_epochs, v_split=v_split, valid_train=True,
                       train_loss_lst=loss_lst, train_mse_lst=mse_lst, train_rmse_lst=rmse_lst, train_mae_lst=mae_lst,
                       val_loss_lst=val_loss, val_mse_lst=val_mse_lst,
                       val_rmse_lst=val_rmse_lst, val_mae_lst=val_mae_lst)
    # save the output of the model and its parameters
    summary_model_path = os.path.join(file_path_to_save_results, f'{png_name}_report.txt')
    # save model summary
    with open(summary_model_path, 'w+') as fh:
        # Pass the file handle in as a lambda function to make it callable
        lstm_model.summary(print_fn=lambda x: fh.write(x + '\n'))
    # save the training modele at every level, same path as summary of model
    model_save_path = os.path.join(file_path_to_save_results, f'{png_name}_saved_model')
    lstm_model.save(model_save_path)
    # plotting the metrics of the model
    plot_metrics(file_path_to_save_results, png_name, repeat_each_epoch=True, loss_lst=loss_lst, mse_lst=mse_lst,
                 rmse_lst=rmse_lst, mae_lst=mae_lst, val_loss=val_loss, val_mse=val_mse_lst, val_rmse=val_rmse_lst, val_mae=val_mae_lst,
                 valid_train=True)

    for file_index in range(len(testing_ds_dict)):
        filename_id = testing_file_names[file_index].split("_")[1]
        png_name = f'LSTM_Model_T{TIMESTEPS}_B{batch_size}_Experiment_{experiment_counter}_File_{filename_id}_{marker_name}'
        mocap_data, testX_scaled, testY, scaler= fetch_test_data(file_index)

        y_pred = lstm_model.predict(testX_scaled, batch_size=1)
        test_predictions = scaler.inverse_transform(y_pred)
        # report performance using root mean square error, MSE and MAE
        test_mse, test_rmse, test_mae = calculate_metrics(testY[:, 0], test_predictions[:, 0])
        test_mse_str = f'\nTest MSE : {test_mse.numpy()} \n'
        test_rmse_str = f'Test RMSE: {test_rmse}  \n'
        test_mae_str = f'Test MAE : {test_mae.numpy()} \n'

        test_metrics = test_mse_str, test_rmse_str, test_mae_str
        create_summary_of_experiment(file_path_to_save_results, png_name, v_split, batch_size,
                                     num_neurons, STATEFUL, repeat_each_epoch, opt, test_metrics)
        prediction_plot = setup_data_to_plot(mocap_data, test_predictions, look_back=TIMESTEPS)
        plot_mocap_predictions(file_path_to_save_results, png_name, mocap_data, prediction_plot, increments=INCREMENTS)

        out = f"Output{experiment_counter}, File:{file_path_to_save_results}\n"
        file_out_path= os.path.join(marker_exp_directory, f"Output_{marker_name}.txt")
        file_out = open(file_out_path, "+a")
        file_out.write(out)
        file_out.close()

        LSTM_dict['Name'].append(png_name)
        LSTM_dict['Marker'].append(marker_name)
        LSTM_dict['loss function'].append(loss_function)
        LSTM_dict['optimizer'].append(opt)
        LSTM_dict['validation split'].append(v_split * 100)
        LSTM_dict['hidden_units'].append(num_neurons)
        LSTM_dict['timesteps'].append(TIMESTEPS)
        LSTM_dict['epochs'].append(num_epochs)
        LSTM_dict['batch size'].append(batch_size)
        LSTM_dict['test mse'].append(test_mse.numpy())
        LSTM_dict['test rmse'].append(test_rmse)
        LSTM_dict['test mae'].append(test_mae.numpy())

        # new dataframes for the columns in each sheet
        LSTM_df = pd.DataFrame(LSTM_dict)
        # new dictionary for all the sheets int he excel file
        excel_sheet_name = f'LSTM_{filename_id}'
        excel_sheet_3 = {excel_sheet_name: LSTM_df}
        LSTM_dict.clear()
        # name of excel file with summary
        excel_name = f"LSTM_T{TIMESTEPS}_B{batch_size}_Experiments_{marker_name}_Results.xlsx"

        excel_name = os.path.join(file_path_to_save_results, excel_name)

        save_to_excel(excel_name, stateful=STATEFUL, excel_sheet=excel_sheet_3, excel_sheet_name=excel_sheet_name)
        for row in rows_keys:
            LSTM_dict[row] = list()

    tf_session = tf.compat.v1.keras.backend.get_session()
    tf.compat.v1.reset_default_graph()
    tf.compat.v1.keras.backend.clear_session()
    K.clear_session()
    tf_session.close()

    plt.close("all")

    del lstm_model

    set_random_seed(seed)


if __name__ == "__main__":
    experiment_counter = 0  # 3 * 20 * 2 * 2 * 7 = 1680 files so thats the last counter value
    for i in range(START, 10):
        run_model(experiment_counter=i)
