"""
@author:Geovanni Hernandez
File that uses multivariate approach that uses all markers of interest: Elbow, Thigh, Shin, Back, ShoulderTop, ShoulderBack, Elbow each having L/R side.
Since each frame is captured at rate of 10msec per frame, there will be 42 components for each frame because each marker has an XYZ component.
This also includes the time factor at every frame. Using the merged borg and mocap file for this implmenetation the goal is to investigate if the
mocap data in realtion to time can be useful to predict the borg scale value of indiviudals.
This implementation uses the LSTM Deep learning model.
"""
import os
from argparse import ArgumentParser

"""Command Line Arguments For Experiments"""
parser = ArgumentParser(description="Begin experiments based on start and end values from cmd")
parser.add_argument("start", type=int, help="n integer for starter value of experiments")
parser.add_argument("end", type=int, help="an integer for ending value of experiments")
parser.add_argument("marker", type=str, help="string for marker name for experiments")
parser.add_argument('timestep', type=int, help="an integer for value of timestep to use in model")
parser.add_argument('num_layers', type=int, help='an integer for value of how many layers for lstm model (max5)')
parser.add_argument("batch", type=int, help='an integer for batch size value of training of experiments')
parser.add_argument("-v", "--verbose", action="store_true", help="output verbosity")
parser.add_argument("--t", "--test", action='store_true', dest='test', help="use 250 values of data to run program")
"""Use Command Line Arguments to choice START & END"""
args = parser.parse_args()
START = args.start
END = args.end
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
from data_preprocessing.get_list_of_files import findFilesInFolder, get_training_data, get_testing_data
from data_preprocessing.preprocessing import multivariate_data_to_sequence
from ml_models.models import define_lstm
from ml_models.save_info import save_to_excel
from ml_models.metrics_for_models import calculate_metrics
from plotting.metric_plot import plot_metrics
from plotting.plot_data import setup_data_to_plot, plot_borg_predictions
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
NUM_FEATURES = (14*3) + 1
BATCH_SIZE = args.batch
NUM_LAYERS = args.num_layers
STATEFUL = False
if args.test is True:
    INCREMENTS = 10
elif args.test is False:
    INCREMENTS = 1000
else:
    INCREMENTS = 0
"""Directory that will be created to save experiment results in"""
#parent_directory = 'C:/Users/geova/Desktop/Filled_Files/experiments/'
parent_directory = '/gpfs/home/g_h62/thesis_project/borg_predictions/experiments'
# parent_directory = '/home/g_h62/conference_paper/experiments/'
# parent_directory = 'D:/HIPE/Clean_qtm/Fill/experiments/'
# parent_directory = '/gpfs/home/g_h62/thesis_project/conference_paper/experiments'

""" Directory for experiment set """
experiment_set = f'LSTM-{NUM_LAYERS}L-Batch{BATCH_SIZE}_BorgPrediction_Experiments'
training_exp_directory = os.path.join(parent_directory, experiment_set)
if not os.path.exists(training_exp_directory):
    os.mkdir(training_exp_directory)

"""Directory for the type of markers being used in the experiment"""
marker_name = args.marker #"RElbowOut X" "All"
marker_exp_directory = os.path.join(training_exp_directory, marker_name)
if not os.path.exists(marker_exp_directory):
    os.mkdir(marker_exp_directory)

new_dir = f'LSTM_Model_T{TIMESTEPS}_B{BATCH_SIZE}_{NUM_LAYERS}L'
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
        np.random.shuffle(training_file_names) # suffle order of names in list for randomization of training per file
        for file_index, file_in_ds in enumerate(training_file_names): # iterate through each file
            """The LSTM network expects the input data (X) to be provided 
            with a specific array structure in the form of: [samples, time steps, features].
            Currently, our data is in the form: [samples, features] and we are framing the 
            problem as one time step for each sample. We can transform the prepared train and test input data into the 
            expected structure using numpy.reshape() as follows:"""
            trainX, trainY = fetch_train_data(file_in_ds, file_index)
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
print('Getting Training Files')
training_path_list, training_filename_list = findFilesInFolder(TRAINING_PARENT_DIR, extension='.csv')
training_ds_dict = get_training_data(training_path_list)
training_file_names = np.array(list(training_ds_dict.keys()))
print('Getting Testing Files')
testing_path_list, testing_filename_list = findFilesInFolder(TESTING_PARENT_DIR, extension='.csv')
testing_ds_dict = get_testing_data(testing_path_list)
testing_file_names = np.array(list(testing_ds_dict.keys()))



def fetch_train_data(training_file_name, train_file_name_index):
    """"""
    global train_data_from_file_np
    train_data_from_file = training_ds_dict[training_file_name]
    if args.test is True:
        train_data_from_file_np = train_data_from_file.values[:250]
    elif args.test is False:
        train_data_from_file_np = train_data_from_file.values

    time_per_frame = train_data_from_file_np[:, 0]
    """Training mocap data, and borg data"""
    train_mocap_data = train_data_from_file_np[:, 1:-1]
    train_borg_data = train_data_from_file_np[:, -1]
    try: # make sure borg data shape is a 1D array
        train_borg_data.shape[1]
    except IndexError:
        train_borg_data = train_borg_data.reshape(-1, 1)
    if NUM_FEATURES == 1: # to avoid error between scaler transform and array shape
        train_mocap_data = train_mocap_data.reshape(len(train_mocap_data), 1)
    """using partial fit to scale data with the mocap data used
        since this will be the first file the index should be 0"""
    if train_file_name_index == 0:
        scaler = MinMaxScaler(feature_range=(-1,1)) # create scaler between -1,1
        scaler.partial_fit(train_mocap_data)
    else: # new data will also be scaled as it comes from the outer loop in fit
        scaler_filename = 'scaler_{}_T{}.gz'.format((train_file_name_index - 1), TIMESTEPS)
        scaler_file_saved = os.path.join(file_path_to_save_results, scaler_filename)
        scaler = load(scaler_file_saved) # load the scaler to then be used by the next file
        scaler.partial_fit(train_mocap_data) # scale the data
    train_mocap_scaled_data = scaler.transform(train_mocap_data)
    train_mocap_scaled_data_with_time = np.column_stack((time_per_frame, train_mocap_scaled_data))
    # using 43 columns of data so using the multivariate method to create data to timeseries, 1 column for time per frame and 42 for mocap data
    trainX, trainY = multivariate_data_to_sequence(train_mocap_scaled_data_with_time, train_borg_data, 0, None, TIMESTEPS, 0, 1, True)
    scaler_filename = 'scaler_{}_T{}.gz'.format(train_file_name_index, TIMESTEPS)
    scaler_file_path = os.path.join(file_path_to_save_results, scaler_filename)
    dump(scaler, scaler_file_path) # save the new scaler that was created when data is fit to it
    return trainX, trainY
def fetch_test_data(test_file_name_index, using_train_files = False):
    '''
         Obtaining the data from the file and doing scaling and splitting into X and Y components to test the model.
        :param test_file_name_index: the index in the dictonary containing the testing file data
        :param using_train_files: boolean so that this method can be used with predicting train files
        :return: the raw mocap data, the X and Y componeents fro the model and the scaler to inverse tranform afterwards
        '''
    global saved_scaler, test_data_from_file_np
    if not using_train_files:
        testing_file_name = testing_file_names[test_file_name_index]
        test_data_from_file = testing_ds_dict[testing_file_name]
    else:
        testing_file_name = training_file_names[test_file_name_index]
        test_data_from_file = training_ds_dict[testing_file_name]

    # mocap_data = mocap_data.values[:250]
    if args.test is True:
        test_data_from_file_np = test_data_from_file.values[:250]
    elif args.test is False:
        test_data_from_file_np = test_data_from_file.values
    time_per_frame = test_data_from_file_np[:, 0]
    """Testing mocap data, and borg data"""
    test_mocap_data = test_data_from_file_np[:, 1:-1]
    test_borg_data = test_data_from_file_np[:, -1]
    try: # make sure borg data shape is a 1D array
        test_borg_data.shape[1]
    except IndexError:
        test_borg_data = test_borg_data.reshape(-1, 1)
    if NUM_FEATURES == 1:
        test_mocap_data = test_mocap_data.reshape(len(test_mocap_data), 1)

    if test_file_name_index == 0:
        scaler_filename = 'scaler_{}_T{}.gz'.format((len(training_ds_dict) - 1), TIMESTEPS) # read scaler file
        saved_scaler = os.path.join(file_path_to_save_results, scaler_filename)

    scaler = load(saved_scaler) # load scaler file that was saved and created during training
    test_mocap_scaled_data = scaler.transform(test_mocap_data) # use scaler that was used for training to transform data
    test_mocap_scaled_data_with_time = np.column_stack((time_per_frame, test_mocap_scaled_data))
    # using 43 columns of data so using the multivariate method to create data to timeseries, 1 column for time per frame and 42 for mocap data
    testX, testY = multivariate_data_to_sequence(test_mocap_scaled_data_with_time, test_borg_data, 0, None, TIMESTEPS, 0, 1, True)
    # testY = scaler.inverse_transform(testY)
    return test_borg_data, testX, testY, scaler



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
    v_split = 0

    loss_lst = list()
    val_loss = list()
    rmse_lst = list()
    val_rmse_lst = list()
    mae_lst = list()
    val_mae_lst = list()
    mse_lst = list()
    val_mse_lst = list()

    # name that will be used for storing file names corresponding to the experiment ran
    png_name = f'LSTM_Model_T{TIMESTEPS}_B{BATCH_SIZE}_{NUM_LAYERS}L_Borg_Prediction_Experiment_{experiment_counter}_{marker_name}'
    # defining the model that will be used
    lstm_model = define_lstm(timesteps=TIMESTEPS, num_features=NUM_FEATURES, num_units=num_neurons,
                             stateful=STATEFUL,num_layers=NUM_LAYERS)

    # flag for plotting
    repeat_each_epoch = True
    # training using model created
    fit_stateless_lstm(lstm_model, batch_size=batch_size, num_epoch=num_epochs, valid_train=False,
                       train_loss_lst=loss_lst, train_mse_lst=mse_lst, train_rmse_lst=rmse_lst, train_mae_lst=mae_lst)
    # save the output of the model and its parameters
    summary_model_path = os.path.join(file_path_to_save_results, f'{png_name}_model_report.txt')
    # save model summary
    with open(summary_model_path, 'w+') as fh:
        # Pass the file handle in as a lambda function to make it callable
        lstm_model.summary(print_fn=lambda x: fh.write(x + '\n'))
    # save the training modele at every level, same path as summary of model
    model_save_path = os.path.join(file_path_to_save_results, f'{png_name}_saved_model')
    lstm_model.save(model_save_path)

    # plotting the metrics of the model
    plot_metrics(file_path_to_save_results, png_name, repeat_each_epoch=True, loss_lst=loss_lst, mse_lst=mse_lst,
                 rmse_lst=rmse_lst, mae_lst=mae_lst,
                 valid_train=False)
    # training file predictions
    for training_file_index in range(len(training_ds_dict)):
        training_filename_id = training_file_names[training_file_index].split("_")[1]
        train_png_name = f'LSTM_Model_T{TIMESTEPS}_B{BATCH_SIZE}_{NUM_LAYERS}L_Borg_Prediction_Experiment_{experiment_counter}_File_{training_filename_id}_{marker_name}'
        train_borg_data, trainX_scaled, trainY, scaler = fetch_test_data(training_file_index, True)
        try:
            trainY.shape[1]
            raise IndexError
        except IndexError:
            print("Need to Reshape 1D array")
            trainY = trainY.reshape(-1, 1)

        y_borg_pred = lstm_model.predict(trainX_scaled, batch_size=1)

        # report performance using root mean square error, MSE and MAE
        train_mse, train_rmse, train_mae = calculate_metrics(trainY[:, 0], y_borg_pred[:, 0])
        train_mse_str = f'\nTrain MSE : {train_mse.numpy()} \n'
        train_rmse_str = f'Train RMSE: {train_rmse}  \n'
        train_mae_str = f'Train MAE : {train_mae.numpy()} \n'

        train_metrics = train_mse_str, train_rmse_str, train_mae_str
        print(f'{training_filename_id} metrics {train_metrics}')
        # create txt file for backup of values and log
        create_summary_of_experiment(file_path_to_save_results, train_png_name, v_split, batch_size,
                                     num_neurons, STATEFUL, repeat_each_epoch, opt, train_metrics)
        # plot the predictions for train data
        train_borg_prediction_plot = setup_data_to_plot(train_borg_data, y_borg_pred, TIMESTEPS)

        plot_borg_predictions(file_path_to_save_results, train_png_name, train_borg_data, train_borg_prediction_plot, increments=INCREMENTS)
        LSTM_dict['Name'].append(train_png_name)
        LSTM_dict['Marker'].append(marker_name)
        LSTM_dict['loss function'].append(loss_function)
        LSTM_dict['optimizer'].append(opt)
        LSTM_dict['validation split'].append(v_split * 100)
        LSTM_dict['hidden_units'].append(num_neurons)
        LSTM_dict['timesteps'].append(TIMESTEPS)
        LSTM_dict['epochs'].append(num_epochs)
        LSTM_dict['batch size'].append(batch_size)
        LSTM_dict['test mse'].append(train_mse.numpy())
        LSTM_dict['test rmse'].append(train_rmse)
        LSTM_dict['test mae'].append(train_mae.numpy())

        # new dataframes for the columns in each sheet
        LSTM_df = pd.DataFrame(LSTM_dict)
        # new dictionary for all the sheets int he excel file
        train_excel_sheet_name = f'LSTM_{training_filename_id}'
        train_excel_sheet_3 = {train_excel_sheet_name: LSTM_df}
        LSTM_dict.clear()
        # name of excel file with summary
        train_pred_excel_name = f"LSTM_T{TIMESTEPS}_B{BATCH_SIZE}_{NUM_LAYERS}L_Train_Borg_Prediction_Experiments_{marker_name}_Results.xlsx"

        train_pred_excel_name_path = os.path.join(file_path_to_save_results, train_pred_excel_name)

        save_to_excel(train_pred_excel_name_path, stateful=STATEFUL, excel_sheet=train_excel_sheet_3, excel_sheet_name=train_excel_sheet_name)
        for row in rows_keys:
            LSTM_dict[row] = list()
    # testing files predictions
    for file_index in range(len(testing_ds_dict)):
        filename_id = testing_file_names[file_index].split("_")[1]
        # name that will be used for storing file names corresponding to the experiment ran
        test_png_name = f'LSTM_Model_T{TIMESTEPS}_B{BATCH_SIZE}_{NUM_LAYERS}L_Borg_Prediction_Experiment_{experiment_counter}_File_{filename_id}_{marker_name}'
        test_borg_data, testX_scaled, testY, scaler= fetch_test_data(file_index)
        try:
            testY.shape[1]
            raise IndexError
        except IndexError:
            print("Need to Reshape 1D array")
            testY = testY.reshape(-1,1)

        y_borg_pred = lstm_model.predict(testX_scaled, batch_size=1)

        # report performance using root mean square error, MSE and MAE
        test_mse, test_rmse, test_mae = calculate_metrics(testY[:, 0], y_borg_pred[:, 0])
        test_mse_str = f'\nTest MSE : {test_mse.numpy()} \n'
        test_rmse_str = f'Test RMSE: {test_rmse}  \n'
        test_mae_str = f'Test MAE : {test_mae.numpy()} \n'

        test_metrics = test_mse_str, test_rmse_str, test_mae_str

        # create txt file for backup of values and log
        create_summary_of_experiment(file_path_to_save_results, test_png_name, v_split, batch_size,
                                     num_neurons, STATEFUL, repeat_each_epoch, opt, test_metrics)
        # plot the predictions for test data
        test_borg_prediction_plot = setup_data_to_plot(test_borg_data, y_borg_pred, TIMESTEPS)

        plot_borg_predictions(file_path_to_save_results, test_png_name, test_borg_data, test_borg_prediction_plot, increments=INCREMENTS)

        out = f"Output{experiment_counter}, File:{file_path_to_save_results}\n"#.format(experiment_counter, file_path_to_save_results)
        file_out_path= os.path.join(file_path_to_save_results, f"Output_{marker_name}.txt")
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
        excel_name = f"LSTM_T{TIMESTEPS}_B{BATCH_SIZE}_{NUM_LAYERS}L_Test_Borg_Prediction_Experiments_{marker_name}_Results.xlsx"

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
    for i in range(START, END):
        run_model(experiment_counter=i)
