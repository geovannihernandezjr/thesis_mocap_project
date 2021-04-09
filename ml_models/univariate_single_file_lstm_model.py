"""
@author Geovanni Hernandez
hyperparameter searching for LSTM model to find for each marker what type parameters might work best.
To predict motion based on only one marker at a timestep of 1.
Univariate timeseries analyis of the marker data to investigate the use of only one marker and one subject file to predict
the mocap data.
"""
import os
import faulthandler
seed = 1
os.environ['PYTHONHASHSEED '] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # to ignore warnings from tensorflow
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
from keras import backend as K
from data_preprocessing.get_list_of_files import get_training_data
from data_preprocessing.preprocessing import univariate_data_to_sequence
from ml_models.models import define_lstm
from ml_models.save_info import save_to_excel
from ml_models.metrics_for_models import calculate_metrics
from plotting.metric_plot import plot_metrics
from plotting.plot_data import plot_single_mocap_file_predictions
from sklearn.preprocessing import MinMaxScaler
from ml_models.set_random_seed import set_random_seed
from argparse import ArgumentParser

"""Command Line Arguments For Experiments"""
parser = ArgumentParser(description="Begin experiments based on start and end values from cmd")
parser.add_argument("start", type=int, help="n integer for starter value of experiments")
parser.add_argument("end", type=int, help="an integer for ending value of experiments")
parser.add_argument("marker", type=str, help="string for marker name for experiments")
parser.add_argument("-v", "--verbose", action="store_true", help="output verbosity")
parser.add_argument("--t", "--test", action='store_true', dest='test', help="use 250 values of data to run program")

"""set up hyperparameters"""
train_split_per = [.60, .70, .80]
v_split = .15
batch_size = 1
neurons_lst = [ 2,  8,  4, 18, 16, 11, 20, 17,  1,  9, 13, 10,  7, 12, 15, 14, 19,
        3,  5,  6]
loss_function = 'mean_squared_error'
optimizers = ['adam', 'sgd']
num_epochs = 10

def model_configs():
    # create model configuration list
    configs = list()
    for opt in optimizers:
        for split_percent in train_split_per:
            for neurons in neurons_lst:
                cfg = [opt, split_percent, neurons]
                configs.append(cfg)
    print(f'Total Configs: {len(configs)}')
    return configs


"""LEAP"""
# TRAINING_FILES = '/gpfs/home/g_h62/thesis_project/conference_paper/Train/*.tsv'
TRAINING_PARENT_DIR = '/gpfs/home/g_h62/thesis_project/conference_paper/Train/'
# TESTING_FILES = '/gpfs/home/g_h62/thesis_project/conference_paper/Test/*.tsv'
# TESTING_PARENT_DIR = '/gpfs/home/g_h62/thesis_project/conference_paper/Test/'
mocap_file = "Lifting_1002_51_9_20_10-10-2019_100%Filled.tsv"

mocap_file_path = os.path.join(TRAINING_PARENT_DIR, mocap_file)
"""Use Command Line Arguments to fill START & END"""
args = parser.parse_args()
START = args.start
END = args.end

TIMESTEPS = 1
NUM_FEATURES = 1
STATEFUL = False
# parent_directory = '/home/g_h62/conference_paper/experiments/'
# parent_directory = 'D:/HIPE/Clean_qtm/Fill/experiments/'
parent_directory = '/gpfs/home/g_h62/thesis_project/conference_paper/experiments'
# Directory for experiment set
experiment_set = 'LSTM-1L-MSE-HYPERExperiments'
training_exp_directory = os.path.join(parent_directory, experiment_set)
if not os.path.exists(training_exp_directory):
    os.mkdir(training_exp_directory)

marker_name = args.marker #"RElbowOut X"
marker_exp_directory = os.path.join(training_exp_directory, marker_name)
if not os.path.exists(marker_exp_directory):
    os.mkdir(marker_exp_directory)


new_dir = f'LSTM_Stateless_Model_T{TIMESTEPS}'
file_path_model_type = os.path.join(marker_exp_directory, new_dir)
if os.path.exists(file_path_model_type):
    pass
else:
    os.mkdir(file_path_model_type)

#
# training_file_list, testing_file_list = get_file_list(TRAINING_FILES, TESTING_FILES)
# training_ds_dict = get_training_data(training_file_list)
# testing_ds_dict = get_testing_data(testing_file_list)
# testing_file_names = np.array(list(testing_ds_dict.keys()))

single_training_ds_dict = get_training_data([mocap_file_path])
file_name = list(single_training_ds_dict.keys())[0]
subject_id = file_name.split("_")[1]
# file_path_to_save_results = os.path.join(file_path_model_type, subject_id)
file_path_subject_id = os.path.join(file_path_model_type, subject_id)

if not os.path.exists(file_path_subject_id):
    os.mkdir(file_path_subject_id)


def fit_lstm_stateless(model, trainX, trainY, batch_size, num_epoch=10, valid_data=None, v_split=None, valid_train=True,
                       train_loss_lst=None, train_mse_lst=None, train_rmse_lst=None, train_mae_lst=None,
                       val_loss_lst=None, val_mse_lst=None,
                       val_rmse_lst=None, val_mae_lst=None):
    """
    This is the implementation to train the model defined.
    :param model: the defined model created
    :param trainX: the array containing the dataset to be used for X position in model
    :param trainY: the array contaiing the dataset of labels or Y position in model
    :param batch_size: the size of batch that will be used when training each sequence
    :param num_epoch: the number of iterations for training the model
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
    """The LSTM network expects the input data (X) to be provided 
                with a specific array structure in the form of: [samples, time steps, features].
                Currently, our data is in the form: [samples, features] and we are framing the 
                problem as one time step for each sample. We can transform the prepared train and test input data into the 
                expected structure using numpy.reshape() as follows:"""
    if True is valid_train:
        model.fit(trainX, trainY, epochs=num_epoch, batch_size=batch_size, validation_split=v_split, verbose=0,
                  shuffle=False)
    else:
        model.fit(trainX, trainY, epochs=num_epoch, batch_size=batch_size, verbose=0,
                  shuffle=False)
    return model


def one_step_forcast(model, X, batch_size):
    # X = X.reshape(1, 1, len(X))

    n_features = 1
    X = X.reshape((1, TIMESTEPS, NUM_FEATURES))
    yhat = model.predict(X, batch_size=batch_size)
    return yhat[0, 0]


def create_summary_of_experiment(file_path_to_save_results, png_name, train_split_per, validation_split, batch_size,
                                 neurons, stateful,
                                 each_epoch, optimizer, train_metrics, test_metrics):
    '''
    This is just to ensure and have some form of low memory backup of results and parameters used
    :param file_path_to_save_results: path to save summary output too
    :param png_name: the name of the png for each experiment
    :param train_split_per: the percent split between trianing and testing
    :param validation_split: the percent validation split
    :param batch_size: the number for batch size used
    :param neurons: the number neurons
    :param stateful: if it is stateful or not
    :param each_epoch: if the model needed to be trainined one epoch at a time
    :param optimizer: the optmizer used for the experiment
    :param train_metrics: the metrics using training split dataset
    :param test_metrics: the metrics using testing split dataset
    :return: save the file to txt
    '''
    summary = f'\t\t\t{png_name}\n' \
              f'\t\ttrain_split_percent: {train_split_per}\n' \
              f'\t\tvalidation_split: {validation_split}\n' \
              f'\t\tbatch_size: {batch_size}\n' \
              f'\t\tneurons: {neurons}\n' \
              f'\t\tStateful: {stateful}\n' \
              f'\t\tTrain per each epoch {each_epoch}\n' \
              f'\t\tOptimizer Used: {optimizer}\n' \
              f'\t\tTrain Metrics: {train_metrics}\n' \
              f'\t\tTest Metrics: {test_metrics}\n'
    filename = "Summary_" + png_name + ".txt"
    file_path = os.path.join(file_path_to_save_results, filename)
    file = open(file_path, 'a+')
    file.write(summary)
    file.close()



rows_keys = ['Name', 'Marker', 'loss function', 'optimizer', 'traintest split percent', 'validation split',
             'hidden_units',
             'timesteps', 'epochs', 'batch size', 'train mse', 'train rmse', 'train mae',
             'test mse', 'test rmse', 'test mae']
lstm_stateless_dict = {}

# create dicontary with empty list. This will make it possible to then create sheets for excel file to save
# results from each run
for row in rows_keys:
    lstm_stateless_dict[row] = list()


data_from_file = single_training_ds_dict[file_name]
mocap_data_from_file = data_from_file.iloc[:, :-1]
print(mocap_data_from_file.iloc[:,-1])
# borg_data_from_file = data_from_file.iloc[:, -1]
mocap_data = mocap_data_from_file[marker_name]

if args.test is True:
    mocap_data = mocap_data.values[:250]
    INCREMENTS = 10

else:
    mocap_data = mocap_data.values
    INCREMENTS = 1000

if NUM_FEATURES == 1:
    mocap_data = mocap_data.reshape(len(mocap_data), 1)

def run_model(experiment_counter=START):
    # loss_function = 'mean_squared_error'
    # opt = 'sgd'
    # num_epochs = 10
    # num_neurons = 14
    # batch_size = 1
    # v_split = .15
    global lstm_stateless_dict
    configs = model_configs()
    # start of loop for parameters
    for config in configs[START:END]:
        opt, split_percent, num_neurons = config

        file_path_to_save_results = os.path.join(file_path_subject_id, opt)
        if not os.path.exists(file_path_to_save_results):
            os.mkdir(file_path_to_save_results)

        experiment_file_name = f'{subject_id}_GRU_Stateless_Model_T{TIMESTEPS}_Experiment_{experiment_counter}_{marker_name}'
        png_name = experiment_file_name + "_" + loss_function + "_" + str(opt) + "_" + str(
            split_percent * 100) + "_" + str(num_neurons)
        # loss_lst = list()
        # val_loss = list()
        # rmse_lst = list()
        # val_rmse_lst = list()
        # mae_lst = list()
        # val_mae_lst = list()
        # mse_lst = list()
        # val_mse_lst = list()

        # beginning of changing the split percent of the mocap data
        train_size = int(len(mocap_data) * split_percent)
        train, test = mocap_data[0:train_size], mocap_data[train_size:]

        # scaling the data
        scaler = MinMaxScaler(feature_range=(-1,1))
        scaler = scaler.fit(train)
        train_scaled = scaler.transform(train)
        test_scaled = scaler.transform(test)

        # Turning the X(t) values into Y(t) using timestep previous values
        trainX, trainY = univariate_data_to_sequence(train_scaled,0, None, TIMESTEPS, 0)
        testX, testY = univariate_data_to_sequence(test_scaled, 0, None,  TIMESTEPS, 0)
        testY = scaler.inverse_transform(testY)
        # use the LSTM model and define it which would be different for each depending on number of neurons
        print("define LSTM")
        lstm_model = define_lstm(timesteps=TIMESTEPS, num_features=NUM_FEATURES, num_units=num_neurons,
                                 batch_size=batch_size,
                                 stateful=STATEFUL, opt=opt)
        repeat_each_epoch = False
        # fit the training data X to Y for 10 epochs
        fit_lstm_stateless(lstm_model, trainX=trainX, trainY=trainY, batch_size=batch_size, num_epoch=num_epochs,
                           v_split=v_split,
                           valid_train=True)
        # save the output of the model and its parameters
        summary_model_path = os.path.join(file_path_to_save_results, f'{png_name}_report.txt')
        # save model summary
        with open(summary_model_path, 'w+') as fh:
            # Pass the file handle in as a lambda function to make it callable
            lstm_model.summary(print_fn=lambda x: fh.write(x + '\n'))
        # plot the mse, rmse, mae metric values for training and validation
        plot_metrics(file_path_to_save_results, png_name, repeat_each_epoch=False,
                     model_metric_history=lstm_model.history.history,
                     valid_train=True)

        # predict the train values to build state of model
        train_predict = lstm_model.predict(trainX, batch_size=1)

        # use the prevous scaler to transform the values back to mocap values
        train_predict_unscaled = scaler.inverse_transform(train_predict)
        trainY = scaler.inverse_transform(trainY)  # inverse scale the values back to use for calculating the metrics
        # calcaulte the metrics from training predictions between trainY values and predictions
        if type(train_predict_unscaled) is list:
            train_mse, train_rmse, train_mae = calculate_metrics(trainY[:, 0], train_predict_unscaled)
        else:
            train_mse, train_rmse, train_mae = calculate_metrics(trainY[:, 0], train_predict_unscaled[:, 0])

        # create the string of each metric
        train_mse_str = f'\nTrain MSE : {train_mse.numpy()} \n'
        train_rmse_str = f'Train RMSE: {train_rmse}  \n'
        train_mae_str = f'Train MAE : {train_mae.numpy()} \n'
        train_metrics = f'{train_mse_str}, {train_rmse_str}, {train_mae_str}'

        # begin the one step forecast of each test value
        test_predictions = list()
        for i in range(len(testX)):
            # make one-step forecast
            # testX, testY = test_scaled[i, 0:-1], test_scaled[i, -1]
            y_pred = one_step_forcast(lstm_model, testX[i], 1)
            # invert scaling
            y_pred = scaler.inverse_transform(y_pred.reshape(1, -1))
            # store forecast
            test_predictions.append(y_pred[0, -1])

        # calculate the metrics between real test values and predicted test values
        if type(test_predictions) is list:
            test_mse, test_rmse, test_mae = calculate_metrics(testY[:, 0], test_predictions)
        else:
            test_mse, test_rmse, test_mae = calculate_metrics(testY[:, 0], test_predictions[:, 0])
        test_mse_str = f'\nTest MSE : {test_mse.numpy()} \n'
        test_rmse_str = f'Test RMSE: {test_rmse}  \n'
        test_mae_str = f'Test MAE : {test_mae.numpy()} \n'
        test_metrics = f'{test_mse_str}, {test_rmse_str}, {test_mae_str}'

        # create txt file for backup of values and log
        create_summary_of_experiment(file_path_to_save_results, png_name, split_percent, v_split, batch_size,
                                     num_neurons, STATEFUL,
                                     repeat_each_epoch, opt, train_metrics, test_metrics)

        # plot the predictionsn for train and test
        plot_single_mocap_file_predictions(file_path=file_path_to_save_results, png_name=png_name,
                                           raw_dataset=mocap_data,
                                           train_predict=train_predict_unscaled, test_predict=test_predictions,
                                           look_back=TIMESTEPS, increments=INCREMENTS)
        out = f"Output{experiment_counter}, File:{file_path_to_save_results}\n"  # .format(experiment_counter, file_path_to_save_results)
        file_out_path = os.path.join(file_path_subject_id, f"Output_StartAt{START}.txt")
        file_out = open(file_out_path, "+a")

        # file_out = open("Output" + ".txt", "+a")
        file_out.write(out)
        file_out.close()

        """lstm_stateless_dict rows_keys: ['Name', 'Marker', 'loss function', 'optimizer', 'traintest split percent', 'validation split', 'hidden_units',
                 'timesteps','epochs', 'batch size', 'train mse', 'train rmse', 'train mae'
                 'test mse', 'test rmse', 'test mae']"""
        lstm_stateless_dict['Name'].append(png_name)
        lstm_stateless_dict['Marker'].append(marker_name)
        lstm_stateless_dict['loss function'].append(loss_function)
        lstm_stateless_dict['optimizer'].append(opt)
        lstm_stateless_dict['traintest split percent'].append(split_percent)
        lstm_stateless_dict['validation split'].append(v_split)
        lstm_stateless_dict['hidden_units'].append(num_neurons)
        lstm_stateless_dict['timesteps'].append(TIMESTEPS)
        lstm_stateless_dict['epochs'].append(num_epochs)
        lstm_stateless_dict['batch size'].append(batch_size)
        lstm_stateless_dict['train mse'].append(train_mse.numpy())
        lstm_stateless_dict['train rmse'].append(train_rmse)
        lstm_stateless_dict['train mae'].append(train_mae.numpy())
        lstm_stateless_dict['test mse'].append(test_mse.numpy())
        lstm_stateless_dict['test rmse'].append(test_rmse)
        lstm_stateless_dict['test mae'].append(test_mae.numpy())

        """Save the results to exxcel file begins"""
        # new dataframes for the columns in each sheet
        Excel_df = pd.DataFrame(lstm_stateless_dict)
        # new dictionary for all the sheets int he excel file
        excel_sheet_name = f'LSTM_Stateless_{subject_id}'
        excel_sheet_3 = {excel_sheet_name: Excel_df}
        lstm_stateless_dict.clear()
        # name of excel file with summary
        excel_name = f"{subject_id}_StartAt_{START}_LSTM_Stateless_T{TIMESTEPS}_Experiments_{marker_name}_Results.xlsx"

        excel_name = os.path.join(file_path_subject_id, excel_name)

        save_to_excel(excel_name, stateful=STATEFUL, excel_sheet=excel_sheet_3, excel_sheet_name=excel_sheet_name)

        for row in rows_keys:
            lstm_stateless_dict[row] = list()
        """Save the results to exxcel file ends"""

        # reset_keras()
        tf_session = tf.compat.v1.keras.backend.get_session()
        tf.compat.v1.reset_default_graph()
        tf.compat.v1.keras.backend.clear_session()
        K.clear_session()
        tf_session.close()

        plt.close("all")

        del lstm_model
        del scaler
        del train, trainY, trainX, train_predict_unscaled, train_predict, train_mse, train_rmse, train_mse_str, train_rmse_str, train_mae_str, train_metrics
        del test, testY, testX, test_scaled, test_predictions, test_mse, test_rmse, test_mae, test_mse_str, test_rmse_str, test_mae_str, test_metrics
        del experiment_file_name, png_name
        set_random_seed(seed)
        experiment_counter += 1


if __name__ == "__main__":
    if args.verbose:
        print(f'start {args.start}\nend {args.end}')

    faulthandler.enable()
    experiment_counter = START  # counter number of which experiment is being conducted
    run_model(experiment_counter)

    fault_file = os.path.join(file_path_subject_id, f'StartAt_{START}_{END}_faults.log')
    fault_file_out = open(fault_file, mode="+a")
    faulthandler.enable(fault_file_out)
    fault_file_out.close()
