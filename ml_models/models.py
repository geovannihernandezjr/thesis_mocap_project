"""
File containing the definition of each model GRU and LSTM
"""""
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from tensorflow.keras.metrics import RootMeanSquaredError
from keras.layers import GRU

def define_lstm(timesteps, num_features=1, num_units=20, batch_size=1, stateful=False, opt='sgd', loss_fun = 'mean_squared_error', num_layers = 1):
    """
    
    :param timesteps: refer to the ticks/sequence in given time that the NN will have memory on. It is how long in time of each of the samples is. i.e. a sample can contain 100-time steps where each time-step could be 1 second for a capture of frames. This could be associated with a word, time frame of signal or a sentence.
    :param num_features:  this is the number of dimensions feed at each time-step. Can be the number of variables in time when predicting/forecasting weather, stocks, or next word. , univariate vs multivariate. i.e. in case of 3D signal, X, Y, and Z signal will measure on each frame or timestep, meaning there will be 3 features.
    :param num_units: refer to the number of units, dimensionality of outer space, positive int
    :param batch_size: may refer to individual training examples. Could be used as 'batch_size' variable hence the count of the samples inputted to the a neural network. Can be seen as how many different examples are feed at once to the NN.
    :param stateful: boolean if 'stateful', the last state for each sample at index i in a batch will be used as initial state for the sample of index i in the following batch
    :param opt: String name of optimizer
    :param loss_fun: String name of objective function for compiling model
    :param num_layers: number of layers to use with the lstm
    :return: a group of layers into an object with training
    """
    model = Sequential()
    return_seq = True
    if stateful is False:
        if num_layers is 1:
            return_seq = False
        model.add(LSTM(num_units, input_shape=(timesteps, num_features), stateful=stateful, return_sequences=return_seq))
        if num_layers > 1:
            if num_layers >= 3:
                model.add(LSTM(32,return_sequences=True))
                if num_layers is 5:
                    model.add(LSTM(32, return_sequences=True))
                    model.add(LSTM(32, return_sequences=True))
            model.add(LSTM(32))

    else:
        model.add(LSTM(num_units, batch_input_shape=(batch_size, timesteps, num_features), stateful=stateful))
    model.add(Dense(1))
    # model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mae', 'mse', RootMeanSquaredError])
    model.compile(optimizer=opt, loss=loss_fun, metrics=['mse', RootMeanSquaredError(name='rmse'), 'mae'])

    return model

def define_gru(timesteps, num_features=1, num_units=None, batch_size=1, stateful=False, opt='sgd', loss_fun = 'mean_squared_error'):
    '''

    :param timesteps: refer to the ticks/sequence in given time that the NN will have memory on. It is how long in time of each of the samples is. i.e. a sample can contain 100-time steps where each time-step could be 1 second for a capture of frames. This could be associated with a word, time frame of signal or a sentence.
    :param num_features:  this is the number of dimensions feed at each time-step. Can be the number of variables in time when predicting/forecasting weather, stocks, or next word. , univariate vs multivariate. i.e. in case of 3D signal, X, Y, and Z signal will measure on each frame or timestep, meaning there will be 3 features.
    :param num_units: refer to the number of units, dimensionality of outer space, positive int
    :param batch_size: may refer to individual training examples. Could be used as 'batch_size' variable hence the count of the samples inputted to the a neural network. Can be seen as how many different examples are feed at once to the NN.
    :param stateful: boolean if 'stateful', the last state for each sample at index i in a batch will be used as initial state for the sample of index i in the following batch
    :param opt: String name of optimizer
    :param loss_fun: String name of objective function for compiling model
    :return: a group of layers into an object with training
    '''
    model = Sequential()
    if stateful is False:
        model.add(GRU(units=num_units, input_shape=(timesteps, num_features), stateful=stateful))
    else:
        model.add(GRU(units=num_units, batch_input_shape=(batch_size, timesteps, num_features), stateful=stateful))
    model.add(Dense(1))
    model.compile(optimizer=opt, loss=loss_fun, metrics=['mse', RootMeanSquaredError(name='rmse'), 'mae'])

    return model
