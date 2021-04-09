'''
@author Geovanni Hernandez
Created to obtain the datasets for training and testing using tensorflow
'''

import tensorflow as tf
import os
from numpy import float32
from data_preprocessing.preprocessing import read_mocap_data
from pandas import read_csv

def get_file_list(training_dir, testing_dir):
    '''

    :param training_dir: the directory containing the training files
    :param testing_dir: the directory containing the testing files
    :return: two lists, one list that contains the path for each training and one for testing
    '''
    training_list_ds = tf.data.Dataset.list_files(training_dir)
    testing_list_ds = tf.data.Dataset.list_files(testing_dir)
    return training_list_ds, testing_list_ds
def get_training_data(training_file_list):
    '''

    :param training_file_list: input the list containing the path to each file
    :return: dictonary containing dataframe of each dataset from training directory
    '''
    training_dataset_dict = {}
    for fileinlist in training_file_list:
        fileinlist_path = ""
        if tf.is_tensor(fileinlist):
            fileinlist_path = fileinlist.numpy().decode() # obtain string from tensorobject
        else:
            fileinlist_path = fileinlist

        if fileinlist_path.endswith(".tsv"):
            print(f'file is a TSV format, {fileinlist_path}')
            datset_content = read_mocap_data(fileinlist_path) # read data and modify it, then save it into variable
            file_name = tf.strings.split(fileinlist, os.sep)[-1].numpy().decode() # using tensorflow os separater to split string and obtain name of file
            training_dataset_dict[file_name] = datset_content
        elif fileinlist_path.endswith('.csv'):
            print(f'file is CSV format, {fileinlist_path}')
            file_name_csv = fileinlist_path.split(os.sep)[-1] # parse to get name of file with extension
            dataset_content_csv = read_csv(fileinlist_path, sep=",", index_col='Frames', dtype=float32)
            training_dataset_dict[file_name_csv] = dataset_content_csv
    return training_dataset_dict
def get_testing_data(testing_file_list):
    '''

    :param testing_file_list: input the list containing the path to each file
    :return: dictonary containing dataframe of each dataset from testing directory
    '''
    testing_dataset_dict = {}
    for fileinlist in testing_file_list:
        fileinlist_path = ""
        if tf.is_tensor(fileinlist):
            fileinlist_path = fileinlist.numpy().decode()  # obtain string from tensorobject

        else:
            fileinlist_path = fileinlist

        if fileinlist_path.endswith(".tsv"):
            print(f'file is a TSV format, {fileinlist_path}')
            dataset_content = read_mocap_data(fileinlist_path)  # read data and modify it, then save it into variable
            file_name = tf.strings.split(fileinlist, os.sep)[-1].numpy().decode()  # using tensorflow os separater to split string and obtain name of file
            testing_dataset_dict[file_name] = dataset_content
        elif fileinlist_path.endswith('.csv'):
            print(f'file is CSV format, {fileinlist_path}')
            file_name_csv = fileinlist_path.split(os.sep)[-1]  # parse to get name of file with extension
            dataset_content_csv = read_csv(fileinlist_path, sep=",", index_col='Frames', dtype=float32)
            testing_dataset_dict[file_name_csv] = dataset_content_csv
    return testing_dataset_dict

