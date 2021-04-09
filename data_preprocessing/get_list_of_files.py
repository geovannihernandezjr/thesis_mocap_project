'''
@author Geovanni Hernandez
Created to obtain the dataset files and create a dictonary to store all data to iterate through.
'''
import os
from data_preprocessing.preprocessing import read_mocap_data
from numpy import array, float32
from pandas import read_csv
def findFilesInFolder(dataset_dir, pathList=None, fileList=None, extension =""):

    """
        Function to find all files and its paths of an extension type in a folder and subfolders using os.walk
    :param dataset_dir: Root directory to find files
    :param pathList:    A list that stores all file paths
    :param fileList:    A list that stores all file names
    :param extension:   Fle extension to find
    :return:             pathList and fileList
    """
    if fileList is None:
        fileList = list()
    if pathList is None:
        pathList = list()
    try:  # Trapping a OSError:  File permissions problem I believe
        for dirpath, dirnames, filenames in os.walk(dataset_dir):
            for file in filenames:  # ITERATE THROUGH FILES IN THE CURRENT DIRECTOR
                if file.endswith(extension):  # FIND FIELS WITH .TSV
                    file_path = os.path.join(dirpath, file) # CREATE PATH USING os.path.join to STORE INTO LIST OF FILE PATHS
                    pathList.append(file_path)
                    fileList.append(file)  # STORE INTO LIST OF TSV FILE NAMES FOUND WITH .TSV EXTENSION


    except OSError:
        print('Cannot access ' + dataset_dir + '.Check permissions, or directory exists')

    return array(pathList), array(fileList)


def get_training_data(training_file_path_list):
    print("GETTING TRAINING DATA")
    """
    Read file from corresponding file path and extract data based on function (tsv) or pandas (csv) into a dataframe.
    Then store each dataframe into a dictonary with the filename as their key. 
    :param training_file_path_list: list containing all file paths considered for training
    :return: a dictonary containing all training data corresponding to file path list
    """
    training_dataset_dict = {}
    for file_path in training_file_path_list:
        if file_path.endswith(".tsv"):
            print("Detected TSV")
            dataset_content = read_mocap_data(file_path)
            file_name = file_path.split(os.sep)[-1]#.split("/")[-1]
            training_dataset_dict[file_name] = dataset_content
        elif file_path.endswith(".csv"):
            print('Detected CSV')
            file_name_csv = file_path.split("/")[-1]
            dataset_content_csv = read_csv(file_path, sep=",", index_col='Frames', dtype=float32)
            training_dataset_dict[file_name_csv] = dataset_content_csv
        print(f'training file obtained {file_path}')
    return training_dataset_dict

def get_testing_data(testing_file_path_list):
    print("GETTING TESTING DATA")

    """
    Read file from corresponding file path and extract data based on function (tsv) or pandas (csv) into a dataframe.
    Then store each dataframe into a dictonary with the filename as their key. 
    :param testing_file_path+list: list containing all file paths considered for training
    :return: a dictonary containing all test data corresponding to file path list
    """
    testing_dataset_dict = {}
    for file_path in testing_file_path_list:
        if file_path.endswith(".tsv"):
            print("Detected TSV")
            dataset_content = read_mocap_data(file_path)
            file_name = file_path.split(os.sep)[-1]#.split("/")[-1]
            testing_dataset_dict[file_name] = dataset_content
        elif file_path.endswith(".csv"):
            print("Detected CSV")
            file_name_csv = file_path.split("/")[-1]
            dataset_content_csv = read_csv(file_path, sep=",", index_col='Frames', dtype=float32)
            testing_dataset_dict[file_name_csv] = dataset_content_csv
        print(f'testing file obtained {file_path}')
    return testing_dataset_dict


