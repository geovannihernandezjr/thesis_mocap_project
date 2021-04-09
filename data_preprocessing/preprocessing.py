"""
@author Geovanni Hernandez
This file is to do preprocessing of data.
read_mocap_data_with_borg - is used to create csv with the data
read_mocap_data - is used to just read in the data for use from TSV file if there is not need for merged data, might be slower
univariate_data_to_sequence and multivariate_data_to_sequence - is to create regular data into time series sequence data
"""
import csv
import numpy as np
from numpy import float32
import pandas as pd
from data_preprocessing.MoCapData import MoCapData
from data_preprocessing.borg_scale_list_from_file import get_borgscale_list
from data_preprocessing.remove_data import remove_markers_not_needed_in_each_df_and_marker_name_lst

def read_mocap_data_with_borg(mocap_filepath, borg_filepath):
    '''
    MoCap and Borg files should correspond to the same experiment conducted. This will cause issues if it isn't. The files are merged together.
    The Borg value is obtained at a rate of 60 secs while the frames for each MoCap capture is at a rate of 100 frames per sec.
    Therefore, the Borg value is interporlated with 100 frames per sec * 60 secs per rpe = 6000 frames per rpe, so every 6000 frames will have
    the same borg value in that 60 sec time span.
    :param mocap_filepath: path to dataset containing MoCap data, usually the .tsv file from QTM
    :param borg_filepath: path to the file containing Borg RPE values for each file, usually a .xlsx file
    :return: a dataframe with merged MoCap and Borg data,
    '''
    borg_index = 0 # index of first number in borg list
    borg_lst_from_excel_file = get_borgscale_list(borg_filepath) # obtain the borg value from reading the file store into list

    markerNameListForDoingMultiIndex = list()
    markerPointsForEachMarker = list()
    moCapDataAttributes = MoCapData()
    borg_scale_data = list()
    # iterate through two files

    with open(mocap_filepath) as tsvfile:  # ('F_76_12_9_6-28-2019_clean.tsv') as tsvfile:

        reader = csv.reader(tsvfile, delimiter='\t')
        for i, row in enumerate(reader):  # all the data from the file for line is assigned to row, enumerate value is assigned to i

            if i <= 10:  # from [0,10], the header for a 3D QTM TSV File
                # store string from first item in list removing underscores doing lower case letters
                header = row[0].lower().replace("_", "")
                # Value is for storing the list second item, it can be a str or list
                value = row[1]
                if 'noofframes' in header:  # Line 0 of header is NO_OF_FRAMES i.e ['NO_OF_FRAMES', 23233]
                    moCapDataAttributes.numOfFrames = int(value)
                elif 'noofcameras' in header:  # Line 1 of header is NO_OF_CAMERAS i.e ['NO_OF_CAMERAS', 12]
                    moCapDataAttributes.numOfCameras = int(value)
                elif 'noofmarkers' in header:  # Line 2 of header is NO_OF_MARKERS i.e ['NO_OF_MARKERS', 41]
                    moCapDataAttributes.numOfMarkers = int(value)
                elif 'frequency' in header:  # Line 3 of header is FREQUENCY    i.e ['FREQUENCY', '100']
                    moCapDataAttributes.frequency = value
                elif 'noofanalog' in header:  # Line 4 of header is NO_OF_ANALOG
                    moCapDataAttributes.numOfAnalog = value
                elif 'analogfrequency' in header:  # Line 5 of header is ANALOG_FREQUENCY
                    moCapDataAttributes.analogFrequency = value
                elif 'description' in header:  # Line 6 of header DESCRIPTION i.e --
                    moCapDataAttributes.description = value
                elif 'timestamp' in header:  # Line 7 of header TIME_STAMP i.e ['TIME_STAMP', 2019-06-28, 12:39:23, 351682]
                    values = row[1:]  # different range since the time stamp contains two elements
                    moCapDataAttributes.timeStamp = values
                elif 'dataincluded' in header:  # Line 8 of header Type of Data Included in File DATA_INCLUDED i.e 3D
                    moCapDataAttributes.dataType = value
                # check string after convert first string in row to lowercase, and remove the underscores (_)
                elif 'markernames' in header:  # Line 9 of header is MARKER_NAMES i.e HeadF|Chest
                    values = row[1:]
                    moCapDataAttributes.markerNames = values  # store all Marker Names into the moCapData() markerNames list
                    for marker in moCapDataAttributes.markerNames:
                        markerNameListForDoingMultiIndex.extend(marker for _ in range(3))  # duplicate marker name 3 times to be used later for creating a multi-index dataframe
                elif 'frame' in row[0].lower() and 'time' in row[1].lower():  # Line 10
                    values = row[2:-1]
                    moCapDataAttributes.markerNamesXYZ = values  # this row contains the string names for each marker x,y,and z position. *Using -1] because an extra character was being rw
                else:
                    print("header not found")
                    print("Check file number of headers ")

            # After header in tsv file
            else:  # from [11,] the header for a 3D QTM TSV File
                framenumber = int(row[0])
                time = float(row[1])
                marker_data_points = row[2:]
                # check size is numOFmarker*3
                moCapDataAttributes.frameNumberList.append(framenumber)  # add frame number from current row to a list for storing the sequential order of frame to be used as index for data frame
                moCapDataAttributes.timeList.append(round(time, 2))  # add time at each time step to list should start @ [0, +10msec] increment by .01 sec
                markerPointsForEachMarker.append(list(map(float, row[2:])))  # create a list of 3D coordinates for each marker, with each row in the list indicting the marker data for the particular frame.row
                # ADDED SECTION WHICH IS USED TO MERGE THE BORG SCALE ##
                # BORG VALUE ADDED AT EACH FRAME AS THE FILE IS BEING READ AND STORED AT EACH ROW ##

                if (framenumber - 1) % 6000 == 0 and time % 60.0000 == 0.0:  # using that 100 frames/sec since the borg scale is per minute, the equivalent in frames is 100 frames / sec * 60 sec / minute = 6000 frames,
                    # therefore every 6000 frames the borg scale should change. Since the frame number begins at 1 the
                    # subtracting one will indicate 6000th frame at every 60 seconds (every minute). The time
                    # and frame number must match.
                    if time == 0.0 and framenumber == 1:  # Used to catch first frame at time 0 so that the
                        # later code does not execute and increase borg index to change the borg value to next row.
                        # The next iteration will continue from top and will merge the borg scale to every frame.
                        borg_scale_data.append(borg_lst_from_excel_file[borg_index])  # for initial frame
                        continue
                    borg_index = borg_index + 1
                    if borg_index > (len(borg_lst_from_excel_file) - 1):  # if the index is greater than max elements in borg scale this
                        borg_index = len(borg_lst_from_excel_file) - 1  # is to keep the borg scale list index from going out of range

                borg_scale_data.append(borg_lst_from_excel_file[borg_index])  # with framenumber starting @ 1
                # BORG MERGE WITH MOCAP DATA ENDS HERE ##
    if len(moCapDataAttributes.frameNumberList) != moCapDataAttributes.numOfFrames:
        print("Something Went Wrong! Check moCapData() frameNumberList is same as total number of frames")

    # create list with marker names and marker names with X,YorZ appended to marker name.
    body_to_index_labels = [markerNameListForDoingMultiIndex, moCapDataAttributes.markerNamesXYZ]

    # create a list of tuples containing the column header and subheader for tbe data frame and for storing data
    # corresponding to X,Y and Z
    marker_label_list_tuples = list(zip(*body_to_index_labels))
    # print(label_tuples)

    # create the structure for the columns (header and subheader) from the list of tuples created
    column_head = pd.MultiIndex.from_tuples(marker_label_list_tuples, names=['Marker_Names', 'Frame'])
    # create a dataframe with the data,using the column headeer struture beforehand. since the x,y, and z data is stored as a list with each row containing all the markers for each frame.
    # with the multi-index structure each list will be split according to the numbee of subheader and the position in the list.
    # since only 3 subheaders are created e.g. HeadF X/Y/Z, 0, 1,2 from data will be used, the next will be 3,4,5 etc.

    mocap_multi_df = pd.DataFrame(markerPointsForEachMarker, index=moCapDataAttributes.frameNumberList, columns=column_head)

    # borg_tup = [('labels', 'borg scale')]
    # borg_multi = pd.MultiIndex.from_tuples(borg_tup)
    # borg_scale_data_df = pd.DataFrame(borg_scale_data, index=moCapDataAttributes.frameNumberList, columns=borg_multi)

    ################ EDITT The multi index DATAFRAMES TO REFLECT REMOVED COLUMNS NOT ARE NOT NEEDED ################

    remove_markers_not_needed_in_each_df_and_marker_name_lst(mocap_multi_df, moCapDataAttributes)

    num_of_markers_after_removing = len(moCapDataAttributes.markerNamesXYZ)/3  # get number of zeros by getting length of marker names after removing the names from list
    moCapDataAttributes.numOfMarkers=num_of_markers_after_removing
    #### now that the multiindex dataframe for mocap has the removed columns which make it easier to remove instead of having to illuminate per x,y or z columns

    # create dataframe after removing uneeded ones with just the marker name xyz label, e.g. RElbowOut X instead of RElbowOut -> RElbowOutX
    time_df = pd.DataFrame(moCapDataAttributes.timeList, index=moCapDataAttributes.frameNumberList, columns=['TimePerFrame'],dtype=float32)
    mocap_df = pd.DataFrame(mocap_multi_df.values, index=moCapDataAttributes.frameNumberList, columns=moCapDataAttributes.markerNamesXYZ, dtype=float32)
    borg_scale_data_df = pd.DataFrame(borg_scale_data, index=moCapDataAttributes.frameNumberList, columns=['borg'])
    # delete the multiindex now that it isn't needed
    del mocap_multi_df
    # delete the attributes for the file might not need them anymore
    del moCapDataAttributes
    # concat all the dataframes to create the full dataset of all the features with borg scale being the last column, the labels
    mocap_borg_merg = pd.concat([time_df, mocap_df, borg_scale_data_df], axis=1)
    mocap_borg_merg.index.name = 'Frames'
    return mocap_borg_merg

def read_mocap_data(mocap_filepath):
    '''
    This type of function will do the same as above but no merging is done with borg value, this is to be used if you just want to read TSV files.
    :param mocap_filepath: path to dataset containing MoCap data, usually the .tsv file from QTM
    :return: dataframe of mocap data only
    '''
    markerNameListForDoingMultiIndex = list()
    markerPointsForEachMarker = list()
    moCapDataAttributes = MoCapData()

    with open(mocap_filepath) as tsvfile:  # ('F_76_12_9_6-28-2019_clean.tsv') as tsvfile:

        reader = csv.reader(tsvfile, delimiter='\t')
        for i, row in enumerate(reader):  # all the data from the file for line is assigned to row, enumerate value is assigned to i

            if i <= 10:  # from [0,10], the header for a 3D QTM TSV File
                # store string from first item in list removing underscores doing lower case letters
                header = row[0].lower().replace("_", "")
                # Value is for storing the list second item, it can be a str or list
                value = row[1]
                if 'noofframes' in header:  # Line 0 of header is NO_OF_FRAMES i.e ['NO_OF_FRAMES', 23233]
                    moCapDataAttributes.numOfFrames = int(value)
                elif 'noofcameras' in header:  # Line 1 of header is NO_OF_CAMERAS i.e ['NO_OF_CAMERAS', 12]
                    moCapDataAttributes.numOfCameras = int(value)
                elif 'noofmarkers' in header:  # Line 2 of header is NO_OF_MARKERS i.e ['NO_OF_MARKERS', 41]
                    moCapDataAttributes.numOfMarkers = int(value)
                elif 'frequency' in header:  # Line 3 of header is FREQUENCY    i.e ['FREQUENCY', '100']
                    moCapDataAttributes.frequency = value
                elif 'noofanalog' in header:  # Line 4 of header is NO_OF_ANALOG
                    moCapDataAttributes.numOfAnalog = value
                elif 'analogfrequency' in header:  # Line 5 of header is ANALOG_FREQUENCY
                    moCapDataAttributes.analogFrequency = value
                elif 'description' in header:  # Line 6 of header DESCRIPTION i.e --
                    moCapDataAttributes.description = value
                elif 'timestamp' in header:  # Line 7 of header TIME_STAMP i.e ['TIME_STAMP', 2019-06-28, 12:39:23, 351682]
                    values = row[1:]  # different range since the time stamp contains two elements
                    moCapDataAttributes.timeStamp = values
                elif 'dataincluded' in header:  # Line 8 of header Type of Data Included in File DATA_INCLUDED i.e 3D
                    moCapDataAttributes.dataType = value
                # check string after convert first string in row to lowercase, and remove the underscores (_)
                elif 'markernames' in header:  # Line 9 of header is MARKER_NAMES i.e HeadF|Chest
                    values = row[1:]
                    moCapDataAttributes.markerNames = values  # store all Marker Names into the moCapData() markerNames list
                    for marker in moCapDataAttributes.markerNames:
                        markerNameListForDoingMultiIndex.extend(marker for _ in range(3))  # duplicate marker name 3 times to be used later for creating a multi-index dataframe
                elif 'frame' in row[0].lower() and 'time' in row[1].lower():  # Line 10
                    values = row[2:-1]
                    moCapDataAttributes.markerNamesXYZ = values  # this row contains the string names for each marker x,y,and z position. *Using -1] because an extra character was being rw
                else:
                    print("header not found")
                    print("Check file number of headers ")

            # After header in tsv file
            else:  # from [11,] the header for a 3D QTM TSV File
                framenumber = int(row[0])
                time = float(row[1])
                marker_data_points = row[2:]
                # check size is numOFmarker*3
                moCapDataAttributes.frameNumberList.append(framenumber)  # add frame number from current row to a list for storing the sequential order of frame to be used as index for data frame
                moCapDataAttributes.timeList.append(round(time, 2))  # add time at each time step to list should start @ [0, +10msec] increment by .01 sec
                markerPointsForEachMarker.append(list(map(float, row[2:])))  # create a list of 3D coordinates for each marker, with each row in the list indicting the marker data for the particular frame.row
    if len(moCapDataAttributes.frameNumberList) != moCapDataAttributes.numOfFrames:
        print("Something Went Wrong! Check moCapData() frameNumberList is same as total number of frames")

    # create list with marker names and marker names with X,YorZ appended to marker name.
    body_to_index_labels = [markerNameListForDoingMultiIndex, moCapDataAttributes.markerNamesXYZ]

    # create a list of tuples containing the column header and subheader for tbe data frame and for storing data
    # corresponding to X,Y and Z
    marker_label_list_tuples = list(zip(*body_to_index_labels))
    # print(label_tuples)

    # create the structure for the columns (header and subheader) from the list of tuples created
    column_head = pd.MultiIndex.from_tuples(marker_label_list_tuples, names=['Marker_Names', 'Frame'])
    #print(f'iter: {f}, filepath: {mocap_filepath}  of points: {np.shape(markerPointsForEachMarker)}')
    # create a dataframe with the data,using the column headeer struture beforehand. since the x,y, and z data is stored as a list with each row containing all the markers for each frame.
    # with the multi-index structure each list will be split according to the numbee of subheader and the position in the list.
    # since only 3 subheaders are created e.g. HeadF X/Y/Z, 0, 1,2 from data will be used, the next will be 3,4,5 etc.

    mocap_multi_df = pd.DataFrame(markerPointsForEachMarker, index=moCapDataAttributes.frameNumberList, columns=column_head)

    # borg_tup = [('labels', 'borg scale')]
    # borg_multi = pd.MultiIndex.from_tuples(borg_tup)
    # borg_scale_data_df = pd.DataFrame(borg_scale_data, index=moCapDataAttributes.frameNumberList, columns=borg_multi)

    ################ EDITT The multi index DATAFRAMES TO REFLECT REMOVED COLUMNS NOT ARE NOT NEEDED ################

    remove_markers_not_needed_in_each_df_and_marker_name_lst(mocap_multi_df, moCapDataAttributes)

    num_of_markers_after_removing = len(moCapDataAttributes.markerNamesXYZ)/3  # get number of zeros by getting length of marker names after removing the names from list
    moCapDataAttributes.numOfMarkers=num_of_markers_after_removing
    #### now that the multiindex dataframe for mocap has the removed columns which make it easier to remove instead of having to illuminate per x,y or z columns

    # create dataframe after removing uneeded ones with just the marker name xyz label, e.g. RElbowOut X instead of RElbowOut -> RElbowOutX

    mocap_df = pd.DataFrame(np.array(mocap_multi_df.values, dtype=np.float32), index=moCapDataAttributes.frameNumberList, columns=moCapDataAttributes.markerNamesXYZ)
    mocap_df.index.name = 'Frames'
    # delete the multiindex now that it isn't needed
    del mocap_multi_df
    # delete the attributes for the file might not need them anymore
    del moCapDataAttributes
    # concat all the dataframes to create the full dataset of all the features with borg scale being the last column, the labels
    # dataset = pd.concat([mocap_multi_df, velocity_df, acceleration_df, borg_scale_data_df], axis=1)
    #dataset.to_csv("Dataset_"+(mocap_filepath.split("/")[-1].split(".")[0])+".csv", sep=",")   # contain pandas dataframe multiindex in column 1 and mocapAttributes in col 2
    # dataset_dic[mocap_filepath.split("/")[-1].split(".")[0]] = [dataset, moCapDataAttributes]#mocap_filepath.split("\\")[-1].split(".")[0]] = [dataset, moCapDataAttributes]  # save dataframes into a dictonary tied to filename after removing the columns not needed

    return mocap_df
    # dataset_dic[mocap_filepath.split("/")[-1].split(".")[0]] = [mocap_df, moCapDataAttributes]



def univariate_data_to_sequence(dataset, start_index, end_index, history_size, target_size):
    '''
    Using only one set of data (univariate) or one column of data feature to create a time sequence of data.
    This will only use the data dataset inputed to create an output of data that will be split into sequences based on
    the history and target size of sequence.
    :param dataset: numpy array that will be the univariate dataset
    :param start_index: the start of where the sequence should begin from the dataset
    :param end_index: the end of where the sequence should end from the dataset
    :param history_size: the amount of previous timestep to consider
    :param target_size: the amount of future timestep to consider
    :return: two numpy arrays, one containing the data based on history size, and one for containing the data
    '''
    data = []
    labels = []
    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):

        indices = range(i-history_size, i) # i.e. subtract 1 from
        # Reshape data from (history_size,) to (history_size, 1)
        data.append(np.reshape(dataset[indices], (history_size, 1))) # reshape dataset to contain the correct shape for the sequence with length of history_size
        labels.append(dataset[i+target_size]) # offset by target_size
    return np.array(data), np.array(labels)

def multivariate_data_to_sequence(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
    '''
    Using mulitple set of data or multiple column of data features to create a multivariate time sequence of data
    :param dataset: numpy array that will be the multivariate dataset, usually containing the full dataset
    :param target: numpy array that will contain the same multivariate dataset or just the dataset that will be used as a target for labels
    :param start_index: this will be the beginning of where the sequence will begin
    :param end_index: This will be the ending of where the sequence will end
    :param history_size: the past observation size, or the previous size of the timesteps to predict based on history
    :param target_size: the future observation size, or the future size of the timestep size
    :param step: number of step between each sequence
    :param single_step: boolean value to assign if each output based on sequence should be single step target
    :return: array of data containing the features or sequences based on history, array of data containing the labels or sequences based on target size
    '''

    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i-history_size, i, step)
        data.append(dataset[indices])

        if single_step:
            labels.append(target[i+target_size])
        else:
            labels.append(target[i:i+target_size])

    return np.array(data), np.array(labels)