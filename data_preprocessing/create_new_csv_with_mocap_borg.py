"""
@author Geovanni Hernandez
Obtain all the files for MoCAp and Borg Rpe files to merge and then save into a new csv file format.
"""
from data_preprocessing.get_list_of_files import findFilesInFolder
from data_preprocessing.preprocessing import read_mocap_data_with_borg
from os import path

def loop_merge_files(mocap_path_list, mocap_filename_list, borg_path_list, borg_filename_list, parent_directory):
    """
    Iterates through corresponding motion capture lists, and borg file lists to merge each file and
    save to a csv for each pair of files.
    File format: Motion_ID_HEIGHT_INTERVAL_WEIGHT_DATEOfExperiment
    :param mocap_path_list: list containing the motion capture paths fo reach file (usually .tsv)
    :param mocap_filename_list: list to just the names of each motion capture file
    :param borg_path_list: list containing the borg data points for each file (usually .xlsx)
    :param borg_filename_list:  list containing the name of each borg file
    :param parent_directory: directory root to where you would save the csv files merged with mocap and borg
    :return: Save merged Dataframe into csv file
    """
    for file_num, mocap_filename in enumerate(mocap_filename_list):
        string_name_of_merged_file = f'{mocap_filename.split(".")[0]}-borg-merged.csv'
        for borg_num, borg_filename in enumerate(borg_filename_list):
            """example file name Lifting_005_51_9_13_8-2-2019_100%Filled.tsv
            need to remove extension and extra text after date since borg data file iwll have the same
            text without that. example borg file name Lifting_005_51_9_13_8-2-2019.xlsx.
            spliting the data with underscore will give the remaining text in separate strings 
            so a joining of these texts before the extra text and extension can be done"""

            mocap_filename_without_extension = "_".join(mocap_filename.split("_")[:-1]) # split mocap file without extension
            borg_filename_without_extension = borg_filename.split(".")[0].rstrip(" ")
            if not mocap_filename_without_extension == borg_filename_without_extension:
                print(f'file do not match {mocap_filename_without_extension} == {borg_filename_without_extension}')
                continue
            else:
                print(f'file match {mocap_filename_without_extension} == {borg_filename_without_extension}')
                print(f'mocap_file {mocap_path_list[file_num]}\nborg_file {borg_path_list[borg_num]}')
                print(f'merged file name {string_name_of_merged_file}')
                mocap_df = read_mocap_data_with_borg(mocap_path_list[file_num], borg_path_list[borg_num])
                path_to_save_csv = path.join(parent_directory, string_name_of_merged_file)
                mocap_df.to_csv(path_to_save_csv)


def main():
    """Parent directory location of files for training/testing for both types of data"""
    # TRAINING_PARENT_DIR = 'D:/HIPE/Clean_qtm/Fill/Train/'
    # BORG_TRAINING_PARENT_DIR = 'D:/HIPE/Clean_qtm/Fill/Train/Borg/'
    # TESTING_PARENT_DIR = 'D:/HIPE/Clean_qtm/Fill/Test/'
    # BORG_TESTING_PARENT_DIR = 'D:/HIPE/Clean_qtm/Fill/Test/Borg/'
    TRAINING_PARENT_DIR = 'C:/Users/geova/Desktop/Filled_Files/Train/1001'
    BORG_TRAINING_PARENT_DIR = 'C:/Users/geova/Desktop/Filled_Files/Train/1001'
    TESTING_PARENT_DIR = 'C:/Users/geova/Desktop/Filled_Files/Test/'
    BORG_TESTING_PARENT_DIR = 'C:/Users/geova/Desktop/Filled_Files/Test/Borg'
    """Obtain the list of files for training of mocap and borg"""
    training_path_list, training_filename_list = findFilesInFolder(TRAINING_PARENT_DIR, extension='.tsv')
    training_borg_path_list, training_borg_filename_list = findFilesInFolder(BORG_TRAINING_PARENT_DIR,
                                                                             extension='.xlsx')
    """Obtain the list of files for testing of mocap and borg"""
    # testing_path_list, testing_filename_list = findFilesInFolder(TESTING_PARENT_DIR, extension='.tsv')
    # testing_borg_path_list, testing_borg_filename_list = findFilesInFolder(BORG_TESTING_PARENT_DIR, extension='.xlsx')

    """use function to iterate of each train file"""
    loop_merge_files(training_path_list, training_filename_list, training_borg_path_list, training_borg_filename_list, TRAINING_PARENT_DIR)
    """Use function to iterate of each test"""
    # loop_merge_files(testing_path_list, testing_filename_list, testing_borg_path_list, testing_borg_filename_list, TESTING_PARENT_DIR)


if __name__ == "__main__":
    main()
