"""
@author: GEovanni Hernandez
Data plotting for different types of plotting needs.
This covers plotting for a single file in which its data is split between train and test.
This also covers plotting for a single file in which only the testing is being plotted on its own.
Finally it also covers the plotting of borg prediction scale.
"""
import matplotlib.pyplot as plt
import os

import numpy as np
from numpy import min, max
from math import floor, ceil


def setup_data_to_plot(dataset, predictions, look_back):
    """
    formatting the
    :param dataset:
    :param predictions:
    :param look_back:
    :return:
    """
    dataset = dataset.reshape(len(dataset), 1) # reshape data into shape that needs to be used for plot need an it in (row, col)
    predictions = np.array(predictions).reshape(len(predictions), 1) # reshape data into shape that needs to be used for plot
    # new_array_with__same_shape_type_as_given_array = np.empty_like(dataset)
    prediction_plot = np.empty_like(dataset)
    prediction_plot[:, :] = np.nan
    prediction_plot[look_back:, :] = predictions # i.e. trainY == BackL[1:train_size+1]
    return prediction_plot
def plot_mocap_predictions(file_path, png_name, raw_dataset, predictions, increments=1000, color='g'):
    fig, ax = plt.subplots(figsize=(20,6))
    ax.set_title("Raw Data vs Prediction\n" + png_name)
    ax.set_xlabel(f"Frames (increments of {increments} frames)")
    ax.set_ylabel("Marker positions (mm)")
    ax.plot(raw_dataset[::increments], color="k", alpha=0.5)
    ax.plot(predictions[::increments], linestyle='dashed', color=color)
    ax.legend(['Raw Data', 'Test Predictions'], loc='upper right')
    all_png_name = png_name + "_predictions.png"
    all_png_name_path = os.path.join(file_path, all_png_name)
    plt.savefig(all_png_name_path, dpi=300)
    plt.clf()
    plt.cla()
    plt.close(fig)

def setup_single_file_data_to_plot(dataset, train_predict, test_predict, look_back=1):
    if type(test_predict) is list:
        test_predict = np.array(test_predict).reshape(len(test_predict), 1)
    train_size = len(train_predict)
    # new_array_with__same_shape_type_as_given_array = np.empty_like(dataset)
    trainPredictPlot = np.empty_like(dataset)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:train_size+look_back, :] = train_predict # i.e. trainY == BackL[1:train_size+1]
    testPredictPlot = np.empty_like(dataset)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[(train_size+1)+look_back:, :] = test_predict # i.e. testY == BackL[train_size+1:-1]
    return trainPredictPlot, testPredictPlot

def plot_single_mocap_file_predictions(file_path, png_name, raw_dataset, train_predict, test_predict, look_back, increments=1000):
    train_predictions, test_predictions = setup_single_file_data_to_plot(raw_dataset, train_predict, test_predict, look_back)
    prediction_test_start_frame_number = (len(train_predict)+1)+look_back
    print(prediction_test_start_frame_number)
    title = "Raw Data vs Prediction\n" + png_name
    fig = plt.figure(figsize=(25, 10))
    plt.suptitle(title)

    # Plot Raw vs Train
    # add_subplot(nrows, ncols, index) index starts at 1 in upper left corner to right
    ax1 = fig.add_subplot(311)
    ax1.title.set_text('Raw Mocap Data vs Train Prediction')
    ax1.plot(raw_dataset[::increments], color='orange', alpha=0.5)
    ax1.plot(train_predictions[::increments], linestyle="dashed", color='b')
    ax1.legend(['Raw Data', 'Train'], loc='upper right')

    # Plot Raw vs Test
    ax2 = fig.add_subplot(312)
    ax2.title.set_text('Raw Mocap Data vs Test Prediction')
    ax2.plot(raw_dataset[::increments], color='orange', alpha=0.5)
    ax2.plot(test_predictions[::increments], linestyle='dashed', color='g')
    ax2.legend(['Raw Data', 'Test'], loc='upper right')

    # plot Raw, Train and Test
    ax3 = fig.add_subplot(313)
    ax3.title.set_text('Raw Data vs Train & Test ')
    first_pt = int(floor(min(raw_dataset / float(increments)))) * increments
    last_pt = int(ceil(max(raw_dataset / float(increments)))) * increments
    mid_pt = int((first_pt + last_pt) / 2)
    prediction_line = int(prediction_test_start_frame_number/increments)
    print(f'predic{prediction_line}')
    ax3.axvline(prediction_line, 0, 1, linewidth=2, color='red')
    text_str_frame = f'Frame={prediction_test_start_frame_number}'
    ax3.text(prediction_line, mid_pt, text_str_frame, bbox=dict(
        facecolor='yellow', alpha=.5), horizontalalignment='center', va='center',
            fontsize=8)
    ax3.plot(raw_dataset[::increments], color='orange', alpha=0.5)
    ax3.plot(train_predictions[::increments], linestyle='dashed', color='b')
    ax3.plot(test_predictions[::increments], linestyle='dashed', color='g')
    ax3.legend([f'Start of Test Prediction @ {text_str_frame}', 'Raw Data', 'Train', 'Test'], loc='upper right')

    plt.xlabel(f"Frames (increments of {increments} frames)")
    for ax in fig.axes:
        ax.set(ylabel='Marker positions (mm)')
    # plt.savefig(png_name, dpi=300)
    # fig, ax = plt.subplots(figsize=(20,6))
    # ax.set_title("Raw Data vs Prediction\n" + png_name)
    # ax.set_xlabel(f"Frames (increments of {increments} frames)")
    # ax.set_ylabel("Marker positions (mm)")
    # ax.plot(raw_dataset[::increments], color="k", alpha=0.5)
    # ax.plot(predictions[::increments], linestyle='dashed', color=color)
    # ax.legend(['Raw Data', 'Test Predictions'], loc='upper right')
    all_png_name = png_name + "_predictions.png"
    all_png_name_path = os.path.join(file_path, all_png_name)
    plt.savefig(all_png_name_path, dpi=300)
    plt.clf()
    plt.cla()
    plt.close(fig)

def setup_data_to_plot(dataset, predictions, look_back):
    dataset = dataset.reshape(len(dataset), 1) # reshape data into shape that needs to be used for plot need an it in (row, col)
    predictions = np.array(predictions).reshape(len(predictions), 1) # reshape data into shape that needs to be used for plot
    # new_array_with__same_shape_type_as_given_array = np.empty_like(dataset)
    prediction_plot = np.empty_like(dataset)
    prediction_plot[:, :] = np.nan
    prediction_plot[look_back:, :] = predictions # i.e. trainY == BackL[1:train_size+1]
    return prediction_plot
def plot_borg_predictions(file_path, png_name, raw_borg, borg_predictions, increments=1000, color='g'):
    fig, ax = plt.subplots(figsize=(20,6))
    title = "Raw Data vs Predictions\n" + png_name

    ax.set_title(title)
    ax.set_xlabel(f"Frames (increments of {increments} frames)")
    ax.set_ylabel("Borg RPE Value")
    ax.plot(raw_borg[::increments], color="k", alpha=0.5)
    ax.plot(borg_predictions[::increments], linestyle='dashed', color=color)
    ax.legend(['Borg Data', 'Borg Prediction'], loc='upper right')
    all_png_name = png_name + "_borg_predictions.png"
    all_png_name_path = os.path.join(file_path, all_png_name)
    plt.savefig(all_png_name_path, dpi=300)
    plt.clf()
    plt.cla()
    plt.close(fig)