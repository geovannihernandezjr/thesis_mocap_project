"""
@author Geovanni Hernandez
Simply to plot the metrics obtained from each epoch.
"""
import matplotlib.pyplot as plt
import os


def setup_figure_ax(x_title='Epoch', y_title=None, figure_title='Insert Title'):
    """
    setup a figue with axis specficed
    :param x_title: if int list of metric xlables ['RMSE', 'MSE', 'MAE']
    :param y_title:
    :param figure_title:
    :return:
    """
    lst_of_metric_plot_options = ['RMSE', 'MSE', 'MAE']
    if int is type(y_title):
        y_title = lst_of_metric_plot_options[y_title]


    fig, ax = plt.subplots()
    fig.canvas.draw()
    ax.set_xlabel(x_title)
    ax.set_ylabel(y_title)
    ax.set_title(figure_title)
    return ax

def plot_metrics(file_path, png_name, repeat_each_epoch=False, model_metric_history=None, loss_lst=None, mse_lst=None,
                 rmse_lst=None
                 , mae_lst=None, val_loss=None, val_mse=None, val_rmse=None, val_mae=None, valid_train=False):
    '''

    :param file_path: path of where the png will be save
    :param png_name: name of the png
    :param repeat_each_epoch: flag to indicate if epoch is repeated for trianing
    :param model_metric_history: using metric history model, which only works if the epoch isn't repeated 1by1
    :param loss_lst: list to contain all the values for loss curve
    :param mse_lst: list to contain all the mean square error values from training
    :param rmse_lst: list to contain all the root mse values from training
    :param mae_lst: list to contain all the mean absolute error from training
    :param val_loss: list to contain the values for loss curve on based on validation data used
    :param val_mse: list to contain the values for the mse values basedo no validation
    :param val_rmse: list to contain the values for the rmse values based on validation
    :param val_mae: list to contain the values for the mean absolute error values based on validation
    :param valid_train: flag to indicate if there is going to be validation metrics to plot with train metrics
    :return:
    '''
    if rmse_lst is None:
        rmse_lst = list()
    if val_rmse is None:
        val_rmse = list()
    if val_mae is None:
        val_mae = list()
    if mse_lst is None:
        mse_lst = list()
    if loss_lst is None:
        loss_lst = list()
    if val_mse is None:
        val_mse = list()
    if val_loss is None:
        val_loss = list()
    if mae_lst is None:
        mae_lst = list()
    global fig
    if repeat_each_epoch == True:
        if valid_train:
            title_valid_metrics = png_name + '\n Train vs Validation'
            fig = plt.figure(figsize=(10, 10))
            plt.suptitle(title_valid_metrics)

            # Plot Loss
            ax1 = fig.add_subplot(221)
            ax1.title.set_text('Loss')
            ax1.plot(loss_lst)
            ax1.plot(val_loss)
            ax1.legend(['Train', 'Validation'], loc='upper left')

            # Plot Mean Square Error
            ax2 = fig.add_subplot(222)
            ax2.title.set_text('MSE')
            ax2.plot(mse_lst)
            ax2.plot(val_mse)
            ax2.legend(['Train', 'Validation'], loc='upper right')

            # plot Root Mean Square Error
            ax3 = fig.add_subplot(223)
            ax3.title.set_text('RMSE')
            ax3.plot(rmse_lst)
            ax3.plot(val_rmse)
            ax3.legend(['Train', 'Validation'], loc='upper left')

            # Plot Mean Absolute Error
            ax4 = fig.add_subplot(224)
            ax4.title.set_text('MAE')
            ax4.plot(mae_lst)
            ax4.plot(val_mae)
            ax4.legend(['Train', 'Validation'], loc='upper right')
            for ax in fig.axes:
                ax.set(xlabel='Epoch')
        else:
            title_valid_metrics = png_name + '\n Train Metrics'
            fig = plt.figure(figsize=(10, 10))
            plt.suptitle(title_valid_metrics)

            # Plot Loss
            ax1 = fig.add_subplot(221)
            ax1.title.set_text('Loss')
            ax1.plot(loss_lst)
            ax1.legend(['Train'], loc='upper left')

            # Plot Mean Square Error
            ax2 = fig.add_subplot(222)
            ax2.title.set_text('MSE')
            ax2.plot(mse_lst)
            ax2.legend(['Train'], loc='upper right')

            # plot Root Mean Square Error
            ax3 = fig.add_subplot(223)
            ax3.title.set_text('RMSE')
            ax3.plot(rmse_lst)
            ax3.legend(['Train'], loc='upper left')

            # Plot Mean Absolute Error
            ax4 = fig.add_subplot(224)
            ax4.title.set_text('MAE')
            ax4.plot(mae_lst)
            ax4.legend(['Train'], loc='upper right')
            for ax in fig.axes:
                ax.set(xlabel='Epoch')

    else:
        if valid_train:
            title_novalid = png_name + '\n Train vs Validation'
            fig = plt.figure(figsize=(10, 10))
            plt.suptitle(title_novalid)

            # Plot Loss
            ax1 = fig.add_subplot(221)
            ax1.title.set_text('Loss')
            ax1.plot(model_metric_history['loss'])
            ax1.plot(model_metric_history['val_loss'])
            ax1.legend(['Train', 'Validation'], loc='upper left')

            # Plot Mean Square Error
            ax2 = fig.add_subplot(222)
            ax2.title.set_text('MSE')
            ax2.plot(model_metric_history['mse'])
            ax2.plot(model_metric_history['val_mse'])
            ax2.legend(['Train', 'Validation'], loc='upper right')

            # plot Root Mean Square Error
            ax3 = fig.add_subplot(223)
            ax3.title.set_text('RMSE')
            ax3.plot(model_metric_history['rmse'])
            ax3.plot(model_metric_history['val_rmse'])
            ax3.legend(['Train', 'Validation'], loc='upper left')

            # Plot Mean Absolute Error
            ax4 = fig.add_subplot(224)
            ax4.title.set_text('MAE')
            ax4.plot(model_metric_history['mae'])
            ax4.plot(model_metric_history['val_mae'])
            ax4.legend(['Train', 'Validation'], loc='upper right')
            for ax in fig.axes:
                ax.set(xlabel='Epoch')
        else:
            title_valid_metrics = png_name + '\n Train Metrics'
            fig = plt.figure(figsize=(10, 10))
            plt.suptitle(title_valid_metrics)

            # Plot Loss
            ax1 = fig.add_subplot(221)
            ax1.title.set_text('Loss')
            ax1.plot(model_metric_history['loss'])
            ax1.legend(['Train'], loc='upper left')

            # Plot Mean Square Error
            ax2 = fig.add_subplot(222)
            ax2.title.set_text('MSE')
            ax2.plot(model_metric_history['mse'])
            ax2.legend(['Train'], loc='upper right')

            # plot Root Mean Square Error
            ax3 = fig.add_subplot(223)
            ax3.title.set_text('RMSE')
            ax3.plot(model_metric_history['rmse'])
            ax3.legend(['Train'], loc='upper left')

            # Plot Mean Absolute Error
            ax4 = fig.add_subplot(224)
            ax4.title.set_text('MAE')
            ax4.plot(model_metric_history['mae'])
            ax4.legend(['Train'], loc='upper right')
            for ax in fig.axes:
                ax.set(xlabel='Epoch')

    png_name = png_name + "_Metrics.png"
    png_path = os.path.join(file_path, png_name)
    plt.savefig(png_path, dpi=300)
    plt.clf()
    plt.cla()
    plt.close(fig)
    del fig
