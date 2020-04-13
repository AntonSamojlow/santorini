"""
Custom tools around the definition and training of a dense neural network.

Written by Anton Samojlow, April 2020. [anton.samojlow@web.de]
"""


import logging
from os import path
from time import time

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

LOGGER = logging.getLogger(__name__)


class DenseModel():
    """Defines a  NN by specifying the list of layersizes: Either an 
    integer or 'BN' (BatchNormalization layer). The fit function is a wrapper
    around the usual tf.keras.Model.fit, i.p. with a customized callback"""
    def __init__(self, layer_sizes, name=None, compile=True,
                 optimizer=tf.keras.optimizers.Adam(0.001),
                 hiddenactivation='swish', dropout_rate=0.1, L1reg_weight=0, L2reg_weight=0):
        self.name = name
        self.layer_sizes = layer_sizes
        # generate tf.keras model
        layers = [tf.keras.Input(shape=(layer_sizes[0],))]
        for layer_size in layer_sizes[1:-1]:
            if layer_size == "BN":
                layers.append(tf.keras.layers.BatchNormalization()(layers[-1]))
            else:
                layers.append(
                    tf.keras.layers.Dense(
                        layer_size, activation=hiddenactivation, kernel_regularizer=tf.keras.regularizers.L1L2(
                            L1reg_weight, L2reg_weight))(layers[-1]))
                layers.append(tf.keras.layers.Dropout(dropout_rate)(layers[-1]))

        out_pi = tf.keras.layers.Dense(layer_sizes[-1][0], activation='softmax', name='pi')(layers[-1])
        out_v = tf.keras.layers.Dense(1, activation='tanh', name='v')(layers[-1])
        self.model = tf.keras.models.Model(inputs=layers[0], outputs=[out_pi, out_v])
        if compile:
            self.model.compile(optimizer=optimizer,
                               loss=['categorical_crossentropy', 'mean_squared_error'],
                               metrics=['accuracy', 'categorical_crossentropy', 'mean_squared_error'])

    def fit(self, train_config):
        """Wrapper around tf.keras.Model.fit.
        
        Parameters: train_config (of type TrainConfig)
        """
        if not isinstance(train_config, TrainConfig):
            raise Exception("given config is not an instance of 'TrainConfig'")
        t0 = time()
        LOGGER.info("-> training the model '{0}'".format(self.name))
        vals = self.model.evaluate(train_config.ds_full, verbose=False)
        LOGGER.info(
            "loss on full data (train+validation) before training: {0:.5f} (pi={1:.5f}|v={2:.5f})".format(vals[0], vals[1], vals[2]))

        trainresult = self.model.fit(train_config.ds_train,
                                     validation_data=train_config.ds_validate,
                                     steps_per_epoch=train_config.stepsperepoch,
                                     epochs=train_config.epochs,
                                     callbacks=train_config.callbacks,
                                     verbose=0)
        vals = self.model.evaluate(train_config.ds_full, verbose=False)
        print()
        LOGGER.info(
            "loss on full data (train+validation) after training: {0:.5f} (pi={1:.5f}|v={2:.5f})".format(vals[0], vals[1], vals[2]))
        t1 = time()
        LOGGER.info("<- training finished in {0} seconds".format(int(t1-t0)))
        return trainresult


class TrainConfig():
    """Argument for the fit-function, collecting training relevant parameters."""
    def __init__(self, epochs, trainsplit=0.8, batchsize=100, buffersize=None, callbacks=None):
        if callbacks is None:
            self.callbacks = EpochDots(report_every=1000, dot_every=25)
        else:
            self.callbacks = callbacks
        self.epochs = epochs
        self.trainsplit = trainsplit
        self.buffersize = buffersize
        self.batchsize = batchsize

    def LoadDatasets(self, folderpath=None, x=None, y_pi=None, y_val=None):
        """Resets tf.data.dataset objects and recalculates stepsperepoch.
        Args: folderpath should contain the three files 'x.csv', 'y_pi.csv', 'y_val.csv'"""
        if folderpath is not None:
            x = np.loadtxt(path.join(folderpath, 'x.csv'), delimiter=',')
            y_pi = np.loadtxt(path.join(folderpath, 'y_pi.csv'), delimiter=',')
            y_val = np.loadtxt(path.join(folderpath, 'y_val.csv'), delimiter=',')

        n_total = int(x.shape[0])
        n_train = int(n_total*self.trainsplit)
        n_validation = int(n_total-n_train)
        buffersize = n_train if self.buffersize is None else self.buffersize

        LOGGER.info('TrainConfig loaded and prepares {0} samples of which {1} are resevered for validation'
                    .format(n_total, n_validation))

        self.stepsperepoch = n_train//self.batchsize

        ds_features = tf.data.Dataset.from_tensor_slices(x)
        ds_labels = tf.data.Dataset.from_tensor_slices((y_pi, y_val))
        dataset = tf.data.Dataset.zip((ds_features, ds_labels))
        dataset.shuffle(buffersize)
        self.ds_full = dataset.take(n_train).cache().batch(self.batchsize)
        self.ds_validate = dataset.take(n_validation).cache().batch(self.batchsize)
        self.ds_train = dataset.skip(n_validation).take(n_train).cache().repeat().batch(self.batchsize)


class EpochDots(tf.keras.callbacks.Callback):
    """A simple callback that prints a "." every epoch, with occasional reports.
    Args:
    report_every: How many epochs between full reports
    dot_every: How many epochs between dots.
    """

    def __init__(self, report_every=100, dot_every=1):
        self.report_every = report_every
        self.dot_every = dot_every

    def on_epoch_end(self, epoch, logs):
        if epoch % self.report_every == 0:
            print()
            print('Epoch: {:d}, train loss: {:0.5f} (pi {:0.5f}, v {:0.5f})'
                  .format(epoch,
                          logs['loss'],
                          logs['pi_loss'],
                          logs['v_loss'],), end='')
            print(', val loss: {:0.5f} (pi {:0.5f}, v {:0.5f})'
                  .format(logs['val_loss'],
                          logs['val_pi_loss'],
                          logs['val_v_loss']), end='')
            print()
        if epoch % self.dot_every == 0:
            print('.', end='', flush=True)


class Tools():
    """
    Collection of tools to display training results or analyse a models
    performance
    """
    class HistoryPlotter(object):
        """A class for plotting a named set of keras-histories.
        The class maintains colors for each key from plot to plot.

        Code is from:
        https://github.com/tensorflow/docs/blob/master/tools/tensorflow_docs/plots
        """
        _COLOR_CYCLE = plt.rcParams['axes.prop_cycle'].by_key()['color']

        def __init__(self, smoothing_std=None):
            self.color_table = {}
            self.smoothing_std = smoothing_std

        def _smooth(self, values):
            """Smooths a list of values by convolving with a gaussian.
            Assumes equal spacing.
            Args:
                values: A 1D array of values to smooth.
                std: Sandard deviation of the gaussian. Units are array elements.
            Returns:
                The smoothed array.
            """
            width = self.smoothing_std * 4
            x = np.linspace(-width, width, 2 * width + 1)
            kernel = np.exp(-(x / 5)**2)

            values = np.array(values)
            weights = np.ones_like(values)

            smoothed_values = np.convolve(values, kernel, mode='same')
            smoothed_weights = np.convolve(weights, kernel, mode='same')

            return smoothed_values / smoothed_weights

        def plot(self, histories, metric, figsize=(8, 6), savepath=None):
            """Plots a {name: history} dictionary of keras histories.
            Colors are assigned to the name-key, and maintained from call to call.
            Training metrics are shown as a solid line, validation metrics dashed.
            Args:
            histories: {name: history} dictionary of keras histories.
            metric: which metric to plot from all the histories.          
            """

            plt.figure(figsize=figsize)

            for name, history in histories.items():
                # Remember name->color asociations.
                if name in self.color_table:
                    color = self.color_table[name]
                else:
                    color = self._COLOR_CYCLE[len(self.color_table)
                                              % len(self._COLOR_CYCLE)]
                    self.color_table[name] = color

                train_value = history.history[metric]
                if self.smoothing_std is not None:
                    train_value = self._smooth(train_value)
                plt.plot(history.epoch, train_value, color=color,
                         label=name.title() + ' Train')

                # added case check for missing validation set
                if 'val_' + metric in history.history.keys():
                    val_value = history.history['val_' + metric]
                    if self.smoothing_std is not None:
                        val_value = self._smooth(val_value)
                    plt.plot(history.epoch, val_value, '--', color=color,
                             label=name.title() + ' Val')

            plt.xlabel('Epochs')
            plt.ylabel(metric.replace('_', ' ').title())
            plt.legend()

            plt.xlim([0, max([history.epoch[-1] for history in histories.values()])])
            y_min = min([min(history.history[metric]) for history in histories.values()])
            y_max = max([max(history.history[metric]) for history in histories.values()])
            if 'val_' + metric in history.history.keys():
                y_min_val = min([min(history.history['val_' + metric]) for history in histories.values()])
                y_max_val = max([max(history.history['val_' + metric]) for history in histories.values()])
                y_min = min(y_min, y_min_val)
                y_max = max(y_max, y_max_val)

            plt.ylim([(1 - 0.1*np.sign(y_min))*y_min, (1 + 0.1*np.sign(y_max))*y_max])
            plt.grid(True)
            if savepath is not None:
                if path.exists(savepath):
                    LOGGER.warning("overriding existing plot '{}'".format(savepath))
                plt.savefig(savepath)
                LOGGER.info("saved plot to '{}'".format(savepath))
