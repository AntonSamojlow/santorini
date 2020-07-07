import logging
import logging.handlers
import multiprocessing
import multiprocessing.queues as mpq
import os
from time import sleep

import numpy as np
import matplotlib.pyplot as plt

import gymdata


class Trainer(multiprocessing.Process):
    def __init__(self, config: gymdata.TrainConfig, gympath: gymdata.GymPath,
                 endevent: multiprocessing.Event,
                 newmodelevent: multiprocessing.Event, logging_q: mpq.Queue):
        super().__init__()
        self.config = config
        self.endevent = endevent
        self.newmodelevent = newmodelevent
        self.logging_q = logging_q
        self.gympath = gympath
        self.logger: logging.Logger

    def run(self):
        # logging setup
        self.logger = logging.getLogger(type(self).__name__)
        self.logger.setLevel(self.config.logging.loglevel)
        logfilepath = os.path.join(self.gympath.log_folder,
                                   "{}.log".format(type(self).__name__))
        rfh = self.config.logging.getRotatingFileHandler(logfilepath)
        qh = logging.handlers.QueueHandler(self.logging_q)
        qh.setLevel(logging.INFO)
        self.logger.addHandler(rfh)
        self.logger.addHandler(qh)
        self.logger.info("started and initialized logger")

        try:
            import tensorflow as tf

            self.logger.info('imported TensorFlow {0}'.format(
                tf.__git_version__))
            tflogger = tf.get_logger()
            tflogger.addHandler(rfh)
            tflogger.setLevel(self.config.logging.tf_loglevel)
            self.logger.info(
                f"changed tensorflow log level to {self.config.logging.tf_loglevel}"
            )

            history_plotter = HistoryPlotter(self.logger)

            class EpochDots(tf.keras.callbacks.Callback):
                """A simple callback that prints a "." every epoch, with occasional reports.
                Args:
                report_every: How many epochs between full reports
                dot_every: How many epochs between dots.
                """
                def __init__(self,
                             logger: logging.Logger,
                             report_every=100,
                             dot_every=1):
                    self.report_every = report_every
                    self.dot_every = dot_every
                    self.logger = logger

                def on_epoch_end(self, epoch, logs):
                    if epoch == 0 or (1 + epoch) % self.report_every == 0:
                        print()
                        msg = 'Epoch: {:d}, train loss: {:0.5f} (pi {:0.5f}, v {:0.5f})'.format(
                            1 + epoch, logs['loss'], logs['pi_loss'],
                            logs['v_loss'])
                        try:
                            msg += ', val loss: {:0.5f} (pi {:0.5f}, v {:0.5f})'.format(
                                logs['val_loss'], logs['val_pi_loss'],
                                logs['val_v_loss'])
                        except KeyError:
                            pass
                        print(msg)
                        self.logger.info(msg)
                        print()
                    if epoch % self.dot_every == 0:
                        print('.', end='', flush=True)

            def try_load_datasets() -> tuple:
                modeliteration: int
                with open(self.gympath.modelinfo_file, 'r') as f:
                    modeliteration = gymdata.ModelInfo.from_json(
                        f.read()).iterationNr
                activefolders = [
                    f for f in os.scandir(self.gympath.gamerecordpool_folder)
                    if f.is_dir() and int(f.name) +
                    self.config.max_sampleage >= modeliteration
                ]
                basepaths = []
                for folder in activefolders:
                    basepaths += [
                        f.path.replace(".x.csv", "")
                        for f in os.scandir(folder.path)
                        if f.is_file() and f.name.endswith(".x.csv")
                    ]
                if len(basepaths) == 0:
                    return Exception("No fresh samples found in {}".format(
                        self.gympath.gamerecordpool_folder))

                x, y_pi, y_val = [], [], []
                try:
                    for basepath in basepaths:
                        x += [
                            np.loadtxt("{0}.{1}".format(basepath, 'x.csv'),
                                       delimiter=',')
                        ]
                        y_pi += [
                            np.loadtxt("{0}.{1}".format(basepath, 'y_pi.csv'),
                                       delimiter=',')
                        ]
                        y_val += [
                            np.loadtxt("{0}.{1}".format(basepath, 'y_val.csv'),
                                       delimiter=',')
                        ]
                    x = np.concatenate(x)
                    y_pi = np.concatenate(y_pi)
                    y_val = np.concatenate(y_val)
                except Exception as exc:
                    return exc

                n_total = min(int(x.shape[0]), self.config.max_samplecount)
                n_validate = int(n_total * self.config.validation_split)
                n_train = n_total - n_validate
                self.logger.debug(
                    "loaded {} samples from files - n_train={}, n_validate={}".
                    format(int(x.shape[0]), n_train, n_validate))
                if n_train < self.config.batchsize:  # should one also check n_validate?
                    return Exception(
                        "Less training samples ({}) than training batchsize ({})"
                        .format(n_train, self.config.batchsize))

                ds_features = tf.data.Dataset.from_tensor_slices(x)
                ds_labels = tf.data.Dataset.from_tensor_slices((y_pi, y_val))
                dataset = tf.data.Dataset.zip((ds_features, ds_labels))
                dataset.shuffle(n_total)

                if n_validate > 0:
                    ds_validate = dataset.take(
                        n_validate).cache().repeat().batch(
                            self.config.batchsize)
                else:
                    ds_validate = None
                ds_train = dataset.skip(n_validate).take(
                    n_train).cache().repeat().batch(self.config.batchsize)
                return (n_train, ds_train, n_validate, ds_validate)

            tf_device = "/cpu:0"
            if self.config.use_gpu:
                if self.config.gpu_memorylimit == None:
                    tf_device = "/gpu:0"
                else:
                    self.logger.debug(
                        "creating new logical gpu with {}MB memory".format(
                            self.config.gpu_memorylimit))
                    tf.config.experimental.set_virtual_device_configuration(
                        tf.config.experimental.list_physical_devices('GPU')[0],
                        [
                            tf.config.experimental.VirtualDeviceConfiguration(
                                memory_limit=self.config.gpu_memorylimit)
                        ])
                    logical_gpus = tf.config.experimental.list_logical_devices(
                        'GPU')
                    self.logger.debug(
                        "available logical gpus: {}".format(logical_gpus))
                    tf_device = logical_gpus[0].name

            MODEL: tf.keras.Model
            self.logger.info("using: {}".format(tf_device))
            with tf.device(tf_device):
                MODEL = tf.keras.models.load_model(self.gympath.model_folder)
                model_iteration: int
                with open(self.gympath.modelinfo_file, 'r') as f:
                    model_iteration = gymdata.ModelInfo.from_json(
                        f.read()).iterationNr
                self.logger.info(
                    "tf.keras.model (iteration {}) loaded from {}".format(
                        model_iteration, self.gympath.model_folder))
                while not self.endevent.is_set():
                    if self.newmodelevent.is_set():
                        self.logger.debug(
                            "waiting for predictor to load new model")
                        sleep(1)
                    else:
                        load_datasets_result = try_load_datasets()
                        if isinstance(load_datasets_result, Exception):
                            self.logger.debug(
                                "failed loading dataset: {}".format(
                                    load_datasets_result))
                            self.logger.debug("sleeping 30 sec")
                            sleep(10)
                        else:
                            n_train, ds_train, n_validate, ds_validate = load_datasets_result
                            if n_train < self.config.min_samplecount:
                                self.logger.debug(
                                    "training dataset ({} entries) below min_samplecount ({}) - sleeping 30 sec"
                                    .format(n_train,
                                            self.config.min_samplecount))
                                sleep(30)
                            else:
                                self.logger.debug("loaded dataset: {}".format(
                                    ds_train.element_spec))
                                trainresult = MODEL.fit(
                                    ds_train,
                                    steps_per_epoch=1 +
                                    n_train // self.config.batchsize,
                                    validation_data=ds_validate,
                                    validation_steps=1 +
                                    n_validate // self.config.batchsize,
                                    epochs=self.config.epochs,
                                    callbacks=EpochDots(logger=self.logger,
                                                        report_every=500,
                                                        dot_every=10),
                                    verbose=0)
                                MODEL.save(self.gympath.model_folder)
                                model_iteration += 1
                                with open(self.gympath.modelinfo_file,
                                          'w+') as f:
                                    f.write(
                                        gymdata.ModelInfo(
                                            model_iteration).as_json(indent=0))
                                    self.logger.info(
                                        'updated model (iteration {})'.format(
                                            model_iteration))
                                self.newmodelevent.set()
                                self.logger.info('newmodelevent was set')

                                history_dump_file = os.path.join(
                                    self.gympath.trainhistories_folder,
                                    "{}.json".format(model_iteration))
                                with open(history_dump_file, 'w+') as f:
                                    gymdata.json.dump(trainresult.history, f)
                                self.logger.debug(
                                    'saved trainhistory (iteration {})'.format(
                                        model_iteration))

                                for metric in ["loss", "pi_loss", "v_loss"]:
                                    history_plotter.plot(
                                        {
                                            f"iteration {model_iteration}":
                                            trainresult
                                        },
                                        metric,
                                        savepath=
                                        f"{self.gympath.trainhistories_folder}/{model_iteration}_{metric}.png"
                                    )

                                MODEL.save_weights(os.path.join(
                                    self.gympath.weights_folder,
                                    "{}".format(model_iteration)),
                                                   save_format='h5')
                                self.logger.debug(
                                    'weights saved (iteration {})'.format(
                                        model_iteration))
            self.logger.info("endevent received - terminating")
        except Exception as exc:
            self.logger.error("exception: {}".format(exc))
            self.logger.info("signaling endevent and terminating")
            self.endevent.set()


class HistoryPlotter(object):
    """A class for plotting a named set of keras-histories.
    The class maintains colors for each key from plot to plot.

    Code is from:
    https://github.com/tensorflow/docs/blob/master/tools/tensorflow_docs/plots
    """
    _COLOR_CYCLE = plt.rcParams['axes.prop_cycle'].by_key()['color']

    def __init__(
        self,
        logger: logging.Logger,
        smoothing_std=None,
    ):
        self.color_table = {}
        self.smoothing_std = smoothing_std
        self.logger = logger

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

    def plot(self, histories, metric, figsize=(20, 10), savepath=None):
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
                color = self._COLOR_CYCLE[len(self.color_table) %
                                          len(self._COLOR_CYCLE)]
                self.color_table[name] = color

            train_value = history.history[metric]
            if self.smoothing_std is not None:
                train_value = self._smooth(train_value)
            plt.plot(history.epoch,
                     train_value,
                     color=color,
                     label=name.title() + ' Train')

            # added case check for missing validation set
            if 'val_' + metric in history.history.keys():
                val_value = history.history['val_' + metric]
                if self.smoothing_std is not None:
                    val_value = self._smooth(val_value)
                plt.plot(history.epoch,
                         val_value,
                         '--',
                         color=color,
                         label=name.title() + ' Val')

        plt.xlabel('Epochs')
        plt.ylabel(metric.replace('_', ' ').title())
        plt.legend()

        plt.xlim(
            [0, max([history.epoch[-1] for history in histories.values()])])
        y_min = min(
            [min(history.history[metric]) for history in histories.values()])
        y_max = max(
            [max(history.history[metric]) for history in histories.values()])
        if 'val_' + metric in history.history.keys():
            y_min_val = min([
                min(history.history['val_' + metric])
                for history in histories.values()
            ])
            y_max_val = max([
                max(history.history['val_' + metric])
                for history in histories.values()
            ])
            y_min = min(y_min, y_min_val)
            y_max = max(y_max, y_max_val)

        plt.ylim([(1 - 0.1 * np.sign(y_min)) * y_min,
                  (1 + 0.1 * np.sign(y_max)) * y_max])
        plt.grid(True)
        if savepath is not None:
            if os.path.exists(savepath):
                self.logger.warning(
                    "HistoryPlotter overriding existing plot '{}'".format(
                        savepath))
            directory = os.path.dirname(savepath)
            if not os.path.exists(directory):
                self.logger.debug(
                    "HistoryPlotter creating directory '{}'".format(directory))
                os.makedirs(directory)
            plt.savefig(savepath)
            self.logger.debug(
                "HistoryPlotter saved plot to '{}'".format(savepath))
