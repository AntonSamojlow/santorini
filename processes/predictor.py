import logging
import logging.handlers
import multiprocessing
import multiprocessing.queues as mpq
import os

import numpy as np

import gymdata


class Predictor(multiprocessing.Process):
    def __init__(self,
                 config: gymdata.PredictConfig,
                 gympath: gymdata.GymPath,
                 endevent: multiprocessing.Event,
                 newmodelevent: multiprocessing.Event,
                 logging_q: mpq.Queue,
                 request_q: mpq.Queue,
                 response_qs: dict,
                 modelsavefolder_override: str = None,
                 modelweightsfolder_override: str = None):

        super().__init__()
        self.logging_q = logging_q
        self.request_q = request_q
        self.response_qs = response_qs
        self.endevent = endevent
        self.newmodelevent = newmodelevent
        self.config = config
        self.gympath = gympath
        self.debug_stats = {'predict_batches': []}
        # two overrides for the evaluator - to be removed during refactoring:
        self.modelsavefolder_override = modelsavefolder_override
        self.modelweightsfolder_override = modelweightsfolder_override

    def run(self):
        # logging setup
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(self.config.logging.loglevel)
        logfilepath = os.path.join(self.gympath.log_folder,
                                   "{}[{}].log".format(self.name, self.pid))
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
                if self.modelsavefolder_override is None:
                    MODEL = tf.keras.models.load_model(
                        self.gympath.model_folder)
                    model_iteration: int
                    with open(self.gympath.modelinfo_file, 'r') as f:
                        model_iteration = gymdata.ModelInfo.from_json(
                            f.read()).iterationNr
                    self.logger.info(
                        "tf.keras.model (iteration {}) loaded from {}, awaiting requests..."
                        .format(model_iteration, self.gympath.model_folder))
                else:
                    MODEL = tf.keras.models.load_model(
                        self.modelsavefolder_override)
                    if self.modelweightsfolder_override is None:
                        self.logger.info(
                            f"tf.keras.model loaded from {self.modelsavefolder_override}, awaiting requests..."
                        )
                    else:
                        MODEL.load_weights(self.modelweightsfolder_override)
                        self.logger.info(
                            f"tf.keras.model loaded from {self.modelsavefolder_override} with weights from {self.modelweightsfolder_override}, awaiting requests..."
                        )

                while not self.endevent.is_set():
                    if self.newmodelevent.is_set():
                        if self.modelsavefolder_override is None:
                            MODEL = tf.keras.models.load_model(
                                self.gympath.model_folder)
                            with open(self.gympath.modelinfo_file) as f:
                                model_iteration = gymdata.ModelInfo.from_json(
                                    f.read()).iterationNr
                            self.newmodelevent.clear()
                            self.logger.info(
                                "tf.keras.model (iteration {}) reloaded from {}, awaiting requests..."
                                .format(model_iteration,
                                        self.gympath.model_folder))
                        else:
                            self.logger.error(
                                "changing models is not supported for the evaluator"
                            )

                    requests = []
                    try:
                        while len(requests) < self.config.batchsize:
                            request = self.request_q.get(
                                block=True,
                                timeout=self.config.trygetbatchsize_timeout)
                            requests += [request]
                    except mpq.Empty:
                        pass
                    if len(requests) > 0:
                        self.debug_stats['predict_batches'].append(
                            len(requests))
                        x = np.array([req[2] for req in requests])
                        self.logger.debug("predicting {} requests: {}".format(
                            len(x), x))
                        predictions = MODEL.predict_on_batch(x)
                        for i in range(len(requests)):
                            requester_process_name = requests[i][0]
                            requester_thread_id = requests[i][1]
                            prediction = [
                                np.array([predictions[0][i]]),
                                np.array([predictions[1][i]])
                            ]
                            # self.logger.debug("returning {} from requester {} to outputq".format(prediction, requesterid))
                            self.response_qs[requester_process_name].put(
                                [requester_thread_id, prediction])

                if len(self.debug_stats['predict_batches']) > 0:
                    self.logger.info(
                        "average predict batch size was {}".format(
                            sum(self.debug_stats['predict_batches']) /
                            len(self.debug_stats['predict_batches'])))
                self.logger.info("endevent received - terminating")
        except Exception as exc:
            self.logger.error("exception: {}".format(exc))
            self.logger.info("signaling endevent and terminating")
            self.endevent.set()
