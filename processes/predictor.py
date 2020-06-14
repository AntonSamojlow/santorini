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
        endevent : multiprocessing.Event, 
        newmodelevent : multiprocessing.Event,
        logging_q : mpq.Queue, 
        request_q : mpq.Queue, 
        response_q : mpq.Queue):
        
        super().__init__()
        self.logging_q = logging_q
        self.request_q = request_q
        self.response_q = response_q
        self.endevent = endevent
        self.newmodelevent = newmodelevent
        self.config = config
        self.gympath = gympath               
        self.debug_stats = {'predict_batches':[]}

    def run(self):
        # logging setup
        self.logger = logging.getLogger(type(self).__name__)
        self.logger.setLevel(self.config.logging.loglevel)
        logfilepath = os.path.join(self.gympath.log_folder,"{}.log".format(type(self).__name__))
        self.config.logging.addRotatingFileHandler(self.logger, logfilepath)
        qh = logging.handlers.QueueHandler(self.logging_q)
        qh.setLevel(logging.INFO)
        self.logger.addHandler(qh)
        self.logger.info("started and initialized logger")
       
        import tensorflow as tf        
        self.logger.info('imported TensorFlow {0}'.format(tf.__git_version__))
        tflogger = tf.get_logger()
        tflogger.addHandler(qh)
        tflogger.setLevel(logging.INFO)   
     
        tf.config.experimental.set_virtual_device_configuration(
            tf.config.experimental.list_physical_devices('GPU')[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])

        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        self.logger.info("available logical gpus: {}".format( logical_gpus))      
        self.logger.info("using: {}".format(logical_gpus[0].name))
        with tf.device(logical_gpus[0].name):
        # with tf.device("/cpu:0"):
            MODEL = tf.keras.models.load_model(self.gympath.currentmodel_folder)    
            self.logger.info("tf.keras.model loaded from {}, waiting for requests...".format(self.gympath.currentmodel_folder))
            while not self.endevent.is_set():
                if self.newmodelevent.is_set():
                    MODEL = tf.keras.models.load_model(self.gympath.currentmodel_folder)    
                    self.newmodelevent.clear()
                    self.logger.info("tf.keras.model reloaded from {}".format(self.gympath.currentmodel_folder))
             
                requests = []
                try:
                    while len(requests) < self.config.batchsize:
                        request = self.request_q.get(block=True, timeout=self.config.trygetbatchsize_timeout)
                        requests += [request]
                except mpq.Empty:
                    pass
                if len(requests) > 0:
                    self.debug_stats['predict_batches'].append(len(requests))
                    x = np.array([cmd[1] for cmd in requests])
                    self.logger.debug("predicting {} requests: {}".format(len(x), x))
                    predictions = MODEL.predict_on_batch(x)                
                    for i in range(len(requests)):
                        requesterid = requests[i][0]
                        prediction = [np.array([predictions[0][i]]),
                            np.array([predictions[1][i]])]
                        # self.logger.debug("returning {} from requester {} to outputq".format(prediction, requesterid))
                        self.response_q.put([requesterid, prediction])
            
            if len(self.debug_stats['predict_batches']) > 0:
                self.logger.info("average predict batch size was {}".format(
                    sum(self.debug_stats['predict_batches'])/len(self.debug_stats['predict_batches'])))
            self.logger.info("endevent received - terminating")
            


   