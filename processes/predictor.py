import logging
import logging.handlers
import multiprocessing
import multiprocessing.queues as mpq

import numpy as np

class Predictor(multiprocessing.Process):
    def __init__(self, 
        logging_q : mpq.Queue, 
        request_q : mpq.Queue, 
        response_q : mpq.Queue,
        endevent : multiprocessing.Event, 
        newmodelevent : multiprocessing.Event,                            
        batchsize : int, 
        modelpath : str, 
        trygetbatchsize_timeout = 0.1):

        super().__init__()
        self.logging_q = logging_q
        self.request_q = request_q
        self.response_q = response_q
        self.endevent = endevent
        self.newmodelevent = newmodelevent
        self.batchsize = batchsize
        self.modelpath = modelpath
        self.trygetbatchsize_timeout = trygetbatchsize_timeout
        self.logger : logging.Logger
        self.debug_stats = {'predict_batches':[]}

    def run(self):
        qh = logging.handlers.QueueHandler(self.logging_q)
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(qh)
        self.logger.info("started and initialized logger")
       
        import tensorflow as tf        
        self.logger.info('imported TensorFlow {0}'.format(tf.__git_version__))
        tflogger = tf.get_logger()
        tflogger.addHandler(qh)
        tflogger.setLevel(logging.INFO)

        # expects a tuple (requesting_thread_name, x) where x is the data to be predicted
        # note: numpy.array(x) should be compatible with the neureal network         
        with tf.device('/gpu:0'):
            MODEL = tf.keras.models.load_model(self.modelpath)    
            self.logger.info("tf.keras.model loaded from {}, waiting for requests...".format(self.modelpath))
            while not self.endevent.is_set():
                if self.newmodelevent.is_set():
                    MODEL = tf.keras.models.load_model(self.modelpath)    
                    self.newmodelevent.clear()
                    self.logger.info("tf.keras.model reloaded from {}".format(self.modelpath))
                
                requests = []
                try:
                    while len(requests) < self.batchsize:
                        request = self.request_q.get(block=True, timeout=self.trygetbatchsize_timeout)
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
            


   