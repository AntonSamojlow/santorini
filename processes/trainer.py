import logging
import logging.handlers
import multiprocessing
import multiprocessing.queues as mpq

from time import sleep


class TrainConfig():
    def __init__(self, batchsize, epochs):
        self.batchsize = batchsize
        self.epochs = epochs

class Trainer(multiprocessing.Process):
    def __init__(self, 
    modelpath : str, 
    config : TrainConfig, 
    endevent : multiprocessing.Event, 
    newmodelevent : multiprocessing.Event, 
    logging_q : mpq.Queue):
        super().__init__()
        self.modelpath = modelpath
        self.config = config
        self.endevent = endevent
        self.newmodelevent = newmodelevent
        self.logging_q = logging_q
        self.logger : logging.Logger
    
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

        while not self.endevent.is_set(): 
            sleep(1)
            pass   
            #   load new training batch
            #   train model
            #   save trained model
            #   ??evaluate??
        self.logger.info("endevent received - terminating")

