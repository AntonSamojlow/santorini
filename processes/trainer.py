import logging
import logging.handlers
import multiprocessing
import multiprocessing.queues as mpq
import os
from time import sleep

import numpy as np

import gymdata

class Trainer(multiprocessing.Process):
    def __init__(self, 
    config: gymdata.TrainConfig,
    gympath: gymdata.GymPath,  
    endevent : multiprocessing.Event, 
    newmodelevent : multiprocessing.Event, 
    logging_q : mpq.Queue):
        super().__init__()
        self.config = config
        self.endevent = endevent
        self.newmodelevent = newmodelevent
        self.logging_q = logging_q
        self.gympath = gympath
        self.logger : logging.Logger
       
    
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
        
        class EpochDots(tf.keras.callbacks.Callback):
            """A simple callback that prints a "." every epoch, with occasional reports.
            Args:
            report_every: How many epochs between full reports
            dot_every: How many epochs between dots.
            """

            def __init__(self,  logger:logging.Logger, report_every=100, dot_every=1,):
                self.report_every = report_every
                self.dot_every = dot_every
                self.logger = logger

            def on_epoch_end(self, epoch, logs):
                if epoch % self.report_every == 0:
                    print()
                    msg = 'Epoch: {:d}, train loss: {:0.5f} (pi {:0.5f}, v {:0.5f})'.format(epoch,
                                logs['loss'],
                                logs['pi_loss'],
                                logs['v_loss'],)
                    print(msg, end='')
                    self.logger.info(msg)
                    # print(', val loss: {:0.5f} (pi {:0.5f}, v {:0.5f})'
                    #     .format(logs['val_loss'],
                    #             logs['val_pi_loss'],
                    #             logs['val_v_loss']), end='')
                    print()
                if epoch % self.dot_every == 0:
                    print('.', end='', flush=True)   

        def load_dataset() -> tf.data.Dataset:
            try:
                gamerecordfolders = [f.path for f in os.scandir(self.gympath.gamerecordpool_folder) if f.is_dir() ]
                folder = gamerecordfolders[-1]
                x = np.loadtxt(os.path.join(folder, '0.x.csv'), delimiter=',')
                y_pi = np.loadtxt(os.path.join(folder, '0.y_pi.csv'), delimiter=',')
                y_val = np.loadtxt(os.path.join(folder, '0.y_val.csv'), delimiter=',')
            except Exception as exc:
                return exc
            ds_features = tf.data.Dataset.from_tensor_slices(x)
            ds_labels = tf.data.Dataset.from_tensor_slices((y_pi, y_val))
            dataset = tf.data.Dataset.zip((ds_features, ds_labels)).cache().repeat().batch(self.config.batchsize)
            dataset.shuffle(int(x.shape[0]))            
            return dataset

        tf.config.experimental.set_virtual_device_configuration(
            tf.config.experimental.list_physical_devices('GPU')[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])

        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        self.logger.info("available logical gpus: {}".format( logical_gpus))      
        self.logger.info("using: {}".format(logical_gpus[0].name))        
        with tf.device(logical_gpus[0].name):
            MODEL = tf.keras.models.load_model(self.gympath.currentmodel_folder)    
            self.logger.info("tf.keras.model loaded from {}".format(self.gympath.currentmodel_folder))
            while not self.endevent.is_set(): 
                dataset = load_dataset()
                if isinstance(dataset, Exception):
                    self.logger.info("failed loading dataset: {}".format(dataset))
                    self.logger.info("retrying in 30 sec...".format(dataset))
                    sleep(30)
                else:
                    self.logger.info("loaded dataset: {}".format(dataset.element_spec))
                    trainresult = MODEL.fit(dataset,
                                     steps_per_epoch=10,
                                     epochs=1000,
                                     callbacks=EpochDots(logger=self.logger,
                                         report_every=500, dot_every=10),
                                     verbose=0)
        
        self.logger.info("endevent received - terminating")


    

      

