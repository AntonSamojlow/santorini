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
        rfh = self.config.logging.getRotatingFileHandler(logfilepath)
        qh = logging.handlers.QueueHandler(self.logging_q)
        qh.setLevel(logging.INFO)
        self.logger.addHandler(rfh)
        self.logger.addHandler(qh)
        self.logger.info("started and initialized logger")

        import tensorflow as tf
        self.logger.info('imported TensorFlow {0}'.format(tf.__git_version__))
        tflogger = tf.get_logger()
        tflogger.addHandler(rfh)
        tflogger.setLevel(logging.WARNING)   
        
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

        def try_load_dataset() -> tf.data.Dataset:
            modeliteration: int
            with open(self.gympath.currentmodelinfo_file, 'r') as f:
                modeliteration = gymdata.ModelInfo.from_json(f.read()).iterationNr 
            activefolders = [f for f in os.scandir(self.gympath.gamerecordpool_folder) 
                if f.is_dir() and int(f.name) + self.config.maxsampleage >= modeliteration]
            basepaths = []
            for folder in activefolders:
                basepaths += [f.path.replace(".x.csv", "")  for f in os.scandir(folder.path) if f.is_file() and f.name.endswith(".x.csv")]
            if len(basepaths) == 0:
                return Exception("No fresh samples found in {}".format(self.gympath.gamerecordpool_folder))
        
            x, y_pi, y_val = [], [], []
            try:   
                for basepath in basepaths:
                    print(basepath)
                    x += [np.loadtxt("{0}.{1}".format(basepath, 'x.csv'), delimiter=',')]
                    y_pi += [np.loadtxt("{0}.{1}".format(basepath, 'y_pi.csv'), delimiter=',')]
                    y_val += [np.loadtxt("{0}.{1}".format(basepath, 'y_val.csv'), delimiter=',')]
                x=np.concatenate(x)
                y_pi=np.concatenate(y_pi)
                y_val=np.concatenate(y_val)
            except Exception as exc:
                return exc
            ds_features = tf.data.Dataset.from_tensor_slices(x)
            ds_labels = tf.data.Dataset.from_tensor_slices((y_pi, y_val))
            dataset = tf.data.Dataset.zip((ds_features, ds_labels)).cache().repeat().batch(self.config.batchsize)
            dataset.shuffle(int(x.shape[0]))            
            return dataset

        tf_device = "/cpu:0"
        if self.config.use_gpu:
            if self.config.gpu_memorylimit == None:
                tf_device = "/gpu:0"
            else:
                self.logger.debug("creating new logical gpu with {}MB memory".format(self.config.gpu_memorylimit))
                tf.config.experimental.set_virtual_device_configuration(
                    tf.config.experimental.list_physical_devices('GPU')[0],
                    [tf.config.experimental.VirtualDeviceConfiguration(
                        memory_limit=self.config.gpu_memorylimit)])
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                self.logger.debug("available logical gpus: {}".format( logical_gpus))      
                tf_device = logical_gpus[0].name
        
        MODEL: tf.keras.Model 
        self.logger.info("using: {}".format(tf_device))
        with tf.device(tf_device):            
            MODEL = tf.keras.models.load_model(self.gympath.currentmodel_folder)    
            model_iteration: int
            with open(self.gympath.currentmodelinfo_file, 'r') as f:
                model_iteration = gymdata.ModelInfo.from_json(f.read()).iterationNr 
            self.logger.info("tf.keras.model (iteration {}) loaded from {}".format(model_iteration, self.gympath.currentmodel_folder))
            while not self.endevent.is_set():
                if self.newmodelevent.is_set():
                    self.logger.debug("waiting for predictor to load new model")
                    sleep(1)
                else:
                    dataset = try_load_dataset()
                    if isinstance(dataset, Exception):
                        self.logger.debug("failed loading dataset: {}".format(dataset))
                        self.logger.debug("retrying in 30 sec")
                        sleep(30)
                    else:
                        self.logger.debug("loaded dataset: {}".format(dataset.element_spec))
                        trainresult = MODEL.fit(dataset,
                                        steps_per_epoch=10,
                                        epochs=1000,
                                        callbacks=EpochDots(logger=self.logger,
                                            report_every=500, dot_every=10),
                                        verbose=0)
                        MODEL.save(self.gympath.currentmodel_folder)
                        model_iteration+=1
                        with open(self.gympath.currentmodelinfo_file, 'w+') as f:
                            f.write(gymdata.ModelInfo(model_iteration).as_json(indent=0))
                            self.logger.info('updated model (iteration {})'.format(model_iteration))
                        self.newmodelevent.set()
                        MODEL.save_weights(os.path.join(self.gympath.weights_folder,"{}".format(model_iteration)), save_format='h5')
        self.logger.info("endevent received - terminating")


    

      

