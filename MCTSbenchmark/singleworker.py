import logging
import multiprocessing
import threading
from time import time, sleep
from random import choice

import numpy as np

import santorini
from MCTSbenchmark.benchmark import Experiment, LookAhead


LOGGER = logging.getLogger(__name__)

class SingleWorker(Experiment):
    def __init__(self, name, parameters):
        super().__init__(name=name)
        self.params = parameters

    @staticmethod
    def _worker_proc(params, loggingq, resultq):
        t_start = time()
        # setup logging
        import logging
        import logging.handlers
        qh = logging.handlers.QueueHandler(loggingq)
        logger = logging.getLogger(SingleWorker._worker_proc.__name__)
        logger.setLevel(logging.DEBUG)
        logger.addHandler(qh)
        logger.info("worker started")     
        
        STATS = Experiment.Statistics()
        ENV = santorini.Environment(dimension=params.dimension, 
                units_per_player=params.unitspp)
        SG = santorini.SanGraph(ENV)        
        ROOTS = [SG.nodes[r] for r in SG.root_names]

        import tensorflow as tf
        logger.info('imported TensorFlow {0}'.format(tf.__git_version__))
        tflogger = tf.get_logger()
        tflogger.addHandler(qh)
        tflogger.setLevel(logging.INFO)

        with tf.device(params.tf_device):
            MODEL = tf.keras.models.load_model(params.modelpath)      
            predictor = lambda nodename: MODEL.predict(np.array([SG.nparray_of(nodename)]))

            t_searchstart = time()
            STATS.startup_time_avg = t_searchstart - t_start
            while(time()- t_searchstart < params.max_runtime_sec):   

                # resetting MCTS (data, graph, timers)
                MCTS = LookAhead(SG.root_copy(), predictor) 

                searchnodename = choice(ROOTS).name    
                logger.debug("{} MCTS runs on Node {} ".format(params.searchcount, searchnodename))

                t0 = time()
                MCTS.run_counted(searchnodename, max_count=params.searchcount)            
                STATS.MCTS_runtimes += [time()-t0]
                STATS.NNpredictions += [MCTS.predictions]
                STATS.NNpredict_times += [MCTS.predicttime]
                STATS.addchildren_times += [MCTS.addchildrentime]

        resultq.put(STATS)

    
    @staticmethod
    def _distributed_logger(q):
        """Entry for a thread handling logging from the workers via queues."""
        while True:
            record = q.get()
            if record == 'TER':
                LOGGER.debug('distributed_logger received {} from logging queue'.format(record))
                break
            rootlogger = logging.getLogger()
            rootlogger.handle(record)


    def run(self):        
        LOGGER.info("singleWorker starting ({})".format(self.params.__dict__))
      
        if self.stats.startup_time_avg > 0:
            LOGGER.warning("Experiment was already executed, skipping request")
            return None 

        mplogger = multiprocessing.get_logger()
        mplogger.setLevel(logging.INFO)

        # prepare queues and workers        
        resultq = multiprocessing.Queue()
        loggingq = multiprocessing.Queue()     

        # run the logginthread and the worker process 
        threading.Thread(target=SingleWorker._distributed_logger, args=(loggingq,)).start()
        process = multiprocessing.Process(target=SingleWorker._worker_proc,
                                             args=(self.params, loggingq, resultq,))
        process.start()
                                             

        while resultq.empty():
            sleep(1)      
        self.stats = resultq.get()
        LOGGER.debug("received results: {}".format(self.stats.__dict__))
        
        LOGGER.debug("waiting for process '{}' to join...".format(process.name))
        process.join(10)
        if process.is_alive():
            LOGGER.warning("process '{}' is still alive after join attempt - terminating it".format(process.name))
            process.terminate()
        else:
            LOGGER.debug("joined")
        
        LOGGER.debug("sending 'TER' signal to logging queue thread...") 

        loggingq.put('TER')       
        LOGGER.debug("calling close() on result queue...")
        resultq.close()
        LOGGER.debug("calling join_thread() on result queue...")
        resultq.join_thread()
        LOGGER.debug("calling close() on logging queue...")
        loggingq.close()
        LOGGER.debug("calling join_thread() on logging queue...")
        loggingq.join_thread()
        LOGGER.info("singleWorker finished")

