import logging
import multiprocessing
import threading
from time import time, sleep
from random import choice

import numpy as np

import santorini
from MCTSbenchmark.benchmark import Experiment, LookAhead, json


LOGGER = logging.getLogger(__name__)

class BatchPredictor(Experiment):
    def __init__(self, name, parameters, threads, batchsize):
        super().__init__(name=name)
        self.params = parameters
        self.threads = threads
        self.batchsize = batchsize

    @property
    def json_string(self):
        return json.dumps({
            'type': str(self.__class__), 
            'name':self.name, 
            'threads':self.threads,
            'batchsize':self.batchsize,
            'parameters':self.params.__dict__, 
            'stats_summary':self.stats.summary,
            'stats_rundetails':self.stats.__dict__
            }, indent=4)


    @staticmethod
    def _predictor_proc(params, batchsize, logginq, 
                                predictor_inq, predictor_outqs, endevent):
        # setup logging
        import logging
        import logging.handlers
        qh = logging.handlers.QueueHandler(logginq)
        logger = logging.getLogger(BatchPredictor._predictor_proc.__name__)
        logger.setLevel(logging.DEBUG)
        logger.addHandler(qh)
        logger.info("predictor started")

        import tensorflow as tf
        logger.info('imported TensorFlow {0}'.format(tf.__git_version__))
        tflogger = tf.get_logger()
        tflogger.addHandler(qh)
        tflogger.setLevel(logging.INFO)

        commands = [] # expcect tuples: (threadNr, nparray)
        with tf.device(params.tf_device):
            MODEL = tf.keras.models.load_model(params.modelpath)    
            logger.info("predictor loaded model and waits for requests")
            while True:
                while len(commands) < batchsize:
                    if predictor_inq.empty() and endevent.is_set():
                        if len(commands) > 0:
                            # endsignal set -> allow preditcion with less than batchsize 
                            break
                        else:
                            sleep(0.1)
                    elif predictor_inq.empty():
                        sleep(0.1)
                    else:
                        commands += [predictor_inq.get()]
                        if commands[-1] == 'TER':
                            while not predictor_inq.empty(): predictor_inq.get()
                            logger.info("received TER signal, drained queues and terminating")
                            return

                x = np.array([cmd[1] for cmd in commands])
                logger.debug("predicting {}".format(x))
                predictions = MODEL.predict_on_batch(x)                
                for i in range(len(commands)):
                    prediction = [np.array([predictions[0][i]]),
                        np.array([predictions[1][i]])]
                    logger.debug("returning {} to outputq Nr {}".format(prediction, commands[i][0]))
                    predictor_outqs[commands[i][0]].put(prediction)
                commands = []
                    

    @staticmethod
    def _worker_thread(params, SG, threadNr, 
                        predictor_inq, predictor_outq, resultq, endevent):
        t_start = time()
        # setup logging
        logger = logging.getLogger(threading.currentThread().getName())
        logger.setLevel(logging.DEBUG)
        logger.info("worker started")
        
        STATS = Experiment.Statistics()
       

        def predictor(nodename):
            predictor_inq.put((threadNr, SG.nparray_of(nodename)))
            while predictor_outq.empty():
                sleep(0.01)
                logger.debug("waiting on queue Nr {}".format(threadNr))
            return predictor_outq.get()


        STATS.startup_time_avg = time() - t_start
        while not endevent.is_set():    
            # resetting MCTS (data, graph, timers)
            MCTS = LookAhead(SG.root_copy(), predictor) 

            searchnodename = choice(list(SG.root_names))
            logger.debug("{} MCTS runs on Node {} ".format(params.searchcount, searchnodename))
            
            t0 = time()
            MCTS.run_counted(searchnodename, max_count=params.searchcount)            
            STATS.MCTS_runtimes += [time()-t0]
            STATS.NNpredictions += [MCTS.predictions]
            STATS.NNpredict_times += [MCTS.predicttime]
            STATS.addchildren_times += [MCTS.addchildrentime]

        resultq.put(STATS)
        logger.info("finished")

    
    @staticmethod
    def _distributed_logger(q):
        """Entry for a thread handling logging from the workers via queues."""
        logger = logging.getLogger(threading.currentThread().getName())
        logger.info('distributed_logger starting')
        while True:
            record = q.get()
            if record == 'TER':
                logger.info('distributed_logger received {} from logging queue'.format(record))
                break
            rootlogger = logging.getLogger()
            rootlogger.handle(record)


    def run(self):        
        LOGGER.info("BatchPredictor starting ({})".format(self.params.__dict__))
        
        if self.stats.startup_time_avg > 0:
            LOGGER.warning("Experiment was already executed, skipping request")
            return None 

        mplogger = multiprocessing.get_logger()
        mplogger.setLevel(logging.INFO)

        # prepare queues, endevent and workers
        endevent = multiprocessing.Event()        
        loggingq = multiprocessing.Queue()     
        resultq = multiprocessing.Queue()
        predictor_inq = multiprocessing.Queue()     
        predictor_outqs = [multiprocessing.Queue() for _ in range(self.threads)]        
     
        # run the loggingthread and the predictor process 
        threading.Thread(target=BatchPredictor._distributed_logger, args=(loggingq,)).start()

        predictor_proc = multiprocessing.Process(
            target=BatchPredictor._predictor_proc,
            args=(self.params, self.batchsize, loggingq, predictor_inq, predictor_outqs, endevent,))
        predictor_proc.start()                                  
        
        # run the MCTS searchthreads
        SG = santorini.SanGraph(santorini.Environment(dimension=self.params.dimension, 
                units_per_player=self.params.unitspp))   
        threads = [threading.Thread(target=BatchPredictor._worker_thread, 
            args=(self.params, SG, i, predictor_inq, predictor_outqs[i], resultq,endevent,))
            for i in range(self.threads)]
        
        for t in threads:
            t.start()
        
        LOGGER.info("run-thread pauses {} sec".format(self.params.max_runtime_sec))
        sleep(self.params.max_runtime_sec)      
        LOGGER.info("run-thread awakened and signals endevent")
        endevent.set()

        while resultq.qsize() < self.threads:
            sleep(1)
        
        LOGGER.info("resultq full, sending 'TER' signal to predictor") 
        predictor_inq.put('TER')       

        while not resultq.empty():
            workerstats = resultq.get()
            LOGGER.debug("received results: {}".format(workerstats.__dict__))        
            self.stats.startup_time_avg += workerstats.startup_time_avg / self.threads
            self.stats.MCTS_runtimes += workerstats.MCTS_runtimes
            self.stats.NNpredictions += workerstats.NNpredictions
            self.stats.NNpredict_times += workerstats.NNpredict_times
            self.stats.addchildren_times += workerstats.addchildren_times
        
        LOGGER.info('initiaing shutdown procedures')
        LOGGER.debug("waiting for threads to join...")
        for t in threads:
            t.join(10)
            if t.is_alive():
                LOGGER.warning("thread '{}' is still alive after join attempt - terminating it".format(t.name))
                t.terminate()
            else:
                LOGGER.debug("thread '{}' joined".format(t.name))

        LOGGER.debug("waiting for process '{}' to join...".format(predictor_proc.name))
        predictor_proc.join(10)
        if predictor_proc.is_alive():
            LOGGER.warning("process '{}' is still alive after join attempt - terminating it".format(predictor_proc.name))
            predictor_proc.terminate()
        else:
            LOGGER.debug("joined")
        
        LOGGER.info("sending 'TER' signal to logging queue thread...") 
        loggingq.put('TER')       
        
        LOGGER.info('closing queues')
        LOGGER.debug("calling close() on predictor input queue...")
        predictor_inq.close()
        LOGGER.debug("calling join_thread() on predictor input queue...")
        predictor_inq.join_thread()

        LOGGER.debug("calling close() on predictor outout queues...")
        for q in predictor_outqs: q.close()
        LOGGER.debug("calling join_thread() on predictor outout queues...")
        for q in predictor_outqs: q.join_thread()

        LOGGER.debug("calling close() on result queue...")
        resultq.close()
        LOGGER.debug("calling join_thread() on result queue...")
        resultq.join_thread()

        LOGGER.debug("calling close() on logging queue...")
        loggingq.close()
        LOGGER.debug("calling join_thread() on logging queue...")
        loggingq.join_thread()
        LOGGER.info("BatchPredictor finished")
