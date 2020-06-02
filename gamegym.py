import logging
import logging.handlers
import multiprocessing
import threading
from queue import Empty, Queue
from os import path, makedirs
from math import sqrt
from random import choice, choices
from time import time, sleep
from datetime import timedelta

import numpy as np

from gamegraph import GameGraph

LOGGER = logging.getLogger(__name__)

def getLogger():
    return LOGGER

class SelfPlayConfig():
    def __init__(self, 
            searchthreadcount : int, 
            searchcount : int, 
            predictionbatchsize : int, 
            exploration_const = 2.0, 
            use_virtualloss = False, 
            sleeptime_blocked_select = 1,
            temperature = 1):
        self.mcts_searchcount = searchcount
        self.mcts_searchthreadcount = searchthreadcount
        self.mcts_exploration_const = exploration_const
        self.mcts_use_virtualloss = use_virtualloss
        self.mcts_sleeptime_blocked_select = sleeptime_blocked_select
        self.predictionbatchsize = predictionbatchsize
        self.temperature = temperature

class TrainConfig():
    def __init__(self, batchsize, epochs):
        self.batchsize = batchsize
        self.epochs = epochs

class GymSettings():
    def __init__(self, mcts_config : SelfPlayConfig, train_config : TrainConfig):
        self.mcts_config = mcts_config
        self.train_config = train_config

class GameRecordPool():
    def __init__(self):
        self.records = []
        

class GameGym():
    def __init__(self, graph, settings, sessionfolder, intialmodelpath = None):
        self.graph = graph
        if not isinstance(settings, GymSettings):
            LOGGER.error("settings need to be of type {}".format(GymSettings.__class__))
        self.settings = settings
        self._sessionfolder = sessionfolder
        if intialmodelpath is None:
            self._modelpath = "{}/models/current/".format(self.sessionfolder)
        else:
            self._modelpath = intialmodelpath

    @property
    def sessionfolder(self):
        return self.sessionfolder

    @property
    def modelpath(self):
        return self._modelpath

    def resume(self):
        # 1. Start - initialize ressources         
        # # events         
        endevent = multiprocessing.Event()
        newmodelevent = multiprocessing.Event()
        # queues and loggingthread
        logging_q = multiprocessing.Queue()
        predict_request_q = multiprocessing.Queue()
        predict_response_q = multiprocessing.Queue()
        # start loggingthread
        loggingthread = threading.Thread(target=self.__class__._distributed_logger,
            args=(logging_q,), name="logging_thread")
        loggingthread.start()
        while not loggingthread.is_alive():
            sleep(0.1)
        LOGGER.info("loggingthread alive")    
        # processes        
        train_proc = TrainProcess(self.modelpath, self.settings.train_config, 
                                            endevent, newmodelevent, logging_q)
        predictor_proc = PredictorProcess(logging_q, predict_request_q, predict_response_q,
                    endevent, newmodelevent, 10, self.modelpath)      
        selfplay_proc = SelfPlayProcess(self.graph, self.settings.mcts_config, endevent, 
                    logging_q, predict_request_q, predict_response_q)

        # 2. Run...
        # start threads and processes
        train_proc.start()
        predictor_proc.start()
        selfplay_proc.start()
        
        threading.Thread(
            target=self._keyboard_reader,
            args=(endevent, ),
            name="keyboard_reader").start()

        while not endevent.is_set():
            sleep(1)

        # 3. Endphase - cleaning up ressources
        LOGGER.info("endevent was set")
        LOGGER.info("sleeping 10 sec...")
        sleep(10)
        logging_q.put('TER')
        sleep(5)

        LOGGER.info("ending")
        return
    
    @staticmethod
    def _keyboard_reader(endevent : threading.Event):
        """Entry for a thread waiting for keyboard input."""
        threadname = threading.currentThread().getName()
        logger = logging.getLogger(threadname)
        logger.setLevel(logging.DEBUG)
        logger.info('keyboard_watcher started')
        while True:
            text = input()
            logger.info("registered input {}".format(text))
            if text == "exit":
                endevent.set()
                return
           
    @staticmethod
    def _distributed_logger(logging_q : Queue):
        """Entry for a thread handling logging from the workers via queues."""
        threadname = threading.currentThread().getName()
        logger = logging.getLogger(threadname)
        logger.info('distributed_logger starting')
        while True:
            record = logging_q.get(block=True)
            if record == 'TER':
                logger.info('received {} from logging queue'.format(record))
                break
            rootlogger = logging.getLogger()
            rootlogger.handle(record)
  

class SelfPlayProcess(multiprocessing.Process):
    def __init__(self,            
            graph : GameGraph, 
            selfplayconfig : SelfPlayConfig, 
            endevent : multiprocessing.Event, 
            logging_q : Queue, 
            predict_request_q : Queue, 
            predict_response_q : Queue):
        super().__init__()
        graph.truncate_to_roots()
        self._mcts_graph = graph.copy()
        self._mcts_searchtable = {}
        self.config = selfplayconfig
        self.endevent = endevent
        self.logging_q = logging_q
        self.predict_request_q = predict_request_q
        self.predict_response_q = predict_response_q
        self.recordpool = GameRecordPool()
        self.logger : logging.Logger
    
    @property
    def mcts_graph(self):
        return self._mcts_graph

    @property
    def mcts_searchtable(self):
        return self._mcts_searchtable

    def run(self):
        qh = logging.handlers.QueueHandler(self.logging_q)
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(qh)
        self.logger.info("started and initialized logger")       

        # ressources shared among threads:
        self.mcts_request_q = Queue()
        self.mcts_response_q = Queue()
        self.mcts_graph_lock = threading.Lock()
        self.terminateevent = threading.Event()

        # search threads: assign their ressources and start
        self.search_threads = [] 
        self.thread_prediction_locks = {}   
        self.thread_predict_response_qs = {}   
        for _ in range(self.config.mcts_searchthreadcount):
            thread_prediction_lock = threading.Lock()
            thread_predict_response_q = Queue()
            thread = threading.Thread(target=self.__class__._MCTSsearcher,
                args=(self.mcts_graph, 
                self.mcts_searchtable, 
                self.config,
                self.terminateevent,
                self.mcts_request_q,
                self.mcts_response_q,
                self.mcts_graph_lock,
                thread_prediction_lock,
                self.predict_request_q,
                thread_predict_response_q,
                self.logging_q,))            
            thread.start()
            while thread.native_id is None: # wait until thread has started
                sleep(0.1)
                pass
            self.search_threads += [thread]
            self.thread_prediction_locks[thread.native_id] = thread_prediction_lock
            self.thread_predict_response_qs[thread.native_id] = thread_predict_response_q    
        
        while not self.endevent.is_set():
            self.recordpool.records += self._selfplay()
            print("current records: {}".format(self.recordpool.records))

        self.terminateevent.set()
        # unlock threads still waiting for predictions and insert fake prediction 
        for t in self.search_threads:
            if t.is_alive() and self.thread_prediction_locks[t.native_id].locked():               
                try:
                    self.thread_predict_response_qs[t.native_id].put([[] , 0.0])
                    self.thread_prediction_locks[t.native_id].release()
                except RuntimeError:
                    logging.warning("failed to release predict_lock for thread {}".format(t.native_id))
                    pass
        
        for t in self.search_threads:
            t.join(timeout=5)

        self.logger.info("endevent received - terminating ({} workers still alive)".format(
            sum([int(t.is_alive()) for t in self.search_threads])))


    def _selfplay(self) -> list:
        self.mcts_graph.truncate_to_roots()
        vertex = choice(tuple(self.mcts_graph.roots))
        self.mcts_searchtable.clear()
        gamelog = []
        while self.mcts_graph.open_at(vertex) or not self.mcts_graph.terminal_at(vertex):  # play a game
            self._MCTSrun(vertex)
            if self.endevent.is_set():
                self.logger.debug("endevent received - cancelling selfplay")                
                return []
            prob_distr = [N**(1/self.config.temperature) for N in self.mcts_searchtable[vertex]['N']] 
            gamelog.append([vertex, prob_distr, None])
            vertex = choices(self.mcts_graph.children_at(vertex), weights=prob_distr)[0]
        for i in range(0, len(gamelog)):  # propagate the result
            sign = 1 - 2*int(i % 2 == len(gamelog) % 2)
            gamelog[i][2] = sign*self.mcts_graph.score_at(vertex)
        return gamelog

    def _MCTSrun(self, vertex):
        self.logger.debug("MCTSrun called on vertex {}".format(vertex))  
        for _ in range(0, self.config.mcts_searchcount):
            self.mcts_request_q.put(vertex)
        replies = 0
        while replies < self.config.mcts_searchcount:
            if self.endevent.is_set():
                self.logger.debug("endevent received - exiting _MCTSrun")               
                return
            if not self.predict_response_q.empty():
                message = self.predict_response_q.get()
                thread_id, prediction = message
                self.thread_predict_response_qs[thread_id].put([tuple(prediction[0][0]),float(prediction[1])])
                self.thread_prediction_locks[thread_id].release()
            if not self.mcts_response_q.empty():
                path = self.mcts_response_q.get()
                with self.mcts_graph_lock:
                    self._update(path)               
                replies += 1
        return

    def _update(self, path):
        end_val = self.mcts_searchtable[path[-1]]['Q']     
        for i in range(0, len(path) - 1): 
            sign = 1 - 2*int(i % 2 == path.__len__() % 2)
            j = self.mcts_graph.children_at(path[i]).index(path[i+1])
            self.mcts_searchtable[path[i]]['N'][j] += 1
            self.mcts_searchtable[path[i]]['Q'] +=\
                (sign*end_val - self.mcts_searchtable[path[i]]['Q'])\
                /sum(self.mcts_searchtable[path[i]]['N'])
        return

    @staticmethod
    def _MCTSsearcher(
            graph : GameGraph, 
            searchtable : dict, 
            config : SelfPlayConfig,
            terminateevent : threading.Event,
            request_q : Queue, 
            response_q : Queue, 
            graphlock : threading.Lock, 
            prediction_lock : threading.Lock, 
            predict_request_q : Queue, 
            predict_response_q : Queue, 
            logging_q : Queue):       
        # setup logging
        thread_id = threading.get_ident()
        qh = logging.handlers.QueueHandler(logging_q)
        logger = logging.getLogger("Thread-{}".format(thread_id))
        logger.setLevel(logging.DEBUG)
        logger.addHandler(qh)
        logger.info("MCTSsearcher started")

        def predictor(vertex):
            prediction_lock.acquire()
            predict_request_q.put((thread_id, graph.numpify(vertex)))
            # waiting for predictor_proc to release the lock
            # alternative: check against lock.locked() ?
            with prediction_lock: 
                # logger.debug("acquired prediction lock, reading from queue...")
                return predict_response_q.get()                 

        def select(vertex):
            # logger.info("select({})".format(vertex))
            if graph.open_at(vertex) or graph.terminal_at(vertex):
                return [vertex]
            while vertex not in searchtable:                
                logger.warning( ("{} not in searchtable: waiting for other threads to finish prediction. " +
                    "Consider increasing virtualloss. Thread sleeps for {} seconds").format(
                        vertex,config.mcts_sleeptime_blocked_select))
                sleep(config.mcts_sleeptime_blocked_select)                
            # check event here since another blocking thread may have been given 'fake' predictions
            if terminateevent.is_set():
                return [vertex]
            
            visits = searchtable[vertex]['N']
            def U(child):
                try:
                    c_val = searchtable[child]['Q']
                except KeyError:
                    c_val = 0
                j = graph.children_at(vertex).index(child)
                prob = searchtable[vertex]['P'][j]
                return c_val - config.mcts_exploration_const*prob*sqrt(sum(visits))/(1+visits[j])
            return [vertex] + select(min(graph.children_at(vertex), key=U))

        def expand(vertex):
            # logger.info("expand({})".format(vertex))
            if graph.open_at(vertex):     
                # block all other threads until graph has been expanded           
                with graphlock: 
                    # logger.debug("acquired graphlock for expanding at {}".format(vertex))
                    if graph.open_at(vertex): # check again before expanding  
                        graph.expand_at(vertex)    
                        # logger.debug("expanded at {}".format(vertex))

                # initialize table statistics
                if graph.terminal_at(vertex): 
                    tableentry = {
                        'N': [], 
                        'Q': graph.score_at(vertex), 
                        'P': []}
                else:
                    prediction = predictor(vertex)
                    tableentry = {
                            'N': [0 for c in graph.children_at(vertex)],
                            'Q': prediction[1],
                            'P': prediction[0][:len(graph.children_at(vertex))]} 
                if vertex not in searchtable.keys():
                    searchtable[vertex] = tableentry                 
            return

        while not terminateevent.is_set():
            try:
                request = request_q.get(block=True, timeout=0.1)                
                path = select(request)    

                expand(path[-1])
                response_q.put(path)
            except Empty:
                pass
        logger.info("terminate event received - terminating")

class TrainProcess(multiprocessing.Process):
    def __init__(self, 
    modelpath, train_config, endevent, newmodelevent, 
    logging_q):
        super().__init__()
        self.modelpath = modelpath
        self.train_config = train_config
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


class PredictorProcess(multiprocessing.Process):
    def __init__(self, 
        logging_q : Queue, 
        request_q : Queue, 
        response_q : Queue,
        endevent : multiprocessing.Event, 
        newmodelevent : multiprocessing.Event,                            
        batchsize : int, 
        modelpath : str, 
        trygetbatchsize_timeout = 0.1):

        super().__init__()
        self.logging_q=logging_q
        self.request_q = request_q
        self.response_q = response_q
        self.endevent = endevent
        self.newmodelevent = newmodelevent
        self.batchsize = batchsize
        self.modelpath = modelpath
        self.trygetbatchsize_timeout = trygetbatchsize_timeout
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
                except Empty:
                    pass
                if len(requests) > 0:
                    x = np.array([cmd[1] for cmd in requests])
                    self.logger.debug("predicting {} requests: {}".format(len(x), x))
                    predictions = MODEL.predict_on_batch(x)                
                    for i in range(len(requests)):
                        requesterid = requests[i][0]
                        prediction = [np.array([predictions[0][i]]),
                            np.array([predictions[1][i]])]
                        # self.logger.debug("returning {} from requester {} to outputq".format(prediction, requesterid))
                        self.response_q.put([requesterid, prediction])
            
            self.logger.info("endevent received - terminating")


   