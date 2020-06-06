import logging
import logging.handlers
import multiprocessing
import multiprocessing.queues as mpq
import threading
import queue

from math import sqrt
from time import time, sleep
from random import choice, choices

from namedqueues import NamedQueue
from gamegraph import GameGraph

class SelfPlayConfig():
    def __init__(self, 
            searchthreadcount : int, 
            searchcount : int, 
            predictionbatchsize : int, 
            exploration_const = 2.0, 
            virtualloss = 0.5, 
            sleeptime_blocked_select = 0.1,
            temperature = 1):
        self.mcts_searchcount = searchcount
        self.mcts_searchthreadcount = searchthreadcount
        self.mcts_exploration_const = exploration_const
        self.mcts_virtualloss = virtualloss
        self.mcts_sleeptime_blocked_select = sleeptime_blocked_select
        self.predictionbatchsize = predictionbatchsize
        self.temperature = temperature

class GameRecordPool():
    def __init__(self):
        self.records = []

class Selfplayer(multiprocessing.Process):
    def __init__(self,
            graph : GameGraph, 
            selfplayconfig : SelfPlayConfig, 
            endevent : multiprocessing.Event, 
            logging_q : mpq.Queue, 
            predict_request_q : mpq.Queue, 
            predict_response_q : mpq.Queue):
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
        self.debug_stats = []

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
        self.mcts_request_q = NamedQueue("mcts_request_q")
        self.mcts_response_q = NamedQueue("mcts_response_q")
        self.mcts_graph_lock = threading.Lock()
        self.terminateevent = threading.Event()
        queues = [self.mcts_request_q, self.mcts_response_q]    

        # search threads: assign their ressources and start
        self.search_threads = [] 
        self.thread_prediction_locks = {}   
        self.thread_predict_response_qs = {}   
        for _ in range(self.config.mcts_searchthreadcount):
            thread_prediction_lock = threading.Lock()
            thread_predict_response_q = NamedQueue("")
            thread = threading.Thread(target=_MCTSsearcher,
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
                self.logging_q,
                self.debug_stats))            
            thread.start()
            while thread.native_id is None: # wait until thread has started
                sleep(0.1)
                pass
            self.search_threads += [thread]
            self.thread_prediction_locks[thread.native_id] = thread_prediction_lock
            thread_predict_response_q.name = "[{}]_predict_response_q".format(thread.native_id)
            self.thread_predict_response_qs[thread.native_id] = thread_predict_response_q    
        
        queues += list(self.thread_predict_response_qs.values())

        while not self.endevent.is_set():
            newrecords = self._selfplay()
            self.logger.info("adding new playrecords: {}".format(newrecords))
            self.recordpool.records += newrecords

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
            t.join(5)
            if t.is_alive():
                self.logger.warning("thread {} has not ended".format(t.name))
            else:
                self.logger.info("thread {} has ended".format(t.name))
        for q in queues:
            if not q.empty():
                self.logger.info("{} items on queue {} - emptying".format(q.qsize(), q.name))
                while not q.empty():
                    q.get()
            if isinstance(q, mpq.Queue):
                q.close()
                q.join_thread()
                self.logger.info("queue {} joined thread".format(q.name))

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
        self.logger.info("MCTS finished {} searches, each thread spent on average {} sec waiting during select".format(
            self.config.mcts_searchcount, 
            float(sum(self.debug_stats)/self.config.mcts_searchthreadcount) ))
        self.debug_stats.clear()
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
            self.mcts_searchtable[path[i]]['VL'] = 0
        return


def _MCTSsearcher(
        graph : GameGraph, 
        searchtable : dict, 
        config : SelfPlayConfig,
        terminateevent : threading.Event,
        request_q : queue.Queue, 
        response_q : queue.Queue, 
        graphlock : threading.Lock, 
        prediction_lock : threading.Lock, 
        predict_request_q : mpq.Queue, 
        predict_response_q : mpq.Queue, 
        logging_q : mpq.Queue,
        debug_stats : list):       
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
        t0 = time()
        while vertex not in searchtable:                
            # logger.warning( ("{} not in searchtable: waiting for other threads to finish prediction. " +
            #     "Consider increasing virtualloss. Thread sleeps for {} seconds").format(
            #         vertex,config.mcts_sleeptime_blocked_select))                
            sleep(config.mcts_sleeptime_blocked_select)                
        debug_stats.append(time()-t0)

        # check event here since another blocking thread may have been given 'fake' predictions
        if terminateevent.is_set():
            return [vertex]
        
        # thread safe operation - else we need another lock   
        searchtable[vertex]['VL'] = config.mcts_virtualloss    

        visits = searchtable[vertex]['N']
        def U(child):
            try:
                c_val = searchtable[child]['Q'] + searchtable[child]['VL']
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
                    'P': [],
                    'VL': 0}
            else:
                prediction = predictor(vertex)
                tableentry = {
                        'N': [0 for c in graph.children_at(vertex)],
                        'Q': prediction[1],
                        'P': prediction[0][:len(graph.children_at(vertex))],
                        'VL': 0} 
            if vertex not in searchtable:
                searchtable[vertex] = tableentry                 
        return
    
    while not terminateevent.is_set():
        try:
            request = request_q.get(block=True, timeout=0.1)
            path = select(request)   
            expand(path[-1])
            response_q.put(path)
        except queue.Empty:
            pass
    logger.info("terminate event received - terminating")
