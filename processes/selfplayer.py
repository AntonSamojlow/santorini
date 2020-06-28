import logging
import logging.handlers
import multiprocessing
import multiprocessing.queues as mpq
import threading
import queue
import os

from math import sqrt
from time import time, sleep
from datetime import datetime
from random import choice, choices
from collections import deque

import numpy as np

import gymdata
from namedqueues import NamedQueue
from gamegraph import GameGraph

class Selfplayer(multiprocessing.Process):
    def __init__(self,
            name: str,
            config: gymdata.SelfPlayConfig, 
            gympath: gymdata.GymPath,
            monitoring_q: mpq.Queue,
            endevent: multiprocessing.Event,          
            logging_q: mpq.Queue,            
            predict_request_q: mpq.Queue, 
            predict_response_q: mpq.Queue,
            graph: GameGraph):
        super().__init__()
        graph.truncate_to_roots()
        self.name = name
        self._mcts_graph = graph.copy()
        self._mcts_searchtable = {}
        self.config = config
        self.endevent = endevent         
        self.logging_q = logging_q
        self.predict_request_q = predict_request_q
        self.predict_response_q = predict_response_q
        self.gympath = gympath
        self.monitoring_q = monitoring_q
        self.logger : logging.Logger
        self.debug_stats = {'select_wait':[], 'predict_wait':[]}
        self._records = deque([])


    @property
    def mcts_graph(self):
        return self._mcts_graph

    @property
    def mcts_searchtable(self):
        return self._mcts_searchtable

    def run(self):
       # logging setup
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(self.config.logging.loglevel)        
        logfilepath = os.path.join(self.gympath.log_folder,"{}[{}].log".format(type(self).__name__, self.pid))
        rfh = self.config.logging.getRotatingFileHandler(logfilepath)
        qh = logging.handlers.QueueHandler(self.logging_q)
        qh.setLevel(logging.INFO)
        self.logger.addHandler(rfh)
        self.logger.addHandler(qh)
        self.logger.info("started and initialized logger")

        try:

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
            for _ in range(self.config.searchthreadcount):
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
                    self.logger,
                    self.name,
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

            t0_statssignal = 0
            gamecount=0
            while not self.endevent.is_set():
                if time()-t0_statssignal > self.config.freq_statssignal:
                    t0_statssignal = time()
                    stats = {
                        'recordcount': len(self._records),
                        'gamecount': gamecount,
                        'select_wait_sum':  sum(self.debug_stats['select_wait']),
                        'select_wait_count': len(self.debug_stats['select_wait']),
                        'predict_wait_sum':  sum(self.debug_stats['predict_wait']),
                        'predict_wait_count': len(self.debug_stats['predict_wait'])
                        }
                    data = (
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        f"{__name__}[{self.pid}]",
                        gymdata.MonitoringLabel.SELFPLAYSTATS,
                        stats)
                    self.monitoring_q.put(data)
                    self.debug_stats['predict_wait'].clear()
                    self.debug_stats['select_wait'].clear()

                newrecords = self._selfplay()
                self.append_records(newrecords)
                self.logger.debug("appended {} new records to game record pool".format(len(newrecords)))
                gamecount+=1
            

            self.terminateevent.set()
            # unlock threads still waiting for predictions and insert fake prediction 
            for t in self.search_threads:
                if t.is_alive() and self.thread_prediction_locks[t.native_id].locked():               
                    try:
                        self.thread_predict_response_qs[t.native_id].put([[] , 0.0])
                        self.thread_prediction_locks[t.native_id].release()
                    except RuntimeError:
                        self.logger.warning("failed to release predict_lock for thread {}".format(t.native_id))
                        pass

            self.logger.debug("writing playrecords to file...")
            self._dump_records()
            self.logger.debug("...done")
            
            for t in self.search_threads:
                t.join(5)
                if t.is_alive():
                    self.logger.warning("thread {} has not ended".format(t.name))
                else:
                    self.logger.debug("thread {} has ended".format(t.name))
            for q in queues:
                if not q.empty():
                    self.logger.debug("{} items on queue {} - emptying".format(q.qsize(), q.name))
                    while not q.empty():
                        q.get()
                if isinstance(q, mpq.Queue):
                    q.close()
                    q.join_thread()
                    self.logger.debug("queue {} joined thread".format(q.name))

            self.logger.info("endevent received - terminating ({} workers still alive)".format(
                sum([int(t.is_alive()) for t in self.search_threads])))
        
        except Exception as exc :
            self.logger.error("exception: {}".format(exc))
            self.logger.info("signaling endevent and terminating")
            self.endevent.set()
                

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
        for _ in range(0, self.config.searchcount):
            self.mcts_request_q.put(vertex)
        replies = 0
        while replies < self.config.searchcount:
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
        self.logger.debug(f"MCTS finished {self.config.searchcount} searches")
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

    def append_records(self, newrecords):
        for r in newrecords:
            self._records.append(r)
        while len(self._records) >= self.config.record_dumpbatchsize:
            self._dump_records(batchsize=self.config.record_dumpbatchsize)

    def _dump_records(self, batchsize=None):
        if len(self._records) < 2:
            self.logger.warning("nothing to dump - records list has one or no elements")
            return
        modeliteration: int
        with open(self.gympath.modelinfo_file, 'r') as f:
            modeliteration = gymdata.ModelInfo.from_json(f.read()).iterationNr 
        folderpath = "{}/{}".format(self.gympath.gamerecordpool_folder, modeliteration)
        try:
            if not os.path.exists(folderpath):
                os.makedirs(folderpath)
        except FileExistsError: #due to several simultaneous processes accessing 
            pass

        if batchsize == None:
            batchsize = len(self._records)
        else:
            batchsize = min(len(self._records), batchsize)        

        records_to_dump =  [self._records.popleft() for _ in range(batchsize)]
        x, val_vec, pi_vec = [], [], []
        for turndata in records_to_dump:
            pi = turndata[1]
            if pi is not None: # filter out terminal states
                x += [self.mcts_graph.numpify(turndata[0])]
                val_vec += [float(turndata[2])]
                # regularize dimension of pi and normalize to a proper probability
                pi = [p/sum(pi) for p in pi]
                for _ in range(len(pi), self.mcts_graph.outdegree_max):
                    pi.append(0)
                pi_vec += [pi]
        prefix = "{}[{}]".format(datetime.now().strftime("%Y-%m-%dT%H_%M_%S"), self.pid)
        np.savetxt(os.path.join(folderpath, '{}.x.csv'.format(prefix)), np.array(x), delimiter=',')
        np.savetxt(os.path.join(folderpath, '{}.y_val.csv'.format(prefix)), np.array(val_vec), delimiter=',')
        np.savetxt(os.path.join(folderpath, '{}.y_pi.csv'.format(prefix)), np.array(pi_vec), delimiter=',')

def _MCTSsearcher(
        graph : GameGraph, 
        searchtable : dict, 
        config : gymdata.SelfPlayConfig,
        terminateevent : threading.Event,
        request_q : queue.Queue, 
        response_q : queue.Queue, 
        graphlock : threading.Lock, 
        prediction_lock : threading.Lock, 
        predict_request_q : mpq.Queue, 
        predict_response_q : mpq.Queue,
        parentlogger: logging.Logger,
        selfplayername: str,
        debug_stats : list):       
    # logging setup
    thread_id = threading.get_ident()
    logger = parentlogger.getChild("Thread-{}".format(thread_id))
    logger.debug("started and initialized logger")

    def predictor(vertex):
        prediction_lock.acquire()
        predict_request_q.put((selfplayername, thread_id, graph.numpify(vertex)))
        # waiting for predictor_proc to release the lock
        # alternative: check against lock.locked() ?
        with prediction_lock: 
            logger.debug("acquired prediction lock, reading from queue...")
            return predict_response_q.get()                 

    def select(vertex):
        logger.debug("select({})".format(vertex))
        if graph.open_at(vertex) or graph.terminal_at(vertex):
            return [vertex]
        t0 = time()
        while vertex not in searchtable:
            sleep(config.sleeptime_blocked_select)                
        debug_stats['select_wait'].append(time()-t0)

        # check event here since another blocking thread may have been given 'fake' predictions
        if terminateevent.is_set():
            return [vertex]
        
        # thread safe operation - else we need another lock   
        searchtable[vertex]['VL'] = config.virtualloss    

        visits = searchtable[vertex]['N']
        def U(child):
            try:
                c_val = searchtable[child]['Q'] + searchtable[child]['VL']
            except KeyError:
                c_val = 0
            j = graph.children_at(vertex).index(child)
            prob = searchtable[vertex]['P'][j]
            return c_val - config.exploration_const*prob*sqrt(sum(visits))/(1+visits[j])
        return [vertex] + select(min(graph.children_at(vertex), key=U))

    def expand(vertex):
        # logger.debug("expand({})".format(vertex))
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
                t0 = time()
                prediction = predictor(vertex)
                debug_stats['predict_wait'].append(time()-t0)
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
    logger.debug("terminate event received - terminating")
