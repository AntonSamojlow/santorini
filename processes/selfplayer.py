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
from random import choice, choices, random
from collections import deque

import numpy as np

import gymdata
from namedqueues import NamedQueue
from gamegraph import GameGraph


class Selfplayer(multiprocessing.Process):
    def __init__(self, name: str, config: gymdata.SelfPlayConfig,
                 gympath: gymdata.GymPath, monitoring_q: mpq.Queue,
                 endevent: multiprocessing.Event, logging_q: mpq.Queue,
                 predict_request_q: mpq.Queue, predict_response_q: mpq.Queue,
                 graph: GameGraph):
        super().__init__()
        graph.truncate_to_roots()
        self.name = name
        self._mcts_graph = graph.deepcopy()
        self._mcts_searchtable = dict()
        self._mcts_current_expansions = set()
        self.config = config
        self.endevent = endevent
        self.logging_q = logging_q
        self.predict_request_q = predict_request_q
        self.predict_response_q = predict_response_q
        self.gympath = gympath
        self.monitoring_q = monitoring_q
        self.logger: logging.Logger
        self.debug_stats = {
            'searches': 0,
            'select_wait': [],
            'predict_wait': []
        }
        self._gamelogs = deque(())
        self.modeliteration: int
        with open(self.gympath.modelinfo_file) as f:
            self.modeliteration = gymdata.ModelInfo.from_json(
                f.read()).iterationNr

    @property
    def mcts_graph(self):
        return self._mcts_graph

    @property
    def mcts_searchtable(self):
        return self._mcts_searchtable

    @property
    def mcts_current_expansions(self):
        return self._mcts_current_expansions

    def run(self):
        # logging setup
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(self.config.logging.loglevel)
        logfilepath = os.path.join(
            self.gympath.log_folder,
            "{}[{}].log".format(type(self).__name__, self.pid))
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
            self.mcts_table_lock = threading.Lock()
            self.mcts_expansion_lock = threading.Lock()
            self.terminateevent = threading.Event()
            queues = [self.mcts_request_q, self.mcts_response_q]

            # search threads: assign their ressources and start
            self.search_threads = []
            self.thread_prediction_locks = {}
            self.thread_predict_response_qs = {}
            for _ in range(self.config.searchthreadcount):
                thread_prediction_lock = threading.Lock()
                thread_predict_response_q = NamedQueue("")
                thread = threading.Thread(target=self._MCTS_worker,
                                          args=(
                                              thread_prediction_lock,
                                              thread_predict_response_q,
                                          ))
                thread.start()
                while thread.native_id is None:  # wait until thread has started
                    sleep(0.1)
                    pass
                self.search_threads += [thread]
                self.thread_prediction_locks[
                    thread.native_id] = thread_prediction_lock
                thread_predict_response_q.name = "[{}]_predict_response_q".format(
                    thread.native_id)
                self.thread_predict_response_qs[
                    thread.native_id] = thread_predict_response_q

            queues += list(self.thread_predict_response_qs.values())

            t0_statssignal = 0
            while not self.endevent.is_set():
                if time() - t0_statssignal > self.config.freq_statssignal:
                    t0_statssignal = time()
                    self._sendstats()
                gamelog = self._selfplay()
                self.cache_records(gamelog)
                self.logger.debug("cached {} new records".format(len(gamelog)))

            self.terminateevent.set()
            # unlock threads still waiting for predictions and insert fake prediction
            for t in self.search_threads:
                if t.is_alive() and self.thread_prediction_locks[
                        t.native_id].locked():
                    try:
                        self.thread_predict_response_qs[t.native_id].put('TER')
                        self.thread_prediction_locks[t.native_id].release()
                    except RuntimeError:
                        self.logger.warning(
                            "failed to release predict_lock for thread {}".
                            format(t.native_id))
                        pass

            self.logger.debug("writing playrecords to file...")
            self._dump_cached_records()
            self.logger.debug("...done")

            for t in self.search_threads:
                t.join(5)
                if t.is_alive():
                    self.logger.warning(f"thread {t.native_id} has not ended")
                else:
                    self.logger.debug(f"thread {t.native_id} has ended")
            for q in queues:
                if not q.empty():
                    self.logger.debug("{} items on queue {} - emptying".format(
                        q.qsize(), q.name))
                    while not q.empty():
                        q.get()
                if isinstance(q, mpq.Queue):
                    q.close()
                    q.join_thread()
                    self.logger.debug("queue {} joined thread".format(q.name))

            self.logger.info(
                "endevent received - terminating ({} workers still alive)".
                format(sum([int(t.is_alive()) for t in self.search_threads])))

        except Exception as exc:
            self.logger.error("exception: {}".format(exc))
            self.logger.info("signaling endevent and terminating")
            self.endevent.set()

    def _sendstats(self):
        stats = {
            'searches': self.debug_stats['searches'],
            'select_wait_sum': sum(self.debug_stats['select_wait']),
            'select_wait_count': len(self.debug_stats['select_wait']),
            'predict_wait_sum': sum(self.debug_stats['predict_wait']),
            'predict_wait_count': len(self.debug_stats['predict_wait'])
        }
        data = (datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                f"{__name__}[{self.pid}]",
                gymdata.MonitoringLabel.SELFPLAYSTATS, stats)
        self.monitoring_q.put(data)
        self.debug_stats['predict_wait'].clear()
        self.debug_stats['select_wait'].clear()
        self.debug_stats['searches'] = 0

    def _selfplay(self) -> list:
        self.mcts_graph.truncate_to_roots()
        vertex = choice(tuple(self.mcts_graph.roots))
        self.mcts_searchtable.clear()
        gamelog = []
        while self.mcts_graph.open_at(
                vertex
        ) or not self.mcts_graph.terminal_at(vertex):  # play a game
            self._MCTSrun(vertex)
            if self.endevent.is_set():
                self.logger.debug("endevent received - cancelling selfplay")
                return []
            prob_distr = [
                N**(1 / self.config.temperature)
                for N in self.mcts_searchtable[vertex]['N']
            ]
            gamelog.append([vertex, prob_distr, None])
            vertex = choices(self.mcts_graph.children_at(vertex),
                             weights=prob_distr)[0]
        for i in range(0, len(gamelog)):  # propagate the result
            sign = 1 - 2 * int(i % 2 == len(gamelog) % 2)
            gamelog[i][2] = sign * self.mcts_graph.score_at(vertex)
        return gamelog

    def _MCTSrun(self, vertex):
        # self.logger.debug("MCTSrun called on vertex {}".format(vertex))
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
                self.thread_predict_response_qs[thread_id].put(
                    [tuple(prediction[0][0]),
                     float(prediction[1])])
                self.thread_prediction_locks[thread_id].release()
            if not self.mcts_response_q.empty():
                try:
                    response = self.mcts_response_q.get()
                    if response == "SEARCH_COMPLETE":
                        self.debug_stats['searches'] += 1
                        replies += 1
                except queue.Empty:
                    pass
        self.logger.debug(
            f"MCTS finished {self.config.searchcount} searches on {vertex} -> {self.mcts_searchtable[vertex]}"
        )
        return

    def cache_records(self, gamelog):
        with open(self.gympath.modelinfo_file) as f:
            currentmodeliteration = gymdata.ModelInfo.from_json(
                f.read()).iterationNr
        self._gamelogs.append(
            (currentmodeliteration, datetime.now().strftime("%Y-%m-%dT%H%M%S"),
             gamelog))
        if currentmodeliteration > self.modeliteration:
            self.modeliteration = currentmodeliteration
            self._dump_cached_records()
        while len(self._gamelogs) >= self.config.gamelog_dump_threshold:
            self._dump_cached_records(
                batchsize=self.config.gamelog_dump_threshold)

    def _dump_cached_records(self, batchsize=None):
        if len(self._gamelogs) < 1:
            self.logger.warning("nothing to dump")
            return
        if batchsize == None:
            batchsize = len(self._gamelogs)
        else:
            batchsize = min(len(self._gamelogs), batchsize)

        self.logger.info(
            f"storing cached batch of {batchsize} gamelogs to disk")

        for entry in [self._gamelogs.popleft() for _ in range(batchsize)]:
            modeliteration, timestamp, records = entry
            if len(records) < 2:
                self.logger.warning(
                    "skipping gamelog with less than 2 entries")
                return

            folderpath = "{}/{}".format(self.gympath.gamerecordpool_folder,
                                        modeliteration)
            try:
                if not os.path.exists(folderpath):
                    os.makedirs(folderpath)
            except FileExistsError:  #due to several simultaneous processes accessing
                pass

            x, val_vec, pi_vec = [], [], []
            for turndata in records:
                pi = turndata[1]
                if pi is not None:  # filter out terminal states
                    x += [self.mcts_graph.numpify(turndata[0])]
                    val_vec += [float(turndata[2])]
                    # regularize dimension of pi and normalize to a proper probability
                    pi = [p / sum(pi) for p in pi]
                    for _ in range(len(pi), self.mcts_graph.outdegree_max):
                        pi.append(0)
                    pi_vec += [pi]
            np.savetxt(os.path.join(folderpath,
                                    f'{timestamp}[{self.pid}].x.csv'),
                       np.array(x),
                       delimiter=',')
            np.savetxt(os.path.join(folderpath,
                                    f'{timestamp}[{self.pid}].y_val.csv'),
                       np.array(val_vec),
                       delimiter=',')
            np.savetxt(os.path.join(folderpath,
                                    f'{timestamp}[{self.pid}].y_pi.csv'),
                       np.array(pi_vec),
                       delimiter=',')

    def _MCTS_worker(self, prediction_lock: threading.Lock,
                     thread_response_q: queue.Queue):
        # logging setup
        thread_id = threading.get_ident()
        logger = self.logger.getChild("Thread-{}".format(thread_id))
        logger.debug("started and initialized logger")

        def select(vertex):
            # logger.debug(f"select({vertex})")
            if self.mcts_graph.open_at(vertex):
                with self.mcts_expansion_lock:
                    if not vertex in self.mcts_current_expansions:
                        self.mcts_current_expansions.add(vertex)
                        # logger.debug(f"added {vertex} to current_expansions, exiting select")
                        return [vertex]

            if vertex in self.mcts_current_expansions:
                # logger.debug("entering wait state for ({})".format(vertex))
                t0 = time()
                while vertex in self.mcts_current_expansions:
                    sleep(0.01)
                    if self.terminateevent.is_set():
                        return []
                self.debug_stats['select_wait'].append(time() - t0)
                # logger.debug("exiting wait state for ({})".format(vertex))

            if self.mcts_graph.terminal_at(vertex):
                # logger.debug("exiting select: {} is terminal".format(vertex))
                return [vertex]

            visits = self.mcts_searchtable[vertex]['N']

            def U(child):
                try:
                    c_val = self.mcts_searchtable[child][
                        'Q'] + self.config.virtualloss * int(
                            child in self._mcts_current_expansions)
                except KeyError:  # excpect error if child is open
                    c_val = (1 - 2 *
                             random()) / 1000  # adding noise for open children
                j = self.mcts_graph.children_at(vertex).index(child)
                prob = self.mcts_searchtable[vertex]['P'][j]
                return c_val - self.config.exploration_const * prob * sqrt(
                    sum(visits)) / (1 + visits[j])

            return [vertex] + select(
                min(self.mcts_graph.children_at(vertex), key=U))

        def expand(vertex) -> bool:
            # logger.debug("expand({})".format(vertex))
            if self.mcts_graph.open_at(vertex):
                # block all other threads until graph has been expanded
                with self.mcts_graph_lock:
                    # logger.debug("acquired graphlock for expanding at {}".format(vertex))
                    self.mcts_graph.expand_at(vertex)
                    # logger.debug("expanded at {}".format(vertex))

                # initialize table statistics
                if self.mcts_graph.terminal_at(vertex):
                    # logger.debug(f"graph.terminal_at({vertex})")
                    tableentry = {
                        'N': [],
                        'Q': self.mcts_graph.score_at(vertex),
                        'P': []
                    }
                else:
                    # logger.debug(f"predicting: {vertex}")
                    t0 = time()
                    prediction_lock.acquire()
                    self.predict_request_q.put(
                        (self.name, thread_id,
                         self.mcts_graph.numpify(vertex)))
                    with prediction_lock:  # waiting here for external proc/thread to release the lock...
                        # logger.debug("acquired prediction lock, reading from queue...")
                        prediction = thread_response_q.get()
                        if prediction == 'TER':
                            return False
                    self.debug_stats['predict_wait'].append(time() - t0)
                    # logger.debug(f"prediction received for: {vertex}")
                    tableentry = {
                        'N': [0 for c in self.mcts_graph.children_at(vertex)],
                        'Q':
                        prediction[1],
                        'P':
                        prediction[0]
                        [:len(self.mcts_graph.children_at(vertex))]
                    }
                self.mcts_searchtable[vertex] = tableentry
                self.mcts_current_expansions.remove(vertex)
            return True

        def update(path):
            # logger.debug("update({})".format(path))
            with self.mcts_table_lock:
                end_val = self.mcts_searchtable[path[-1]]['Q']
                for i in range(0, len(path) - 1):
                    sign = 1 - 2 * int(i % 2 == path.__len__() % 2)
                    j = self.mcts_graph.children_at(path[i]).index(path[i + 1])
                    self.mcts_searchtable[path[i]]['N'][j] += 1
                    self.mcts_searchtable[path[i]]['Q'] +=\
                        (sign*end_val - self.mcts_searchtable[path[i]]['Q'])\
                        /sum(self.mcts_searchtable[path[i]]['N'])
            # logger.debug("finished update({})".format(path))
            return

        while not self.terminateevent.is_set():
            try:
                request = self.mcts_request_q.get(block=True, timeout=0.1)
                path = select(request)
                if expand(path[-1]):
                    update(path)
                    self.mcts_response_q.put('SEARCH_COMPLETE')
                # logger.debug("main thread was signalled: search complete")
            except queue.Empty:
                # logger.debug("failed to read from request_q")
                pass
        logger.debug("terminate event received - terminating")
