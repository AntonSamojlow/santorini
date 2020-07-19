import dataclasses
import logging
import logging.handlers
import multiprocessing
import multiprocessing.queues as mpq
from os import scandir
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


class Evaluator(multiprocessing.Process):
    def __init__(self, name: str, config: gymdata.EvaluateConfig,
                 gympath: gymdata.GymPath, monitoring_q: mpq.Queue,
                 endevent: multiprocessing.Event, logging_q: mpq.Queue,
                 predict_request_qs: 'list[mpq.Queue]', predict_response_q: mpq.Queue,
                 graph: GameGraph):
        super().__init__()
        graph.truncate_to_roots()
        self.name = name
        self._mcts_graphs = [graph.deepcopy(), graph.deepcopy()]
        self._mcts_searchtables = [dict(), dict()]
        self._mcts_current_expansions = set()
        self.config = config
        self.endevent = endevent
        self.logging_q = logging_q
        self.predict_request_qs = predict_request_qs
        self.predict_response_q = predict_response_q
        self.gympath = gympath
        self.monitoring_q = monitoring_q
        self.logger: logging.Logger
        self.debug_stats = {
            'searches': 0,
            'select_wait': [],
            'predict_wait': []
        }
        self.scores: 'list[int]' = []

    @property
    def mcts_graphs(self):
        return self._mcts_graphs

    @property
    def mcts_searchtables(self):
        return self._mcts_searchtables

    @property
    def mcts_current_expansions(self):
        return self._mcts_current_expansions

    def run(self):
        # logging setup
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(self.config.logging.loglevel)
        logfilepath = os.path.join(
            self.gympath.log_folder,
            "{}[{}].log".format(self.name, self.pid))
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
            starting_player = 0
            while not self.endevent.is_set():
                if time() - t0_statssignal > self.config.freq_statssignal:
                    t0_statssignal = time()
                    self._sendstats()
                result = self._pit(starting_player=starting_player)
                # store score from model1-perspective and alternate starting player  
                self.scores += [ (1-2*starting_player)*result]
                starting_player = (starting_player+1)%2


            self.logger.info(f"total score: {sum(self.scores)} out of {len(self.scores)} games")
            

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
            'predict_wait_count': len(self.debug_stats['predict_wait']),
            'win_perc_model1' : float(1/2 + sum(self.scores)/(2*len(self.scores))) if len(self.scores)>0 else 0
        }
        data = (datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                f"{__name__}[{self.pid}]",
                gymdata.MonitoringLabel.SELFPLAYSTATS, stats)
        self.monitoring_q.put(data)
        self.debug_stats['predict_wait'].clear()
        self.debug_stats['select_wait'].clear()
        self.debug_stats['searches'] = 0

    def _pit(self, starting_player:int = 0) -> int:
        for graph in self._mcts_graphs:
            graph.truncate_to_roots()
       
        vertex = choice(tuple(self.mcts_graphs[0].roots))
        current_player = starting_player

        for table in self.mcts_searchtables:
            table.clear()
        gamelog = []

        while True:  # play a game
            # self.logger.debug(f"self.mcts_graphs[current_player].open_at(vertex)={self.mcts_graphs[current_player].open_at(vertex)}")
            # if not (self.mcts_graphs[current_player].open_at(vertex)):
            #     self.logger.debug(f"self.mcts_graphs[current_player].terminal_at(vertex)={self.mcts_graphs[current_player].terminal_at(vertex)}")
            
            self._MCTSrun(current_player, vertex)
            if self.endevent.is_set():
                self.logger.debug("endevent received - cancelling selfplay")
                return 0
            N_sum = sum(self.mcts_searchtables[current_player][vertex]['N'])
            prob_distr = [
                (N/N_sum)**(1 / self.config.temperature)
                for N in self.mcts_searchtables[current_player][vertex]['N']
            ]
            gamelog.append([vertex, prob_distr, None])
            vertex = choices(self.mcts_graphs[current_player].children_at(vertex),
                             weights=prob_distr)[0]

            game_finished = (not self.mcts_graphs[current_player].open_at(vertex)) and (
                self.mcts_graphs[current_player].terminal_at(vertex))

            current_player = (current_player + 1)%2
            # TODO: Add proper method to gamegraph for adding open nodes
            if vertex not in self.mcts_graphs[current_player]._childrentable:
                self.mcts_graphs[current_player]._childrentable[vertex] = None

            if game_finished:
                break
            
            

        score = self.mcts_graphs[current_player].score_at(vertex)
        startingplayerscore = score*(-1 + 2*int(starting_player == current_player)) 
        self.logger.debug(f"player #{current_player} scores {score}: {[e[0] for e in gamelog]}")
        self.logger.debug(f"returning score {startingplayerscore} for player #{starting_player}")
        return startingplayerscore

    def _MCTSrun(self, player_nr:int, vertex):
        # self.logger.debug("MCTSrun called on vertex {}".format(vertex))
        for _ in range(0, self.config.searchcount):
            self.mcts_request_q.put((player_nr,vertex))
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
            f"MCTS finished {self.config.searchcount} searches (player #{player_nr}) on {vertex} -> {self.mcts_searchtables[player_nr][vertex]}"
        )
        return

    def _MCTS_worker(self, prediction_lock: threading.Lock,
                     thread_response_q: queue.Queue):
        # logging setup
        graph: GameGraph
        table: dict
        player_nr: int
        thread_id = threading.get_ident()
        logger = self.logger.getChild("Thread-{}".format(thread_id))
        logger.debug("started and initialized logger")

        def select(vertex):
            # logger.debug(f"select({vertex})")
            if graph.open_at(vertex):
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

            if graph.terminal_at(vertex):
                # logger.debug("exiting select: {} is terminal".format(vertex))
                return [vertex]

            visits = table[vertex]['N']

            def U(child):
                try:
                    c_val = table[child][
                        'Q'] + self.config.virtualloss * int(
                            child in self._mcts_current_expansions)
                except KeyError:  # excpect error if child is open
                    c_val = (1 - 2 *
                             random()) / 1000  # adding noise for open children
                j = graph.children_at(vertex).index(child)
                prob = table[vertex]['P'][j]
                return c_val - self.config.exploration_const * prob * sqrt(
                    sum(visits)) / (1 + visits[j])

            return [vertex] + select(
                min(graph.children_at(vertex), key=U))

        def expand(vertex) -> bool:
            # logger.debug("expand({})".format(vertex))
            if graph.open_at(vertex):
                # block all other threads until graph has been expanded
                with self.mcts_graph_lock:
                    # logger.debug("acquired graphlock for expanding at {}".format(vertex))
                    graph.expand_at(vertex)
                    # logger.debug("expanded at {}".format(vertex))

                # initialize table statistics
                if graph.terminal_at(vertex):
                    # logger.debug(f"graph.terminal_at({vertex})")
                    tableentry = {
                        'N': [],
                        'Q': graph.score_at(vertex),
                        'P': []
                    }
                else:
                    # logger.debug(f"predicting: {vertex}")
                    t0 = time()
                    prediction_lock.acquire()
                    self.predict_request_qs[f"Predictor-{player_nr}"].put(
                        (self.name, thread_id,
                         graph.numpify(vertex)))
                    with prediction_lock:  # waiting here for external proc/thread to release the lock...
                        # logger.debug("acquired prediction lock, reading from queue...")
                        prediction = thread_response_q.get()
                        if prediction == 'TER':
                            return False
                    self.debug_stats['predict_wait'].append(time() - t0)
                    # logger.debug(f"prediction received for: {vertex}")
                    tableentry = {
                        'N': [0 for c in graph.children_at(vertex)],
                        'Q':
                        prediction[1],
                        'P':
                        prediction[0]
                        [:len(graph.children_at(vertex))]
                    }
                table[vertex] = tableentry
                self.mcts_current_expansions.remove(vertex)
            return True

        def update(path):
            # logger.debug("update({})".format(path))
            with self.mcts_table_lock:
                end_val = table[path[-1]]['Q']
                for i in range(0, len(path) - 1):
                    sign = 1 - 2 * int(i % 2 == path.__len__() % 2)
                    j = graph.children_at(path[i]).index(path[i + 1])
                    table[path[i]]['N'][j] += 1
                    table[path[i]]['Q'] +=\
                        (sign*end_val - table[path[i]]['Q'])\
                        /sum(table[path[i]]['N'])
            # logger.debug("finished update({})".format(path))
            return

        while not self.terminateevent.is_set():
            try:
                player_nr, request = self.mcts_request_q.get(block=True, timeout=0.1)
                graph = self.mcts_graphs[player_nr]
                table = self.mcts_searchtables[player_nr]
                path = select(request)
                if expand(path[-1]):
                    update(path)
                    self.mcts_response_q.put('SEARCH_COMPLETE')
                # logger.debug("main thread was signalled: search complete")
            except queue.Empty:
                # logger.debug("failed to read from request_q")
                pass
        logger.debug("terminate event received - terminating")
