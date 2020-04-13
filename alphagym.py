"""
Tool (Gym) that given an instance of gamesearch.GameGraph, will generate 
training examples for a neural network (NN) from selfplay. Also allows a numpy
version of the data, ready for training by a NN, to be saved to file - asumming 
that the NN input is fed from gamesearch.GameGraph.nparray_of().

Approach inspired/mimics parts of the AlphaZero training loop.

Written by Anton Samojlow, April 2020. [anton.samojlow@web.de]
"""

import logging
import multiprocessing
import threading
from os import path, makedirs
from math import sqrt
from random import choice, choices
from time import time, sleep
from datetime import timedelta

import numpy as np

import gamesearch

LOGGER = logging.getLogger(__name__)


class Gym():
    """Provides methods to generate play records and evalute the performance of
    two predictors: functions mapping gamesearch.GameGraph.Node.name -> [pi, v], 
    the NN output where pi and v are numpy arrays of shape (1,DIM_PI) and (1,1)"""

    def __init__(self, game_graph, dim_pi, temperature=None, TS_runs=100):
        self.lookahead = LookAhead(game_graph, None)
        self.temperature = temperature
        self.TS_runs = TS_runs
        self.dim_pi = dim_pi  # Note: Maybe autoparse?

    def search_plays(self, node):
        """Runs a search on the current node, returning a probability
        distribution over the ordered children."""

        # If 'node' has not yet been visited, TS_runs is effectively
        # reduced by one (the first run will be used to initialize
        # lookout.data[node.name]...) and this method returns the
        # invalid probability [0, 0, ...]. To avoid this:
        runs = self.TS_runs + int(node.name not in self.lookahead.data.keys())

        self.lookahead.run_counted(node, runs)
        if self.temperature is None:
            return[1 if i == np.argmax(self.lookahead.data[node.name]['N'])
                   else 0 for i in range(0, node.children.__len__())]
        else:
            return [N**(1/self.temperature) for N in self.lookahead.data[node.name]['N']]

    def saveplaydata(self, gamelogs, folderpath):
        """Save data to .csv files, ready to be readby numpy.loadtxt.
        
        Note: Pi values will be normalized to meet the value of 'Gym.dim_pi'"""
        x, val_vec, pi_vec = [], [], []
        count = sum([len(l) for l in gamelogs])
        LOGGER.info('savePlayData() - saving {0} samples to folder {1}'
                    .format(count, path.abspath(folderpath)))

        for log in gamelogs:
            for turndata in log:
                pi = turndata[1]
                if pi is not None:
                    x += [self.lookahead.game_graph.nparray_of(turndata[0])]
                    val_vec += [float(turndata[2])]
                    # regularize dimension of pi and normalize to a proper probability
                    pi = [p/sum(pi) for p in pi]
                    for _ in range(len(pi), self.dim_pi):
                        pi.append(0)
                    pi_vec += [pi]

        if not path.exists(folderpath):
            LOGGER.warning("the outputpath '{}' did not exists, creating it"
                           .format(folderpath))
            makedirs(folderpath)
        
        np.savetxt(path.join(folderpath, 'x.csv'),  np.array(x), delimiter=',')
        np.savetxt(path.join(folderpath, 'y_val.csv'), np.array(val_vec), delimiter=',')
        np.savetxt(path.join(folderpath, 'y_pi.csv'), np.array(pi_vec), delimiter=',')

        LOGGER.info('...done')

    def _pit(self, predictor1, predictor2):
        """Returns the gamelog of a playthrough between two predictors:
        functions mapping gamesearch.GameGraph.Node.name -> [pi, v], the NN 
        output where pi and v are numpy arrays of shape (1,DIM_PI) and (1,1)."""

        predictors = [predictor1, predictor2]
        root_name = choice(tuple(self.lookahead.graph.root_names))
        node = self.lookahead.game_graph.nodes[root_name]

        gamelog = []
        while not node.is_terminal:  # play a game
            self.lookahead.predict_fct = predictors[len(gamelog) % 2]  # assign player
            prob = self.search_plays(node)
            gamelog.append([node.name, prob, None])
            node = choices(node.children, weights=prob)[0]
        end_val = node.value
        gamelog.append([node.name, None, end_val])
        for i in range(0, len(gamelog)):  # propagate the result
            sign = 1 - 2*int(i % 2 == len(gamelog) % 2)
            gamelog[i][2] = sign*end_val
        return gamelog

    class PitWorkerConfig():
        """Argument of the method alphagym.Gym._pit_cpuworker"""
        def __init__(self, modelpath1, modelpath2, commandq, resultq, loggingq):
            self.modelpath1 = modelpath1
            self.modelpath2 = modelpath2
            self.commandq = commandq
            self.resultq = resultq
            self.loggingq = loggingq

    def pitrun(self, jobcount, modelpath1, modelpath2, savepath=None, livesignal_seconds=300):
        """Pits two tf.keras models 'jobcount'-times against each other, returning the gamelog.
        
        Data is generated from two tf.keras models saved under 'modelpaths'. 
        If 'savepath' is specified, the gamelogs are saved using the method 
        Gym.saveplaydata. Status is written to the log each livesignal_seconds"""
        import tensorflow as tf
        LOGGER.debug('using TensorFlow {}'.format(tf.__git_version__))
        LOGGER.debug("checking visible devices: {}".format(tf.config.experimental.get_visible_devices()))

        tf_loglevel = tf.get_logger().level
        tf.get_logger().setLevel('ERROR')
        Model1 = tf.keras.models.load_model(modelpath1)
        Model2 = tf.keras.models.load_model(modelpath2)
        tf.get_logger().setLevel(tf_loglevel)

        def predictor1(nodename): return Model1.predict(
            np.array([self.lookahead.game_graph.nparray_of(nodename)]))
        def predictor2(nodename): return Model2.predict(
            np.array([self.lookahead.game_graph.nparray_of(nodename)]))

        results = []
        done_record = [(0, time())]  # (jobsdone, time_live): record to calculate ETA
        for jobnr in range(jobcount):
            if int(time() - done_record[-1][1]) >= livesignal_seconds:
                # eta calculation: average of processing speed over last 5 live-signals
                done_record.append((jobnr, time()))
                if len(done_record) > 5:
                    done_record.remove(done_record[0])
                processing_speed_sec = sum(
                    [(done_record[j][0] - done_record[j-1][0])/(done_record[j][1] - done_record[j-1][1])
                     for j in range(1, len(done_record))])/(len(done_record)-1)
                if processing_speed_sec > 0:
                    eta = timedelta(seconds=int((jobcount-jobnr)/processing_speed_sec))
                else:
                    eta = None
                LOGGER.info("{0} jobs left to process, estimated finish in {1}"
                            .format(jobcount-jobnr, eta))
            LOGGER.debug('pit in process...')
            results += [self._pit(predictor1, predictor2)]
            LOGGER.debug('..finished')

        if savepath is not None:
            LOGGER.debug("saving results to the folder {}"
                         .format(path.abspath(savepath)))
            self.saveplaydata(results, path.abspath(savepath))
            LOGGER.debug("...done saving results")

        return results

    def _pit_cpuworker(self, config):
        """Entry point for processsed spawned by Gym.pitrun_cpudistributed()."""
        # check argument
        if not isinstance(config, Gym.PitWorkerConfig):
            raise Exception("invalid valued for parameter config - expected" +
                            "an instance of Gym.PitWorkerConfig")
        # setup logging
        import logging
        import logging.handlers
        qh = logging.handlers.QueueHandler(config.loggingq)
        logger = logging.getLogger(self._pit_cpuworker.__name__)
        logger.setLevel(logging.DEBUG)
        logger.addHandler(qh)
        logger.info("pit_worker started")

        # setup tensorflow - assuming this function runs in a seperate process
        import tensorflow as tf
        logger.debug('using TensorFlow {}'.format(tf.__git_version__))
        with tf.device('/cpu:0'):
            # logger.debug("disabling GPU for TensorFlow")
            # tf.config.set_visible_devices([],'GPU') # disable the GPU for this worker
            # logger.debug("checking visible devices: {}".format(tf.config.experimental.get_visible_devices()))

            # we temporarily reducing the tensorflow logging - ignore some warnings
            # due to custom gradients, as we only do prediction here
            tf_loglevel = tf.get_logger().level
            tf.get_logger().setLevel('ERROR')
            Model1 = tf.keras.models.load_model(config.modelpath1)
            Model2 = tf.keras.models.load_model(config.modelpath2)
            tf.get_logger().setLevel(tf_loglevel)

            def predictor1(nodename): return Model1.predict(
                np.array([self.lookahead.game_graph.nparray_of(nodename)]))
            def predictor2(nodename): return Model2.predict(
                np.array([self.lookahead.game_graph.nparray_of(nodename)]))

            logger.debug("finished loading tensorflow models, waiting for commands")
            while True:
                x = config.commandq.get()
                logger.debug('received {} from command queue'.format(x))
                if x == 'TER':
                    logging.info('teminating (received TER command)')
                    break
                elif x == 'PIT':
                    logger.debug('working...')
                    result = self._pit(predictor1, predictor2)
                    logger.debug('...done working, reporting to result queue')
                    config.resultq.put(result)

    def pitrun_cpudistributed(self, jobcount, workercount, modelpath1, modelpath2,
                              multiprocessingloglevel=logging.INFO, savepath=None, livesignal_seconds=300):
        """Pits two tf.keras models 'jobcount'-times against each other, returning the gamelog.
        
        The task is parallized and distributed on 'workercount'-many processes.
        Data is generated from two tf.keras models saved under 'modelpaths'. 
        If 'savepath' is specified, the gamelogs are saved using the method 
        Gym.saveplaydata. Status is written to the log each livesignal_seconds."""
        def distributed_logger(q):
            """Entry for a thread handling logging from the workers via queues."""
            while True:
                record = q.get()
                if record == 'TER':
                    LOGGER.debug('distributed_logger received {} from logging queue'.format(record))
                    break
                rootlogger = logging.getLogger()
                rootlogger.handle(record)

        # prepare logging
        LOGGER.info("pit_distributed started for {0} workers and {1} jobs ".
                    format(workercount, jobcount))
        mplogger = multiprocessing.get_logger()
        mplogger.setLevel(multiprocessingloglevel)

        # prepare queues and workers
        commandq = multiprocessing.Queue()
        resultq = multiprocessing.Queue()
        loggingq = multiprocessing.Queue()
        for _ in range(jobcount):
            commandq.put('PIT')
        for _ in range(workercount):
            commandq.put('TER')
        LOGGER.info("inserted {} items onto the command queue".format(commandq.qsize()))

        pitworkerconfig = Gym.PitWorkerConfig(modelpath1, modelpath2,
                                              commandq, resultq, loggingq)
        processes = [multiprocessing.Process(target=self._pit_cpuworker,
                                             args=(pitworkerconfig,))
                     for i in range(workercount)]
        LOGGER.info("created {} processes".format(len(processes)))

        # run the job and handle result
        threading.Thread(target=distributed_logger, args=(loggingq,)).start()
        for p in processes:
            p.start()

        done_record = [(0, time())]  # (jobs done, live_signal): records to calculate ETA
        # output alive info and eta during the run
        while not commandq.empty():
            if int(time() - done_record[-1][1]) >= livesignal_seconds:
                # eta calculation: average of processing speed over last 5 live-signals
                done_record.append((int(jobcount-commandq.qsize()), time()))
                if len(done_record) > 5:
                    done_record.remove(done_record[0])
                processing_speed_sec = sum(
                    [(done_record[j][0] - done_record[j-1][0])/(done_record[j][1] - done_record[j-1][1])
                     for j in range(1, len(done_record))])/(len(done_record)-1)
                if processing_speed_sec > 0:
                    eta = timedelta(seconds=int(commandq.qsize()/processing_speed_sec))
                else:
                    eta = None
                LOGGER.info("{0} jobs left to process, estimated finish in {1}"
                            .format(commandq.qsize(), eta))

                # check workers alive
                alivecount = sum([int(p.is_alive()) for p in processes])
                if alivecount < workercount:
                    LOGGER.warning("only {0} out of {1} workers are alive"
                                   .format(sum([int(p.is_alive()) for p in processes]), workercount))
                if alivecount == 0:
                    LOGGER.warning("no workers alive, emptying command queue and terminating worker processes")
                    while not commandq.empty():
                        commandq.get()
                    for p in processes:
                        p.terminate()

            else:
                sleep(1)  # check only each second to save cpu cycles

        LOGGER.debug("command queue is empty")
        results = []
        while not resultq.empty():
            results.append(resultq.get())
        LOGGER.info("received all results")
        if savepath is not None:
            LOGGER.debug("saving results to the folder {}"
                         .format(path.abspath(savepath)))
            self.saveplaydata(results, path.abspath(savepath))
            LOGGER.debug("...done saving results")

        # clean up step
        for p in processes:
            LOGGER.debug("waiting for process '{}' to join".format(p.name))
            p.join()
            LOGGER.debug("process '{}' has joined".format(p.name))
        LOGGER.debug("sending 'TER' signal to logging queue thread...")
        loggingq.put('TER')
        LOGGER.debug("calling close() on command queue...")
        commandq.close()
        LOGGER.debug("calling join_thread() on command queue...")
        commandq.join_thread()
        LOGGER.debug("calling close() on result queue...")
        resultq.close()
        LOGGER.debug("calling join_thread() on result queue...")
        resultq.join_thread()
        LOGGER.debug("calling close() on logging queue...")
        loggingq.close()
        LOGGER.debug("calling join_thread() on logging queue...")
        loggingq.join_thread()

        LOGGER.debug("pit_distributed finished, returning results")
        return results


class LookAhead(gamesearch.SEU):
    """Lookaahead function within the 'alphazero trainloop': The expand-step of the usual MCTS search is skipped,
    and replaced by an estimate form the Neural Network predictor.

    Properties:
        - game_graph: instance of gamesearch.GameGraph
        - predict_fct: Maps 'game_graph.nodes.key' -> [pi, v] where pi and v are numpy 
                arrays of shape (1,PI_DIM) and (1,1)        
        - explore_cst: constant for the LCB1-type function used during selection
        - data: a dictionary with keys 'N' (number of visits), 'V' (current value estimate) 
                and 'P' (probability weights, retruned by the Neural Network predictor)
    """

    def __init__(self, game_graph, predict_fct, data=None, explore_cst=2.0):
        """Args: 
        - game_graph: Instance of a gamesearch.GameGraph
        - predict_fct: Maps 'game_graph.nodes.key' -> [pi, v] where
            pi and v are numpy arrays of shape (1,DIM_PI) and (1,1)"""
        gamesearch.SEU.__init__(self, game_graph, data=data)
        if not isinstance(game_graph, gamesearch.GameGraph):
            raise Exception("Validation failed: '{}' is not an instance of gamesearch.GameGraph".format(game_graph))

        self.game_graph = game_graph
        self.predict_fct = predict_fct
        self.explore_cst = explore_cst

    def select(self, node):
        if node.is_open:
            self.graph.add_children(node)
        if node.is_terminal:
            return [node]

        try:
            visits = self.data[node.name]['N']
            if visits == []:
                return [node]
        except KeyError:
            self.data[node.name] = {'N': [], 'V': 0, 'P': None}
            return [node]

        def U(child):
            c_val = self.data[child.name]['V']
            c_visits = 1 + visits[node.children.index(child)]
            prob = self.data[node.name]['P'][node.children.index(child)]
            return c_val - self.explore_cst*prob*sqrt(sum(visits))/c_visits

        return [node] + self.select(min(node.children, key=U))

    def expand(self, node):
        """No expansion takes place."""
        return [node]

    def update(self, path):
        endnode = path[-1]
        # set values for the endnode and its children if not terminal
        if endnode.is_terminal:
            end_val = endnode.value
            self.data[endnode.name] = {'N': None, 'V': end_val, 'P': []}
        elif self.data[endnode.name]['N'] == []:
            prediction = self.predict_fct(endnode.name)
            end_val = float(prediction[1])
            self.data[endnode.name] = {'N': [0 for c in endnode.children],
                                       'V': end_val,
                                       'P': list(prediction[0][0][:len(endnode.children)])}
            for child in endnode.children:
                if child.name not in self.data:
                    self.data[child.name] = {'N': [], 'V': 0, 'P': None}
        else:
            raise Exception("Unexpected case in method LookAhead.update")

        # update the path
        for i in range(0, path.__len__() - 1):
            sign = 1 - 2*int(i % 2 == path.__len__() % 2)
            j = path[i].children.index(path[i+1])
            self.data[path[i].name]['N'][j] += 1
            self.data[path[i].name]['V'] +=\
                (sign*end_val-self.data[path[i].name]['V'])\
                / sum(self.data[path[i].name]['N'])
