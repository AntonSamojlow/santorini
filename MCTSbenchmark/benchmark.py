"""
Provides the framework for experiments to benchmark the performance of
MCTS search functions/architectures.   

Written by Anton Samojlow, May 2020. [anton.samojlow@web.de]
"""

import logging
import json
from os import path, makedirs
from math import sqrt
from random import choice, choices
from time import time, sleep
from datetime import timedelta
import numpy as np

import gamesearch
import santorini

LOGGER = logging.getLogger(__name__)

class Experiment():
    DIM_PI_dict = { (2,1) : 4, # key := (dimension, units_per_player)
                    (3,1) : 27, (3,2) : 28,
                    (4,1) : 46, (4,2) : 66,
                    (5,1) : 63, (5,2) : 100,
                    (6,1) : 64, (6,2) : 115 }

    def __init__(self, name):
        self.name = name
        self.params = Experiment.Parameters(None)
        self.stats = Experiment.Statistics()

    @property
    def json_string(self):
        return json.dumps({
            'type': str(self.__class__), 
            'name':self.name, 
            'parameters':self.params.__dict__, 
            'stats_summary':self.stats.summary,
            'stats_rundetails':self.stats.__dict__
            }, indent=4)

    class Statistics():
        def __init__(self):
            self.startup_time_avg = 0
            self.MCTS_runtimes = []
            self.NNpredictions = []
            self.NNpredict_times = []
            self.addchildren_times = []

        @property
        def summary(self):
            N = len(self.MCTS_runtimes)
            totalpredicts = sum(self.NNpredictions)            
            return {
                'total_runs' : N, 
                'avg_MCTS_runtime' : sum(self.MCTS_runtimes)/N,
                'avg_runtimespent_predict' : sum(self.NNpredict_times)/N,
                'avg_runtimespent_addchildren' : sum(self.addchildren_times)/N,
                'totalpredictions' : totalpredicts,
                'avg_timeperpredict' : sum(self.NNpredict_times)/totalpredicts
               }    

     
    class Parameters():
        def __init__(self, modelpath, max_runtime_sec = 5, searchcount = 100, tf_device = '/gpu:0',
                    dimension = 3, unitspp = 1):
            self.modelpath = modelpath
            self.max_runtime_sec = max_runtime_sec
            self.searchcount = searchcount
            self.tf_device = tf_device
            self.dimension = dimension
            self.unitspp = unitspp

        def copy(self):
            return Experiment.Parameters(
                modelpath=self.modelpath,
                max_runtime_sec=self.max_runtime_sec,         
                searchcount = self.searchcount,
                tf_device = self.tf_device,
                dimension = self.dimension,
                unitspp = self.unitspp
            )

    def run(self):
        return None
  
class LookAhead(gamesearch.SEU):
    """Lookaahead function within the 'alphazero trainloop': The expand-step of the usual MCTS search is skipped,
    and replaced by an estimate for the Neural Network predictor.

    Properties:
        - game_graph: instance of gamesearch.GameGraph
        - predict_fct: Maps 'game_graph.nodes.key' -> [pi, v] where pi and v are numpy 
                arrays of shape (1,PI_DIM) and (1,1)        
        - explore_cst: constant for the LCB1-type function used during selection
        - data: a dictionary with keys 'N' (number of visits), 'Q' (current value estimate) 
                and 'P' (probability weights, retruned by the Neural Network predictor)
    """

    def __init__(self, game_graph, predict_fct, data=None, explore_cst=2.0):
        """Args: 
        - game_graph: Instance of a gamesearch.GameGraph
        - predict_fct: Maps 'game_graph.nodes.key' -> [pi, v] where
            pi and v are numpy arrays of shape (1,DIM_PI) and (1,1)"""
        if not isinstance(game_graph, gamesearch.GameGraph):
            raise Exception("Validation failed: '{}' is not an instance of gamesearch.GameGraph".format(game_graph))
        gamesearch.SEU.__init__(self, game_graph, data=data)
       
        self.predict_fct = predict_fct
        self.explore_cst = explore_cst
        self.reset_counters()

    def reset_counters(self):
        self.predicttime = 0.0
        self.predictions = 0
        self.addedchildren = 0
        self.addchildrentime = 0.0

    def select(self, node):
        if node.is_open or node.is_terminal:
            return [node]       

        visits = self.data[node.name]['N']
        def U(child):
            try:
                c_val = self.data[child.name]['Q']
            except KeyError:
                c_val = 0
            j = node.children.index(child)
            prob = self.data[node.name]['P'][j]
            return c_val - self.explore_cst*prob*sqrt(sum(visits))/(1+visits[j])

        return [node] + self.select(min(node.children, key=U))

    def expand(self, node):
        if node.is_open:
            t0 = time()
            self.graph.add_children(node)           
            self.addchildrentime += float(time()-t0)
            self.addedchildren += 1
            # initialize the node statistics
            if node.is_terminal:
                self.data[node.name] = {'N': [], 'Q': node.value, 'P': []}
            else:
                t0 = time()
                prediction = self.predict_fct(node.name)
                self.predicttime += float( time() - t0 )
                self.predictions += 1
                val = float(prediction[1])
                self.data[node.name] = {
                    'N': [0 for c in node.children],
                    'Q': val,
                    'P': list(prediction[0][0][:len(node.children)])}
        return [node]

    def update(self, path):   
        end_val = self.data[path[-1].name]['Q']     
        for i in range(0, len(path) - 1): 
            sign = 1 - 2*int(i % 2 == path.__len__() % 2)
            j = path[i].children.index(path[i+1])
            self.data[path[i].name]['N'][j] += 1
            self.data[path[i].name]['Q'] +=\
                (sign*end_val - self.data[path[i].name]['Q'])\
                /sum(self.data[path[i].name]['N'])
