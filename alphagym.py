"""
Implements the alphazero training loop. Uses tools
provided by gamesearch.py.

Gamespecific knowledge is to be passed as an instance
of the class gamesearch.GameGraph

Here, we define two classes: Network and Gym...
"""
from random import choices, choice
from math import sqrt

import h5py
import tensorflow as tf
import numpy as np
from tensorflow import keras


from gamesearch import SEU

class Network():
    """A simple fully connected (dense) neural network with softmax activation,
    output (pi,v) and loss functions (cat. cross-entropy, mean squared. The
    method 'predict' is to be overwritten in a concrete implementation.)"""
    def __init__(self, dim_in, dim_range_pi, learning_rate=0.001, model_compile=True):
        self.dim_in = dim_in
        self.dim_range_pi = dim_range_pi

        input_states = keras.Input(shape=(self.dim_in,))
        layer = keras.layers.Dense(self.dim_in, activation='softmax')(input_states)
        out_pi = keras.layers.Dense(self.dim_range_pi, activation='softmax', name='pi')(layer)
        out_v = keras.layers.Dense(1, activation='tanh', name='v')(layer)

        self.model = keras.models.Model(inputs=input_states,
                                        outputs=[out_pi, out_v])
        if model_compile:
            self.model.compile(loss=['categorical_crossentropy',
                                 'mean_squared_error'],
                                optimizer=keras.optimizers.Adam(learning_rate))

    def predict(self, x):
        """Input x : Some node.name of the corresponding gamegraph"""
        return [[float(i == 0) for i in range(0, self.dim_range_pi)],
                choice([-1, 1])]

    def save(self, name):
        self.model.save_weights(name)

    def load(self, name):
        self.model.load_weights(name)

    def print_info(self):
        self.model.summary()

class Gym():
    """ - missing docstring - """
    def __init__(self, game_graph, network):
        self.network = network
        self.lookout = self.LookOut(game_graph, network)

    class LookOut(SEU):
        def __init__(self, game_graph, network, data=None, explore_cst=2.0):
            SEU.__init__(self, game_graph, data=data)
            self.network = network
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
                [prob, end_val] = self.network.predict(endnode.name)
                self.data[endnode.name] = {'N': [0 for c in endnode.children],
                                           'V': end_val, 'P': prob}
                for child in endnode.children:
                    if child.name not in self.data:
                        self.data[child.name] = {'N':[], 'V':0, 'P': None}

            # update the path
            for i in range(0, path.__len__() - 1):
                sign = 1 - 2*int(i%2 == path.__len__()%2)
                j = path[i].children.index(path[i+1])
                self.data[path[i].name]['N'][j] += 1
                self.data[path[i].name]['V'] +=\
                     (sign*end_val-self.data[path[i].name]['V'])\
                        /sum(self.data[path[i].name]['N'])

    def search_plays(self, node, temp=None, TS_runs=100):
        """Runs a search on the current node, returning a probability
        distribution over the ordered children."""

        # If 'node' has not yet been visited, TS_runs is effectively
        # reduced by one (the first run will be used to initialize
        # lookout.data[node.name]...). Then this method might return
        # the invalid probability [0, 0, ...]
        if node.name not in self.lookout.data.keys():
            TS_runs += 1

        self.lookout.run_counted(node, TS_runs)

        if temp is None:
            return[1 if i == np.argmax(self.lookout.data[node.name]['N'])
                   else 0 for i in range(0, node.children.__len__())]
        else:
            prob_weights = [N**temp for N in self.lookout.data[node.name]['N']]
            return [x / sum(prob_weights) for x in prob_weights]

    def selfplay(self, temp=None, TS_runs=100):
        """Generates and returns the log [...(node, value)...] of one game,
        starting at a random root node. Play decisions are determined
        by the treesearch with samplecount TS_runs"""
        root_name = choice(tuple(self.lookout.graph.root_names))

        node = self.lookout.graph.nodes[root_name]
        log = []
        while not node.is_terminal: # play a game
            prob = self.search_plays(node, temp=temp, TS_runs=TS_runs)
            log.append([node.name, prob, None])
            node = choices(node.children, weights=prob)[0]

        end_val = node.value
        log.append([node.name, [1.0], end_val])
        for i in range(0, log.__len__()): # propagate the result
            sign = 1 - 2*int(i%2 == log.__len__()%2)
            log[i][2] = sign*end_val

        return log


