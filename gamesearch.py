"""
Collection of search algorithms for two-player games. The central object
upon which the algorithms operate is the 'GameGraph':

It has at least one root (no ingoing edges) and the property that each edge
increases the 'level' (internal counter of a node, for roots it equals 0)
by _exactly_ one. So every path from a fixed node to any root has the same
length, the 'level' of the node.

Most two-player games can be modelled as such a graph. Transition
probabilities can be implemenetd as weights on the edges.

Since for most games, it is infeasible (or impossible) to specify the full
graph, we allow an 'incomplete' description: A node N may be 'open', that is
its children are not specified, but they may be computed by some rule, which
is defined in the method 'add_children'. Note in particular that an 'open' Node
is not terminal (the latter means add_children has been run and returned empty).

Written by Anton Samojlow, December 2018. [anton.samojlow@web.de]
Updated April 2020.

Example 1, classes SEU and ExampleGraph:
    Execute the most basic MCTS on a simple graph with two roots.

    EG = ExampleGraph()
    MCTS = SEU(EG)
    print('roots:',EG.root_names)
    EG.print_subtree('root1', 3, data=MCTS.data)
    EG.print_subtree('root2', 3, data=MCTS.data)
    MCTS.run_counted('root1', 100)
    MCTS.run_counted('root2', 100)
    EG.save('out.json')
    GameGraph.load('out.json').print_subtree('root1', 3, data=MCTS.data)
    GameGraph.load('out.json').print_subtree('root2', 3, data=MCTS.data)

Example 2, classes LCB1 and ExampleGraph:
    Execute an MCTS, guided by a lower confidence bound (LCB1 algorithm).

    In the previous example, replace the line
        MCTS = SEU(EG)
    by
        MCTS = LCB1(EG)

Example 3, classes Alphabeta and ExampleGraph:
    Run the alphabeta-algorithm with transposition table.

    EG = ExampleGraph()
    AB = Alphabeta(EG)
    AB.tabled(EG.nodes['root1'], 2)
    AB.tabled(EG.nodes['root2'], 4)
    EG.print_subtree('root1', 4, data=AB.data)
    EG.print_subtree('root2', 4, data=AB.data)
"""

from random import choice, shuffle
from math import sqrt, log
from time import time
from datetime import datetime
from os.path import isfile
import json
import logging
from numpy import array as nparray

LOGGER = logging.getLogger(__name__)


class GameGraph():
    """Represents a Graph of a two-player game.

    The Graph may be 'open': There exist nodes which have not been expanded yet,
    meaning it is not known whether they have children or are terminal. To
    implement a specific game, inherit and override the methods. Better
    avoid overriding class Node, i.p. changing the properties is not advised.
    Then run the search algorithms provided in this module.

    Subclasses:
        Node

    Attributes:
        desc (str):         Description of the Graph.
        nodes (dct):        An index of the form {Node.name : Node}, holding
                            all known Nodes of the Graph.
        root_names (set):   Holds the root names.

    Methods:
        add_children, node_from_array, print_subtree, to_json, from_json,
        save, load
    """
    class Node():
        """Node in the Graph.

        This is the basic unit of the Graph. Better NOT override the properties.
        Moreover, instead of attaching data to the  Node, consider adding
        strucuture/variables/methods to GameGraph itself.

        Attributes:
            name (str):     Identifier.
            parents ([Node,...])
            children ([Node,...])
            value (int):    Value of the Node.

        Logic/Assumptions:
            n is open       <=> n.children = None
            n is terminal   <=> n.children = []
            n is terminal    => n.value = +1/-1
            [!] n.value == +1/-1 DOES NOT imply that n is terminal
        """

        def __init__(self, name, parents=None, children=None, value=0):
            self.name = name
            self.parents = parents
            self.children = children
            self.value = value

        @property
        def is_open(self):
            """See GameGraph.Node.__doc__ and gamesearch.__doc__"""
            return self.children is None

        @property
        def is_terminal(self):
            """See GameGraph.Node.__doc__ and gamesearch.__doc__"""
            return self.children == []       

    def __init__(self, desc=None, root_names=None, nodes=None):
        if desc is None:
            self.desc = 'instance of class GameGraph'
        else:
            self.desc = desc
        if root_names is None:
            self.root_names = {}
        else:
            self.root_names = set(root_names)
        if nodes is None:
            self.nodes = {}
        else:
            self.nodes = nodes

    def add_children(self, node):
        """
        Adds child nodes to node.children and updates GameGraph.nodes.

        Recipe:
        If called on a node N that is not open, return False.
        Else: Determine and add all child nodes {C}, instantiating 'self.Node'.
        If the child is new, add it to the graphs register 'self.nodes'.
        Add N to C.parents. Set N.children = {C} and return True.
        Note: If the list of children is empty, the N.value must be +1/-1.
        """
        if not node.is_open:
            return False
        node.children = []
        return True

    def print_subtree(self, node, max_depth=1, __current_depth=0,
                      value_digits=1, data=None):
        """
        Displays the sub_tree (up to 'max_depth') from 'node'.

        Arguments:
            node (Node or Node.name):   Node from which the subtree is shown.
            max_depth (int):    Maximal depth shown. Default: 1
            __current_depth:    [!] Should not be changed manually. User for
                                proper formating of tree structure smybols.
            value_digits (int): Decimal digits shown for 'node.value'.
            data (dct):         Information about each Node can be passed here.
                                Expected Format: {Node.name : D}, where D is
                                a dictionary {info : value}. Default: None.
        """
        if isinstance(node, str):
            node = self.nodes[node]
        if data is None:
            data = {}
        node_info = str(str(node.name) + ' ('   # assemble the string
                        + str(round(node.value, value_digits)
                              ).ljust(2+value_digits, ' ')
                        + '| ')
        try:  # add data
            for key in data[node.name]:
                if __current_depth == 0:
                    node_info += str(key) + ':'
                node_info += str(data[node.name][key])+', '
        except KeyError:
            pass
        node_info = node_info[:-2]
        node_info += ')'

        if node.is_open:
            node_info += ' ...'
        if node.is_terminal:
            node_info += ' [T]'

        if __current_depth == 0:  # print header
            print(''.ljust(2+node_info.__len__(), '-'))
            print('...: open, [T]: terminal')
            print('# ', end='')
        else:    # add tree structure symbol
            for _ in range(0, __current_depth-1):
                print('.   ', end='')
            print('|--', end='')
        print(node_info)  # print the string
        if max_depth > 0 and not node.is_open:  # expand a level
            for child in list(node.children):
                self.print_subtree(child, max_depth-1, __current_depth+1,
                                   value_digits, data)
        if __current_depth == 0:  # print footer
            print(''.ljust(2+node_info.__len__(), '-'))

    def nodename_of(self, array):
        """Return the nodename, regardless whether the node has been explored
        yet. Inverse of nparray_of."""
        return "".join([chr(x) for x in array])
    
    def nparray_of(self, name):
        """Return the nparray, regardless whether the node has been explored
        yet. Inverse of nodename_of."""
        return nparray([ord(c) for c in name])

    def to_json(self, indent=4):
        """Turn the Graph into a (relatively) compact JSON string. May need
        adjustmemt for specific games."""
        class GameGraphEncoder(json.JSONEncoder):
            """Custom Encode for class GameGraph."""

            def default(self, obj):  # pylint: disable=E0202
                if isinstance(obj, GameGraph):
                    return {'__GameGraph__': True,
                            'desc': obj.desc,
                            'root_names': list(obj.root_names),
                            # we store a list of nodes instead of dictionary
                            'nodes': list(obj.nodes.values())}
                if isinstance(obj, GameGraph.Node):
                    return {'__GameGraph.Node__': True,
                            'name': obj.name,
                            # To reduce the size of the JSON string, we replace
                            # Nodes->Node.name in Node.children & Node.parents
                            # The method from_json reconstructs this process
                            'parents': None if obj.parents is None
                                       else [p.name for p in obj.parents],
                            'children': None if obj.children is None 
                                        else [c.name for c in obj.children],
                            'value': obj.value}
                return json.JSONEncoder.default(self, obj)
        return json.dumps(self, cls=GameGraphEncoder, indent=indent)

    def save(self, path):
        """Saves the Graph as JSON to a file at 'path'.

        Does not overwrite 'path'. Create a timestamped copy if 'path' exists.
        """
        if isfile(path):
            path = path+'_'+str(datetime.now().strftime("%y%m%d_%H%M%S_%f"))
            LOGGER.warning('filepath exists, changed to {0}', path)
        with open(path, 'w', encoding='utf-8', newline=None) as file:
            file.write(self.to_json())

    @classmethod
    def from_json(cls, string):
        """Returns the GameGraph from a JSON string, inverse of 'to_json'.

        [!] Must be reimplemented for other graphs with different attributes.
        """
        def decode(dct):
            if "__GameGraph__" in dct:
                return cls(dct['desc'], root_names=dct['root_names'],
                           nodes={n.name: n for n in dct['nodes']})

            if "__GameGraph.Node__" in dct:
                del dct["__GameGraph.Node__"]
                return cls.Node(**dct)

        graph = json.loads(string, object_hook=decode)
        # rebuilding the references between the GameGraph.Node instances:
        for node in graph.nodes.values():
            if not node.is_open:
                node.children = [graph.nodes[c] for c in node.children]
            if not node.parents is None:
                node.parents = [graph.nodes[p] for p in node.parents]
        return graph

    @classmethod
    def load(cls, path):
        """Loads the Graph from a file created by 'GameGraph.save()'."""
        try:
            file = open(path, 'r', encoding='utf-8', newline=None)
            graph = GameGraph.from_json(file.read())
        except Exception as exc:
            LOGGER.warning('failed to load GameGraph from {0}, exception: {1}'
                           .format(path, exc))
        finally:
            file.close()
        return graph


class ExampleGraph(GameGraph):
    """An example of a GameGraph."""

    def __init__(self):
        desc = 'Example of a GameGraph'
        root_names = {'root1', 'root2'}
        nodes = {'root1': self.Node('root1'), 'root2': self.Node('root2')}
        GameGraph.__init__(self, desc, root_names, nodes)

    def add_children(self, node):
        if not node.is_open:
            return False
        children_dict = {'root1': ['a', 'b'],
                         'root2': ['c', 'd'],
                         'a': ['1', '2'],
                         'b': ['3', '4'],
                         'c': ['4', '5'],
                         'd': ['5', '6'],
                         '1': ['1.1', '1.2'], '2': ['2.1'],
                         '3': ['3.1'], '4': ['4.1', '4.2'],
                         '5': ['5.1', '5.2'], '6': ['6.1']}
        value_dict = {'1.1': 1, '1.2': 1, '2.1': -1,
                      '3.1': 1, '4.1': 1, '4.2': -1,
                      '5.1': 1, '5.2': -1, '6.1': 1}
        node.children = []
        if node.name in children_dict:
            for c_name in children_dict[node.name]:
                if c_name in self.nodes:
                    self.nodes[c_name].parents += [node]
                else:
                    self.nodes.update({c_name: self.Node(c_name,
                                                         parents=[node])})
                node.children += [self.nodes[c_name]]
        else:
            node.value = value_dict[node.name]
        return True


class SEU():
    """ Blueprint for search algorithms "Selection, Expansion and Updating".

    Basic examples of a SEU type algorithms are Monte-Carlo tree search (MCTS)
    or "alpha-zero" TS algorithms. This class is both an example of a MCTS and
    a blueprint. As EXAMPLE: It implements a MCTS, counting for each node
    the number of times it has been visited and the number of times it lead
    to a win. This allows to estimate the win-percentage if both players play
    uniformly random. As BLUEPRINT: To implement another algorithm, override
    the methods SEU.select, SEU.expand and SEU.update, as for example in the
    class LCB1. Please also provide a description of the collected data.

    Attributes:
        graph (GameGraph)
        data (dct):     Of the form {Node.name: {'wins': int, 'visits': int}}.

    Methods:
        select, expand, update:         override for other algorithms
        run, run_timed, run_counted:    these use select, expand, update
    """

    def __init__(self, game_graph, data=None):
        self.graph = game_graph
        if data is None:
            self.data = {}
        else:
            self.data = data

    def select(self, node):
        """Selection phase of a MCTS algorithm.

        Returns:    A list of vertices that starts at node
                    and ends in a leaf node.
        """
        if node.is_open or node.is_terminal:
            return [node]
        return [node] + self.select(choice(node.children))

    def expand(self, node):
        """Expansion phase of a MCTS algorithm.

        Returns:    A branch (list of nodes) that starts at node
                    and ends in a terminal node.
        """
        if node.is_open:
            self.graph.add_children(node)
        if node.is_terminal:
            return [node]
        return [node] + self.expand(choice(node.children))

    def update(self, path):
        """Update (Backpropagation) phase of a MCTS algorithm.

        Input:  A branch (list of nodes) that ends in a terminal node.

        This function updates the information along the branch according
        to the outcome of the game and the value the algorithm is estimating.
        """
        root_won = (path[-1].value == 1) != (path.__len__() % 2 == 0)
        for i in range(0, path.__len__()):
            if path[i].name not in self.data:
                self.data[path[i].name] = {'wins': 0, 'visits': 0}
            self.data[path[i].name]['visits'] += 1
            if root_won == (i % 2 == 0):
                self.data[path[i].name]['wins'] += 1

    def run(self, root):
        """Executes the sequence 'select, expand, update'.

        Arguments:
            root (Node or Node.name)
        """
        if isinstance(root, str):
            root = self.graph.nodes[root]
        path = self.select(root)
        path = path[:-1] + self.expand(path[-1])
        self.update(path)

    def run_timed(self, root, max_sec):
        """Executes sequences of 'select, expand, update' for max_sec seconds.

        Arguments:
            root (Node or Node.name)
            max_sec (float)
        The runtime may exceed max_sec by the excecution time of one sequence.
        """
        t_start = time()
        while time() - t_start < max_sec:
            self.run(root)

    def run_counted(self, root, max_count):
        """Executes max_count times the sequence 'select, expand, update'.

        Arguments:
            root (Node or Node.name)
            max_count (int)
        """
        for _ in range(0, max_count):
            self.run(root)


class LCB1(SEU):
    """Monte-Carlo TS with selection via the LCB1-algorithm.

    For a Node n, it sets n.value to an estimate of the best value V,
    given that both players play optimal.

    Attributes:
        LCB_cst (float):    Constant in the lower confidence bound, the
                            tradeoff between exploitation (LCB_cst << 1) and
                            exploration (LCB_cst >> 1).

    Attributes:
        data (dct):     Of the form {Node.name: {'wins': int, 'visits': int}}.

    Methods:
        select, expand, update:         override for other algorithms
        run, run_timed, run_counted:    these use select, expand, update
    """

    def __init__(self, game_graph, data=None, explore_cst=2.0):
        SEU.__init__(self, game_graph, data=data)
        self.explore_cst = explore_cst

    def select(self, node, N=None):
        if node.is_open or node.is_terminal:
            return [node]

        try:
            visits = self.data[node.name]['visits']
        except KeyError:
            visits = 1

        # We reset N if select is called on a root or if the current node
        # has more visits than its parent.
        if N is None or N < visits:
            N = visits

        def lcb1(node):
            try:
                values = self.data[node.name]
            except KeyError:
                values = {'wins': 0, 'visits': 1}
            return values['wins']/values['visits']\
                - sqrt(self.explore_cst*log(N)/(values['visits']))

        return [node] + self.select(min(node.children, key=lcb1), N=N)


class Alphabeta():
    """
    Implements alphabeta search algorithms for zero-sum games.

    Computes the child that maximizes the attainable Node.value for the
    active player in 'depth' turns, given that the opponent plays equally.

    Attributes:
        sortf_fct:      Function, on input a Node n, it returns a (possibly
                        heuristic) value. Default: Node.value.
        data (dict):    Entries {node.name: (d, flag, score, play)} where
                        d:      searchdepth
                        flag:   'exact', 'lowerbound' or 'upperbound'
                        score:  known alphabeta value to depth d
                        play:   child.name for the optimal child (to depth d)

    Note:
        self.table requires Node.name to be a unique identifier.
        Moreover, when passing an existing table, it should be compatible
        with the graph that is about to be analyzed.

    Methods:
        basic, tabled:  Both Take optional boolean arguments 'moveordered'
                        and 'shuffled', which default to False. In case
                        of moverordered=True, shuffled is ignored. Both
                        moveordering and shuffling operate on a copy, thus
                        preserving the original order of Node.children.
    """

    def __init__(self, graph, sort_fct=None, data=None):
        self.graph = graph
        if sort_fct is None:
            self.sort_fct = lambda node: node.value
        else:
            self.sort_fct = sort_fct

        if data is None:
            self.data = {}
        else:
            self.data = data

    def basic(self, node, depth, _alpha=-1, _beta=1,
              moveordered=False, shuffled=False):
        """Basic alphabeta algorithm.

        Returns:    (bestvalue (float), bestchild (Node)).

        Arguments:
            node (GameGraph.Node)
            depth (int)
            _alpha, _beta (int)
            moveordered (bool, default False):  Uses self.sort_fct iff True
            shuffled (bool, default False):     Randomizes child order
                                                iff True AND moveordred=False
        """
        if depth == 0:
            return (node.value, None)
        if node.is_open:
            self.graph.add_children(node)
        if node.is_terminal:
            return (node.value, None)

        bestval, bestchild = -1, None
        if moveordered:
            children = sorted(node.children, key=self.sort_fct)
        else:
            children = list(node.children)
            if shuffled:
                shuffle(children)

        for child in children:
            val = - self.basic(child, depth-1,
                               _alpha=-_beta, _beta=-max(_alpha, bestval),
                               moveordered=moveordered, shuffled=shuffled)[0]
            if val >= bestval:
                bestval, bestchild = val, child
            if bestval >= _beta:
                break
        return (bestval, bestchild)

    def tabled(self, node, depth, _alpha=-1, _beta=1,
               moveordered=False, shuffled=False):
        """Alphabeta with transposition table.

        Returns:    (bestvalue (float), bestchild (Node)).

        Arguments:
            node (GameGraph.Node)
            depth (int)
            _alpha, _beta (int)
            moveordered (bool, default False):  Uses self.sort_fct iff True
            shuffled (bool, default False):     Randomizes child order
                                                iff True AND moveordred=False
        """
        if depth == 0:
            return (node.value, None)
        if node.is_open:
            self.graph.add_children(node)
        if node.is_terminal:
            return (node.value, None)

        try:
            entry = self.data[node.name]
            if entry['depth'] >= depth:
                if entry['flag'] == 'lowerbound':
                    _alpha = max(_alpha, entry['score'])
                elif entry['flag'] == 'upperbound':
                    _beta = min(_beta, entry['score'])
                # the stored bounds might imply a beta-cutoff:
                if entry['flag'] == 'exact' or _alpha >= _beta:
                    if entry['play'] is None:
                        bestchild = None
                    else:
                        bestchild = next(child for child in node.children
                                         if child.name == entry['play'])
                    return (entry['score'], bestchild)
        except KeyError:
            pass

        bestval, bestchild = -1, None

        if moveordered:
            children = sorted(node.children, key=self.sort_fct)
        else:
            children = list(node.children)
            if shuffled:
                shuffle(children)

        for child in children:
            val = - self.tabled(child, depth-1,
                                _alpha=-_beta, _beta=-max(_alpha, bestval),
                                moveordered=moveordered, shuffled=shuffled)[0]
            if val >= bestval:
                bestval, bestchild = val, child
            if bestval >= _beta:
                break

        depth_table, flag, score = depth, 'exact', bestval
        play = None if bestchild is None else bestchild.name

        if bestval <= _alpha:    # 'beta'-cutoff appeared for some child
            flag = 'upperbound'
        elif bestval >= _beta:   # beta-cutoff appeared at this node
            flag = 'lowerbound'
        self.data[node.name] = {'depth': depth_table,
                                'flag': flag,
                                'score': score,
                                'play': play}
        return (bestval, bestchild)
