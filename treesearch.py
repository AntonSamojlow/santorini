"""
Collection of Tree Search (TS) algorithms for two-player games.

Written by Anton Samojlow, December 2018. [anton.samojlow@web.de]
...
"""
from math import sqrt, log
from time import time
from random import choice, shuffle

class Node():
    """
    Node of a tree, on which the algorithms operate.

    Assumptions:
        1. The tree is defined by overriding two functions,
        - init_value:   Used to onitialise the attribute Node.value.
                        On input Node n, returns a number between -1 and -1.
        - grow: Expands the tree by one level from the given node. On input
                Node n, assigns to n.children the list of child Nodes [c,...]
                such that c.children = [] and c.parent = n.
        2. A node is terminal :<=> BOTH n.children = [] and n.value = +1/-1.
        3. A node is a leaf (of the current tree) :<=> n.children = [].

    Attributes:
        name (str):     Identifier for the node.
        parent (Node)
        children ([Node,...])
        value (float):  The value the TS is trying to compute, usually
                        either the win-probability or the value V of the
                        game's underlying Markov decision process.
        visits (int):   Number of times the TS has visited the node.

    Methods:
        init_value, grow, print()
    """
    def __init__(self, name, parent = None):
        self.name = name
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = float(self.init_value())

    def init_value(self):
        return 0

    def grow(self):
        self.children = []

    def print(self, max_depth = None, __current_depth = 0, value_digits = 3):
        """
        Displays the tree from the given node.

        Arguments:
            max_depth (int):    Limits the depth up to which the tree is shown.
                                By default (None), the whole tree is displayed.
            __current_depth:    Used for proper formatting and should not be
                                changed manually.
        """

        node_info = str(str(self.name) + ' ('
              + str(round(self.value, value_digits)).ljust(3+value_digits, ' ')
              + '|' + str(self.visits) + ')')

        if __current_depth == 0:  # header
            print(''.ljust(2+node_info.__len__(), '-'))
            print('* ', end='')
        else:    # add tree structure symbol
            for _ in range(0, __current_depth-1): print('.   ', end='')
            print('|--', end='')
        print(node_info)  # print node data
        if max_depth == None:
            for child in list(self.children):
                child.print(None, __current_depth+1)  # expand a level
        elif max_depth > 0:
            for child in list(self.children):
                child.print(max_depth-1, __current_depth+1) # expand a level
        if __current_depth == 0: # footer
            print(''.ljust(2+node_info.__len__(), '-'))

class ExampleTree(Node):
    """
    An example tree to test the algorithms.
    """
    def grow(self):
        structure = {'root' : ['c1', 'c2'],
                           'c1': ['c11', 'c12'],
                           'c2': ['c21'],
                           'c11': ['c111', 'c112'],
                           'c12': ['c121'],
                           'c21': ['c211', 'c212']
                    }

        self.children = [ExampleTree(val, parent=self)
                                for val in structure[self.name]]

    def init_value(self):
        values = {'c111' : 1, 'c112' : -1, 'c121' : 1,
                        'c211': 1, 'c212' : -1
                    }
        if self.name in values.keys(): return values[self.name]
        else: return 0

class SEU():
    """
    Blueprint for TS of the type "Selection, Expansion and Updating".

    Basic examples of SEU are Monte-Carlo TS (MCTS) or "alpha-zero" TS
    algorithms. This class in particular implements a MCTS which given
    a Node n, computes n.value as an estimate of the win-percentage if
    both players play totally random. To implement another algorithm,
    override the methods SEU.select, SEU.expand and SEU.update, as for
    example in the class MCTS_LCB below.

    Methods:
        select, expand, update:     override for other algorithms
        run, run_timed, run_counted:    these use select, expand, update
    """
    def __init__(self):
        None

    def select(self, node):
        if node.children == []: return [node]
        return [node] + self.select(choice(node.children))

    def expand(self, node):
        if (node.value == 1 or node.value == -1): return [node]
        else: node.grow()
        return [node] + self.expand(choice(node.children))

    def update(self, path):
        root_won = (path[-1].value == 1) != (path.__len__() % 2 == 0)
        for i in range(0, path.__len__()-1):
            path[i].visits += 1
            i_winner = (root_won == (i % 2 == 0))
            path[i].value += (int(i_winner)-path[i].value)/path[i].visits
        path[-1].visits += 1

    def run(self, root):
        path = self.select(root)
        path = path[:-1] + self.expand(path[-1])
        self.update(path)

    def run_timed(self, root, max_sec):
        t_start = time()
        while time() - t_start < max_sec:
            self.run(root)

    def run_counted(self, root, max_count):
        for _ in range(0, max_count): self.run(root)

class MCTS_LCB(SEU):
    """
    Monte-Carlo TS with selection via the LCB1-algorithm for a zero-sum game.

    For a Node n, it sets n.value to an estimate of the best value V,
    given that both players play optimal.

    Attributes:
        LCB_cst (float):    Constant in the lower confidence bound, the
                            tradeoff between exploitation (LCB_cst << 1) and
                            exploration (LCB_cst >> 1).

    Methods:
        run, run_timed, run_counted

    Usage Example:
        E = ExampleTree('root')
        MCTS_LCB(LCB_cst = 10).run_timed(E, 1)
        E.print()
    """
    def __init__(self, LCB_cst=2.0):
        self.LCB_cst = LCB_cst

    def select(self, node, N=None):
        if N == None: N = 1 + node.visits # set N if select is called on root
        if node.children == []: return [node]
        LCB = lambda node : node.value -\
                                sqrt(self.LCB_cst*log(N)/(1+node.visits))
        return [node] + self.select(min(node.children, key = LCB), N=N)

    def update(self, path):
        root_won = (path[-1].value == 1) != (path.__len__() % 2 == 0)
        for i in range(0, path.__len__()-1):
            path[i].visits += 1
            i_winner = (root_won == (i % 2 == 0))
            path[i].value += (2*int(i_winner)-1 -path[i].value)/path[i].visits
        path[-1].visits += 1

class Alphabeta():
    """
    Implements alphabeta search algorithms for zero-sum games.

    Computes the child that maximizes the attainable Node.value for the
    active player in 'depth' turns, given that the opponent plays equally.

    Attributes:
        sortf_fct:      Function, on input a Node n, it returns a (possibly
                        heuristic) value. Default: Node.value.
        table (dict):   Entries {node.name: (d, flag, score, play)} where
                        d:      searchdepth
                        flag:   'exact', 'lowerbound' or 'upperbound'
                        score:  known alphabeta value to depth d
                        play:   child.name for the optimal child (to depth d)

    Note:
        self.table requires Node.name to be a unique identifier.
        Moreover, when passing an existing table, it should be
        compatible with the tree that is about to be analyzed.

    Methods:
        basic, tabled:  Both Take optional boolean arguments 'moveordered'
                        and 'shuffled', which default to False. In case
                        of moverordered=True, shuffled is ignored.

    Usage Example:
        E = ExampleTree('root')
        result = Alphabeta().basic(E, 3)
        print(result[0], result[1].name)
    """
    def __init__(self, sort_fct=None , table=None):
        if sort_fct == None: self.sort_fct = lambda node: node.value
        else: self.sort_fct = sort_fct

        if table == None: self.table = {}
        else: self.table = table

    def basic(self, node, depth, _alpha=-1, _beta=1,
                moveordered=False, shuffled=False):
        """Basic alphabeta algorithm.

        Returns:    (bestvalue (float), bestchild (Node)).

        Arguments:
            moveordered (bool, default False):  Uses self.sort_fct iff True
            shuffled (bool, default False):     Randomizes child order
                                                iff True AND moveordred=False
        """
        if depth == 0: return (node.value, None)
        if node.children == []:
            if node.value == 1 or node.value == -1: return (node.value, None)
            else: node.grow()

        bestval, bestchild = -1, None
        if moveordered:
            children = sorted(node.children, key = self.sort_fct)
        else:
            children = node.children
            if shuffled: shuffle(children)

        for child in children:
            val = - self.basic(child, depth-1,
                             _alpha=-_beta, _beta=-max(_alpha, bestval),
                             moveordered=moveordered, shuffled=shuffled)[0]
            if val >= bestval: bestval, bestchild = val, child
            if bestval >= _beta: break
        return (bestval, bestchild)

    def tabled(self, node, depth, _alpha=-1, _beta=1,
                moveordered=False, shuffled=False):
        """Alphabeta with transposition table.

        Returns:    (bestvalue (float), bestchild (Node)).

        Arguments:
            moveordered (bool, default False):  Uses self.sort_fct iff True
            shuffled (bool, default False):     Randomizes child order
                                                iff True AND moveordred=False
        """
        if depth == 0: return (node.value, None)
        if node.children == []:
            if node.value == 1 or node.value == -1: return (node.value, None)
            else: node.grow()

        if node.name in self.table.keys():
            (d, flag, score, play) = self.table[node.name]
            if d >= depth:
                if flag == 'lowerbound': _alpha = max(_alpha, score)
                elif flag == 'upperbound': _beta = min(_beta, score)
                # the stored bounds might imply a beta-cutoff:
                if flag == 'exact' or _alpha >= _beta:
                    if play == None:  bestchild = None
                    else: bestchild = next(child for child in node.children
                                                    if child.name == play)
                    return (score, bestchild)

        bestval, bestchild = -1, None

        if moveordered:
            children = sorted(node.children, key = self.sort_fct)
        else:
            children = node.children
            if shuffled: shuffle(children)

        for child in children:
            val = - self.tabled(child, depth-1,
                            _alpha=-_beta, _beta=-max(_alpha, bestval),
                            moveordered=moveordered, shuffled=shuffled)[0]
            if val >= bestval: bestval, bestchild = val, child
            if bestval >= _beta: break

        d, flag, score = depth, 'exact', bestval,
        play = None if bestchild == None else bestchild.name

        if bestval <= _alpha:    # 'beta'-cutoff appeared for some child
            flag = 'upperbound'
        elif bestval >= _beta:   # beta-cutoff appeared at this node
            flag = 'lowerbound'
        self.table[node.name]=(d, flag, score, play)

        return (bestval, bestchild)
