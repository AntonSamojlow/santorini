"""
Implements the santorini boardgame. And provides an abstraction of
the game to graph structure (see module gamesearch).

Written by Anton Samojlow, November 2018. [anton.samojlow@web.de]
Updated April 2020.

Example 1, class Environment:
    The following code displays a random santorini gamestate,
    then computes its children and displays them together with
    some heuristic value. This can for example be used to generate
    and analyse the game tree or study the Markov decision process.

    env = Environment(dimension=3, units_per_player=1)
    parent = env.random_state(turn=2)
    parent.print()
    print('*** children & heuristic value: ***')
    for child in env.get_children(parent):
        child.print()
        print(env.heuristic_value(child))

Example 2, class Game:
    The following code lets a human play via the console against
    a uniformly random AI opponent. The log is stored in a file.
    This can also be used to pit AIs against each other (setting
    verbose to 'false'). More sophisticated AIs can be implemented
    by deriving the class Player and overriding Player.choose_play(),
    as implemeneted for the class HumanViaConsole.

    env = Environment(dimension=3, units_per_player=1)
    AI = Player(env)
    Human = HumanViaConsole(env)
    Game([Human, AI], env, verbose=True).save("gamelog.o")

Example 3, class SanGraph:
    The following executes a LCB1-guided MCTS for 1 sec on the
    GameGraph for Santorini with dimension 2 and 1 unit per player.
    It also computes the alphabeta value of some root to depth 13.
    See also gamesearch.py.

    SG = SanGraph(Environment(dimension=2, units_per_player=1))
    MCTS = gamesearch.LCB1(SG)
    print('roots:', SG.root_names)
    ROOT = SG.nodes[choice(list(SG.root_names))]
    SG.print_subtree(ROOT, 2, data=MCTS.data)
    MCTS.run_timed(ROOT, 1)
    SG.save('sangraph.json')
    SanGraph.load('sangraph.json').print_subtree(ROOT, 2, data=MCTS.data)

    AB = gamesearch.Alphabeta(SG)
    AB.tabled(ROOT, 13)
    SG.print_subtree(ROOT, 2, data=AB.data)
"""

from math import sqrt
from random import randint, choice
from time import time
from os.path import isfile
import sys
import json
import logging

import numpy
import gamesearch


LOGGER = logging.getLogger(__name__)


class State():
    """Represents a game state.

    Attributes:
        board:          A dictionary {(row, col) : heightlevel (int)}
        units_player:    A list [(row, col), ...])
        units_opponent:  A list [(row, col), ...])

    Methods:
        string, array, print,
        from_string (a classmethod),
        from_array  (a classmethod)
    """

    def __init__(self, board, units_player, units_opponent):
        self.board = board
        self.units_player = sorted(units_player)
        self.units_opponent = sorted(units_opponent)

    def string(self):
        """Returns a string of the form '0000|0110|0011'"""
        brd, u_p, u_o = '', '', ''
        for pos in self.board.keys():
            brd += str(self.board[pos])
        for pos in self.units_player:
            u_p += str(pos[0]) + str(pos[1])
        for pos in self.units_opponent:
            u_o += str(pos[0]) + str(pos[1])
        return brd + '|' + u_p + '|' + u_o

    @classmethod
    def from_string(cls, string):
        """Returns a State, inverse of State.string()"""
        [brd, u_p, u_o] = string.split('|')
        dimension = int(sqrt(brd.__len__()))
        units_per_player = int(u_p.__len__()/2)
        board, units_player, units_opponent = {}, [], []
        for row in range(0, dimension):
            for col in range(0, dimension):
                board[(row, col)] = int(brd[row*dimension + col])
        for i in range(0, units_per_player):
            units_player += [(int(u_p[0+2*i]), int(u_p[1+2*i]))]
            units_opponent += [(int(u_o[0+2*i]), int(u_o[1+2*i]))]
        return State(board, units_player, units_opponent)

    def array(self):
        """Returns a numpy-array (dim=1 and length= board_dim** + 2*units_per_player)"""
        temp = list(self.board.values())
        for unit in self.units_player:
            temp.append(unit[0])
            temp.append(unit[1])
        for unit in self.units_opponent:
            temp.append(unit[0])
            temp.append(unit[1])
        return numpy.array(temp)

    @classmethod
    def from_array(cls, array, environment):
        """Returns a State, inverse of State.array(). Needs the board_dim."""
        board, dim = {}, environment.dimension
        units_pp = int((len(array) - dim*dim)/2)
        for row in range(0, dim):
            for col in range(0, dim):
                board[(row, col)] = array[row*dim + col]
        units_player = [(array[i], array[i+1])
                        for i in range(dim*dim, dim*dim+units_pp, 2)]
        units_opponent = [(array[i], array[i+1])
                          for i in range(dim*dim+units_pp, dim*dim+2*units_pp, 2)]
        return State(board, units_player, units_opponent)

    def print(self, file=None, initials=None):
        """Displays the state to sys.stdout or append it to a file.

        Arguments:
            file (str, default=None):   Filepath to which the state will be
                                        appended. If None, this is sys.stdout.
            initials:   A list of two strings of ONE character each, use
                        obvious initals for the players. Default: ['P', 'O'].
        """
        if initials is None:
            initials = ['P', 'O']
        dimension = int(sqrt(self.board.__len__()))
        out = open(file, 'a', encoding='utf-8',
                   newline=None) if file else sys.stdout

        for _ in range(0, int(3*dimension/2)):
            out.write('-')
        out.write(initials[0])
        for _ in range(0, int(1/2 + 3*dimension/2)-1):
            out.write('-')
        out.write('\n')
        for row in range(0, dimension):
            for col in range(0, dimension):
                if (row, col) in self.units_player:
                    out.write(initials[0])
                elif (row, col) in self.units_opponent:
                    out.write(initials[1])
                else:
                    out.write(' ')
                out.write(str(self.board[(row, col)]) + ' ')
            out.write('\n')
        for _ in range(0, 3*dimension):
            out.write('-')
        out.write('\n')

        if out is not sys.stdout:
            out.close()


class Environment():
    """Represents the abstract game environment.

    Attributes:
        dimension (int, default=5)
        units_per_player (int, default=2)
        neighbours:     A dictionary {(row, col) : [(row, col), ...]}

    Methods:
        get_moves, do_move, get_plays, do_play, get_children,
        equiv_class, random_state, win_in, lose_in,
        heuristic_value
    """

    def __init__(self, dimension=5, units_per_player=2):
        self.dimension = dimension
        self.units_per_player = units_per_player

        # computing the neighbour positions
        self.neighbours = {}
        for row in range(0, dimension):
            for col in range(0, dimension):
                neighbours = [(row+x, col+y) for x in [-1, 0, 1]
                              for y in [-1, 0, 1]
                              if row+x >= 0 and row+x < dimension
                              and col+y >= 0 and col+y < dimension
                              and (x != 0 or y != 0)]
                self.neighbours.update({(row, col): neighbours})

        # computing the possible start states (all possible unit positions)
        positions = list(self.neighbours.keys())
        u_list = [(p, ) for p in positions]
        for _ in range(0, 2*self.units_per_player-1):  # add other units
            u_list = [(*l, p) for p in positions for l in u_list if p not in l]
        u_list = sorted({(tuple(sorted(u[:units_per_player])),
                          tuple(sorted(u[units_per_player:])))
                         for u in u_list})  # removing duplicates and sorting
        self.start_states = [State({p: 0 for p in positions}, u[0], u[1])
                             for u in u_list]

    def get_moves(self, state):
        """Lists all moves '(unitToMove, destination)'. Does NOT check if state is terminal."""
        moves = []
        for pos in state.units_player:
            for dest in self.neighbours[pos]:
                if state.board[dest] <= 3\
                        and not dest in state.units_player + state.units_opponent\
                        and state.board[dest] - state.board[pos] <= 1:
                    moves.append((pos, dest))
        return moves

    def do_move(self, move, state):
        """Returns a new State after moving. Active player does NOT change."""
        (unit, dest) = move
        new_u_p = state.units_player.copy()
        new_u_p.remove(unit)
        new_u_p.append(dest)
        return State(state.board, new_u_p, state.units_opponent)

    def get_plays(self, state):
        """Lists all legal plays '(unitToMove, destination, buildPos)'."""
        plays = []
        if self.score(state) != 0:  # no plays if terminal state
            return plays

        moves = self.get_moves(state)
        for (unit, dest) in moves:
            plays.append((unit, dest, unit))
            for build in self.neighbours[dest]:
                if build not in state.units_player + state.units_opponent\
                        and state.board[build] < 4:
                    plays.append((unit, dest, build))
        return plays

    def do_play(self, play, state):
        """Returns a new State after playing. Active player IS changed."""
        (unit, dest, build) = play
        new_u_p = state.units_player.copy()
        new_u_p.remove(unit)
        new_u_p.append(dest)
        new_brd = {pos: state.board[pos] + int(pos == build)
                   for pos in self.neighbours}
        return State(new_brd, state.units_opponent, new_u_p)

    def get_children(self, parent):
        """Returns a SORTED list of legal children of the parent state."""
        return sorted([self.do_play(pos, parent)
                       for pos in self.get_plays(parent)],
                      key=lambda s: s.string())

    def equiv_class(self, state):
        """Returns symmetrically equivalent states. May contain duplicates."""

        def hor(pos):
            return (self.dimension - 1 - pos[0], pos[1])

        def diag(pos):
            return (pos[1], pos[0])
        trafos = [diag, lambda p: hor(diag(p)), lambda p: diag(hor(diag(p))),
                  lambda p: hor(diag(hor(diag(p)))),
                  hor, lambda p: diag(hor(p)), lambda p: hor(diag(hor(p))),
                  lambda p: p]

        def transform_by(trafo):
            new_brd = {trafo(p): state.board[p] for p in self.neighbours}
            new_u_p = [trafo(p) for p in state.units_player]
            new_u_o = [trafo(p) for p in state.units_opponent]
            return State(new_brd, new_u_p, new_u_o)

        return [transform_by(trafo) for trafo in trafos]

    def random_state(self, turn=None):
        """Returns a random state at the specified turn.

        Note: If turn is unspecified (None, the default) or if it chosen
        higher than maxTurns, it is set to be at random in [0, maxTurns/2].
        """
        max_turns = 1+4*(self.dimension**2 - self.units_per_player)
        if turn is None or turn >= max_turns:
            turn = randint(0, int(max_turns/2))
        u_p, u_o = [], []
        for rep in range(0, 2*self.units_per_player):
            while True:
                pos = (randint(0, self.dimension-1),
                       randint(0, self.dimension-1))
                if pos not in u_p + u_o:
                    break
            if rep < self.units_per_player:
                u_p.append(pos)
            else:
                u_o.append(pos)
        filling = 0
        board = {pos: 0 for pos in self.neighbours}
        while filling < turn:
            pos = (randint(0, self.dimension-1), randint(0, self.dimension-1))
            if board[pos] < 4 - 2*int(pos in u_p+u_o):
                board[pos] += 1
                filling += 1
        return State(board, u_p, u_o)

    def score(self, state):
        """Returns +1/-1 if active player won/lost, 0 else. Inconsistent if two or more units are on level 3."""
        for pos in state.units_player:
            if state.board[pos] == 3:
                return 1
        for pos in state.units_opponent:
            if state.board[pos] == 3:
                return -1
        if self.get_moves(state).__len__() == 0:
            return -1
        return 0

    def win_in(self, k, state):
        """
        Returns true iff active player has a (up to) k-turn
        winning strategy. Searchdepth: 2k-1.
        """
        if k <= 0:
            return False
        moves = self.get_moves(state)
        if moves.__len__() == 0:
            return False
        for move in [self.do_move(M, state) for M in moves]:
            if 3 in [move.board[p] for p in move.units_player]:
                return True
        for child in self.get_children(state):
            if self.lose_in(k-1, child):
                return True
        return False

    def lose_in(self, k, state):
        """
        Returns true iff for any play, the opponent has a k-turn
        winning strategy. Searchdepth: 2k
        """
        # assuming k >= 0
        moves = self.get_moves(state)
        if moves.__len__() == 0:
            return True
        if k > 0:
            for move in [self.do_move(M, state) for M in moves]:
                if 3 in [move.board[p] for p in move.units_player]:
                    return False
            for child in self.get_children(state):
                if not self.win_in(k, child):
                    return False
            return True
        return False

    def heuristic_value(self, state):
        """Returns a heuristic value, satisfying the zero-sum property."""
        # Check if a player has a won.
        # Does not [!] check for illegal state of two units on level 3
        score = self.score(state)
        if score != 0:
            return score

        # compute a heightscore of the units
        value = sum([state.board[u] for u in state.units_player])
        value -= sum([state.board[u] for u in state.units_opponent])
        value /= (1 + 2*state.units_opponent.__len__())  # normalize

        return value


class Player:
    """Blueprint for a Player.

    Use inheritance to define an AI or interface for human play, see
    for example the class HumanViaConsole below.

    Attributes:
        info (str)
        env (Environment)
    """

    def __init__(self, environment, info='random play'):
        self.info = info
        self.env = environment

    def choose_play(self, state):
        """Function {State s} -> {State r: reachable from s}"""
        return choice(self.env.get_plays(state))


class HumanViaConsole(Player):
    """A human that plays via the console."""

    def __init__(self, environment, info='human'):
        Player.__init__(self, environment=environment, info=info)

    def choose_play(self, state):
        """
        Asks to input a valid play for the active player.
        Returns it as '(unit (int), dest (int), build (int))'
        """
        unit, dest, build = None, None, None

        valid_plays = self.env.get_plays(state)
        print(set(valid_plays))
        unit_positions = list({u for (u, d, b) in valid_plays})
        print('Choose a unit to move from', unit_positions, '...')
        unit = self.input_pos(unit_positions)

        destinations = list({d for (u, d, b) in valid_plays if u == unit})
        print('Choose a destination from', destinations, '...')
        dest = self.input_pos(destinations)

        build_positions = [
            b for (u, d, b) in valid_plays if u == unit and d == dest]
        print('Choose where to build from', build_positions, '...')
        build = self.input_pos(build_positions)
        return (unit, dest, build)

    def input_pos(self, valid_list):
        """Asks to input a coordinate from 'validList'."""
        error = "Valid coordinates are 'xy' with x=row, y=column being integers in [0,4].\nTry again..."

        while True:
            try:
                raw_input = input()
                if raw_input == 'exit':
                    exit()
                row, col = int(raw_input[0]), int(raw_input[1])
            except IndexError:
                print(error)
                continue
            except ValueError:
                print(error)
                continue
            if raw_input.__len__() != 2:
                print(error)
            elif (row, col) not in valid_list:
                print((row, col), 'is not from', valid_list, '\nTry again...')
            else:
                return (row, col)


class Game:
    """Each instance represents a played game.

    The game is played between instances of the 'class Players'.
    It is played when calling __init__([1stPlayer, 2ndPlayer], environment).

    Attributes:
        players [Player, Player]
        startState (State):     Starting state with units already positioned
        plays:                  Lists the plays (unit, dest, build) made
        result (int):           Equals 1 (2) if the first (second) player won
        playtime (float)

    Methods:
        save(path)
    """

    def __init__(self, players, environment, start_state=None, verbose=False):
        """Plays a game between players.

        Arguments:
            players [Player, Player]
            environment (Environment)
            startState (State, default: None)
            verbose (bool, default: False)
        """

        if start_state is None:
            self.start_state = choice(environment.start_states)
        self.players = players
        self.plays = []
        self.result = 0
        self.playtime = 0

        if verbose:
            print('* start of game')
            print('* first player:', self.players[0].info)
            print('* second player:', self.players[1].info)

        active = 0
        currentstate = self.start_state

        if verbose:
            currentstate.print(initials=['A', 'B'])
        time0 = time()
        while True:
            self.plays += [players[active].choose_play(currentstate)]
            currentstate = environment.do_play(self.plays[-1], currentstate)
            active = 0 if active == 1 else 1
            if verbose:
                if active == 0:
                    currentstate.print(initials=['A', 'B'])
                else:
                    currentstate.print(initials=['B', 'A'])

            score = environment.score(currentstate)
            if score != 0:
                self.playtime = round(time()-time0, 3)
                if ((score == 1 and active == 0) or
                        (score == -1 and active == 1)):
                    self.result = 1
                else:
                    self.result = 2
                if verbose:
                    print('* Player', self.result, 'won!')
                break

    def save(self, path, form='text'):
        """Appends the game information to a file.

        Arguments:
            path (str):     The file path.
            format (str):   Default: 'text'. Other option: 'json'.
        """

        if form is 'text':
            with open(path, 'a', newline=None) as file:
                file.write('\n')
                file.write('1st player:  '+self.players[0].info+'\n')
                file.write('2nd player:  '+self.players[1].info+'\n')
                file.write('winner:      '+str(self.result)+'\n')
                file.write('start state: '+self.start_state.string()+'\n')
                plays_str = ''.join(str(P)
                                    for P in self.plays).replace(')(', '|')
                plays_str = plays_str.replace(',', '').replace(
                    ')', '').replace('(', '').replace(' ', '')
                file.write('plays:       '+plays_str+'\n')
                file.write('playtime:    '+str(self.playtime)+' s\n')
        elif form is 'json':
            entry = {'1st player': self.players[0].info,
                     '2nd player': self.players[1].info,
                     'winner': self.result,
                     'start state': self.start_state.string(),
                     'plays': self.plays,
                     'playtime': self.playtime}
            if isfile(path):
                with open(path, 'r', encoding='utf-8', newline=None) as file:
                    json_dict = json.load(file)
                    json_dict['Gamelogs'] += [entry]
                    save_data = json_dict
            else:
                save_data = {'Gamelogs': [entry]}
            with open(path, 'w', encoding='utf-8', newline=None) as file:
                json.dump(save_data, file, skipkeys=True, indent=4)
        return False


class SanGraph(gamesearch.GameGraph):
    """Graph for a Santorini Game.

    Attributes:
        desc (str):     Description of the Graph.
        nodes (dct):    An index of the form {Node.name : Node}, holding
                        all known Nodes of the Graph.
        root_names (set)
        env (Environment)

    Methods:
        add_children, print_subtree, to_json, from_json, save, load,
    """

    def __init__(self, env, desc=None, root_names=None, nodes=None):
        if desc is None:
            desc = 'GameGraph for Santorini, dim: '+str(env.dimension) +\
                   ', units/player:'+str(env.units_per_player)
        if root_names is None:
            root_names = {s.string() for s in env.start_states}
            if nodes is None:
                nodes = {}
            for name in root_names:
                if name not in nodes:
                    nodes.update({name: self.Node(name)})

        gamesearch.GameGraph.__init__(self, desc, root_names, nodes)
        self.env = env

    def root_copy(self):
        """
        Returns a copy of the current graph containing only open roots.
        """
        return SanGraph(env=self.env)

    def nodename_of(self, array):
        """Return the nodename, regardless whether the node has been explored yet"""
        return State.from_array(array, self.env).string()

    def nparray_of(self, name):
        """Return the nparray, regardless whether the node has been explored yet"""
        return State.from_string(name).array()

    def add_children(self, node):
        if not node.is_open:
            LOGGER.warning('SanGraph.add_children(node) called on non-open node')
            return False

        node.children = []
        for c_state in self.env.get_children(State.from_string(node.name)):
            c_name = c_state.string()
            try:
                node.children += [self.nodes[c_name]]
                self.nodes[c_name].parents += [node]
            except KeyError:
                child_node = self.Node(c_name, parents=[node],
                                       value=self.env.score(c_state))
                self.nodes[c_name] = child_node
                node.children += [child_node]
        return True

    @classmethod
    def from_json(cls, string, env):
        """Returns the GameGraph from a JSON string, inverse of 'to_json'."""
        def decode(dct):
            if "__GameGraph__" in dct:
                return cls(env, dct['desc'], root_names=dct['root_names'],
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
    def load(cls, path, env):
        """Loads the Graph from a file created by 'GameGraph.save()'."""
        try:
            file = open(path, 'r', encoding='utf-8', newline=None)
            graph = cls.from_json(file.read(), env)
        except Exception as exc:
            LOGGER.warning('failed to load SanGraph from {0}', path)
        finally:
            file.close()
        return graph
