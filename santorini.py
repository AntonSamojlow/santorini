"""
Implements the santorini boardgame. See usage examples below.

Written by Anton Samojlow, November 2018. [anton.samojlow@web.de]

Example 1, class Environment:
    The following code displays a random santorini gamestate,
    then computes its children and displays them together with
    some heuristic value. This can for example be used to generate
    and analyse the game tree or study the Markov decision process.

    env = Environment(dimension=3, unitsPerPlayer=1)
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

    env = Environment(dimension=3, unitsPerPlayer=1)
    AI = Player(env)
    Human = HumanViaConsole(env)
    Game([Human, AI], env, verbose=True).save("gamelog.o")
"""

from math import sqrt
from random import randint, choice
from time import time
import sys


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
                board[(row, col)] = int(brd[col*dimension + row])
        for i in range(0, units_per_player):
            units_player += [(int(u_p[0+i]), int(u_p[1+i]))]
            units_opponent += [(int(u_o[0+i]), int(u_o[1+i]))]
        return State(board, units_player, units_opponent)

    def array(self):
        """Returns [[height,...], [[row, col],... ], [[row, col],... ]]"""
        u_p = [list(p) for p in self.units_player]
        u_o = [list(p) for p in self.units_opponent]
        return [list(self.board.values()), u_p, u_o]

    @classmethod
    def from_array(cls, array):
        """Returns a State, inverse of State.array()"""
        [brd, u_p, u_o] = array
        board, dimension = {}, int(sqrt(brd.__len__()))
        for row in range(0, dimension):
            for col in range(0, dimension):
                board[(row, col)] = brd[col*dimension + row]
        units_player = [(p[0], p[1]) for p in u_p]
        units_opponent = [(p[0], p[1]) for p in u_o]
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
        unitsPerPlayer (int, default=2)
        neighbours:     A dictionary {(row, col) : [(row, col), ...]}

    Methods:
        get_moves, do_move, get_plays, do_play, get_children,
        equiv_class, random_state, win_in, lose_in,
        heuristic_value
    """

    def __init__(self, dimension=5, unitsPerPlayer=2):
        self.dimension = dimension
        self.units_per_player = unitsPerPlayer
        self.neighbours = {}
        for row in range(0, dimension):
            for col in range(0, dimension):
                neighbours = [(row+x, col+y) for x in [-1, 0, 1]
                              for y in [-1, 0, 1]
                              if row+x >= 0 and row+x < dimension
                              and col+y >= 0 and col+y < dimension
                              and (x != 0 or y != 0)]
                self.neighbours.update({(row, col): neighbours})

    def get_moves(self, state):
        """Lists all moves '(unitToMove, destination)'.

        [!] Does NOT check if state is terminal.
        """
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
        """Returns the list of legal children of the parent state."""
        return [self.do_play(pos, parent) for pos in self.get_plays(parent)]

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
        """Returns +1/-1 if active player won/lost, 0 else.

        [!] Inconsistent if two or more units are on level 3.
        """
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
        error = "[!] Valid coordinates are 'xy' with x=row, y=column"\
            + " being integers in [0,4].\nTry again..."

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
            self.start_state = environment.randomState(turn=0)
        self.players = players
        self.plays = []
        self.result = 0
        self.playtime = 0

        if verbose:
            print('* start of game')
            print('* first player:', self.players[0].info)
            print('* second player:', self.players[1].info)

        activeplayer = self.players[0]
        currentstate = self.start_state

        if verbose:
            currentstate.print(playerInitials=['A', 'B'])
        time0 = time()
        while True:
            play = activeplayer.choose_play(currentstate)
            self.plays.append(play)
            currentstate = environment.do_play(play, currentstate)
            activeplayer = self.players[int(
                not bool(self.players.index(activeplayer)))]
            if verbose:
                if activeplayer == self.players[0]:
                    currentstate.print(playerInitials=['A', 'B'])
                else:
                    currentstate.print(playerInitials=['B', 'A'])

            score = environment.score(currentstate)
            if score != 0:
                self.playtime = round(time()-time0, 3)
                if ((score == 1 and activeplayer == self.players[0]) or
                        (score == -1 and activeplayer == self.players[1])):
                    self.result = 1
                else:
                    self.result = 2
                if verbose:
                    print('* Player', self.result, 'won!')
                break

    def save(self, path):
        """Appends the game information to a file."""
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
