"""
Implements the santorini boardgame. See usage examples below.

Written by Anton Samojlow, November 2018. [anton.samojlow@web.de]

Example 1, class Environment:
    The following code displays a random santorini gamestate,
    then computes its children and displays them together with
    some heuristic value. This can for example be used to generate
    and analyse the game tree or study the Markov decision process.

    env = Environment(dimension=3, unitsPerPlayer=1)
    parent = env.randomState(turn=2)
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
        unitsPlayer:    A list [(row, col), ...])
        unitsOpponent:  A list [(row, col), ...])

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
        b, u_p, u_o = '', '', ''
        for p in self.board.keys():
            b += str(self.board[p])
        for p in self.units_player:
            u_p += str(p[0]) + str(p[1])
        for p in self.units_opponent:
            u_o += str(p[0]) + str(p[1])
        return b + '|' + u_p + '|' + u_o

    @classmethod
    def from_string(cls, string):
        """Returns a State, inverse of State.string()"""
        [b, u_p, u_o] = string.split('|')
        dimension = int(sqrt(b.__len__()))
        units_per_player = int(u_p.__len__()/2)
        board, unitsPlayer, unitsOpponent = {}, [], []
        for row in range(0, dimension):
            for col in range(0, dimension):
                board[(row, col)] = int(b[col*dimension + row])
        for i in range(0, units_per_player):
            unitsPlayer += [(int(u_p[0+i]), int(u_p[1+i]))]
            unitsOpponent += [(int(u_o[0+i]), int(u_o[1+i]))]
        return State(board, unitsPlayer, unitsOpponent)

    def array(self):
        """Returns [[height,...], [[row, col],... ], [[row, col],... ]]"""
        uP = [list(p) for p in self.units_player]
        uO = [list(p) for p in self.units_opponent]
        return [list(self.board.values()), uP, uO]

    @classmethod
    def from_array(cls, array):
        """Returns a State, inverse of State.array()"""
        [b, uP, uO] = array
        board, dimension = {}, int(sqrt(b.__len__()))
        for row in range(0, dimension):
            for col in range(0, dimension):
                board[(row, col)] = b[col*dimension + row]
        unitsPlayer = [(p[0], p[1]) for p in uP]
        unitsOpponent = [(p[0], p[1]) for p in uO]
        return State(board, unitsPlayer, unitsOpponent)

    def print(self, file=None, playerInitials=['P', 'O']):
        """Displays the state to sys.stdout, or optionally to 'file'."""
        dimension = int(sqrt(self.board.__len__()))
        out = open(file, 'a', encoding='utf-8',
                   newline=None) if file else sys.stdout
        for _ in range(0, int(3*dimension/2)):
            out.write('-')
        out.write(playerInitials[0])
        for _ in range(0, int(1/2 + 3*dimension/2)-1):
            out.write('-')
        out.write('\n')
        for row in range(0, dimension):
            for col in range(0, dimension):
                if (row, col) in self.units_player:
                    out.write(playerInitials[0])
                elif (row, col) in self.units_opponent:
                    out.write(playerInitials[1])
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
    """

    def __init__(self, dimension=5, unitsPerPlayer=2):
        self.dimension = dimension
        self.unitsPerPlayer = unitsPerPlayer
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
        """Returns a list of moves '(unitToMove, destination)'."""
        moves = []
        for p in state.unitsPlayer:
            for dest in self.neighbours[p]:
                if state.board[p] <= 3\
                        and not dest in state.unitsPlayer + state.unitsOpponent\
                        and state.board[dest] - state.board[p] <= 1:
                    moves.append((p, dest))
        return moves

    def do_move(self, move, state):
        """Returns a new State after moving. Active player NOT changed."""
        (unitPos, dest) = move
        newUnitsPlayer = state.unitsPlayer.copy()
        newUnitsPlayer.remove(unitPos)
        newUnitsPlayer.append(dest)
        return State(state.board, newUnitsPlayer, state.unitsOpponent)

    def get_plays(self, state):
        """Returns a list of plays '(unitToMove, destination, buildPos)'."""
        plays = []
        moves = self.get_moves(state)
        for (unitPos, dest) in moves:
            plays.append((unitPos, dest, unitPos))
            for buildPos in self.neighbours[dest]:
                if buildPos not in state.unitsPlayer + state.unitsOpponent\
                        and state.board[buildPos] != 4:
                    plays.append((unitPos, dest, buildPos))
        return plays

    def do_play(self, play, state):
        """Returns a new State after playing. Active player IS changed."""
        (unitPos, dest, buildPos) = play
        newUnitsPlayer = state.unitsPlayer.copy()
        newUnitsPlayer.remove(unitPos)
        newUnitsPlayer.append(dest)
        newBoard = {p: state.board[p] + int(p == buildPos)
                    for p in self.neighbours.keys()}
        return State(newBoard, state.unitsOpponent, newUnitsPlayer)

    def get_children(self, parent):
        """Returns the list of children of the parent state."""
        return [self.do_play(p, parent) for p in self.get_plays(parent)]

    def equiv_class(self, state):
        """Returns symmetrically equivalent states. May contain duplicates."""

        def H(p): return (self.dimension - 1 - p[0], p[1])

        def D(p): return (p[1], p[0])
        symTrafos = [D, lambda p: H(D(p)), lambda p: D(H(D(p))),
                        lambda p: H(D(H(D(p)))),
                     H, lambda p: D(H(p)), lambda p: H(D(H(p))),
                        lambda p: p]

        def transformBy(trafo, s=state):
            newBoard = {trafo(p): s.board[p] for p in self.neighbours.keys()}
            newUnitsPlayer = [trafo(p) for p in s.unitsPlayer]
            newUnitsOpponent = [trafo(p) for p in s.unitsOpponent]
            return State(newBoard, newUnitsPlayer, newUnitsOpponent)
        return [transformBy(trafo) for trafo in symTrafos]

    def randomState(self, turn=None):
        """Returns a random state at the specified turn.

        Note: If turn is unspecified (None, the default) or if it chosen
        higher than maxTurns, it is set to be at random in [0, maxTurns/2].
        """
        maxTurns = 1+4*(self.dimension**2 - self.unitsPerPlayer)
        if turn == None or turn >= maxTurns:
            turn = randint(0, int(maxTurns/2))
        uP, uO = [], []
        for rep in range(0, 2*self.unitsPerPlayer):
            while True:
                p = (randint(0, self.dimension-1),
                     randint(0, self.dimension-1))
                if p not in uP + uO:
                    break
            if rep < self.unitsPerPlayer:
                uP.append(p)
            else:
                uO.append(p)
        filling = 0
        board = {p: 0 for p in self.neighbours.keys()}
        while filling < turn:
            p = (randint(0, self.dimension-1), randint(0, self.dimension-1))
            if board[p] < 4 - 2*int(p in uP+uO):
                board[p] += 1
                filling += 1
        return State(board, uP, uO)

    def score(self, state):
        """Returns +1/-1 if active player won/lost, 0 else.

        [!] Inconsistent if two or more units are on level 3.
        """
        for p in state.unitsPlayer:
            if state.board[p] == 3:
                return 1
        for p in state.unitsOpponent:
            if state.board[p] == 3:
                return -1
        if self.get_moves(state).__len__() == 0:
            return -1
        return 0

    def exists_winIn(self, k, state):
        """
        Returns true iff active player has a (up to) k-turn
        winning strategy. Searchdepth: 2k-1.
        """
        if k <= 0:
            return False
        Moves = self.get_moves(state)
        if Moves.__len__() == 0:
            return False
        for m in [self.do_move(M, state) for M in Moves]:
            if 3 in [m.board[p] for p in m.units_player]:
                return True
        for c in self.get_children(state):
            if self.exists_loseIn(k-1, c):
                return True
        return False

    def exists_loseIn(self, k, state):
        """
        Returns true iff for any play, the opponent has a k-turn
        winning strategy. Searchdepth: 2k
        """
        # assuming k >= 0
        Moves = self.get_moves(state)
        if Moves.__len__() == 0:
            return True
        if k > 0:
            for m in [self.do_move(M, state) for M in Moves]:
                if 3 in [m.board[p] for p in m.units_player]:
                    return False
            for c in self.get_children(state):
                if not self.exists_winIn(k, c):
                    return False
            return True
        return False

    def heuristic_value(self, state):
        """Returns a heuristic value of the gamestate."""
        # Check if a player has a won.
        # Does not [!] check for illegal state of two units on level 3
        score = self.score(state)
        if score != 0:
            return score

        Moves = self.get_moves(state)
        # heightscore of possible moves
        value = sum([state.board[m[1]] for m in Moves])/(3*Moves.__len__())
        # subtract height of opponents current units
        value -= sum([state.board[u] for u in state.unitsOpponent]
                     )/(3*state.unitsOpponent.__len__())
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
        self.info = info
        self.env = environment

    def choose_play(self, state):
        """
        Asks to input a valid play for the active player.
        Returns it as '(unit (int), dest (int), build (int))'
        """
        unitPos, dest, build = None, None, None

        validPlays = self.env.get_plays(state)
        print(set(validPlays))
        unitPositions = list({u for (u, d, b) in validPlays})
        print('Choose a unit to move from', unitPositions, '...')
        unitPos = self.inputCoord(unitPositions)

        destinations = list({d for (u, d, b) in validPlays if u == unitPos})
        print('Choose a destination from', destinations, '...')
        dest = self.inputCoord(destinations)

        buildPositions = [
            b for (u, d, b) in validPlays if u == unitPos and d == dest]
        print('Choose where to build from', buildPositions, '...')
        build = self.inputCoord(buildPositions)
        return (unitPos, dest, build)

    def inputCoord(self, validList):
        """Asks to input a coordinate from 'validList'."""
        error = "[!] Valid coordinates are 'xy' with x=row, y=column"\
            + " being integers in [0,4].\nTry again..."

        while True:
            try:
                rawIn = input()
                if rawIn == 'exit':
                    exit()
                x, y = int(rawIn[0]), int(rawIn[1])
            except IndexError:
                print(error)
                continue
            except ValueError:
                print(error)
                continue
            if rawIn.__len__() != 2:
                print(error)
            elif (x, y) not in validList:
                print((x, y), 'is not from', validList, '\nTry again...')
            else:
                return (x, y)


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
        save(file)
    """

    def __init__(self, players, environment, startState=None, verbose=False):
        """Plays a game between players.

        Arguments:
            players [Player, Player]
            environment (Environment)
            startState (State, default: None)
            verbose (bool, default: False)
        """

        self.startState = startState if startState is not None\
            else environment.randomState(turn=0)
        self.players = players
        self.plays = []
        self.result = 0
        self.playtime = 0

        if(verbose):
            print('* start of game')
            print('* first player:', self.players[0].info)
            print('* second player:', self.players[1].info)

        activeplayer = self.players[0]
        currentstate = self.startState

        if(verbose):
            currentstate.print(playerInitials=['A', 'B'])
        t0 = time()
        while True:
            play = activeplayer.choose_play(currentstate)
            self.plays.append(play)
            currentstate = environment.do_play(play, currentstate)
            activeplayer = self.players[int(
                not bool(self.players.index(activeplayer)))]
            if(verbose):
                if activeplayer == self.players[0]:
                    currentstate.print(playerInitials=['A', 'B'])
                else:
                    currentstate.print(playerInitials=['B', 'A'])

            score = environment.score(currentstate)
            if score != 0:
                self.playtime = round(time()-t0, 3)
                if ((score == 1 and activeplayer == self.players[0]) or
                    (score == -1 and activeplayer == self.players[1])):
                    self.result = 1
                else: self.result = 2
                if(verbose): print('* Player', self.result, 'won!')
                break

    def save(self, file, format='text'):
        """Appends the game information to a file."""
        if format == 'text':
            with open(file, 'a', newline=None) as f:
                f.write('\n')
                f.write('1st player:  '+self.players[0].info+'\n')
                f.write('2nd player:  '+self.players[1].info+'\n')
                f.write('winner:      '+str(self.result)+'\n')
                f.write('start state: '+self.startState.string()+'\n')
                strPlays = ''.join(str(P)
                                   for P in self.plays).replace(')(', '|')
                strPlays = strPlays.replace(',', '').replace(
                    ')', '').replace('(', '').replace(' ', '')
                f.write('plays:       '+strPlays+'\n')
                f.write('playtime:    '+str(self.playtime)+' s\n')
