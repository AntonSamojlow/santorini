from math import sqrt
from random import randint
import sys

class State():
    def __init__(self, board, unitsPlayer, unitsOpponent):
        self.board = board
        self.unitsPlayer = unitsPlayer
        self.unitsOpponent = unitsOpponent

    def string(self):
        b, uP, uO = '', '', ''
        for p in self.board.keys(): b += str(self.board[p])
        for p in self.unitsPlayer: uP += str(p[0]) + str(p[1])
        for p in self.unitsOpponent: uO += str(p[0]) + str(p[1])
        return b + '|' + uP + '|' + uO

    @classmethod
    def from_string(self, string):
        [b, uP, uO] = string.split('|')
        dimension, unitsPerPlayer = int(sqrt(b.__len__())), int(uP.__len__()/2)
        board, unitsPlayer, unitsOpponent = {}, [], []
        for row in range(0, dimension):
            for col in range(0, dimension):
                board[(row, col)] = int(b[col*dimension + row])
        for i in range(0, unitsPerPlayer):
            unitsPlayer += [(int(uP[0+i]), int(uP[1+i]))]
            unitsOpponent += [(int(uO[0+i]), int(uO[1+i]))]
        return State(board, unitsPlayer, unitsOpponent)
    
    def array(self):
        uP = [list(p) for p in self.unitsPlayer]
        uO = [list(p) for p in self.unitsOpponent]
        return [list(self.board.values()), uP, uO]

    @classmethod
    def from_array(self, array):
        [b, uP, uO] = array
        board, dimension = {}, int(sqrt(b.__len__()))
        for row in range(0, dimension):
            for col in range(0, dimension):
                board[(row, col)] = b[col*dimension + row]
        unitsPlayer = [(p[0], p[1]) for p in uP]
        unitsOpponent = [(p[0], p[1]) for p in uO]
        return State(board, unitsPlayer, unitsOpponent)

    def print(self, file=None, playerInitials=['P', 'O']):
        """Outputs a representation of the state to sys.stdout, or optionally to 'file'."""
        dimension = int(sqrt(self.board.__len__()))
        out = open(file, 'a', encoding='utf-8', newline=None) if file else sys.stdout        
        for _ in range(0, int(3*dimension/2)): out.write('-')
        out.write(playerInitials[0])
        for _ in range(0, int(1/2+ 3*dimension/2)-1): out.write('-')
        out.write('\n')
        for row in range(0, dimension):
            for col in range(0, dimension):
                if (row, col) in self.unitsPlayer: 
                    out.write(playerInitials[0])                     
                elif (row, col) in self.unitsOpponent: 
                    out.write(playerInitials[1])
                else: 
                    out.write(' ')
                out.write(str(self.board[(row, col)]) + ' ')                                    
            out.write('\n')
        for _ in range(0,3*dimension): out.write('-')
        out.write('\n')

        if out is not sys.stdout:
            out.close()    
    
    def __eq__(self, other):
        """Overriding default: Returns true iff the fields entries agree irresp of the ordering of 'units'."""
        return self.string() == other.string()

class Environment():
    def __init__(self, dimension=5, units_per_player=2):        
        self.dimension = dimension
        self.unitsPerPlayer = units_per_player
        self.neighbours = {}
        for row in range(0, dimension):
            for col in range(0, dimension):
                neighbours = [(row+x, col+y) for x in [-1,0,1] for y in [-1,0,1] 
                    if row+x >= 0 and row+x < dimension and col+y >= 0 and col+y < dimension 
                    and (x != 0 or y != 0)]
                self.neighbours.update({(row,col) : neighbours})

    def get_moves(self, state):
        """Return a list of moves '(unitToMove, destination)'."""
        moves = []
        for p in state.unitsPlayer:
            for dest in self.neighbours[p]:
                if state.board[p] <= 3\
                        and not dest in state.unitsPlayer + state.unitsOpponent\
                        and state.board[dest] - state.board[p] <= 1:
                    moves.append((p,dest))
        return moves

    def do_move(self, move, state):
        """Returns a new state after executing the move. Active player is not changed."""
        (unitPos, dest) = move 
        newUnitsPlayer = state.unitsPlayer.copy()
        newUnitsPlayer.remove(unitPos)
        newUnitsPlayer.append(dest)
        return State(state.board, newUnitsPlayer, state.unitsOpponent)

    def get_plays(self, state):
        """Return a list of plays '(unitToMove, destination, buildPos)'."""
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
        """Returns a new state after executing the play. Active player is changed."""
        (unitPos, dest, buildPos) = play
        newUnitsPlayer = state.unitsPlayer.copy()
        newUnitsPlayer.remove(unitPos)
        newUnitsPlayer.append(dest)
        newBoard = {p : state.board[p] + int(p == buildPos) for p in self.neighbours.keys()}
        return State(newBoard, state.unitsOpponent, newUnitsPlayer) 
    
    def get_children(self, parent):
        """Returns the list of children of the parent state."""
        return [self.do_play(p, parent) for p in self.get_plays(parent)]

    def equiv_class(self, state):
        """Lists all symmetrically equivalent states. [!] May contain duplicates."""
        def H(p): return (self.dimension -1 - p[0], p[1])
        def D(p): return (p[1], p[0])
        symTrafos = [D, lambda p: H(D(p)), lambda p: D(H(D(p))), lambda p: H(D(H(D(p)))),
                     H, lambda p: D(H(p)), lambda p: H(D(H(p))), lambda p: p]
        def transformBy(trafo, s=state):
            newBoard = {trafo(p) : s.board[p] for p in self.neighbours.keys()}
            newUnitsPlayer = [trafo(p) for p in s.unitsPlayer]
            newUnitsOpponent = [trafo(p) for p in s.unitsOpponent]
            return State(newBoard, newUnitsPlayer, newUnitsOpponent)        
        return [transformBy(trafo) for trafo in symTrafos]

    def randomState(self, turn=None):
        """If turn is unspecified, it is chosen at random in [0, maxTurns/2]"""
        maxTurns = 1+4*(self.dimension**2 - self.unitsPerPlayer )
        if turn == None or turn >= maxTurns: turn = randint(0, int(maxTurns/2))
        uP, uO = [], []
        for rep in range(0, 2*self.unitsPerPlayer):
            while True: 
                p = (randint(0, self.dimension-1), randint(0, self.dimension-1))
                if p not in uP + uO : break
            if rep < self.unitsPerPlayer: uP.append(p)
            else: uO.append(p)
        filling = 0
        board = { p : 0 for p in self.neighbours.keys()}
        while filling < turn:
            p = (randint(0, self.dimension-1), randint(0, self.dimension-1))
            if board[p] < 4 - 2*int( p in uP+uO): 
                board[p] += 1
                filling += 1
        return State(board, uP, uO)

    def score(self, state):
        """Returns +1/-1 if active player won/lost, 0 else. [!] Inconsistent if two or more units are on level 3."""        
        for p in state.unitsPlayer: 
            if state.board[p] == 3: return 1
        for p in state.unitsOpponent: 
            if state.board[p] == 3: return -1  
        if self.get_moves(state).__len__() == 0: return -1
        return 0 

    def exists_winIn(self, k, state):     
        """Returns true iff active player has a (up to) k-turn winning strategy. Searchdepth: 2k-1:"""
        if k<=0: return False 
        Moves = self.get_moves(state)
        if Moves.__len__() == 0: return False        
        for m in [self.do_move(M, state) for M in Moves]:
            if 3 in [m.board[p] for p in m.unitsPlayer]: return True                
        for c in self.get_children(state):                    
            if self.exists_loseIn(k-1, c): return True
        return False

    def exists_loseIn(self, k, state):
        """Returns true iff for any play, the opponent has a k-turn winning strategy. Searchdepth: 2k"""
        # assuming k >= 0
        Moves = self.get_moves(state)
        if Moves.__len__() == 0: return True
        if k > 0:
            for m in [self.do_move(M, state) for M in Moves]:
                if 3 in [m.board[p] for p in m.unitsPlayer]: return False
            for c in self.get_children(state):      
                if not self.exists_winIn(k, c): return False
            return True
        return False

    def heuristic_value(self, state):
        """Returns a heuristic value of the gamestate, from the perspective of the active(!) player."""
        # Check if a player has a won. 
        #     Does not [!] check for the non-valid situation of two units on level 3      
        score = self.score(state)
        if score != 0: return score
        
        Moves = self.get_moves(state)
        # heightscore of possible moves
        value = sum([state.board[m[1]] for m in Moves])/(3*Moves.__len__())
        # subtract height of opponents current units
        value -= sum([state.board[u] for u in state.unitsOpponent])/(3*state.unitsOpponent.__len__())
        return value
        
        

       

e = Environment(dimension=2, units_per_player=1)
s = e.randomState(turn=10)
s.print()
print(e.heuristic_value(s))