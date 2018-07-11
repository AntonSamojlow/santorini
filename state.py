# To Do:
#   - add a logger, see https://docs.python.org/3/howto/logging.html#logging-basic-tutorial
#   - a methods that checks two states for symmetries: 
#           player<->~player, 90 degree -rotations, mirroring (diagonals, horizontal, vertical)

from enum import Flag, auto
from copy import deepcopy
from random import randint
from itertools import chain
import sys
#------------------------------------------------------------------------------
# the main class
#------------------------------------------------------------------------------

class state:
    """| class state:
| Provides tools to check, analyse, display and save/export a Santorini gamestate.
| Class variables:
|   - nborDict          A dictionary {(i,j) : [(i,j),...]}, to be initialised manually(!). 
|                       For fast lookup of neighbor positions.    
| Instance variables:   
|   - activePlayer:     Instance of class player(Flag), either 'RED' or 'BLUE'.
|   - boardheight:      A 5x5-tuple with integer values between 0 and 4.
|   - units:            A dictionary {(i,j) : player(Flag)}. Lists the units with their position and owner.      
|
| Methods (described in their own docstring):              
|   init_nborGrid, atRandom, __eq__, valid, print(file=None),
|   evolveByMove, evolveByBuild(pos), invertPlayers, winIn(k), loseIn(k), value, 
|   alphabeta(k,alpha,beta)     
"""

    #--- class variables ------------------------------------------------------
    nborDict = {}    
    
    class player(Flag):       
        RED, BLUE = auto(), auto()
    
    #--- constructor and class methods ----------------------------------------
    def __init__(self, activePlayer=None, boardheight=((0,)*5,)*5, units={}):        
        self.activePlayer = activePlayer
        self.boardheight = boardheight
        self.units = units
        
    @classmethod
    def init_nborGrid(self):
        """Populates the class variable 'nborGrid', !MUST! have been called once."""
        def neighboursOf(pos):
            # compute the neighbouring positions relative to 'pos'
            returnList = []
            for i in range(-1, 2):
                for j in range(-1, 2):
                    if pos[0]+i in range(0, 5) and pos[1]+j in range(0, 5) and not(i == 0 and j == 0):
                        returnList.append((pos[0]+i, pos[1]+j))
            return returnList        
        # populate the dictionary
        if self.nborDict.__len__() == 0:
            for i in range(0, 5):
                for j in range(0, 5):
                    self.nborDict.update({(i,j): neighboursOf((i,j))})
        else: print('[!] state.init_nborGrid() called, but nborDict is already set')

    @classmethod
    def atRandom(self):
        """Returns an instance with random field entries and which passes 'self.valid()'."""
        activePlayer = self.player(True)
        if randint(0, 1) == 0: activePlayer = ~self.player(True)
        units={}
        for rep in range(0,4):
            while True: 
                i, j = randint(0,4), randint(0,4)
                if (i,j) not in units.keys(): break
            if rep < 2: units.update({(i,j) : activePlayer})
            else: units.update({(i,j) : ~activePlayer})
        boardheight = tuple(tuple(randint(0, 2) if (i,j) in units.keys() else randint(0, 4)\
                                for j in range(0, 5)) for i in range(0, 5))
        return state(activePlayer=activePlayer, boardheight=boardheight, units=units)

    #--- methods --------------------------------------------------------------
    def __eq__(self, other):
        """Overriding default: Returns true iff the fields entries agree irresp of the ordering of 'units'."""
        if self.activePlayer != other.activePlayer or self.boardheight != other.boardheight: return False
        for unitPos in self.units.keys():
            if unitPos not in other.units.keys(): return False
            elif self.units[unitPos] != other.units[unitPos]: return False
        return True

    def valid(self):
        """Returns true iff the instance is 'syntactically correct', as defined in the comments of this method."""
        #   1) activePlayer is of type player-Enum
        if not isinstance(self.activePlayer, self.player):
            return False

        #   2) boardheight is a 5x5 tuple with values in {0,1,2,3,4}
        if not (isinstance(self.boardheight, tuple) and self.boardheight.__len__() == 5):
            return False
        else:
            for row in self.boardheight:
                if not (isinstance(row, tuple) and row.__len__() == 5):
                    return False
                else:
                    for pos in row:
                        if pos not in range(0, 5, 1):
                            return False

        #   3) units is a dict with keys 'pos' and entries 'player'
        if not (isinstance(self.units, dict) and self.units.__len__() == 4):
            return False
        else:
            for posKey in self.units.keys():
                if not (isinstance(posKey, tuple) and posKey.__len__() == 2\
                                            and isinstance(self.units[posKey], self.player)):
                    return False
                elif not (posKey[0] in range(0, 5, 1) and posKey[1] in range(0, 5, 1)):
                    return False

        #   4) each player should have exactly two units
        if not (list(self.units.values()).count(self.player.BLUE) == 2 \
                and list(self.units.values()).count(self.player.RED) == 2):
            return False

        #   5) units are not allowed on a boardheight of level 4 or 3 (winning situation)
        for pos in self.units.keys():
            if self.boardheight[pos[0]][pos[1]] >= 3:
                return False

        # All the problems should (!) have been caught, so we return TRUE
        return True

    def print(self, file=None):
        """Outputs a representation of the state to sys.stdout, or optionally to 'file'."""
        out = open(file, 'a', encoding='utf-8') if file else sys.stdout        
        out.write('----turn: '+self.activePlayer.name[0]+'----\n')
        for i in range(0, 5):
            for j in range(0, 5):
                if (i,j) in list(self.units.keys()): out.write(self.units[(i,j)].name[0])
                else: out.write(' ')
                out.write(str(self.boardheight[i][j])+' ')                                    
            out.write('\n')
        for i in range(0,15): out.write('-')
        out.write('\n')

        if out is not sys.stdout:
            out.close()    

    def evolveByMove(self):
        """Lists all '(s,p)', where 's' is a state the active player can reach by moving the unit at 'p'."""
        returnList = []
        for currentPos in [p for p in list(self.units.keys()) if self.units[p] == self.activePlayer]:
            for p in self.nborDict[currentPos]:
                if self.boardheight[p[0]][p[1]] <= 3 and not p in list(self.units.keys()) \
                        and self.boardheight[p[0]][p[1]]-self.boardheight[currentPos[0]][currentPos[1]] <= 1:
                    newUnitDict = deepcopy(self.units)
                    del newUnitDict[currentPos]
                    newUnitDict.update({p:self.activePlayer})
                    newstate = state(activePlayer=self.activePlayer, boardheight=self.boardheight, units=newUnitDict)
                    returnList.append((newstate, p))
        return returnList

    def evolveByBuild(self, pos):
        """Return a list of states the active player can reach by building with a unit at 'pos'."""
        returnList = []        
        for p in self.nborDict[pos]:
            if self.boardheight[p[0]][p[1]] <= 3 and not p in list(self.units.keys()):
                newboardheight = tuple(tuple((self.boardheight[i][j]+1 if i == p[0] and j == p[1] \
                                                                    else self.boardheight[i][j]) \
                                                    for j in range(0, 5)) for i in range(0, 5))
                newstate = state(activePlayer=~self.activePlayer, boardheight=newboardheight, units=self.units)
                returnList.append(newstate)
        return returnList

    def invertPlayers(self):
        """Returns a new instance with activePlayer -> ~activePlayer and {pos :  player} -> {pos : ~player}."""
        unitsNew = {pos:~self.units[pos] for pos in self.units.keys()}
        return state(activePlayer=~self.activePlayer, boardheight=self.boardheight, units=unitsNew) 

    def winIn(self, k):     
        """Returns true iff active player has a (up to) k-turn winning strategy. Searchdepth: 2k-1:"""
        if k<=0: return False       
        
        Moves = self.evolveByMove()
        if Moves.__len__() == 0: return False        
    
        for m in [M[0] for M in Moves]:
            if 3 in [m.boardheight[pos[0]][pos[1]] for pos in m.units.keys()]: return True
                
        for BM in chain(*[m.evolveByBuild(p) for (m,p) in Moves]):                    
            if BM.loseIn(k-1): return True
        
        return False

    def loseIn(self, k):
        """Returns true iff for any play, the opponent has a k-turn winning strategy. Searchdepth: 2k"""
        # assuming k >= 0
        Moves = self.evolveByMove()
        if Moves.__len__() == 0: return True
        if k > 0:
            for m in [M[0] for M in Moves]:
                if 3 in [m.boardheight[pos[0]][pos[1]] for pos in m.units.keys()]: return False

            for BM in chain(*[m.evolveByBuild(p) for (m,p) in Moves]):
                if not BM.winIn(k): return False
            return True
        
        return False

    def value(self):
        """Returns an estimated value of the state, from the perspective of the active(!) player."""
        # Check if a player has a won. 
        #     Does not [!] check for the non-valid situation of two units on level 3      
        for pos in self.units.keys():
            if self.boardheight[pos[0]][pos[1]] == 3:
                if self.units[pos] == self.activePlayer: return 1
                else: return -1
        if self.evolveByMove().__len__() == 0: return -1

        # Compute some indicators 
        Moves = self.evolveByMove()
        turnNr = sum( chain(*[ self.boardheight[i] for i in range(0, 5)] ) ) # max: 93
        oppPos, ownPos = [], []
        for pos in self.units.keys():
            if self.units[pos] == self.activePlayer: ownPos.append(pos)
            else: oppPos.append(pos)
        
        # (1)   number of possible moves (min: 0, max: 16)
        I1 = 1/16*Moves.__len__()
        a1 = 1/3*(1/50*turnNr if turnNr <= 50 else 1)

        # (2)   summed heightscore of possible moves (min: 0, max: 3*number of moves => WIN)
        I2 = 0
        for m in [M[0] for M in Moves]:
            I2 += sum([int(m.units[pos] == self.activePlayer)*m.boardheight[pos[0]][pos[1]] \
                                        for pos in list(m.units.keys())])
        I2 = I2/(3*Moves.__len__())
        a2 = 1/3

        # (3)   number of adjacent opponents units that are not higher (min: 0, max: 2)
        I3 = 0
        for o in oppPos: 
            for p in ownPos: 
                if p in self.nborDict[o] and self.boardheight[p[0]][p[1]] >= self.boardheight[o[0]][o[1]] : 
                    I3 += 1/2
                    break
        a3 = 1/3

        return a1*I1 + a2*I2 + a3*I3 
         

    def alphabeta(self, depth, alpha, beta):
        """Negamax-algorithm to depth k with alpha-beta pruning, values estimated by 'state.value()'."""
        val = self.value()
        if depth == 0 or  val == -1 or val == 1: return val

        bestval = -1
        for BM in chain(*[m.evolveByBuild(p) for (m,p) in self.evolveByMove()]):
            bestval = max(bestval, -BM.alphabeta(depth-1, -beta, -alpha))
            alpha= max(alpha, bestval)
            if beta <= alpha: break
        return bestval
 