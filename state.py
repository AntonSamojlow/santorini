# To Do:
#   - add a logger, see https://docs.python.org/3/howto/logging.html#logging-basic-tutorial
#   - a methods that checks two states for symmetries: 
#          90 degree -rotations, mirroring (diagonals, horizontal, vertical)

from random import randint
import sys
#------------------------------------------------------------------------------
# the main class
#------------------------------------------------------------------------------

class state:
    """| class state:
| Provides tools to check, analyse, display and save/export a Santorini gamestate.
| Class variables:
|   - abTable           Transposition table {state.toString() : (depth, falg, score, play)} for alphabeta-searches
|   - nborDict          A dictionary {(i,j) : [(i,j),...]}, for lookup of neighbor positions.    
| Instance variables:   
|   - boardheight:      A 5x5-tuple with integer values between 0 and 4.
|   - units:            A dictionary {(i,j) : bool}, positions as key and entry True iff owner is the active player.      
|
| Methods (described in their own docstring):              
|   atRandom, fromString(string), __eq__, toString, valid, print(file=None),
|   listMoves, generateMove, listPlays, generatePlay,
|   winIn(k), loseIn(k), value, ab(depth,alpha,beta), ab_order(depth,alpha,beta,sortfunction),
|   ab_order_table(depth,alpha,beta,sortfunction)       
"""

    #--- class variables ------------------------------------------------------
    abTable = {} 

    nborDict = {(0, 0): [(0, 1), (1, 0), (1, 1)], 
                (0, 1): [(0, 0), (0, 2), (1, 0), (1, 1), (1, 2)], 
                (0, 2): [(0, 1), (0, 3), (1, 1), (1, 2), (1, 3)],
                (0, 3): [(0, 2), (0, 4), (1, 2), (1, 3), (1, 4)], 
                (0, 4): [(0, 3), (1, 3), (1, 4)], 
                (1, 0): [(0, 0), (0, 1), (1, 1), (2, 0), (2, 1)], 
                (1, 1): [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)], 
                (1, 2): [(0, 1), (0, 2), (0, 3), (1, 1), (1, 3), (2, 1), (2, 2), (2, 3)], 
                (1, 3): [(0, 2), (0, 3), (0, 4), (1, 2), (1, 4), (2, 2), (2, 3), (2, 4)], 
                (1, 4): [(0, 3), (0, 4), (1, 3), (2, 3), (2, 4)], 
                (2, 0): [(1, 0), (1, 1), (2, 1), (3, 0), (3, 1)], 
                (2, 1): [(1, 0), (1, 1), (1, 2), (2, 0), (2, 2), (3, 0), (3, 1), (3, 2)], 
                (2, 2): [(1, 1), (1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2), (3, 3)], 
                (2, 3): [(1, 2), (1, 3), (1, 4), (2, 2), (2, 4), (3, 2), (3, 3), (3, 4)], 
                (2, 4): [(1, 3), (1, 4), (2, 3), (3, 3), (3, 4)], 
                (3, 0): [(2, 0), (2, 1), (3, 1), (4, 0), (4, 1)], 
                (3, 1): [(2, 0), (2, 1), (2, 2), (3, 0), (3, 2), (4, 0), (4, 1), (4, 2)], 
                (3, 2): [(2, 1), (2, 2), (2, 3), (3, 1), (3, 3), (4, 1), (4, 2), (4, 3)], 
                (3, 3): [(2, 2), (2, 3), (2, 4), (3, 2), (3, 4), (4, 2), (4, 3), (4, 4)], 
                (3, 4): [(2, 3), (2, 4), (3, 3), (4, 3), (4, 4)], 
                (4, 0): [(3, 0), (3, 1), (4, 1)], 
                (4, 1): [(3, 0), (3, 1), (3, 2), (4, 0), (4, 2)], 
                (4, 2): [(3, 1), (3, 2), (3, 3), (4, 1), (4, 3)], 
                (4, 3): [(3, 2), (3, 3), (3, 4), (4, 2), (4, 4)], 
                (4, 4): [(3, 3), (3, 4), (4, 3)]}   
    
    #--- constructor and class methods ----------------------------------------
    def __init__(self, boardheight=((0,)*5,)*5, unitDic={}):        
        self.boardheight = boardheight
        self.unitDic = unitDic
        
    @classmethod
    def atRandom(self, turn=None):
        """Returns an instance with random field entries and which passes 'self.valid()'."""
        unitDic={}
        for rep in range(0,4):
            while True: 
                i, j = randint(0,4), randint(0,4)
                if (i,j) not in unitDic.keys(): break
            if rep < 2: unitDic[(i,j)] = True
            else: unitDic[(i,j)] = False

        if turn == None or turn > 92:
            boardheight = tuple(tuple(randint(0, 2) if (i,j) in unitDic.keys() else randint(0, 4)\
                                for j in range(0, 5)) for i in range(0, 5))
        else:
            filling = 0
            bh = [[0 for i in range(0, 5)] for j in range(0, 5)]            
            while filling < turn:
                (row, col) = (randint(0,4), randint(0,4))                
                if bh[row][col] < 4 - 2*int((row, col) in unitDic.keys()):                    
                    bh[row][col] += 1
                    filling += 1        
            for i in range(0,5): bh[i] = tuple(bh[i])  
            boardheight = tuple(bh)
        return state(boardheight=boardheight, unitDic=unitDic)

    @classmethod
    def fromString(self, string):
        """Returns the state from a string 'boardheight|unitpos', generated from state.toString(self)."""    
        [bh, u] = string.split('|')

        unitDic = {}
        unitDic[(int(u[0]), int(u[1]))] = True
        unitDic[(int(u[2]), int(u[3]))] = True
        unitDic[(int(u[4]), int(u[5]))] = False
        unitDic[(int(u[6]), int(u[7]))] = False

        boardheight = []
        bh = bh.split(',')
        for row in bh: 
            r = []
            for j in range(0,5): r.append(int(row[j]))
            boardheight.append(tuple(r))
        boardheight = tuple(boardheight)

        return state(unitDic=unitDic, boardheight=boardheight)


    #--- methods --------------------------------------------------------------    
    def toString(self):
        """Generates a short and readable string-ID 'boardheight|unitpos' of the state."""        
        u = ''
        bh = ''
        units_player = sorted([pos for pos in self.unitDic.keys() if self.unitDic[pos]], \
                                                    key=lambda k: (k[0], k[1]))
        units_opponent = sorted([pos for pos in self.unitDic.keys() if not self.unitDic[pos]],\
                                                    key=lambda k: (k[0], k[1]))
             
        for pos in units_player:
            u += str(pos[0])+str(pos[1])    
        for pos in units_opponent:
            u += str(pos[0])+str(pos[1])     

        for i in range(0, 5):
            for j in range(0, 5): bh += str(self.boardheight[i][j])
            if i < 4: bh += ','
        return bh+'|'+u    
    
    def __eq__(self, other):
        """Overriding default: Returns true iff the fields entries agree irresp of the ordering of 'units'."""
        return self.toString() == other.toString()
   
    def valid(self):
        """Returns true iff the instance is 'syntactically correct', as defined in the comments of this method."""
        #   1) boardheight is a 5x5 tuple with values in {0,1,2,3,4}
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

        #   2) units is a dict with keys '(i,j)' and boolean entries
        if not (isinstance(self.unitDic, dict) and self.unitDic.__len__() == 4):
            return False
        else:
            for posKey in self.unitDic.keys():
                if not (isinstance(posKey, tuple) and posKey.__len__() == 2\
                                            and isinstance(self.unitDic[posKey], bool)):
                    return False
                elif not (posKey[0] in range(0, 5, 1) and posKey[1] in range(0, 5, 1)):
                    return False

        #   3) each player should have exactly two units
        if not (list(self.unitDic.values()).count(True) == 2 \
                and list(self.unitDic.values()).count(False) == 2):
            return False

        #   4) units are not allowed on a boardheight of level 4 or 3 (winning situation)
        for pos in self.unitDic.keys():
            if self.boardheight[pos[0]][pos[1]] >= 3:
                return False

        # All the problems should (!) have been caught, so we return TRUE
        return True

    def print(self, file=None, playerInitials={ True: 'p' , False: 'o'}):
        """Outputs a representation of the state to sys.stdout, or optionally to 'file'."""
        out = open(file, 'a', encoding='utf-8', newline=None ) if file else sys.stdout        
        for i in range(0,15): out.write('-')
        out.write('\n')
        for i in range(0, 5):
            for j in range(0, 5):
                if (i,j) in self.unitDic.keys():                     
                    out.write(playerInitials[self.unitDic[(i,j)]])
                else: out.write(' ')
                out.write(str(self.boardheight[i][j])+' ')                                    
            out.write('\n')
        for i in range(0,15): out.write('-')
        out.write('\n')

        if out is not sys.stdout:
            out.close()    

    def listMoves(self):
        """Return a list of moves '(unitToMove, destination)'."""
        moves = []
        for unitPos in [p for p in list(self.unitDic.keys()) if self.unitDic[p]]:
            for dest in self.nborDict[unitPos]:
                if self.boardheight[dest[0]][dest[1]] <= 3\
                        and not dest in list(self.unitDic.keys())\
                        and self.boardheight[dest[0]][dest[1]]-self.boardheight[unitPos[0]][unitPos[1]] <= 1:
                    moves.append((unitPos,dest))
        return moves
   
    def executeMove(self, move):
        """Returns a new state after executing the move. Active player is not changed."""
        (unitPos, dest) = move 
        newunitDic = {pos : self.unitDic[pos] for pos in self.unitDic.keys() if pos != unitPos}
        newunitDic[dest] = True
        return state(boardheight=self.boardheight, unitDic=newunitDic)
                    
    def listPlays(self):
        """Return a list of plays '(unitToMove, destination, buildPos)'."""
        plays = []
        moves = self.listMoves()        
        for move in moves:
            plays.append((move[0],move[1],move[0]))
            for buildPos in self.nborDict[move[1]]:
                if self.boardheight[buildPos[0]][buildPos[1]] != 4\
                        and buildPos not in list(self.unitDic.keys()):
                    plays.append((move[0],move[1],buildPos))
        return plays
    
    def executePlay(self, play):
        """Returns a new state after executing the play. Active player is changed."""
        (unitPos, dest, build) = play
        newunitDic = {pos : not self.unitDic[pos] for pos in self.unitDic.keys() if pos != unitPos}
        newunitDic[dest] = False
        newboardheight = tuple(tuple((self.boardheight[i][j]+1 if (i, j) == (build[0], build[1]) \
                                                                    else self.boardheight[i][j]) \
                                                    for j in range(0, 5)) for i in range(0, 5))
        return state(boardheight=newboardheight, unitDic=newunitDic)       

    def winIn(self, k):     
        """Returns true iff active player has a (up to) k-turn winning strategy. Searchdepth: 2k-1:"""
        if k<=0: return False       
        
        Moves = self.listMoves()
        if Moves.__len__() == 0: return False        
    
        for m in [self.executeMove(M) for M in Moves]:
            if 3 in [m.boardheight[pos[0]][pos[1]] for pos in m.unitDic.keys()]: return True
                
        for play in [self.executePlay(P) for P in self.listPlays()]:                    
            if play.loseIn(k-1): return True
        
        return False

    def loseIn(self, k):
        """Returns true iff for any play, the opponent has a k-turn winning strategy. Searchdepth: 2k"""
        # assuming k >= 0
        Moves = self.listMoves()
        if Moves.__len__() == 0: return True
        if k > 0:
            for m in [self.executeMove(M) for M in Moves]:
                if 3 in [m.boardheight[pos[0]][pos[1]] for pos in m.unitDic.keys()]: return False

            for play in [self.executePlay(P) for P in self.listPlays()]:      
                if not play.winIn(k): return False
            return True
        
        return False

    def value(self):
        """Returns an estimated value of the state, from the perspective of the active(!) player."""
        # Check if a player has a won. 
        #     Does not [!] check for the non-valid situation of two units on level 3      
        for pos in self.unitDic.keys():
            if self.boardheight[pos[0]][pos[1]] == 3: return -1+2*int(self.unitDic[pos])               
        if self.listMoves().__len__() == 0: return -1

        # Compute some indicators 
        Moves = self.listMoves()
        turnNr = sum( [ sum(self.boardheight[i]) for i in range(0, 5)] ) # max: 93
        
        units_player = [pos for pos in self.unitDic.keys() if self.unitDic[pos]]
        units_opponent = [pos for pos in self.unitDic.keys() if not self.unitDic[pos]]
        
        # (1)   number of possible moves (min: 0, max: 16)
        I1 = 1/16*Moves.__len__()
        a1 = 1/3*(1/50*turnNr if turnNr <= 50 else 1)

        # (2)   summed heightscore of possible moves (min: 0, max: 3*number of moves => WIN)
        I2 = 0
        for m in [self.executeMove(M) for M in Moves]:
            I2 += sum([int(m.unitDic[pos])*m.boardheight[pos[0]][pos[1]] \
                                        for pos in list(m.unitDic.keys())])
        I2 = I2/(3*Moves.__len__())
        a2 = 1/3

        # (3)   number of adjacent opponents units that are not higher (min: 0, max: 2)
        I3 = 0
        for o in units_opponent: 
            for p in units_player: 
                if p in self.nborDict[o] and self.boardheight[p[0]][p[1]] >= self.boardheight[o[0]][o[1]] : 
                    I3 += 1/2
                    break
        a3 = 1/3

        return a1*I1 + a2*I2 + a3*I3 
        

    def ab(self, depth, alpha=-1, beta=1):
        """Simple alphabeta. Returns (v,p), where p=play that leads to v='ab()'."""
        bestval = self.value()
        if depth == 0 or  bestval == -1 or bestval == 1: return (bestval, None)
        
        bestval, bestplay = -1, None
        plays = self.listPlays()
        for play in plays:
            val =  -self.executePlay(play).ab(depth-1, alpha=-beta, beta=-max(alpha, bestval))[0]
            if val >= bestval: bestval, bestplay = val, play
            if bestval >= beta: break
        
        return (bestval, bestplay)

    def ab_order(self, depth, alpha=-1, beta=1, sortfunction = lambda playdata: playdata[1].value()):
        """Alphabeta with ordering. Returns (v,p), where p=play that leads to v='ab()'."""
        bestval = self.value()
        if depth == 0 or  bestval == -1 or bestval == 1: return (bestval, None)
        
        bestval, bestplay = -1, None
        playdata = sorted([(P,self.executePlay(P)) for P in self.listPlays()] , key=sortfunction) 
        for (play,newstate) in playdata:
            val =  -newstate.ab_order(depth-1, alpha=-beta, beta=-max(alpha, bestval), 
                                                    sortfunction = sortfunction)[0]
            if val >= bestval: bestval, bestplay = val, play
            if bestval >= beta: break
        
        return (bestval, bestplay)


    def ab_order_table(self, depth, alpha=-1, beta=1, sortfunction = lambda playdata: playdata[1].value()):
        """Alphabeta with ordering and transposition table. Returns (v,p), where p=play that leads to v='ab()'."""
        bestval = self.value()
        if depth == 0 or  bestval == -1 or bestval == 1: return (bestval, None)
                    
        ID = self.toString()
        if ID in self.abTable.keys():
            (d, flag, score, play) = self.abTable[ID]
            if d >= depth:
                if flag == 'exact': return (score, play)
                elif flag == 'lowerbound': alpha = max(alpha, score)
                elif flag == 'upperbound': beta = min(beta, score)
        
        bestval, bestplay = -1, None
        playdata = sorted([(P,self.executePlay(P)) for P in self.listPlays()] , key=sortfunction) 
        for (play,newstate) in playdata:
            val =  -newstate.ab_order_table(depth-1, alpha=-beta, beta=-max(alpha, bestval), 
                                                    sortfunction = sortfunction)[0]
            if val >= bestval: bestval, bestplay = val, play
            if bestval >= beta: break
        
        d, flag, score, play = depth, 'exact', bestval, bestplay
        if bestval <= alpha:
            flag = 'upperbound'
        elif bestval >= beta:
            flag = 'lowerbound'
        self.abTable[ID]=(d,flag,score,play)

        return (bestval, bestplay)