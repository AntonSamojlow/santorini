from random import randint
import sys

class state:
    """class state:
Provides tools to check, display and save/export a Santorini gamestate.
Class variables:
  - nborDict          A dictionary {(i,j) : [(i,j),...]}, for lookup of neighbor positions.    
Instance variables:   
  - boardheight:      A 5x5-tuple with integer values between 0 and 4.
  - units:            A dictionary {(i,j) : bool}, positions as key and entry True iff owner is the active player.      
Methods (described in their own docstring):              
  atRandom, fromString(string), __eq__, toString, equivClass, reprString, print(file=None),
  listMoves, executeMove, listPlays, executePlay, winIn(k), loseIn(k), score, heuristicValue 
Written by Anton Samojlow, August 2018
"""

    #--- class variables ------------------------------------------------------
    
    # The following dict is preset for a 5x5 board. For a state wrt other board sizes, call state.resetNborDict().
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
    def atRandom(self, turn=None, unitsPP=2, rows=5, cols=5):
        """Returns an instance with random field entries and which passes 'self.valid()'."""
        unitDic={}
        for rep in range(0,2*unitsPP):
            while True: 
                i, j = randint(0,rows-1), randint(0,cols-1)
                if (i,j) not in unitDic.keys(): break
            if rep < unitsPP: unitDic[(i,j)] = True
            else: unitDic[(i,j)] = False
        
        maxTurns = 1+4*(rows*cols-unitsPP)
        if turn == None or turn > maxTurns:
            boardheight = tuple(tuple(randint(0, 2) if (row,col) in unitDic.keys() else randint(0, 4)\
                                for col in range(0, cols)) for row in range(0, rows))
        else:
            filling = 0
            bh = [[0 for col in range(0, cols)] for row in range(0, rows)]   
            while filling < turn:
                (row, col) = (randint(0,rows-1), randint(0,cols-1))  
                if bh[row][col] < 4 - 2*int((row, col) in unitDic.keys()):                    
                    bh[row][col] += 1
                    filling += 1        
            boardheight = tuple( tuple(bh[row][col] for col in range(0,cols)) for row in range(0,rows))
        return state(boardheight=boardheight, unitDic=unitDic)

    @classmethod
    def fromString(self, string):
        """Returns the state from a string 'boardheight|unitpos', generated from state.toString(self)."""    
        [bh, u] = string.split('|')

        unitDic = {}
        unitsPP = int(u.__len__()/2)        
        for j in range(0, 2*unitsPP, 2): 
            unitDic[(int(u[0 + j]), int(u[1 + j]))] = bool(j<unitsPP)
       
        bh = bh.split(',')       
        boardheight = tuple(tuple( int(entry) for entry in row ) for row in bh)

        return state(unitDic=unitDic, boardheight=boardheight)


    #--- methods --------------------------------------------------------------    
    def toString(self):
        """Generates a short and readable string-ID 'boardheight|unitpos' of the state."""        
        u = ''
        units_player = sorted([pos for pos in self.unitDic.keys() if self.unitDic[pos]], \
                                                    key=lambda k: (k[0], k[1]))
        units_opponent = sorted([pos for pos in self.unitDic.keys() if not self.unitDic[pos]],\
                                                    key=lambda k: (k[0], k[1]))
        for pos in units_player:
            u += str(pos[0])+str(pos[1])    
        for pos in units_opponent:
            u += str(pos[0])+str(pos[1])     

        bh = ''        
        for row in range(0, self.boardheight.__len__()):
            for col in range(0, self.boardheight[row].__len__()): bh += str(self.boardheight[row][col])
            if row < self.boardheight.__len__()-1: bh += ','
        return bh+'|'+u    
    
    def __eq__(self, other):
        """Overriding default: Returns true iff the fields entries agree irresp of the ordering of 'units'."""
        return self.toString() == other.toString()



    def equivClass(self):
        """Lists all symmetrically equivalent states for a SQUARE board. [!] Can (rarely) contain duplicates."""
        def H(p): return (self.boardheight.__len__() -1 -p[0], p[1])
        def D(p): return (p[1], p[0])
        symTrafos = [D, lambda p: H(D(p)), lambda p: D(H(D(p))), lambda p: H(D(H(D(p)))),
                     H, lambda p: D(H(p)), lambda p: H(D(H(p))), lambda p: p]

        def transformBy(trafo, s=self):
            newbh = [[0 for col in range(0,s.boardheight[0].__len__())] for row in range(0,s.boardheight.__len__())]
            for row in range(0,s.boardheight.__len__()):
                for col in range(0,s.boardheight[0].__len__()):
                    newbh[trafo((row,col))[0]][trafo((row,col))[1]] = s.boardheight[row][col] 
            newbh = tuple( tuple(row) for row in newbh)
            newunits = {trafo(unitPos) : s.unitDic[unitPos] for unitPos in s.unitDic.keys()}
            return state(boardheight=newbh, unitDic=newunits)        
        return [transformBy(trafo) for trafo in symTrafos]

    def reprString(self):
        """Returns the self.string() from equivClass which is lowest wrt alphabetical order."""
        return sorted([s.toString() for s in self.equivClass()])[0]


    def print(self, file=None, playerInitials={ True: 'P' , False: 'O'}):
        """Outputs a representation of the state to sys.stdout, or optionally to 'file'."""
        rows = self.boardheight.__len__()
        cols = self.boardheight[0].__len__()

        out = open(file, 'a', encoding='utf-8', newline=None ) if file else sys.stdout        
        for i in range(0, int(3*cols/2) -1 ): out.write('-')
        out.write(playerInitials[True])
        for i in range(0, int(1/2+ 3*cols/2) -1): out.write('-')
        out.write('\n')
        for row in range(0, rows):
            for col in range(0, cols):
                if (row,col) in self.unitDic.keys():                     
                    out.write(playerInitials[self.unitDic[(row,col)]])
                else: out.write(' ')
                out.write(str(self.boardheight[row][col])+' ')                                    
            out.write('\n')
        for i in range(0,3*cols): out.write('-')
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
        newboardheight = tuple(tuple((self.boardheight[row][col]+1 if (row, col) == (build[0], build[1]) \
                                                                    else self.boardheight[row][col]) \
                                      for col in range(0, self.boardheight[0].__len__())) 
                                for row in range(0, self.boardheight.__len__()))
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
    
    def resetNborDict(self):
        """Resets the class variable state.nborDict based on the dimensions of the calling state."""   
        rows, cols = self.boardheight.__len__(), self.boardheight[0].__len__()
        self.nborDict.clear()
        for row in range(0,rows):
            for col in range(0,cols):
                nbors=[]
                for var in [(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]:
                    if row+var[0] >=0 and row+var[0] < rows and col+var[1] >=0 and col+var[1] < cols:
                        nbors.append((row+var[0],col+var[1]))
                self.nborDict.update({(row,col) : nbors})
    
    def score(self):
        """Returns +1/-1 if active player won/lost, 0 else. [!] Inconsistent if two or more units are on level 3."""        
        for pos in self.unitDic.keys():
            if self.boardheight[pos[0]][pos[1]] == 3: return -1+2*int(self.unitDic[pos])               
        if self.listMoves().__len__() == 0: return -1
        return 0         

    def heuristicValue(self):
        """Returns a heuristic value of the gamestate, from the perspective of the active(!) player."""
        # Check if a player has a won. 
        #     Does not [!] check for the non-valid situation of two units on level 3      
        score = self.score()
        if score != 0: return score

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