class alphabeta():    
    """class alphabeta:
Implements the alphabeta search for a two-player game. Assumes that the value 
of 'gamestate' is always given from the active players perspective and that
-1 = loss, +1 = win. 
Required instance variables, to be passed on __init__:   
  - value:        A function: {gamestate} -> [-1,1] 
  - getPlays:     A function: {gamestate} -> {valid plays}
  - makePlay:     A function: { g,p | p valid play for g} -> {gamestate}
Optional instance variables: 
  - sortKey:      A function: { g,p | p valid play for g} -> {reals}
                  This is the movereordering. Defaults to g.value()
  - ID_fct:       A function: {g} -> {IDs}
                  Used as key for the transposition table.  
                  Hence, symmetry reductions can be used here. 
                  [!] If not set, the call self.tabled(...) will crash.
  - transpTable:  A transposition table {ID : depth, flag, score, play}.
                  The value of flag is 'exact', 'lowerbound' or 'upperbound'.
                  If no table is given, an empty one will be created.

Methods (described in their own docstring): 
  basic, reordered, tabled
Written by Anton Samojlow, August 2018
"""
    def __init__(self, value_fct, getPlays_fct, makePlay_fct, 
                 sortKey = None, ID_fct = None, abTable={}):
        self.value = value_fct
        self.getPlays = getPlays_fct
        self.makePlay = makePlay_fct
        
        if sortKey is not None: self.sortKey = sortKey
        else: self.sortKey = lambda childData: self.value(childData[0])  
        
        self.ID = ID_fct
        self.transpTable = abTable
        
    def basic(self, gamestate, depth, alpha=-1, beta=1):
        """Basic alphabeta algorithm. Returns (v,p), where p=play that leads to v='ab()'."""
        bestval = self.value(gamestate)
        if depth == 0 or  bestval == -1 or bestval == 1: return (bestval, None)

        bestval, bestplay = -1, None
        for play in self.getPlays(gamestate):
            val =  - self.basic(self.makePlay(gamestate, play), depth-1, 
                             alpha=-beta, beta=-max(alpha, bestval))[0]
            if val >= bestval: bestval, bestplay = val, play
            if bestval >= beta: break

        return (bestval, bestplay)

    def reordered(self, gamestate, depth, alpha=-1, beta=1):
        """Alphabeta with ordering. Returns (v,p), where p=play that leads to v='ab()'."""
        bestval = self.value(gamestate)
        if depth == 0 or  bestval == -1 or bestval == 1: return (bestval, None)
        
        bestval, bestplay = -1, None        
        childData = sorted([(self.makePlay(gamestate, P), P) for P in self.getPlays(gamestate)], key=self.sortKey) 
        for (newstate, play) in childData:
            val =  -self.reordered(newstate, depth-1, alpha=-beta, beta=-max(alpha, bestval))[0]
            if val >= bestval: bestval, bestplay = val, play
            if bestval >= beta: break
        
        return (bestval, bestplay)

    def tabled(self, gamestate, depth, alpha=-1, beta=1):
        """Alphabeta with ordering AND transposition table. Returns (v,p), where p=play that leads to v='ab()'."""
        bestval = self.value(gamestate)
        if depth == 0 or  bestval == -1 or bestval == 1: return (bestval, None)
                    
        ID = self.ID(gamestate)
        
        if ID in self.transpTable.keys():
            (d, flag, score, play) = self.transpTable[ID]
            if d >= depth:
                if flag == 'exact': return (score, play)
                elif flag == 'lowerbound': alpha = max(alpha, score)
                elif flag == 'upperbound': beta = min(beta, score)
        
        bestval, bestplay = -1, None
        childData = sorted([(self.makePlay(gamestate, P), P) for P in self.getPlays(gamestate)], key=self.sortKey) 
        for (newstate, play) in childData:
            val =  -self.tabled(newstate ,depth-1, alpha=-beta, beta=-max(alpha, bestval))[0]
            if val >= bestval: bestval, bestplay = val, play
            if bestval >= beta: break
        
        d, flag, score, play = depth, 'exact', bestval, bestplay
        if bestval <= alpha:
            flag = 'upperbound'
        elif bestval >= beta:
            flag = 'lowerbound'
        self.transpTable[ID]=(d,flag,score,play)

        return (bestval, bestplay)