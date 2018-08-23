from math import sqrt, log
from time import time
from random import shuffle, choice


class MCTS:
    """class alphabeta:
Implements a basic Monte Carlo tree search with LCB (lower confidence bound, 
since we seek to [!] MINIMIZE [!] the opponents winning chance).
Required instance variables, to be passed on __init__:
  - getChildren:    A function: node n -> ALL childs of n
  - getRandomChild: A function: node n -> ONE random child of n
  - isTerminal:     A function: node n -> True iff n is terminal, else False
  - isWinningLeaf:  A function: node n -> True iff 'n = win', else False
Optional instance variables: '
  - LCB_cst:        Constant of the UCB1-algorithm, defaults to 2
  - statistics:     Saved statistics can be passed here, defaults to empty dict
Methods (described in their own docstring):
  run, printStats
Written by Anton Samojlow, August 2018             
"""

    def __init__(self, getChildren_fct, getRandomChild_fct, 
                 isTerminal_fct, isWin_fct, statistics={}, LCB_cst=2):
        
        self.statistics = statistics       
        self.LCB_cst = LCB_cst

        self.getChildren = getChildren_fct
        self.getRandomChild = getRandomChild_fct
        self.isTerminal = isTerminal_fct
        self.isWininingLeaf = isWin_fct

    def run(self, root, maxSec):
        """Runs the MC tree search for maxSec."""
        def LCB(node):
            (v,n) = self.statistics[node]
            return v/n - sqrt(self.LCB_cst*log(Nplays)/n)
        
        def select(parent):
            if self.isTerminal(parent): return [parent]
            children = self.getChildren(parent)
            shuffle(children)
            for child in children:
                if child not in self.statistics.keys():
                    return [parent, child] 
            
            return [parent] + select(min(children, key=LCB))
        
        def simulate(parent):
            currentnode = parent
            randomplays = [currentnode]
            while not self.isTerminal(currentnode):
                currentnode = self.getRandomChild(currentnode)
                randomplays.append(currentnode)            
            return randomplays

        if root in self.statistics.keys(): Nplays = self.statistics[root][1]
        else: Nplays = 0 
        t0=time()
        print('* Running MCTS for', maxSec, 'seconds...')
        
        while time() - t0  < maxSec:   
            record = []
            Nplays += 1    

            record += select(root)
            leaf = record[-1] 

            if not self.isTerminal(leaf): 
                record += simulate(leaf)  
            
            rootPlayerWon = (self.isWininingLeaf(record[-1])) != (record.__len__() % 2 == 0)

            for i in range(0, record.__len__()):
                if record[i] not in self.statistics.keys(): self.statistics.update({record[i]: [0,0]})
                self.statistics[record[i]][1] += 1
                if rootPlayerWon == (i % 2 == 0):
                    self.statistics[record[i]][0] += 1
    
    def printStats(self, root, NrBestChildren=10):
        """Shows statistics for root and its best (fewest wins) children."""
        if root not in self.statistics.keys():
            print('* no MCTS stats for', root)
            return None
        
        print('* Displaying MCTS stats for rootnode:')        
        print(root)
        print('* It has been encountered',self.statistics[root][1],'times.',
                'Estimate of winning chance:', 
                round(100*self.statistics[root][0]/self.statistics[root][1], 3),'%')
        
        childStats = [ (child,round(100*(1-  self.statistics[child][0]/ self.statistics[child][1]),3)) \
                        for child in self.getChildren(root)]
        childStats.sort( key = lambda e: -e[1])
        print('* Stats for the best',NrBestChildren,'childnodes:')
        for i in range(0, min(NrBestChildren, childStats.__len__())):
            print(childStats[i][0], ':', childStats[i][1], '%')
    
    
