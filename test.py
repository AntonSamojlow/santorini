from state import*
from alphabeta import*
from mcts import*
from game import*


def state_test():
    print('* example for the use of state.py *')
    print('* a randomly chosen state at turn 20:')
    s=state.atRandom(turn=20)
    s.print()
    print('* its string:',s.toString())
    print('* its equiv. class:', [t.toString() for t in s.equivClass()])
    print('* its unique representative:', s.reprString())
    print('* the state reconstructed from its string:')
    state.fromString(s.toString()).print()
    print('* score:',s.score())
    print('* heuristic value:',s.heuristicValue())
    print('* winIn(2):',s.winIn(2))
    print('* loseIn(2):',s.loseIn(2))
    randplay = choice(s.listPlays())
    print('* a randomly chosen play:',randplay)
    print('* the new state generated from the play is:')
    s.executePlay(randplay).print()
    print('* end of example')
    print('')

def mcts_test():        
    def getChildren(stateStr):
        s = state.fromString(stateStr)
        return [c.toString() for c in [s.executePlay(p) for p in s.listPlays()]]

    def getRandomChild(stateStr):
        s = state.fromString(stateStr)
        return  s.executePlay(choice(s.listPlays())).toString()

    def isTerminal(stateStr):
        score = state.fromString(stateStr).score()
        return score == 1 or score == -1 

    def isWin(stateStr):
        return state.fromString(stateStr).score() == 1

    MonteTrial = MCTS(getChildren_fct=getChildren, getRandomChild_fct=getRandomChild,
                        isTerminal_fct=isTerminal, isWin_fct=isWin)

    print('* example for the use of mcts.py *')
    print('* randomly picked the state:')
    s=state.atRandom()
    s.print()
    MonteTrial.run(s.toString(),10)
    MonteTrial.printStats(s.toString()) 
    print('* end of example')
    print('')

def alphabeta_test():
    ab = alphabeta(lambda s : s.heuristicValue(), 
                   lambda s : s.listPlays(),
                   lambda s,p : s.executePlay(p),
                   ID_fct= lambda s: s.reprString())
    
    print('* example for the use of alphabeta.py *')
    print('* randomly picked the state:')
    s=state.atRandom()
    s.print()
    for depth in range(1,4):
        print('* alphabeta with move ordering and transposition table at depth')
        print('* at depth', depth, ':', ab.tabled(s, depth))
    print('* end of example')
    print('')

def game_test():        
    class abAI(player):
        def __init__(self, depth =2):
            self.depth =depth     
            self.info = "Alphabeta AI with move ordering and transposition table. Searchdepth: "+str(self.depth)
            self.ab = alphabeta( lambda s : s.heuristicValue(), 
                                        lambda s : s.listPlays(),
                                        lambda s,p : s.executePlay(p),
                                        ID_fct= lambda s: s.reprString() )

        def choseUnitPos(self, unitDic):
            while True:
                unitPos = (randint(0,4), randint(0,4))
                if unitPos not in unitDic.keys(): break
            return unitPos

        def getPlay(self, state):
            return self.ab.tabled(state,self.depth)[1]

    print('* example for the use of game_test.py *')
    g = game((abAI(),abAI()), printProgress=True)
    g.save('out.txt')
    print('* the game has been saved to "out.txt"')
    print('* Now, a game human vs AI.*')
    print('* NOTE: if prompted for an input, you may type "exit" to quit')
    g = game((abAI(),humanConsole()), printProgress=True)
    g.save('out.txt')
    print('* the game has been saved to "out.txt"')
    print('* end of example')
    print('')
