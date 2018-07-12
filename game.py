from state import*
from time import time
class game:
    """| class game:
| DOCSTRING
"""
    
    #--- constructor ----------------------------------------------------------
    def __init__(self, activePlayer=None, boardheight=((0,)*5,)*5, units={}):        
        self.state = state()
        state.init_nborGrid()
        self.HUMAN = state.player.BLUE
        self.AI = state.player.RED
    
    #--- methods --------------------------------------------------------------
    def AIplay(self, depth=2):
        """Returns a new state, according to the play of the AI."""
        plays = list(chain(*[m.evolveByBuild(p) for (m,p) in self.state.evolveByMove()]))
        
        minvalue, minplay = 1, plays[0]
        for play in plays:
            val = play.alphabeta(depth-1, -1, 1)
            if val == -1: return play
            if minvalue > val: minvalue, minplay = val, play            
        return minplay    
    
    def run(self):
        print('* --- start of game --- ')
        print('* Playing: HUMAN as', self.HUMAN.name, 'versus the AI as', self.AI.name)
        
        # set starting player
        self.state.activePlayer = self.HUMAN if bool(randint(0,2)) else self.AI
        print('*',self.state.activePlayer.name,'starts')        
        
        # choose starting positions
        freePos = [(i,j) for i in range(0,5) for j in range(0,5)]      
        for i in range(0,4):
            if self.state.activePlayer == self.HUMAN:
                print('Choose a starting position for a unit...')
                unitPos = self.inputCoord(freePos)
                freePos.remove(unitPos)
                self.state.units.update({unitPos : self.state.activePlayer})
                self.state.activePlayer = ~self.state.activePlayer
            else:
                while True:
                    unitPos = (randint(1,3), randint(1,3))
                    if unitPos in freePos:
                        freePos.remove(unitPos)
                        self.state.units.update({unitPos : self.state.activePlayer})
                        self.state.activePlayer = ~self.state.activePlayer
                        print('* The AI places a unit at',unitPos)                        
                        break

        # loop until a player has won
        turnNR = 0  
        AIsearchdepth = 2
        while True:            
            turnNR += 1
            self.state.print()
            
            if turnNR == 10 or turnNR == 30:
                AIsearchdepth += 1
                print('* The AI received support from the future!')

            if self.state.activePlayer == self.HUMAN:
                print('* It is your turn,', self.HUMAN.name)
                self.state = self.executePlay(*self.inputPlayFor(self.HUMAN))
            else:
                print('* The AI looks for human weaknesses...')
                t0 = time()
                self.state = self.AIplay(depth=AIsearchdepth)
                print('*    ...and finished in only',round(time()-t0,1),'s.')
            
            val = self.state.value()            
            if val == 1:
                self.state.print()
                if self.state.activePlayer == self.HUMAN: print('* The HUMAN player wins!')
                else: print('* The AI crushed you!')
                break
            if val == -1:
                self.state.print()
                if self.state.activePlayer == self.AI: print('* The HUMAN player wins!')
                else: print('* The AI crushed you!')
                break        
        print('')        
        print('* --- end of game ---')
        return None

    def inputPlayFor(self, player):
        unitPos, dest, build = None, None, None

        unitPositions = [p for p in self.state.units.keys() if self.state.units[p] == player]        
        unitMoves = {}
        for unit in unitPositions:
            moves = []
            for p in self.state.nborDict[unit]:
                if self.state.boardheight[p[0]][p[1]] <= 3 and not p in list(self.state.units.keys()) \
                        and self.state.boardheight[p[0]][p[1]]-self.state.boardheight[unit[0]][unit[1]] <= 1:
                    moves.append(p)
            if moves.__len__() == 0: unitPositions.remove(unit)
            else: unitMoves.update({unit : moves})

        print('Choose a unit to move from', unitPositions, '...')
        unitPos = self.inputCoord(unitPositions)
        
        print('Choose a destination from', unitMoves[unitPos], '...')
        dest = self.inputCoord(unitMoves[unitPos])
           
        buildPositions = []
        for p in self.state.nborDict[dest]:
            if self.state.boardheight[p[0]][p[1]] <= 3 and (p==unitPos or not p in list(self.state.units.keys())):
                buildPositions.append(p)
      
        print('Choose where to build from', buildPositions, '...')
        build = self.inputCoord(buildPositions)

        return (unitPos, dest, build)

    def inputCoord(self, validList):
        error = "[!] Valid coordinates are 'xy' with x=row, y=column being integers in [0,4].\
                \nTry again..."

        while True:
            try:
                rawIn = input()
                if rawIn == 'exit': exit()                
                x, y = int(rawIn[0]), int(rawIn[1])
            except IndexError:
                print(error)
                continue
            except ValueError:
                print(error)
                continue
            if rawIn.__len__() != 2:
                print(error)         
            elif (x,y) not in validList:
                print((x,y), 'is not from', validList,'\nTry again...')
            else:
                return (x,y)

    def executePlay(self, unitPos, dest, build):
        newUnitDict = deepcopy(self.state.units)
        del newUnitDict[unitPos]
        newUnitDict.update({dest : self.state.activePlayer}) 
        newboardheight = tuple(tuple((self.state.boardheight[i][j]+1 if i == build[0] and j == build[1] \
                                                                    else self.state.boardheight[i][j]) \
                                                    for j in range(0, 5)) for i in range(0, 5))
        return state(activePlayer=~self.state.activePlayer, boardheight=newboardheight, units=newUnitDict)
      
    
