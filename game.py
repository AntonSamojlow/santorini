from state import *
from time import time

class player:
    """class player:
Blueprint for a player.
"""
    def __init__(self, info='random play'):     
        self.info = info

    def choseUnitPos(self, unitDic):
        while True:
            unitPos = (randint(0,4), randint(0,4))
            if unitPos not in unitDic.keys(): break
        return unitPos

    def getPlay(self, state):
        plays = state.listPlays()
        return plays[randint(0,plays.__len__()-1)]

class humanConsole(player):
    """class human(player):
A human that plays via the console.
"""
    def __init__(self, info='human'):     
        self.info = info

    def choseUnitPos(self, unitDic):
        """Asks to input a unit position'."""
        print('Choose a starting position for a unit...')
        freePos = [(i,j) for i in range(0,5) for j in range(0,5) if (i,j) not in unitDic.keys()]
        return self.inputCoord(freePos)

    def getPlay(self, state):
        """Asks to input a valid play for the active player and returns it as '(unit, dest, build)'."""
        unitPos, dest, build = None, None, None

        validPlays = state.listPlays()
        unitPositions = list({u for (u, d, b) in validPlays})
        print('Choose a unit to move from', unitPositions, '...')
        unitPos = self.inputCoord(unitPositions)
        
        destinations = [d for (u, d, b) in validPlays if u == unitPos]
        print('Choose a destination from', destinations, '...')
        dest = self.inputCoord(destinations)
            
        buildPositions = [b for (u, d, b) in validPlays if u == unitPos and d == dest]
        print('Choose where to build from', buildPositions, '...')
        build = self.inputCoord(buildPositions)
        return (unitPos, dest, build)

    def inputCoord(self, validList):        
        """Asks to input a coordinate from 'validList'."""
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


class game:
    """class game:
Performs a game between two players when calling __init__((firstplayer, secondplayer)).
Note that this version currently only supports 5x5-boards
Class variables:
  - nborDict          A dictionary {(i,j) : [(i,j),...]}, for lookup of neighbor positions.    
Instance variables:   
  - startState:       The starting state after the units have been positioned
  - players:          Tuple (firstplayer, secondplayer). Note the first player has index 0, the second index 1
  - plays:            List of plays made, which are of the form (unit, dest, build)
  - result            Equals 1 (2) if the first (second) player has won.
  - playtime          Time it tool to compute the plays
Methods (described in their own docstring):              
  save(file)
"""
    #--- constructor and class methods ----------------------------------------
    def __init__(self, players, startState=None, printProgress=False):        
        self.startState = startState if startState is not None else state()       
        self.players = players  # 
        self.plays = []
        self.result = 0     
        self.playtime = 0


        if(printProgress): 
            print('* start of game')
            print('* first player:', self.players[0].info)
            print('* second player:', self.players[1].info)

        if self.startState.unitDic.__len__() == 0:
            unitDic = {}          
            for i in range(0, 2):
                for playerNr in range(0, 2):
                    unitPos = self.players[playerNr].choseUnitPos(unitDic)
                    unitDic[unitPos] = not bool(playerNr)         
            self.startState.unitDic = unitDic

        currentstate = self.startState        
        if(printProgress): currentstate.print(playerInitials={True:'A', False:'B'})

        activeplayer = self.players[0]
        t0 = time()
        while True:
            play = activeplayer.getPlay(currentstate)
            self.plays.append(play)
            currentstate = currentstate.executePlay(play)
            
            activeplayer = self.players[int(not bool(self.players.index(activeplayer)))]
            if(printProgress): currentstate.print(playerInitials={activeplayer == self.players[0]:'A', \
                                                                         not activeplayer == self.players[0]:'B'})
            

            v = currentstate.score()
            if v == -1 or v == 1:
                self.result = 1 + int(bool(self.players.index(activeplayer)) == bool((1+v)/2))
                if(printProgress): print('* Player', self.result, 'won!')
                self.playtime = round(time()-t0,3)
                break

    #--- methods --------------------------------------------------------------   

    def save(self, file, format='text'):
        """Appends the game information to a file."""
        if format == 'text':
            with open(file, 'a', newline = None) as f:
                f.write('\n')
                f.write('1st player:  '+self.players[0].info+'\n')
                f.write('2nd player:  '+self.players[1].info+'\n')
                f.write('winner:      '+str(self.result)+'\n')
                f.write('start state: '+self.startState.toString()+'\n')
                strPlays = ''.join(str(P) for P in self.plays ).replace(')(','|')
                strPlays =  strPlays.replace(',','').replace(')','').replace('(','').replace(' ','')
                f.write('plays:       '+strPlays+'\n')
                f.write('playtime:    '+str(self.playtime)+' s\n')
                
                