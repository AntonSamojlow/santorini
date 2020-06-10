import logging
import json

from random import choice
from numpy import array

LOGGER = logging.getLogger(__name__)

def getLogger():
    return LOGGER

class GameGraph():
    def __init__(self, 
        description="asd", 
        childrentable = {}, 
        roots = (),
        outdegree_max = 4):
        self._childrentable = childrentable
        self.roots = roots
        self.outdegree_max = outdegree_max
        self.description = description

        # examplestructure
        if childrentable == {}:
            self._childrentable= {
                "a" : ("1", "2"),
                "b" : ("3",),
                "c" : ("1", "4"),
                "1" : None,
                "2" : (),
                "3" : None,
                "4" : None}
        if roots == ():
            self.roots = ("a", "b", "c")

    @property
    def vertices(self):
        return set(self._childrentable.keys())

    @property
    def edges(self):
        edges = []
        for p,children in self._childrentable.items():
            if children is not None: 
                edges += [(p,c) for c in children]
        return set(edges)
    
    def as_json(self, indent=2) -> str:       
        """Turn the Graph into a (relatively) compact JSON string. May need
        adjustmemt for specific games / subclassed GameGraphs."""
        class GameGraphEncoder(json.JSONEncoder):
            """Custom Encode for class GameGraph."""

            def default(self, obj):  # pylint: disable=E0202
                if isinstance(obj, GameGraph):                  
                    return {
                        obj.__class__.__name__: True,
                        'description' : obj.description,
                        'childrentable' : obj._childrentable,
                        'outdegree_max': obj.outdegree_max,
                        'roots' : obj.roots}
                return json.JSONEncoder.default(self, obj)
        return json.dumps(self, cls=GameGraphEncoder, indent=indent)

    @classmethod
    def from_json(cls, string) -> 'GameGraph':
        """Returns the GameGraph from a JSON string, inverse of 'to_json'.

        [!] Must be reimplemented for other graphs with different attributes.
        """
        def decode(dct):
            if cls.__name__ in dct:      
                return cls(
                    description=dct['description'],
                    childrentable=dct['childrentable'], 
                    roots=tuple(dct['roots']),
                    outdegree_max=int(dct['outdegree_max']))
            return dct
        return json.loads(string, object_hook=decode)     

    def expand_at(self, vertex):
        if self.open_at(vertex):
            self._childrentable[vertex] = tuple([vertex + str(i) 
                    for i in range(choice(range(self.outdegree_max)))])            
            for c in self._childrentable[vertex]:
                self._childrentable[c] = None            
        else:
            raise VertexNotOpen("{}.expand_at called on non-open vertex {}".format(
                                        self.__class__, vertex))
           
    def open_at(self, vertex):        
        return self._childrentable[vertex] is None

    def terminal_at(self, vertex):
        if self.open_at(vertex):
            raise VertexOpen("{}.terminal_at called on open vertex {}".format(
                                        self.__class__, vertex)) 
        return len(self._childrentable[vertex]) == 0

    def score_at(self, vertex) -> float:
        return 0.0

    def children_at(self, vertex, autoexpand = False) -> tuple:
        if self.open_at(vertex):
            if autoexpand:
                self.expand_at(vertex)
            else:
                raise VertexOpen("{}.children_at called on open vertex {}".format(
                                        self.__class__, vertex))           
        return self._childrentable[vertex]
    
    def edges_at(self, vertex):
        returnlist = []
        for p,c in self.edges:
            if p==vertex or c==vertex:
                returnlist += [(p,c)]
        return set(returnlist)

    def numpify(self, vertex):
        return array([ord(c) for c in vertex])

    def unnumpify(self, np_array):
        return "".join([chr(int(c)) for c in np_array])

    def equivalenceclass_of(self, vertex):
        return {vertex} 

    def representative_of(self, vertex):
        return vertex

    def copy(self) -> 'GameGraph':
        return self.from_json(self.as_json(indent=0))
    
    def truncate_to_roots(self):
        self._childrentable = { r:None for r in self.roots}
        return

    def print_subtree_at(self, vertex, max_depth=1, __current_depth=0, data=None):
        """
        Displays the sub_tree (up to 'max_depth') from 'node'.

        Arguments:
            vertex:             vertex from which the subtree is shown.
            max_depth (int):    Maximal depth shown. Default: 1
            __current_depth:    [!] Should not be changed manually. User for
                                proper formating of tree structure smybols.
            data (dct):         Information about each Vertex can be passed here.
                                Expected Format: {vertex : D}, where D is
                                a dictionary {info : value}. Default: None.
        """

        if data is None:
            data = {}
        
        # assemble the string
        node_info = str(vertex) + ' ('
        try:  # add data
            for key in data[vertex]:
                if __current_depth == 0:
                    node_info += str(key) + ':'
                node_info += str(data[vertex][key])+', '
            node_info = node_info[:-2] + ')'
        except KeyError:
            node_info = node_info[:-2]
            pass
        

        if self.open_at(vertex):
            node_info += ' ...'
        elif self.terminal_at(vertex):
            node_info += ' [T]'

        if __current_depth == 0:  # print header
            print(''.ljust(2+node_info.__len__(), '-'))
            print('...: open, [T]: terminal')
            print('# ', end='')
        else:    # add tree structure symbol
            for _ in range(0, __current_depth-1):
                print('.   ', end='')
            print('|--', end='')
        print(node_info)  # print the string
        if max_depth > 0 and not self.open_at(vertex):  # expand a level
            for child in list(self.children_at(vertex)):
                self.print_subtree_at(child, max_depth-1, __current_depth+1, data)
        if __current_depth == 0:  # print footer
            print(''.ljust(2+node_info.__len__(), '-'))

class VertexOpen(Exception):
    """Operation not allowed on an open vertex."""
    pass

class VertexNotOpen(Exception):
    """Operation not allowed on an non-open vertex."""
    pass