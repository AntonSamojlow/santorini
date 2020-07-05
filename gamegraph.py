"""
GameGraph defines the required graph structure of a game to integrates into the GameGym framework

Written by Anton Samojlow 2020. [anton.samojlow@web.de]

The vertices of the graph must be hashable. All roots of the graph must be given
in a tuple (fixed order of roots). For a given vertex, a dictionary holds the tuple
of its children (fixed order of children). Two important concepts for each vertex:
- 'vertex open' :<=> childrens = None, meaning the vertex has not been explored yet.   
- 'vertex terminal' :<=> childrens = empty, meaning the vertex has no children
Note that 'vertex terminal' implies 'ertex not open'. These concepts allow an incomplete
description of the graph, since it is often unfeasible to compute the full graph of
a sufficiently rich game. Instead, the _rule_ of computing the children is provided
in the method 'expand_at'.

Example code snippet:

E = ExampleGameGraph()
v = E.roots[0]
for _ in range(10):
    if E.open_at(v):
        E.expand_at(v)
    elif E.terminal_at(v):
        break
    else:
        v=E.children_at(v)[0]

for r in E.roots:
    E.print_subtree_at(r,10)
"""

import logging
import json

from random import choice # only neded for the ExampleGameGraph
from numpy import array
from abc import ABC, abstractmethod

LOGGER = logging.getLogger(__name__)


def getLogger():
    return LOGGER


class GameGraph(ABC):
    """See docstring for the module"""
    def __init__(self,
                 description: str,
                 outdegree_max: int,
                 roots: tuple,
                 childrentable: dict):
        self.description = description   
        self.outdegree_max = outdegree_max
        self.roots = roots
        self._childrentable = childrentable
    
    @property
    def vertices(self) -> set:
        """The set of vertices (without an ordering)"""
        return set(self._childrentable.keys())

    @property
    def edges(self) -> set:
        """The set of edges (without an ordering)"""
        edges = []
        for p, children in self._childrentable.items():
            if children is not None:
                edges += [(p, c) for c in children]
        return set(edges)

    # ---------- abstract methods: need to be implemented for each graph ----------
    @abstractmethod
    def expand_at(self, vertex):
        """Expands the graph at an open node. Should raise an exception 'VertexNotOpen' else"""
        if self.open_at(vertex):
            raise NotImplementedError("Not implemented")
        else:
            raise VertexNotOpen(f"{self.__class__}.expand_at called on NON-open vertex {vertex}")   

    @abstractmethod
    def score_at(self, vertex) -> float:
        """Usual convention for the Score: in the interval [-1,+1], representing the win chance from the given vertex"""
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def numpify(self, vertex) -> 'array':
        """Ecnodes the vertex into a 'numpy.ndarray', for input to neural networks"""
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def unnumpify(self, np_array: array):
        """Inverse of the method numpify, returntype is the type of a key in the childrentable"""
        raise NotImplementedError("Not implemented")
    
    def deepcopy(self):
        """Returns a new copy of the graph with the same structure."""
        raise NotImplementedError("Not implemented")
    
    # ---------- methods that _might_ need to be adjusted (overwritten) for specific graphs ----------
    def equivalenceclass_of(self, vertex) -> set:
        """Possible equivalence class of a graph"""
        return {vertex}

    def representative_of(self, vertex):
        """Representative vertex of its equivalence class"""
        return vertex
    
    def truncate_to_roots(self):
        """Resets the graph to its roots (all non-open)"""
        self._childrentable = {r: None for r in self.roots}
        return        

    @classmethod
    def from_json(cls, string) -> 'GameGraph':
        """Inverse of 'as_json'. Restores tuples if they have been serialized as separate object."""        
        def decode(dct):  
            for subclass in GameGraph.__subclasses__():                
                if f"__{subclass.__name__}__" in dct:
                    content = GameGraph.SerializationTools.tuplify(dict(dct))
                    content.pop(f"__{subclass.__name__}__")
                    return subclass(*content.values())
            return dct
        return json.loads(string, object_hook=decode)

    def as_json(self, indent=2, hint_tuples = True):
        """Serializes the graph to JSON. If hint_tuples is true, tuples will be serialized as separate object."""
        class TupleHintingEncoder(json.JSONEncoder):  
            def default(self, obj):                
                if isinstance(obj, GameGraph):
                    content = {f"__{obj.__class__.__name__}__": True}
                    content.update(obj.__dict__)
                    if hint_tuples:
                        return GameGraph.SerializationTools.hint_tuples(content)
                    else:
                        return content
                return json.JSONEncoder.default(self, obj)
        return json.dumps(self, cls=TupleHintingEncoder, indent=indent)

    # ---------- methods that should _not_ be overwritten ----------

    def open_at(self, vertex) -> bool:
        """A vertex is open iff its children have not been explored yet (call expand_at to do so)."""
        return self._childrentable[vertex] is None

    def terminal_at(self, vertex) -> bool:
        """A veretx is terminal iff it has no children. Raises an exsception if called on an open vertex."""
        if self.open_at(vertex):
            raise VertexOpen(f"{self.__class__}.terminal_at called on open vertex {vertex}")
        return len(self._childrentable[vertex]) == 0

    def children_at(self, vertex, autoexpand=False) -> tuple:
        """Returns the ordered tuple of children of the given vertex."""
        if self.open_at(vertex):
            if autoexpand:
                self.expand_at(vertex)
            else:
                raise VertexOpen(
                    f"{ self.__class__}.children_at called on open vertex {vertex}")
        return self._childrentable[vertex]

    def edges_at(self, vertex) -> set:
        """Returns the (unordered) set of edges at the given vertex""" 
        returnlist = []
        for p, c in self.edges:
            if p == vertex or c == vertex:
                returnlist += [(p, c)]
        return set(returnlist)

    def print_subtree_at(self,
                         vertex,
                         max_depth=1,
                         data:dict =None):
        """
        Displays the sub_tree (up to 'max_depth') from 'node'.

        Arguments:
        - vertex:             vertex from which the subtree is shown.
        - max_depth:        Maximal depth shown. Default: 1       
        - data:             Information about each Vertex can be passed here.
                                Expected Format: {vertex : D}, where D is
                                a dictionary {info : value}. Default: None.
        """
        
        if data is None:
            data = {}

        def print_at_depth(vertex,
                            max_depth=1,                          
                            current_depth: int=0):
            # assemble the string
            node_info = str(vertex) + ' ('
            try:  # add data
                for key in data[vertex]:
                    if current_depth == 0:
                        node_info += str(key) + ':'
                    node_info += str(data[vertex][key]) + ', '
                node_info = node_info[:-2] + ')'
            except KeyError:
                node_info = node_info[:-2]
                pass

            if self.open_at(vertex):
                node_info += ' ...'
            elif self.terminal_at(vertex):
                node_info += ' [T]'

            if current_depth == 0:  # print header
                print(''.ljust(2 + node_info.__len__(), '-'))
                print('...: open, [T]: terminal')
                print('# ', end='')
            else:  # add tree structure symbol
                for _ in range(0, current_depth - 1):
                    print('.   ', end='')
                print('|--', end='')
            print(node_info)  # print the string
            if max_depth > 0 and not self.open_at(vertex):  # expand a level
                for child in list(self.children_at(vertex)):
                    print_at_depth(child, max_depth - 1, current_depth + 1)
            if current_depth == 0:  # print footer
                print(''.ljust(2 + node_info.__len__(), '-'))

        print_at_depth(vertex, max_depth, 0)

    class SerializationTools():
        @classmethod
        def tuplify(cls, obj):
            """Walks through a dictionary, converting all 'tuple-hinted' entries to tuples (inverse to 'hint-tuples')"""
            if isinstance(obj, dict):
                if '__tuple__' in obj:
                    return tuple(cls.tuplify(e) for e in obj['items'])
                else:
                    return {cls.tuplify(k) : cls.tuplify(v) for k,v in obj.items()}
            if isinstance(obj, list):
                return list(cls.tuplify(e) for e in obj)    
            if isinstance(obj, tuple):
                return tuple(cls.tuplify(e) for e in obj)
            return obj

        @classmethod
        def hint_tuples(cls, item):
            """Walks through tuples, lists and dicts, converting tuples to dictionaries with entry {'__tuple__' : True} """
            if isinstance(item, tuple):
                return {'__tuple__': True, 
                        'items': [cls.hint_tuples(e) for e in item]}
            if isinstance(item, list):
                return [cls.hint_tuples(e) for e in item]
            if isinstance(item, dict):
                return {key: cls.hint_tuples(value) for key, value in item.items()}
            else:
                return item
 

class VertexOpen(Exception):
    """Operation not allowed on an open vertex."""
    pass

class VertexNotOpen(Exception):
    """Operation not allowed on an non-open vertex."""
    pass

class ExampleGameGraph(GameGraph):
    def __init__(self,
                description = "Example of a GameGraph implementation",
                outdegree_max = 5,
                roots =  ("a", "b", "c"),
                childrentable = {"a": ("1", "2"),
                                "b": ("3", ),
                                "c": ("1", "4"),
                                "1": None,
                                "2": (),
                                "3": None,
                                "4": None}):
        super().__init__(description, outdegree_max, roots, childrentable)
    
    def expand_at(self, vertex):
        if self.open_at(vertex):
            self._childrentable[vertex] = tuple([
                vertex + str(i)
                for i in range(choice(range(self.outdegree_max)))
            ])
            for c in self._childrentable[vertex]:
                self._childrentable[c] = None
        else:
            raise VertexNotOpen(f"{self.__class__}.expand_at called on NON-open vertex {vertex}")   

    def score_at(self, vertex) -> float:
        return 0.0
   
    def numpify(self, vertex):
        return array([ord(c) for c in vertex])
 
    def unnumpify(self, np_array):
        return "".join([chr(int(c)) for c in np_array])

    def deepcopy(self):
        return ExampleGameGraph.from_json(self.as_json())


