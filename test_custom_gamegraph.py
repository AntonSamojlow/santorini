import pytest
from gamegraph import ExampleGameGraph, GameGraph, VertexNotOpen, VertexOpen
from random import choice, sample
from numpy import ndarray

# Import the custom graph to be tested
from santorini import SanGraph, Environment

# parameter to limit the number of checks for large (custom) GameGraphs
MAX_VERTEX_CHECK = 100


@pytest.fixture
def Graph():
    G = SanGraph(Environment(5, 2))

    # import santorini
    # G = santorini.SanGraph(santorini.Environment(5,2))

    v = choice(list(G.roots))
    while True:
        try:
            v = choice(G.children_at(v, autoexpand=True))
        except IndexError:
            break
    return G


def test_numpify_unnumpify(Graph: GameGraph):
    for v in Graph.vertices:
        assert Graph.unnumpify(Graph.numpify(v)) == v


def test_tojson_fromjsom(Graph: GameGraph):
    G = Graph.__class__.from_json(Graph.as_json())
    assert G.description == Graph.description
    assert G.roots == Graph.roots
    for v, c in G._childrentable.items():
        Graph._childrentable[v] == c
    for v, c in Graph._childrentable.items():
        G._childrentable[v] == c


def test_raise_open_nonopen(Graph: GameGraph):
    for v in sample(Graph.vertices,
                    k=min(MAX_VERTEX_CHECK, len(Graph.vertices))):
        if Graph.open_at(v):
            with pytest.raises(VertexOpen):
                Graph.children_at(v)
            with pytest.raises(VertexOpen):
                Graph.terminal_at(v)
        else:
            with pytest.raises(VertexNotOpen):
                Graph.expand_at(v)


def test_attributetypes(Graph: GameGraph):
    assert isinstance(Graph.description, str)
    assert isinstance(Graph.edges, set)
    assert isinstance(Graph.outdegree_max, int)
    assert isinstance(Graph.roots, tuple)
    assert isinstance(Graph.vertices, set)
    assert isinstance(Graph._childrentable, dict)
    for v in sample(Graph.vertices,
                    k=min(MAX_VERTEX_CHECK, len(Graph.vertices))):
        if not Graph.open_at(v):
            children = Graph.children_at(v)
            assert isinstance(children, tuple)
            for c in children:
                assert isinstance(c, type(v))
    for e in Graph.edges:
        assert isinstance(e, tuple)
    for v, c in Graph._childrentable.items():
        assert isinstance(c, tuple) or c == None


def test_returntypes(Graph: GameGraph):
    assert isinstance(Graph.as_json(), str)
    assert Graph.truncate_to_roots() == None
    for v in sample(Graph.vertices,
                    k=min(MAX_VERTEX_CHECK, len(Graph.vertices))):
        assert isinstance(Graph.children_at(v, autoexpand=True), tuple)
        assert isinstance(Graph.edges_at(v), set)
        assert isinstance(Graph.equivalenceclass_of(v), set)
        assert isinstance(Graph.representative_of(v), type(v))
        assert isinstance(Graph.score_at(v), float)
        if not Graph.open_at(v):
            assert isinstance(Graph.terminal_at(v), bool)
        assert isinstance(Graph.numpify(v), ndarray)
    assert isinstance(Graph.deepcopy(), Graph.__class__)
