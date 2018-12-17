# santorini
Tinkering with a simple AI for Santorini.

## santorini.py
Implements the Santorini boardgame. Base classes are `Environment` and `State`. See docstring for useage examples.

## gamesearch.py
Collection of search algorithms (i.p. Monte-Carlo and alphabeta type algorithms) for two-player games. See docstring for useage examples. The central object upon which the algorithms operate is a special (finite) directed graph (see class `GameGraph`):

It has at least one root (no ingoing edges) and the property that each edge increases the 'level' (internal counter of a node, for roots it equals 0) by _exactly_ one. The graph may in particular have roots, but every path from a fixed node N to any root has the same length, equalling the 'level' of N.

Most two-player games can be modelled as such a graph. Transition  probabilities can be modelled by weights on the edges.

Since for many games, it is infeasible (or impossible) to specify the full graph, we allow an 'incomplete' description: A node N may be *_open_*, that is its children are not specified, but they may be computed by some rule,which is defined in the method _add_children_. Note in particular that an open Node is never terminal (the latter means _add_children_ computed an empty list).