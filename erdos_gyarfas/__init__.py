"""EG-Search: an automated search & reinforcement-learning framework for
exploring potential counterexamples to the Erdos-Gyarfas conjecture.

The Erdos-Gyarfas conjecture (1995) asserts that every graph with minimum
degree at least 3 contains a simple cycle whose length is a power of two.
This package implements

* a Numba-accelerated power-of-two cycle detection / counting engine,
* a cubic-graph (delta = 3) representation built on a perfect matching of
  half-edges, with O(1) 2-opt edge swaps,
* exact structural filters derived from the literature (P13-free, claw-free,
  planar, 2-edge-connected),
* simulated annealing / tabu search / DQN / actor-critic drivers,
* analysis, persistence (GraphML + JSON) and plotting utilities.
"""

from __future__ import annotations

__version__ = "1.0.0"

__all__ = [
    "core",
    "fitness",
    "search",
    "analysis",
    "experiments",
    "__version__",
]
