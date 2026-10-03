"""Search agents for the Erdos-Gyarfas counterexample hunt.

All drivers explore the same neighbourhood -- 2-opt edge swaps on the
half-edge matching of a cubic graph -- and differ only in how they choose the
next move:

===========================  ==================================================
Driver                       Move policy
===========================  ==================================================
:mod:`.sa`                   Metropolis acceptance on a geometric temperature
                             schedule (inner loop compiled to machine code).
:mod:`.tabu`                 best-of-neighbourhood with a tabu list on the
                             swapped stub pair plus aspiration.
:mod:`.rl`                   a learned policy: DQN (Q-learning over a move
                             slate) or actor-critic (policy gradient with a
                             value baseline).
===========================  ==================================================
"""

from .base import SearchResult, SearchHistory, MoveStats
from .sa import SimulatedAnnealer, girth_anneal
from .tabu import TabuSearch
from .rl import DQNAgent, ActorCriticAgent, RLSearch

__all__ = [
    "SearchResult",
    "SearchHistory",
    "MoveStats",
    "SimulatedAnnealer",
    "girth_anneal",
    "TabuSearch",
    "DQNAgent",
    "ActorCriticAgent",
    "RLSearch",
]
