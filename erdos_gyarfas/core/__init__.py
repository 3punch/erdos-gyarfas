"""Core graph engine: representation, cycle counting, structural filters."""

from .graph import CubicGraph, random_cubic_graph, random_bipartite_cubic_graph
from .cycles import (
    count_cycles_of_length,
    count_power_of_two_cycles,
    PowerOfTwoReport,
    girth,
    girth_via_bfs,
)
from .structure import (
    is_claw_free,
    is_planar,
    longest_induced_path,
    is_two_edge_connected,
    is_biconnected,
    has_induced_path,
)

__all__ = [
    "CubicGraph",
    "random_cubic_graph",
    "random_bipartite_cubic_graph",
    "count_cycles_of_length",
    "count_power_of_two_cycles",
    "PowerOfTwoReport",
    "girth",
    "girth_via_bfs",
    "is_claw_free",
    "is_planar",
    "longest_induced_path",
    "is_two_edge_connected",
    "is_biconnected",
    "has_induced_path",
]
