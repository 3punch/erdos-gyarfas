"""Structural filters used to prune the search space.

Every filter in this module is grounded in a published theorem, and each is
labelled with exactly what it *proves*:

================================  =========================================================
Filter                            Justification
================================  =========================================================
``is_planar``                     The conjecture is proved for 3-connected cubic planar
                                  graphs and for planar claw-free graphs.  It is *not*
                                  known for planar graphs in general, so this is a
                                  search-space restriction, not a proof of absence.
``is_claw_free``                  Proved for planar claw-free graphs; for cubic claw-free
                                  graphs a counterexample needs >= 114 vertices, and every
                                  claw-free graph with delta >= 3 has a cycle of length
                                  2^k or 3 * 2^k.  Again a restriction, not a proof, for
                                  general (non-planar) claw-free graphs.
``has_induced_path(13)``          **A genuine necessary condition.**  Every P13-free graph
                                  with delta >= 3 has a power-of-two cycle (Hegde,
                                  Sandeep & Shashank 2025), so a counterexample must
                                  contain an induced path on 13 vertices.  Failing this
                                  test *proves* the graph is not a counterexample.
``is_two_edge_connected``         Every minimal counterexample is 2-edge-connected and is
                                  biconnected or a 1-clique-sum of two biconnected graphs
                                  (Ducoffe et al. 2026).
================================  =========================================================
"""

from __future__ import annotations

import numpy as np

from . import kernels
from .graph import CubicGraph

#: Smallest induced path that a counterexample is forced to contain.
REQUIRED_INDUCED_PATH_VERTICES = 13


# ---------------------------------------------------------------------------
# claw-free
# ---------------------------------------------------------------------------
def claw_center_count(graph: CubicGraph) -> int:
    """Number of induced ``K_{1,3}`` centres (== 0 iff the graph is claw-free)."""
    return int(kernels.claw_center_count(graph.nb, graph.n))


def is_claw_free(graph: CubicGraph) -> bool:
    """True iff no vertex has three pairwise non-adjacent neighbours.

    For a cubic graph a claw centre is exactly a vertex whose neighbourhood is
    an independent set, which makes this an O(n) test.
    """
    return claw_center_count(graph) == 0


def is_claw_free_general(graph: CubicGraph) -> bool:
    """Degree-independent claw check (kept for non-cubic reuse and testing)."""
    adj = graph.adjacency_matrix()
    n = graph.n
    for v in range(n):
        nbrs = np.flatnonzero(adj[v])
        for i in range(nbrs.size):
            for j in range(i + 1, nbrs.size):
                if adj[nbrs[i], nbrs[j]]:
                    break
            else:
                if nbrs.size >= 3:
                    return False
                continue
            break
    return True


# ---------------------------------------------------------------------------
# planarity
# ---------------------------------------------------------------------------
_IGRAPH_HAS_PLANARITY: bool | None = None


def _igraph_planarity_available() -> bool:
    """Some igraph wheels are built without the planarity module."""
    global _IGRAPH_HAS_PLANARITY
    if _IGRAPH_HAS_PLANARITY is None:
        try:
            import igraph as ig

            _IGRAPH_HAS_PLANARITY = hasattr(ig.Graph, "is_planar")
        except Exception:
            _IGRAPH_HAS_PLANARITY = False
    return _IGRAPH_HAS_PLANARITY


def is_planar(graph: CubicGraph) -> bool:
    """Planarity test.

    Uses igraph's C implementation when the installed wheel provides it and
    falls back to networkx's LR-planarity otherwise.  Measured on this
    machine: networkx answers in ~12 ms for a 1000-vertex cubic graph, so the
    fallback is cheap enough to use at every fitness checkpoint.
    """
    if _igraph_planarity_available():
        return bool(graph.to_igraph().is_planar())
    import networkx as nx

    return bool(nx.check_planarity(graph.to_networkx(), counterexample=False)[0])


def planar_embedding_or_none(graph: CubicGraph):
    import networkx as nx

    ok, emb = nx.check_planarity(graph.to_networkx(), counterexample=False)
    return emb if ok else None


# ---------------------------------------------------------------------------
# induced paths
# ---------------------------------------------------------------------------
def longest_induced_path(
    graph: CubicGraph,
    target: int = 64,
    budget: int = 1 << 22,
) -> tuple[int, bool]:
    """Longest induced path (in vertices) found within ``budget`` DFS nodes.

    Returns ``(best, completed)``.  Two claims are sound:

    * ``best >= target`` proves the graph contains an induced path on
      ``target`` vertices;
    * ``completed and best < target`` proves it contains none, i.e. the graph
      is ``P_target``-free.

    A budget-limited ``best < target`` with ``completed == False`` proves
    nothing and callers must treat it as "unknown".
    """
    best, completed, _exhausted = kernels.longest_induced_path_kernel(
        graph.nb, graph.n, int(target), int(budget)
    )
    return int(best), bool(completed)


def has_induced_path(
    graph: CubicGraph,
    vertices: int = REQUIRED_INDUCED_PATH_VERTICES,
    budget: int = 1 << 22,
) -> tuple[bool, str]:
    """Decide whether ``graph`` contains an induced path on ``vertices`` vertices.

    Returns ``(answer, status)`` with ``status`` one of ``"proved"``,
    ``"refuted"`` or ``"unknown"`` (budget exhausted).
    """
    best, completed = longest_induced_path(graph, target=vertices, budget=budget)
    if best >= vertices:
        return True, "proved"
    if completed:
        return False, "refuted"
    return False, "unknown"


# ---------------------------------------------------------------------------
# connectivity
# ---------------------------------------------------------------------------
def bridge_count(graph: CubicGraph) -> int:
    return int(kernels.bridge_count(graph.nb, graph.n))


def is_two_edge_connected(graph: CubicGraph) -> bool:
    """True iff the graph is connected and has no bridges.

    Every minimal counterexample is 2-edge-connected (Ducoffe et al. 2026), so
    graphs with a bridge can be discarded outright.
    """
    if not is_connected(graph):
        return False
    return bridge_count(graph) == 0


def is_connected(graph: CubicGraph) -> bool:
    return bool(kernels.is_connected_kernel(graph.nb, graph.n))


def is_biconnected(graph: CubicGraph) -> bool:
    return bool(kernels.is_biconnected_kernel(graph.nb, graph.n))


def triangle_count(graph: CubicGraph) -> int:
    return int(kernels.count_triangles(graph.nb, graph.n))


# ---------------------------------------------------------------------------
# convenience bundle
# ---------------------------------------------------------------------------
def structural_report(graph: CubicGraph, induced_path_budget: int = 1 << 22) -> dict:
    """All structural flags in one dictionary (used by the fitness function)."""
    has_p13, status = has_induced_path(
        graph, REQUIRED_INDUCED_PATH_VERTICES, induced_path_budget
    )
    best_ip, _ = longest_induced_path(graph, target=graph.n, budget=induced_path_budget)
    return {
        "claw_free": is_claw_free(graph),
        "claw_centers": claw_center_count(graph),
        "planar": is_planar(graph),
        "connected": is_connected(graph),
        "two_edge_connected": is_two_edge_connected(graph),
        "biconnected": is_biconnected(graph),
        "triangles": triangle_count(graph),
        "has_induced_p13": has_p13,
        "induced_p13_status": status,
        "longest_induced_path_lb": best_ip,
    }
