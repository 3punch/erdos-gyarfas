"""Measurements reported for every candidate and every scaling point."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from ..core.cycles import PowerOfTwoReport, count_power_of_two_cycles, girth_via_bfs
from ..core.graph import CubicGraph
from ..core.structure import (
    claw_center_count,
    is_biconnected,
    is_planar,
    is_two_edge_connected,
    longest_induced_path,
)


@dataclass
class SpectralSummary:
    """Spectral descriptors of the adjacency and Laplacian matrices."""

    lambda_max: float
    lambda_2: float                 # second largest adjacency eigenvalue
    lambda_min: float
    spectral_gap: float             # 3 - lambda_2, the expansion gap
    algebraic_connectivity: float   # second smallest Laplacian eigenvalue
    energy: float                   # sum |lambda_i|
    n_eigenvalues_used: int = 0

    def as_dict(self) -> dict:
        return {
            "lambda_max": self.lambda_max,
            "lambda_2": self.lambda_2,
            "lambda_min": self.lambda_min,
            "spectral_gap": self.spectral_gap,
            "algebraic_connectivity": self.algebraic_connectivity,
            "energy": self.energy,
        }


def spectral_summary(graph: CubicGraph, k: int = 6) -> SpectralSummary:
    """Top/bottom eigenvalues via a sparse Lanczos solve.

    For a 3-regular graph ``lambda_max = 3`` exactly, so the interesting
    quantity is ``lambda_2``: the Alon-Boppana bound puts the best possible
    expansion at ``2*sqrt(2) ~ 2.828``, and graphs near that bound (Ramanujan
    graphs) are the ones whose cycle-length distribution most closely follows
    the random-graph model -- which is exactly the regime in which power-of-two
    cycles are unavoidable in large numbers.
    """
    import scipy.sparse as sp
    from scipy.sparse.linalg import eigsh

    n = graph.n
    edges = graph.edges()
    rows = np.array([e[0] for e in edges] + [e[1] for e in edges], dtype=np.int32)
    cols = np.array([e[1] for e in edges] + [e[0] for e in edges], dtype=np.int32)
    data = np.ones(rows.size, dtype=np.float64)
    A = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    k = max(2, min(k, n - 2))
    try:
        top = eigsh(A, k=k, which="LA", return_eigenvectors=False)
        bot = eigsh(A, k=2, which="SA", return_eigenvectors=False)
        lam2 = float(np.sort(top)[-2])
        lmin = float(np.sort(bot)[0])
        lmax = float(np.sort(top)[-1])
    except Exception:
        ev = np.linalg.eigvalsh(A.toarray())
        lmax, lam2, lmin = float(ev[-1]), float(ev[-2]), float(ev[0])
    deg = np.asarray(A.sum(axis=1)).ravel()
    D = sp.diags(deg)
    L = D - A
    try:
        small = eigsh(L, k=2, which="SM", return_eigenvectors=False)
        aconn = float(np.sort(small)[1])
    except Exception:
        lev = np.linalg.eigvalsh(L.toarray())
        aconn = float(np.sort(lev)[1])
    energy = float(np.abs(np.linalg.eigvalsh(A.toarray())).sum()) if n <= 1200 else float("nan")
    return SpectralSummary(
        lambda_max=lmax,
        lambda_2=lam2,
        lambda_min=lmin,
        spectral_gap=3.0 - lam2,
        algebraic_connectivity=aconn,
        energy=energy,
        n_eigenvalues_used=k,
    )


@dataclass
class GraphMetrics:
    """Everything the experiment log records about one candidate."""

    n: int
    m: int
    girth: int
    pow2_counts: dict[int, int]
    pow2_certified: dict[int, bool]
    pow2_total: int
    pow2_distinct: list[int]
    smallest_pow2: int | None
    claw_centers: int
    claw_free: bool
    planar: bool
    two_edge_connected: bool
    biconnected: bool
    longest_induced_path_lb: int
    spectral: SpectralSummary
    density_per_vertex: float
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        d = {
            "n": self.n,
            "m": self.m,
            "girth": self.girth,
            "pow2_total": self.pow2_total,
            "pow2_distinct_lengths": self.pow2_distinct,
            "smallest_pow2_present": self.smallest_pow2,
            "claw_centers": self.claw_centers,
            "claw_free": self.claw_free,
            "planar": self.planar,
            "two_edge_connected": self.two_edge_connected,
            "biconnected": self.biconnected,
            "longest_induced_path_lb": self.longest_induced_path_lb,
            "cycle_density_per_vertex": self.density_per_vertex,
        }
        for L, c in self.pow2_counts.items():
            d[f"c{L}"] = c
            d[f"c{L}_certified"] = self.pow2_certified.get(L)
        d.update(self.spectral.as_dict())
        d.update(self.extra)
        return d


def measure(
    graph: CubicGraph,
    report: PowerOfTwoReport | None = None,
    induced_path_budget: int = 1 << 22,
    extra: Mapping | None = None,
) -> GraphMetrics:
    """Compute the full metric bundle for one graph."""
    rep = report or count_power_of_two_cycles(graph)
    ip, _completed = longest_induced_path(
        graph, target=graph.n, budget=induced_path_budget
    )
    return GraphMetrics(
        n=graph.n,
        m=graph.num_edges(),
        girth=girth_via_bfs(graph),
        pow2_counts=dict(zip(rep.lengths, rep.counts)),
        pow2_certified=dict(zip(rep.lengths, rep.certified)),
        pow2_total=rep.total,
        pow2_distinct=rep.distinct_lengths,
        smallest_pow2=rep.smallest_present,
        claw_centers=claw_center_count(graph),
        claw_free=claw_center_count(graph) == 0,
        planar=is_planar(graph),
        two_edge_connected=is_two_edge_connected(graph),
        biconnected=is_biconnected(graph),
        longest_induced_path_lb=ip,
        spectral=spectral_summary(graph),
        density_per_vertex=rep.total / graph.n,
        extra=dict(extra or {}),
    )
