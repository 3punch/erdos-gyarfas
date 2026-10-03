"""Reference cubic graphs and literature benchmarks.

These serve two purposes: they give the search a set of structurally
interesting starting points, and they turn published results into *checkable*
targets for the engine.

The most useful is :func:`benchmark_no_c4_c8`: Markstrom's computer search
found 24-vertex cubic graphs whose only power-of-two cycles are 16-cycles (one
of the four is planar).  A search that cannot reproduce "C4 = C8 = 0 at n = 24"
is not working; one that can, at every n, has a calibrated baseline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core.cycles import count_power_of_two_cycles, girth_via_bfs
from .core.graph import CubicGraph, mobius_ladder, prism_graph
from .fitness import FitnessConfig
from .search import SimulatedAnnealer, girth_anneal


def generalized_petersen(n: int, k: int) -> CubicGraph:
    """The generalized Petersen graph ``G(n, k)`` on ``2n`` vertices."""
    edges = []
    for i in range(n):
        edges.append((i, (i + 1) % n))
        edges.append((n + i, n + (i + k) % n))
        edges.append((i, n + i))
    return CubicGraph.from_edges(2 * n, edges)


def petersen() -> CubicGraph:
    return generalized_petersen(5, 2)


def heawood() -> CubicGraph:
    """The (3,6)-cage: bipartite, girth 6, 14 vertices."""
    return generalized_petersen(7, 2)


def mobius_kantor() -> CubicGraph:
    """G(8,3): the Moebius-Kantor graph, girth 6, 16 vertices."""
    return generalized_petersen(8, 3)


def dodecahedron() -> CubicGraph:
    """G(10,2): the dodecahedral graph, girth 5, planar, 20 vertices."""
    return generalized_petersen(10, 2)


def desargues() -> CubicGraph:
    """G(10,3): the Desargues graph, girth 6, 20 vertices."""
    return generalized_petersen(10, 3)


def nauru() -> CubicGraph:
    """G(12,5): the Nauru graph, girth 6, 24 vertices."""
    return generalized_petersen(12, 5)


def reference_library() -> dict[str, CubicGraph]:
    """A fixed set of well-known cubic graphs, keyed by name."""
    return {
        "petersen_G5_2": petersen(),
        "heawood_G7_2": heawood(),
        "mobius_kantor_G8_3": mobius_kantor(),
        "dodecahedron_G10_2": dodecahedron(),
        "desargues_G10_3": desargues(),
        "nauru_G12_5": nauru(),
        "prism_24": prism_graph(24),
        "mobius_ladder_24": mobius_ladder(24),
    }


@dataclass
class BenchmarkResult:
    n: int
    target: str
    achieved: dict
    success: bool
    girth: int
    planar: bool
    seconds: float
    note: str

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            "target": self.target,
            "achieved": self.achieved,
            "success": self.success,
            "girth": self.girth,
            "planar": self.planar,
            "seconds": self.seconds,
            "note": self.note,
        }


def benchmark_no_c4_c8(
    n: int = 24,
    moves: int = 300_000,
    restarts: int = 6,
    seed: int = 0,
    config: FitnessConfig | None = None,
) -> BenchmarkResult:
    """Try to build a cubic graph on ``n`` vertices with no C4 and no C8.

    Markstrom reported four such graphs on 24 vertices whose only power-of-two
    cycles have length 16.  This is the smallest non-trivial target the engine
    can be checked against.
    """
    import time

    cfg = config or FitnessConfig(length_weights={4: 8.0, 8: 1.0})
    t0 = time.perf_counter()
    best = None
    best_total = None
    rng = np.random.default_rng(seed)
    for r in range(restarts):
        from .core.graph import random_cubic_graph

        g = random_cubic_graph(n, np.random.default_rng(seed * 1000 + r))
        g, _ = girth_anneal(g, moves=max(moves // 3, 1), seed=seed + r,
                            girth_target=9, config=cfg)
        res = SimulatedAnnealer(cfg, T0=1.0, T1=0.02).run(
            g, moves=moves, seed=seed + r
        )
        rep = count_power_of_two_cycles(res.graph, lengths=[4, 8], cap=4096)
        total = sum(rep.counts)
        if best_total is None or total < best_total:
            best, best_total = res.graph, total
        if total == 0:
            break
    elapsed = time.perf_counter() - t0
    rep = count_power_of_two_cycles(best)
    from .core.structure import is_planar

    achieved = dict(zip(rep.lengths, rep.counts))
    ok = achieved.get(4, 0) == 0 and achieved.get(8, 0) == 0
    return BenchmarkResult(
        n=n,
        target="cubic graph with C4 = C8 = 0",
        achieved=achieved,
        success=ok,
        girth=girth_via_bfs(best),
        planar=is_planar(best),
        seconds=elapsed,
        note=(
            "matches Markstrom's 24-vertex examples (only power-of-two cycles "
            "are 16-cycles)" if ok and n == 24 else
            f"best of {restarts} restarts"
        ),
    )
