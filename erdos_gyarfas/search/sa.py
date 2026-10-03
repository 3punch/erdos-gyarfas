"""Simulated annealing over 2-opt swaps.

The whole inner loop runs inside a single Numba kernel
(:func:`erdos_gyarfas.search.kernels.sa_search`), so a move costs about
25 microseconds at n = 1000 and the driver can afford hundreds of thousands of
moves per run.  The objective is the length-weighted number of power-of-two
cycles, maintained *exactly* by incremental deltas rather than recounted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.cycles import count_power_of_two_cycles
from ..core.graph import CubicGraph
from ..fitness import FitnessConfig
from . import kernels as K
from .base import MoveStats, SearchHistory, SearchResult


@dataclass
class SimulatedAnnealer:
    """Metropolis search with a geometric temperature schedule."""

    config: FitnessConfig
    T0: float = 0.5
    T1: float = 0.01
    track_lengths: tuple[int, ...] = (4, 8)
    log_every: int = 1000

    def _arrays(self, n: int):
        lengths = [L for L in self.track_lengths if 3 <= L <= n]
        return (
            np.array(lengths, dtype=np.int64),
            np.full(len(lengths), self.config.count_cap, dtype=np.int64),
            np.full(len(lengths), self.config.budget, dtype=np.int64),
            np.array([self.config.weight(L) for L in lengths], dtype=np.float64),
        )

    def seed_counts(self, graph: CubicGraph) -> np.ndarray:
        lengths, caps, _b, _w = self._arrays(graph.n)
        rep = count_power_of_two_cycles(
            graph, lengths=list(lengths), cap=self.config.count_cap,
            budget=self.config.budget,
        )
        return np.array(rep.counts, dtype=np.int64)

    def run(
        self,
        graph: CubicGraph,
        moves: int = 200_000,
        seed: int = 0,
        T0: float | None = None,
        T1: float | None = None,
    ) -> SearchResult:
        g = graph.copy()
        n = g.n
        lengths, caps, budgets, weights = self._arrays(n)
        counts = self.seed_counts(g)
        onpath = np.zeros(n, dtype=np.int8)
        st = np.zeros(5, dtype=np.int64)
        n_log = max(1, moves // max(self.log_every, 1))
        log = np.zeros((n_log, 4), dtype=np.float64)
        out_counts = np.zeros_like(counts)
        stats = np.zeros(5, dtype=np.float64)

        hist = SearchHistory()
        t0 = __import__("time").perf_counter()
        best = K.sa_search(
            g.nb, g.pairing, n, int(moves),
            float(T0 if T0 is not None else self.T0),
            float(T1 if T1 is not None else self.T1),
            int(seed), weights, lengths, caps, budgets, counts,
            onpath, st, int(self.log_every), log, out_counts, stats,
        )
        elapsed = __import__("time").perf_counter() - t0

        rows = log[log[:, 0] > 0]
        for r in rows:
            hist.add(int(r[0]), float(r[2]), float(r[3]))

        mv = MoveStats(
            proposed=int(stats[0]),
            accepted=int(stats[1]),
            improved=int(stats[2]),
            rejected=int(stats[0] - stats[1]),
            seconds=elapsed,
        )
        # re-derive the tracked counts from the returned best graph so the
        # reported numbers never depend on the incremental bookkeeping alone
        verify = count_power_of_two_cycles(
            g, lengths=list(lengths), cap=self.config.count_cap,
            budget=self.config.budget,
        )
        return SearchResult(
            algorithm="simulated_annealing",
            graph=g,
            best_loss=float(stats[4]),
            final_loss=float(stats[3]),
            stats=mv,
            history=hist,
            meta={
                "T0": self.T0,
                "T1": self.T1,
                "track_lengths": list(map(int, lengths)),
                "tracked_counts": dict(zip(map(int, lengths), map(int, out_counts))),
                "verified_counts": dict(zip(verify.lengths, verify.counts)),
                "counts_match_incremental": verify.counts == list(map(int, out_counts)),
            },
        )


def girth_anneal(
    graph: CubicGraph,
    moves: int = 100_000,
    seed: int = 0,
    T0: float = 3.0,
    T1: float = 0.05,
    girth_target: int = 9,
    weight_base: float = 4.0,
    config: FitnessConfig | None = None,
) -> tuple[CubicGraph, dict]:
    """Push the girth up by penalising *every* cycle shorter than the target.

    High-girth starting points matter enormously: a random cubic graph has
    about two 4-cycles and sixteen 8-cycles (the means of the limiting Poisson
    law, 2^(L-1)/L), and the power-of-two objective is dominated by exactly
    those lengths.  Rather than maintaining a fragile girth bound, this
    anneals the vector of counts of 3-, 4-, ..., (girth_target-1)-cycles,
    which is monotone in the girth and reuses the already-validated delta
    machinery.
    """
    from ..core.cycles import girth_via_bfs

    cfg = config or FitnessConfig()
    g = graph.copy()
    lengths = [L for L in range(3, girth_target) if L <= g.n]
    arr_lengths = np.array(lengths, dtype=np.int64)
    caps = np.full(len(lengths), cfg.count_cap, dtype=np.int64)
    budgets = np.full(len(lengths), cfg.budget, dtype=np.int64)
    # Girth is a *minimum*, so the shortest cycles must be eliminated first.
    # Weighting length L by weight_base**(girth_target-1-L) makes the schedule
    # effectively lexicographic: triangles before 4-cycles, 4-cycles before
    # 5-cycles, and so on.
    weights = np.array(
        [weight_base ** (girth_target - 1 - L) for L in lengths], dtype=np.float64
    )
    rep = count_power_of_two_cycles(
        g, lengths=lengths, cap=cfg.count_cap, budget=cfg.budget
    )
    counts = np.array(rep.counts, dtype=np.int64)
    n_log = max(1, moves // 1000)
    log = np.zeros((n_log, 4), dtype=np.float64)
    out_counts = np.zeros_like(counts)
    stats = np.zeros(5, dtype=np.float64)
    K.sa_search(
        g.nb, g.pairing, g.n, int(moves), float(T0), float(T1), int(seed),
        weights, arr_lengths, caps, budgets, counts,
        np.zeros(g.n, dtype=np.int8), np.zeros(5, dtype=np.int64),
        1000, log, out_counts, stats,
    )
    verify = count_power_of_two_cycles(
        g, lengths=lengths, cap=cfg.count_cap, budget=cfg.budget
    )
    return g, {
        "moves": int(moves),
        "girth_target": girth_target,
        "short_cycle_counts_before": dict(zip(lengths, rep.counts)),
        "short_cycle_counts_after": dict(zip(lengths, verify.counts)),
        "exact_girth": girth_via_bfs(g),
    }
