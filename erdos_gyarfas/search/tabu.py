"""Tabu search over 2-opt swaps.

At each step a neighbourhood of ``k`` random swaps is scored with the exact
incremental delta kernel; the best admissible move is taken.  A move is tabu
when it undoes one of the last ``tenure`` swaps, unless it beats the best loss
seen so far (aspiration criterion).
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from ..core.graph import CubicGraph
from ..fitness import FitnessConfig
from .base import MoveStats, SearchHistory, SearchResult
from .moves import MoveEvaluator


def _tabu_key(swap: tuple[int, int]) -> tuple[int, int]:
    return (min(swap), max(swap))


@dataclass
class TabuSearch:
    """Best-of-neighbourhood local search with a short-term tabu list."""

    config: FitnessConfig
    neighbourhood: int = 24
    tenure: int = 40
    log_every: int = 100
    track_lengths: tuple[int, ...] = (4, 8)

    def run(
        self,
        graph: CubicGraph,
        moves: int = 20_000,
        seed: int = 0,
    ) -> SearchResult:
        rng = np.random.default_rng(seed)
        g = graph.copy()
        ev = MoveEvaluator(g, self.config, track_lengths=self.track_lengths)
        tabu: deque[tuple[int, int]] = deque(maxlen=self.tenure)
        tabu_set: set[tuple[int, int]] = set()

        cur_loss = ev.loss()
        best_loss = cur_loss
        best_graph = g.copy()
        best_counts = ev.counts.copy()
        hist = SearchHistory()
        t0 = time.perf_counter()
        no_improve = 0
        improved = 0

        for it in range(1, moves + 1):
            ev.proposed += 1
            cands = ev.propose_k(rng, self.neighbourhood)
            if not cands:
                continue
            scored = []
            for mv in cands:
                d = ev.peek(mv)
                scored.append((ev.loss_after(d), mv, d))
            scored.sort(key=lambda t: t[0])

            chosen = None
            for loss_after, mv, d in scored:
                key = _tabu_key(mv)
                if key in tabu_set and loss_after >= best_loss:
                    continue
                chosen = (loss_after, mv, d)
                break
            if chosen is None:
                chosen = scored[0]

            loss_after, mv, d = chosen
            ev.commit(mv, d)
            cur_loss = loss_after
            key = _tabu_key(mv)
            if key in tabu_set:
                tabu.append(key)
            else:
                if len(tabu) == tabu.maxlen:
                    tabu_set.discard(tabu[0])
                tabu.append(key)
                tabu_set.add(key)

            if cur_loss < best_loss - 1e-12:
                best_loss = cur_loss
                best_graph = g.copy()
                best_counts = ev.counts.copy()
                no_improve = 0
                improved += 1
            else:
                no_improve += 1

            if it % self.log_every == 0:
                hist.add(it, cur_loss, best_loss, no_improve=float(no_improve))

        elapsed = time.perf_counter() - t0
        mv_stats = MoveStats(
            proposed=ev.proposed,
            accepted=ev.accepted,
            improved=improved,
            rejected=ev.proposed - ev.accepted,
            seconds=elapsed,
        )
        return SearchResult(
            algorithm="tabu_search",
            graph=best_graph,
            best_loss=best_loss,
            final_loss=cur_loss,
            stats=mv_stats,
            history=hist,
            meta={
                "neighbourhood": self.neighbourhood,
                "tenure": self.tenure,
                "no_improve_at_end": no_improve,
                # the counts must describe the *returned* (best) graph, not
                # wherever the walk happened to finish
                "tracked_counts": dict(
                    zip(map(int, ev.lengths), map(int, best_counts))
                ),
                "final_counts": dict(zip(map(int, ev.lengths), map(int, ev.counts))),
            },
        )
