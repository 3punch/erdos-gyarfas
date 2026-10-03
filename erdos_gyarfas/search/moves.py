"""Move generation and incremental evaluation.

:class:`MoveEvaluator` owns a cubic graph plus the ``nb_old`` snapshot that
the delta kernels need, and offers two operations:

* :meth:`peek` -- tentatively apply a swap, return the exact change in the
  tracked cycle counts (plus cheap structural features), then roll it back;
* :meth:`commit` -- apply a swap for real and update the snapshot.

Because a swap touches exactly four half-edges, both operations cost a handful
of array writes on top of the delta kernel itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core import kernels as core_kernels
from ..core.cycles import count_power_of_two_cycles
from ..core.graph import CubicGraph
from ..fitness import FitnessConfig
from . import kernels as K

FEATURE_NAMES = (
    "delta_c4",
    "delta_c8",
    "delta_loss",
    "new_short_cycle",
    "claw_delta",
    "cur_loss",
    "frac_accepted",
    "log_n",
)


@dataclass
class MoveEvaluator:
    """Exact incremental bookkeeping for the Python-level drivers."""

    graph: CubicGraph
    config: FitnessConfig
    track_lengths: tuple[int, ...] = (4, 8)
    girth_cap: int = 24

    def __post_init__(self) -> None:
        g = self.graph
        self.n = g.n
        self.lengths = np.array(
            [L for L in self.track_lengths if 3 <= L <= g.n], dtype=np.int64
        )
        self.nl = len(self.lengths)
        self.caps = np.full(self.nl, self.config.count_cap, dtype=np.int64)
        self.budgets = np.full(self.nl, self.config.budget, dtype=np.int64)
        self.weights = np.array(
            [self.config.weight(int(L)) for L in self.lengths], dtype=np.float64
        )
        self.nb_old = g.nb.copy()
        self.onpath = np.zeros(self.n, dtype=np.int8)
        self.st = np.zeros(5, dtype=np.int64)
        self._out = np.zeros(self.nl, dtype=np.int64)
        rep = count_power_of_two_cycles(
            g, lengths=list(map(int, self.lengths)),
            cap=self.config.count_cap, budget=self.config.budget,
        )
        self.counts = np.array(rep.counts, dtype=np.int64)
        self.proposed = 0
        self.accepted = 0

    # ------------------------------------------------------------------ loss
    def loss(self) -> float:
        total = 0.0
        for i in range(self.nl):
            c = self.counts[i]
            if c > self.caps[i]:
                c = self.caps[i]
            total += self.weights[i] * c
        return total / self.n

    def loss_after(self, delta: np.ndarray) -> float:
        total = 0.0
        for i in range(self.nl):
            c = self.counts[i] + delta[i]
            if c < 0:
                c = 0
            if c > self.caps[i]:
                c = self.caps[i]
            total += self.weights[i] * c
        return total / self.n

    # ------------------------------------------------------------- proposal
    def propose(self, rng: np.random.Generator, attempts: int = 8):
        return self.graph.propose_swap(rng, attempts=attempts)

    def propose_k(self, rng: np.random.Generator, k: int, attempts: int = 8):
        """``k`` distinct candidate swaps (the RL "slate")."""
        out = []
        seen: set[tuple[int, int]] = set()
        tries = 0
        while len(out) < k and tries < k * attempts:
            tries += 1
            mv = self.propose(rng, attempts=1)
            if mv is None:
                continue
            key = (min(mv), max(mv))
            if key in seen:
                continue
            seen.add(key)
            out.append(mv)
        return out

    # ------------------------------------------------------------------ peek
    def peek(self, swap: tuple[int, int], with_features: bool = False):
        """Delta counts for ``swap`` (rolled back afterwards)."""
        g = self.graph
        s1, s2 = int(swap[0]), int(swap[1])
        a = s1 // 3
        b = int(g.nb[s1])
        c = s2 // 3
        d = int(g.nb[s2])
        p1 = int(g.pairing[s1])
        p2 = int(g.pairing[s2])
        K.swap_apply_unchecked(g.pairing, g.nb, s1, s2)
        K.delta_fast(
            g.nb, self.nb_old, self.n, a, b, c, d,
            self.lengths, self.caps, self.budgets, self.onpath, self.st, self._out,
        )
        delta = self._out.copy()
        feats = None
        if with_features:
            sc = min(
                int(core_kernels.shortest_cycle_through_edge(
                    g.nb, self.n, a, d, self.girth_cap)),
                int(core_kernels.shortest_cycle_through_edge(
                    g.nb, self.n, c, b, self.girth_cap)),
            )
            claw = int(K.claw_delta(g.nb, self.nb_old, self.n, a, b, c, d))
            feats = (sc, claw)
        # roll the move back
        g.nb[s1] = self.nb_old[s1]
        g.nb[s2] = self.nb_old[s2]
        g.nb[p1] = self.nb_old[p1]
        g.nb[p2] = self.nb_old[p2]
        g.pairing[s1] = p1
        g.pairing[p1] = s1
        g.pairing[s2] = p2
        g.pairing[p2] = s2
        return (delta, feats) if with_features else delta

    # ---------------------------------------------------------------- commit
    def commit(self, swap: tuple[int, int], delta: np.ndarray | None = None) -> None:
        g = self.graph
        s1, s2 = int(swap[0]), int(swap[1])
        p1 = int(g.pairing[s1])
        p2 = int(g.pairing[s2])
        K.swap_apply_unchecked(g.pairing, g.nb, s1, s2)
        if delta is None:
            a = s1 // 3
            b = int(self.nb_old[s1])
            c = s2 // 3
            d = int(self.nb_old[s2])
            K.delta_fast(
                g.nb, self.nb_old, self.n, a, b, c, d,
                self.lengths, self.caps, self.budgets, self.onpath, self.st,
                self._out,
            )
            delta = self._out
        self.counts = np.maximum(self.counts + delta, 0)
        self.nb_old[s1] = g.nb[s1]
        self.nb_old[s2] = g.nb[s2]
        self.nb_old[p1] = g.nb[p1]
        self.nb_old[p2] = g.nb[p2]
        self.accepted += 1

    # -------------------------------------------------------------- features
    def feature_vector(self, delta: np.ndarray, feats: tuple[int, int] | None,
                       temperature: float = 0.0) -> np.ndarray:
        sc, claw = feats if feats is not None else (self.girth_cap + 1, 0)
        d4 = delta[0] if self.nl > 0 else 0
        d8 = delta[1] if self.nl > 1 else 0
        cur = self.loss()
        frac = self.accepted / max(self.proposed, 1)
        return np.array(
            [
                d4 / 4.0,
                d8 / 16.0,
                (self.loss_after(delta) - cur) * self.n,
                sc / self.girth_cap,
                claw / 4.0,
                cur,
                frac,
                np.log(max(self.n, 2)) / 8.0,
            ],
            dtype=np.float32,
        )
