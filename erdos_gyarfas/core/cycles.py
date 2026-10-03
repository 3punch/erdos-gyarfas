"""Public API for power-of-two cycle detection.

Central object
--------------
:class:`PowerOfTwoReport` records, for every power of two ``2^k <= n``, how
many cycles of that length the graph has and -- crucially -- whether that
number is *certified*.  A report may only be used to claim a counterexample
when every entry is certified: an uncertified zero means only "the budget ran
out before we found one".
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, asdict
from typing import Iterable, Sequence

import numpy as np

from . import kernels
from .graph import CubicGraph

#: Largest power of two we ever look for by default (2^11 = 2048).
MAX_LENGTH = 2048

DEFAULT_CAP = 1 << 20
DEFAULT_BUDGET = 1 << 26


def power_of_two_lengths(n: int, max_length: int = MAX_LENGTH) -> list[int]:
    """All powers of two in ``[4, min(n, max_length)]``.

    A graph on ``n`` vertices cannot contain a cycle longer than ``n``, so a
    counterexample on ``n`` vertices must avoid exactly this finite set.
    """
    out = []
    k = 2
    while True:
        L = 1 << k
        if L > min(n, max_length):
            break
        out.append(L)
        k += 1
    return out


def girth_via_bfs(graph: CubicGraph, max_girth: int = 64) -> int:
    """Exact girth of ``graph`` (``max_girth + 1`` if the girth is larger)."""
    return int(kernels.girth_via_bfs(graph.nb, graph.n, max_girth))


def girth(graph: CubicGraph, max_girth: int = 64) -> int:
    return girth_via_bfs(graph, max_girth)


def count_cycles_of_length(
    graph: CubicGraph,
    length: int,
    cap: int = DEFAULT_CAP,
    budget: int = DEFAULT_BUDGET,
) -> tuple[int, bool]:
    """Count simple cycles of length exactly ``length``.

    Returns ``(count, certified)``.  ``certified`` is False when the count
    saturates the cap or the DFS budget is exhausted, in which case ``count``
    is a lower bound.
    """
    n = graph.n
    if length < 3 or length > n:
        return 0, True
    dist = np.empty(n, dtype=np.int32)
    queue = np.empty(n, dtype=np.int32)
    onpath = np.zeros(n, dtype=np.int8)
    bip = np.zeros(n, dtype=np.int8)
    kernels.component_bipartite_flags(graph.nb, n, bip)
    st = np.zeros(5, dtype=np.int64)
    total = 0
    for s in range(n):
        kernels.bfs_from(graph.nb, n, np.int32(s), dist, queue)
        cnt, done = kernels.count_cycles_from_start_kernel(
            graph.nb, n, length, cap - total, budget, np.int32(s),
            dist, onpath, bip[s], st,
        )
        total += cnt
        if done == 0:
            return total, False
        if total >= cap:
            return total, False
    return total, True


@dataclass
class PowerOfTwoReport:
    """Cycle counts for every power of two up to the order of the graph."""

    n: int
    lengths: list[int]
    counts: list[int]
    certified: list[bool]
    elapsed: float = 0.0
    cap: int = DEFAULT_CAP

    # ------------------------------------------------------------ properties
    @property
    def total(self) -> int:
        """Total number of power-of-two cycles (a lower bound if not certified)."""
        return sum(self.counts)

    @property
    def all_certified(self) -> bool:
        return all(self.certified)

    @property
    def is_counterexample(self) -> bool:
        """True iff this graph is a *certified* Erdos-Gyarfas counterexample."""
        return self.all_certified and self.total == 0

    @property
    def distinct_lengths(self) -> list[int]:
        """Powers of two that actually occur as a cycle length."""
        return [L for L, c in zip(self.lengths, self.counts) if c > 0]

    @property
    def smallest_present(self) -> int | None:
        """Smallest power of two occurring as a cycle length (None if none)."""
        for L, c in zip(self.lengths, self.counts):
            if c > 0:
                return L
        return None

    @property
    def num_distinct(self) -> int:
        return len(self.distinct_lengths)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["total"] = self.total
        d["all_certified"] = self.all_certified
        d["is_counterexample"] = self.is_counterexample
        d["distinct_lengths"] = self.distinct_lengths
        d["smallest_present"] = self.smallest_present
        return d

    def summary(self) -> str:
        parts = []
        for L, c, ok in zip(self.lengths, self.counts, self.certified):
            parts.append(f"C{L}={c}{'' if ok else '?'}")
        tag = "CERTIFIED COUNTEREXAMPLE" if self.is_counterexample else ""
        return f"[{', '.join(parts)}] total={self.total} {tag}".strip()


def count_power_of_two_cycles(
    graph: CubicGraph,
    lengths: Sequence[int] | None = None,
    cap: int = DEFAULT_CAP,
    budget: int = DEFAULT_BUDGET,
    early_stop: bool = False,
) -> PowerOfTwoReport:
    """Count cycles of every power-of-two length up to ``graph.n``.

    The sweep shares one BFS distance layer per start vertex across all
    lengths, so the O(n m) BFS overhead is paid once rather than once per
    length -- important at n = 1000.

    Parameters
    ----------
    early_stop:
        if True, stop scanning a length as soon as one cycle is found.  The
        resulting counts are then only indicators ("0" or ">= 1"), which is
        enough for the jackpot test and much cheaper.
    """
    n = graph.n
    if lengths is None:
        lengths = power_of_two_lengths(n)
    lengths = [int(L) for L in lengths if 3 <= L <= n]
    nl = len(lengths)
    if nl == 0:
        return PowerOfTwoReport(n, [], [], [], 0.0, cap)
    caps = np.full(nl, 1 if early_stop else cap, dtype=np.int64)
    budgets = np.full(nl, budget, dtype=np.int64)
    out_count = np.zeros(nl, dtype=np.int64)
    out_done = np.ones(nl, dtype=np.int64)
    t0 = time.perf_counter()
    kernels.count_pow2_cycles_kernel(
        graph.nb, n,
        np.array(lengths, dtype=np.int64),
        caps, budgets, out_count, out_done,
    )
    elapsed = time.perf_counter() - t0
    counts = [int(c) for c in out_count]
    certified = [bool(d) and (c < (1 if early_stop else cap)) for d, c in zip(out_done, counts)]
    if early_stop:
        counts = [0 if c == 0 else 1 for c in counts]
    return PowerOfTwoReport(n, list(lengths), counts, certified, elapsed, cap)


def delta_power_of_two_counts(
    graph: CubicGraph,
    swap: tuple[int, int],
    lengths: Sequence[int],
    caps: Sequence[int],
    budgets: Sequence[int],
    dist: np.ndarray | None = None,
    onpath: np.ndarray | None = None,
    bip: np.ndarray | None = None,
) -> np.ndarray:
    """Exact change in cycle counts caused by a 2-opt swap.

    A 2-opt swap replaces edges ``(a, b), (c, d)`` by ``(a, d), (c, b)``.  A
    cycle can only be created or destroyed if it uses one of the two removed
    or one of the two added edges, so

        delta_L = |{L-cycles using (a,d) or (c,b)}|_new
                - |{L-cycles using (a,b) or (c,d)}|_old

    which is what this function computes -- exactly, not approximately.  The
    caller is responsible for choosing ``lengths`` small enough that the
    per-move cost stays affordable (the engine uses L in {4, 8} incrementally
    and recounts longer lengths at checkpoints).
    """
    s1, s2 = int(swap[0]), int(swap[1])
    nb_new = graph.nb
    a, b = s1 // 3, int(nb_new[s1])
    c, d = s2 // 3, int(nb_new[s2])
    # Undo the swap to recover the old adjacency.
    g_old = graph.copy()
    g_old.apply_swap(s1, s2)
    nb_old = g_old.nb
    n = graph.n
    if dist is None:
        dist = np.empty(n, dtype=np.int32)
        queue = np.empty(n, dtype=np.int32)
        onpath = np.zeros(n, dtype=np.int8)
        bip = np.zeros(n, dtype=np.int8)
        kernels.component_bipartite_flags(nb_new, n, bip)
    queue = np.empty(n, dtype=np.int32)
    st = np.zeros(5, dtype=np.int64)
    out = np.zeros(len(lengths), dtype=np.int64)
    kernels.bfs_from(nb_new, n, np.int32(a), dist, queue)
    parity_ok = bip[a]
    for i, L in enumerate(lengths):
        L = int(L)
        if L == 4:
            add = kernels.count_four_cycles_using_two(nb_new, a, d, b, c, True)
            rem = kernels.count_four_cycles_using_two(nb_old, a, b, c, d, True)
            out[i] = add - rem
            continue
        add = kernels.count_using_two_edges(
            nb_new, n, L, a, d, b, c, caps[i], budgets[i], dist, onpath,
            parity_ok, st, queue,
        )
        rem = kernels.count_using_two_edges(
            nb_old, n, L, a, b, c, d, caps[i], budgets[i], dist, onpath,
            parity_ok, st, queue,
        )
        out[i] = add - rem
    return out
