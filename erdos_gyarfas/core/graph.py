"""Cubic graph representation and generators.

The state of the search is a **3-regular simple graph**, represented as a
perfect matching on ``3n`` half-edges.  This is the classical configuration
model view and it is what makes the search cheap: a 2-opt edge swap is a
double transposition of two matching entries, i.e. O(1) time, and it
preserves ``delta(G) = 3`` *by construction* -- no degree repair step is ever
needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Iterator, Sequence

import numpy as np

from . import kernels


class InvalidCubicGraph(ValueError):
    """Raised when an edge list does not describe a 3-regular simple graph."""


def _check_degrees_even(n: int) -> None:
    if n < 4:
        raise InvalidCubicGraph(f"a 3-regular simple graph needs n >= 4, got {n}")
    if (3 * n) % 2 != 0:
        raise InvalidCubicGraph(f"a 3-regular graph on {n} vertices has 3n/2 edges; n must be even")


@dataclass
class CubicGraph:
    """A 3-regular simple graph on ``n`` vertices.

    Attributes
    ----------
    n:
        number of vertices.
    nb:
        ``int32[3n]``; neighbours of ``v`` are ``nb[3v] .. nb[3v+2]``.
    pairing:
        ``int32[3n]``; the perfect matching on half-edges, ``pairing[pairing[p]] == p``.
    """

    n: int
    nb: np.ndarray
    pairing: np.ndarray
    _adj_cache: np.ndarray | None = field(default=None, repr=False)

    # ------------------------------------------------------------------ ctor
    @classmethod
    def from_pairing(cls, n: int, pairing: np.ndarray) -> "CubicGraph":
        pairing = np.ascontiguousarray(pairing, dtype=np.int32)
        nb = (pairing // 3).astype(np.int32)
        g = cls(n, nb, pairing)
        g.validate()
        return g

    @classmethod
    def from_edges(cls, n: int, edges: Iterable[Sequence[int]]) -> "CubicGraph":
        """Build from an explicit edge list (used for reference graphs)."""
        _check_degrees_even(n)
        deg = np.zeros(n, dtype=np.int64)
        slots: list[list[int]] = [[] for _ in range(n)]
        seen: set[tuple[int, int]] = set()
        for e in edges:
            u, v = int(e[0]), int(e[1])
            if u == v:
                raise InvalidCubicGraph(f"loop at {u}")
            if not (0 <= u < n and 0 <= v < n):
                raise InvalidCubicGraph(f"edge {e} out of range for n={n}")
            key = (u, v) if u < v else (v, u)
            if key in seen:
                raise InvalidCubicGraph(f"double edge {key}")
            seen.add(key)
            deg[u] += 1
            deg[v] += 1
            slots[u].append(v)
            slots[v].append(u)
        bad = np.flatnonzero(deg != 3)
        if bad.size:
            raise InvalidCubicGraph(
                f"not 3-regular: {bad.size} vertices have degree != 3 "
                f"(e.g. vertex {bad[0]} has degree {deg[bad[0]]})"
            )
        nb = np.empty(3 * n, dtype=np.int32)
        for v in range(n):
            nb[3 * v : 3 * v + 3] = sorted(slots[v])
        # Half-edge matching: stub (v -> w) is matched with its twin (w -> v).
        slot = {(v, int(nb[3 * v + k])): 3 * v + k for v in range(n) for k in range(3)}
        pairing = np.empty(3 * n, dtype=np.int32)
        for (v, w), p in slot.items():
            q = slot.get((w, v))
            if q is None:
                raise InvalidCubicGraph("could not pair all half-edges")
            pairing[p] = q
            pairing[q] = p
        return cls(n, nb, pairing)

    # ------------------------------------------------------------ validation
    def validate(self) -> None:
        n, nb, pairing = self.n, self.nb, self.pairing
        if nb.shape != (3 * n,) or pairing.shape != (3 * n,):
            raise InvalidCubicGraph("array shape mismatch")
        if np.any(nb == np.repeat(np.arange(n, dtype=np.int32), 3)):
            raise InvalidCubicGraph("graph contains a loop")
        if np.any(pairing == np.arange(3 * n, dtype=np.int32)):
            raise InvalidCubicGraph("matching contains a fixed point")
        if np.any(pairing[pairing] != np.arange(3 * n, dtype=np.int32)):
            raise InvalidCubicGraph("pairing is not an involution")
        if np.any(pairing // 3 != nb):
            raise InvalidCubicGraph("nb and pairing disagree")
        for v in range(n):
            a, b, c = (int(nb[3 * v + k]) for k in range(3))
            if len({a, b, c}) != 3:
                raise InvalidCubicGraph(f"vertex {v} has a repeated neighbour")

    # ------------------------------------------------------------- accessors
    def edges(self) -> list[tuple[int, int]]:
        out = []
        for p in range(3 * self.n):
            q = int(self.pairing[p])
            u, v = p // 3, q // 3
            if u < v:
                out.append((u, v))
        return sorted(out)

    def adjacency_matrix(self) -> np.ndarray:
        if self._adj_cache is None:
            a = np.zeros((self.n, self.n), dtype=np.int8)
            nb = self.nb
            for v in range(self.n):
                for k in range(3):
                    a[v, int(nb[3 * v + k])] = 1
            self._adj_cache = a
        return self._adj_cache

    def invalidate_cache(self) -> None:
        self._adj_cache = None

    def copy(self) -> "CubicGraph":
        return CubicGraph(self.n, self.nb.copy(), self.pairing.copy())

    def degree_sequence(self) -> np.ndarray:
        return np.full(self.n, 3, dtype=np.int32)

    def num_edges(self) -> int:
        return 3 * self.n // 2

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CubicGraph):
            return NotImplemented
        return self.n == other.n and np.array_equal(self.nb, other.nb)

    # ---------------------------------------------------------------- moves
    def valid_swap(self, s1: int, s2: int) -> bool:
        return bool(kernels.swap_is_valid(self.nb, np.int32(s1), np.int32(s2)))

    def apply_swap(self, s1: int, s2: int) -> bool:
        """Apply a 2-opt swap in place.  Returns False if it was not admissible."""
        ok = kernels.swap_apply(self.pairing, self.nb, np.int32(s1), np.int32(s2))
        self._adj_cache = None
        return bool(ok)

    def propose_swap(self, rng: np.random.Generator, attempts: int = 8):
        """Sample a uniformly random admissible 2-opt swap, or ``None``."""
        n3 = 3 * self.n
        for _ in range(attempts):
            s1 = int(rng.integers(n3))
            s2 = int(rng.integers(n3))
            if kernels.swap_is_valid(self.nb, np.int32(s1), np.int32(s2)):
                return s1, s2
        return None

    def random_neighbor_state(self, rng: np.random.Generator) -> "CubicGraph":
        """One random 2-opt step, returning a fresh graph (used by RL drivers)."""
        g = self.copy()
        mv = g.propose_swap(rng)
        if mv is None:
            return g
        g.apply_swap(*mv)
        return g

    def connected_components(self) -> list[list[int]]:
        adj = self.adjacency_matrix()
        n = self.n
        seen = np.zeros(n, dtype=bool)
        comps: list[list[int]] = []
        for s in range(n):
            if seen[s]:
                continue
            stack, comp = [s], []
            seen[s] = True
            while stack:
                u = stack.pop()
                comp.append(u)
                for w in np.flatnonzero(adj[u]):
                    if not seen[w]:
                        seen[w] = True
                        stack.append(int(w))
            comps.append(sorted(comp))
        return comps

    # ------------------------------------------------------------- igraph IO
    def to_igraph(self):
        import igraph as ig

        return ig.Graph(n=self.n, edges=self.edges(), directed=False)

    def to_networkx(self):
        import networkx as nx

        return nx.Graph(self.edges())


# ---------------------------------------------------------------------------
# generators
# ---------------------------------------------------------------------------
def random_cubic_pairing(n: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform random perfect matching on ``3n`` stubs (configuration model)."""
    _check_degrees_even(n)
    order = rng.permutation(3 * n).astype(np.int32)
    pairing = np.empty(3 * n, dtype=np.int32)
    for i in range(0, 3 * n, 2):
        a, b = order[i], order[i + 1]
        pairing[a] = b
        pairing[b] = a
    return pairing


def _is_simple_cubic(n: int, nb: np.ndarray) -> bool:
    for v in range(n):
        b = 3 * v
        a0, a1, a2 = int(nb[b]), int(nb[b + 1]), int(nb[b + 2])
        if a0 == v or a1 == v or a2 == v:
            return False
        if a0 == a1 or a0 == a2 or a1 == a2:
            return False
    return True


def random_cubic_graph(n: int, rng: np.random.Generator | None = None,
                       max_tries: int = 10000) -> CubicGraph:
    """Uniform-ish random simple 3-regular graph via rejection sampling.

    The probability that a random pairing is simple tends to ``exp(-(d^2-1)/4)
    = e^-2`` for ``d = 3``, so a handful of retries suffices.
    """
    _check_degrees_even(n)
    rng = rng or np.random.default_rng()
    for _ in range(max_tries):
        pairing = random_cubic_pairing(n, rng)
        nb = (pairing // 3).astype(np.int32)
        if _is_simple_cubic(n, nb):
            return CubicGraph(n, np.ascontiguousarray(nb), pairing)
    raise RuntimeError(f"failed to build a simple cubic graph on {n} vertices")


def random_bipartite_cubic_graph(n: int, rng: np.random.Generator | None = None,
                                 max_tries: int = 100000) -> CubicGraph:
    """Random 3-regular bipartite graph on ``n`` vertices (``n`` divisible by 4).

    Built by taking the union of three random perfect matchings between the
    two colour classes, which is 3-regular bipartite by construction; we retry
    until the union is simple.
    """
    if n % 4 != 0:
        raise InvalidCubicGraph("a 3-regular bipartite graph needs n divisible by 4")
    rng = rng or np.random.default_rng()
    half = n // 2
    left = np.arange(half, dtype=np.int32)
    right = np.arange(half, n, dtype=np.int32)
    for _ in range(max_tries):
        partners: dict[int, list[int]] = {int(v): [] for v in range(n)}
        ok = True
        for _ in range(3):
            perm = rng.permutation(half)
            for i in range(half):
                u = int(left[i])
                v = int(right[perm[i]])
                partners[u].append(v)
                partners[v].append(u)
        for v, lst in partners.items():
            if len(set(lst)) != 3:
                ok = False
                break
        if not ok:
            continue
        nb = np.empty(3 * n, dtype=np.int32)
        for v in range(n):
            nb[3 * v : 3 * v + 3] = sorted(partners[v])
        slot = {(v, int(nb[3 * v + k])): 3 * v + k for v in range(n) for k in range(3)}
        pairing = np.empty(3 * n, dtype=np.int32)
        for (v, w), p in slot.items():
            q = slot.get((w, v))
            if q is None:
                raise InvalidCubicGraph("could not pair all half-edges")
            pairing[p] = q
            pairing[q] = p
        return CubicGraph(n, nb, pairing)
    raise RuntimeError(f"failed to build a simple bipartite cubic graph on {n} vertices")


def prism_graph(n: int) -> CubicGraph:
    """Generalised prism ``K_2 x C_m`` on ``n = 2m`` vertices (2-connected cubic)."""
    if n % 2 or n < 6:
        raise InvalidCubicGraph("prism needs an even n >= 6")
    m = n // 2
    edges = []
    for i in range(m):
        edges.append((i, (i + 1) % m))
        edges.append((m + i, m + (i + 1) % m))
        edges.append((i, m + i))
    return CubicGraph.from_edges(n, edges)


def mobius_ladder(n: int) -> CubicGraph:
    """Moebius ladder ``M_n`` on an even number ``n >= 6`` of vertices."""
    if n % 2 or n < 6:
        raise InvalidCubicGraph("Moebius ladder needs an even n >= 6")
    m = n // 2
    edges = []
    for i in range(m):
        edges.append((i, (i + 1) % m))
        edges.append((m + i, m + (i + 1) % m))
        edges.append((i, m + (m - 1 - i)))
    return CubicGraph.from_edges(n, edges)
