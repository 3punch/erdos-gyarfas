"""Z_k voltage-graph lifts engineered to cancel short (power-of-two) cycles.

Given a base graph B and a voltage assignment φ mapping each directed arc to
Z_k (with φ(v,u) = −φ(u,v)), the *derived graph* (lift) B^φ has vertex set
V(B) × Z_k and an edge (u,g) — (v, g+φ(u,v)) for every arc u→v.  If B is
d-regular the lift is d-regular on |V(B)|·k vertices.

Why this can cancel 2^k cycles: a closed walk of length L in B with net voltage
s lifts to a cycle whose length is L · k / gcd(s, k).  If every short cycle of
B is given a net voltage that *generates* Z_k (gcd(s,k)=1), its lifts have
length L·k, pushing the girth of the lift up by a factor of k and removing the
short power-of-two cycles of B.  The engine below builds lifts and lets the
early-exit checker test whether any power-of-two cycle survives.
"""

from __future__ import annotations

import numpy as np


def build_lift(
    base_edges: list[tuple[int, int]],
    n_base: int,
    k: int,
    voltages: dict[tuple[int, int], int],
):
    """Return (indptr, indices, meta) for the Z_k lift of the base graph.

    ``voltages`` maps an undirected edge (min,max) to its voltage in Z_k; the
    reverse arc gets the negation.
    """
    n = n_base * k
    edges = []
    for (u, v) in base_edges:
        a = voltages[(min(u, v), max(u, v))] % k
        for g in range(k):
            edges.append((u * k + g, v * k + ((g + a) % k)))
    indptr, indices = _csr_from_edges(n, edges)
    meta = {"family": "voltage_lift", "n_base": n_base, "k": k, "n": n,
            "m": len(edges)}
    return indptr, indices, meta


def _csr_from_edges(n: int, edges: list[tuple[int, int]]):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    indptr = np.zeros(n + 1, dtype=np.int64)
    for v in range(n):
        indptr[v + 1] = indptr[v] + len(adj[v])
    indices = np.empty(int(indptr[n]), dtype=np.int32)
    for v in range(n):
        indices[indptr[v]:indptr[v + 1]] = sorted(adj[v])
    return indptr, indices


def random_voltages(base_edges, k: int, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    return {(min(u, v), max(u, v)): int(rng.integers(1, k)) for u, v in base_edges}


def unit_voltages(base_edges, k: int) -> dict:
    """Every edge gets voltage 1 (a cyclic 'helical' lift)."""
    return {(min(u, v), max(u, v)): 1 % k for u, v in base_edges}
