"""CNF/SMT encoding of the Erdős–Gyárfás counterexample search.

A candidate counterexample on ``n`` vertices is a simple graph with

* **Constraint A** — minimum degree δ(G) ≥ 3;
* **Constraint B** — no cycle of length L for every power of two L ≤ n
  (L ∈ {4, 8, 16, 32, 64, …}).

This module builds the propositional encoding.  Edge variables ``e_{uv}``
(1-based DIMACS ids) are the atoms; δ ≥ 3 is a cardinality constraint and
"no L-cycle" is a conjunction of blocking clauses, one per L-cycle of the
complete graph K_n (each clause says "at least one edge of this cycle is
absent").

Cycle enumeration is combinatorial — K_n has C(n,L)·(L-1)!/2 distinct
L-cycles — so the caller passes a ``clause_cap`` and the enumerator raises
:class:`ClauseBudgetExceeded` once the count would exceed it.  This is the
honest tractability frontier of the enumeration encoding: L = 4 is cheap up
to n ≈ 70, L = 8 only up to n ≈ 14, and L ≥ 16 is never enumerable.
"""

from __future__ import annotations

from itertools import combinations, permutations
from typing import Iterator, Sequence


class ClauseBudgetExceeded(RuntimeError):
    """Raised when a cycle-exclusion constraint needs more clauses than allowed."""


def edge_var(u: int, v: int, n: int) -> int:
    """1-based DIMACS id of the edge variable e_{uv} (u != v, order-free)."""
    if u > v:
        u, v = v, u
    return 1 + u * n + v


def n_edge_vars(n: int) -> int:
    return n * n + 1  # generous upper bound on the id space


def degree_literals(v: int, n: int) -> list[int]:
    """Literals e_{uv} for all u != v (the edges incident to v)."""
    return [edge_var(min(v, u), max(v, u), n) for u in range(n) if u != v]


def count_l_cycles(n: int, L: int) -> int:
    """Number of distinct L-cycles in K_n = C(n,L) * (L-1)!/2."""
    if L > n or L < 3:
        return 0
    from math import comb, factorial

    return comb(n, L) * factorial(L - 1) // 2


def enumerate_l_cycles(n: int, L: int, cap: int | None = None) -> Iterator[list[tuple[int, int]]]:
    """Yield each L-cycle of K_n once, as a list of its L edges.

    Canonical form: the cycle is written starting at its smallest vertex, and
    the second vertex is smaller than the last (kills the reflection), so each
    undirected cycle appears exactly once.
    """
    if L > n or L < 3:
        return
    if cap is not None and count_l_cycles(n, L) > cap:
        raise ClauseBudgetExceeded(
            f"K_{n} has {count_l_cycles(n, L)} {L}-cycles (> cap {cap})"
        )
    for verts in combinations(range(n), L):
        v0 = verts[0]
        rest = verts[1:]
        for perm in permutations(rest):
            # reflection symmetry: second < last
            if perm[0] > perm[-1]:
                continue
            cycle = (v0,) + perm
            edges = [(cycle[i], cycle[(i + 1) % L]) for i in range(L)]
            yield [(min(a, b), max(a, b)) for a, b in edges]


def blocking_clause(edges: Sequence[tuple[int, int]], n: int) -> list[int]:
    """Clause forbidding one cycle: OR over its edges of NOT e_{uv}."""
    return [-edge_var(u, v, n) for (u, v) in edges]


def powers_of_two_upto(n: int, max_length: int = 64) -> list[int]:
    out, L = [], 4
    while L <= n and L <= max_length:
        out.append(L)
        L *= 2
    return out


def build_cnf(
    n: int,
    lengths: Sequence[int],
    clause_cap: int = 5_000_000,
    enforce_delta3: bool = True,
):
    """Return ``(clauses, card_constraints, info)``.

    * ``clauses`` -- list of DIMACS clauses (list of signed ints) for the
      cycle-exclusion constraints;
    * ``card_constraints`` -- list of ``(literals, bound)`` "at least"
      cardinality constraints (one per vertex for δ ≥ 3);
    * ``info`` -- dict with per-length clause counts.

    Raises :class:`ClauseBudgetExceeded` if any length's cycle count exceeds
    ``clause_cap``.
    """
    clauses: list[list[int]] = []
    info: dict[str, object] = {"n": n, "per_length": {}}
    for L in lengths:
        cnt = count_l_cycles(n, L)
        info["per_length"][L] = cnt
        if cnt > clause_cap:
            raise ClauseBudgetExceeded(
                f"n={n} L={L}: {cnt} cycles exceeds cap {clause_cap}"
            )
        for edges in enumerate_l_cycles(n, L):
            clauses.append(blocking_clause(edges, n))
    cards = []
    if enforce_delta3:
        for v in range(n):
            cards.append((degree_literals(v, n), 3))
    info["n_clauses"] = len(clauses)
    info["n_cards"] = len(cards)
    return clauses, cards, info
