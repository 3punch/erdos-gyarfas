"""Targeted large 3-regular graph generators (N = 10 000 -> 100 000).

Every generator returns a CSR pair ``(indptr, indices)`` plus a small metadata
dict; the caller runs the early-exit checker on the CSR.

Families
--------
* ``random_cubic``     -- configuration model; the "generic" baseline.
* ``cayley_dihedral``  -- Cay(D_m, {r, r^-1, s}), 2m vertices.
* ``cayley_psl2``      -- Cay(PSL(2,q), {a, a^-1, b}) with a unipotent and an
                          involution, q(q^2-1)/2 vertices (74 412 for q = 53).
* ``cayley_perm``      -- Cay(S_n, {(1 2), cycle, cycle^-1}), n! vertices.
* ``high_girth``       -- random cubic + budgeted girth annealing, engineered
                          to kill C4/C8 so the checker probes C16 and beyond.

All Cayley graphs are vertex-transitive and, for the groups used here, connected
and non-planar at these sizes, so they sit inside the region the exclusion
filters keep.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .kernels import build_csr


def _csr_from_degree3(n: int, nb: np.ndarray):
    """CSR from a fixed-stride degree-3 neighbour array."""
    indptr = (np.arange(0, 3 * n + 1, 3)).astype(np.int64)
    return indptr, nb.astype(np.int32)


# ---------------------------------------------------------------------------
# random cubic (baseline)
# ---------------------------------------------------------------------------
def random_cubic(n: int, seed: int = 0):
    from ..core.graph import random_cubic_graph

    g = random_cubic_graph(n, np.random.default_rng(seed))
    indptr, indices = _csr_from_degree3(n, g.nb)
    return indptr, indices, {"family": "random_cubic", "n": n}


# ---------------------------------------------------------------------------
# dihedral Cayley
# ---------------------------------------------------------------------------
def cayley_dihedral(m: int, seed: int = 0):
    """Cayley graph of the dihedral group D_m = <r, s | r^m, s^2, srs=r^-1>.

    Vertices are encoded as integers: ``i`` for r^i and ``m + i`` for s r^i.
    Generators {r, r^-1, s} give a 3-regular graph on 2m vertices.
    """
    n = 2 * m
    nb = np.empty(3 * n, dtype=np.int32)

    def mul_r(x: int, e: int) -> int:
        if x < m:
            return (x + e) % m
        # s r^i * r^e = s r^(i - e)
        return m + ((x - m - e) % m)

    for x in range(n):
        nb[3 * x] = mul_r(x, 1)          # r
        nb[3 * x + 1] = mul_r(x, -1)     # r^-1
        if x < m:
            nb[3 * x + 2] = m + x        # s : r^i -> s r^i
        else:
            nb[3 * x + 2] = x - m        # s : s r^i -> r^i
    indptr, indices = _csr_from_degree3(n, nb)
    return indptr, indices, {"family": "cayley_dihedral", "n": n, "m": m}


# ---------------------------------------------------------------------------
# PSL(2, q) Cayley
# ---------------------------------------------------------------------------
def _mat_mult(a, b, q):
    return (
        (a[0] * b[0] + a[1] * b[2]) % q,
        (a[0] * b[1] + a[1] * b[3]) % q,
        (a[2] * b[0] + a[3] * b[2]) % q,
        (a[2] * b[1] + a[3] * b[3]) % q,
    )


def _mat_inv(a, q):
    # det = 1, so inverse is [[d,-b],[-c,a]]
    return (a[3] % q, (-a[1]) % q, (-a[2]) % q, a[0] % q)


def _canon(a, q):
    neg = ((-a[0]) % q, (-a[1]) % q, (-a[2]) % q, (-a[3]) % q)
    return a if a <= neg else neg


def cayley_psl2(q: int, seed: int = 0):
    """Cubic Cayley graph of PSL(2,q).

    Elements are SL(2,q) matrices quotiented by {I, -I}, canonically ordered.
    Generators: a = [[1,1],[0,1]] (unipotent), b = [[0,1],[-1,0]] (an
    involution in PSL).  S = {a, a^-1, b} is symmetric of size 3, so the graph
    is 3-regular on q(q^2-1)/2 vertices.
    """
    I = (1, 0, 0, 1)
    A = (1, 1, 0, 1)
    Ainv = _mat_inv(A, q)
    B = (0, 1, q - 1, 0)
    gens = [A, Ainv, B]

    index = {_canon(I, q): 0}
    order = [_canon(I, q)]
    from collections import deque

    dq = deque([_canon(I, q)])
    while dq:
        g = dq.popleft()
        for s in gens:
            h = _canon(_mat_mult(g, s, q), q)
            if h not in index:
                index[h] = len(order)
                order.append(h)
                dq.append(h)
    n = len(order)
    # build nb by right-multiplication
    nb = np.empty(3 * n, dtype=np.int32)
    for gi, g in enumerate(order):
        for k, s in enumerate(gens):
            nb[3 * gi + k] = index[_canon(_mat_mult(g, s, q), q)]
    indptr, indices = _csr_from_degree3(n, nb)
    return indptr, indices, {
        "family": "cayley_psl2",
        "n": n,
        "q": q,
        "expected_order": q * (q * q - 1) // 2,
    }


# ---------------------------------------------------------------------------
# symmetric-group Cayley
# ---------------------------------------------------------------------------
def cayley_perm(n_sym: int, seed: int = 0):
    """Cubic Cayley graph of S_n with {(1 2), (1 2 ... n), (1 2 ... n)^-1}."""
    n = n_sym
    ident = tuple(range(n))

    def compose(p, s):
        return tuple(p[s[i]] for i in range(n))

    trans = list(ident)
    trans[0], trans[1] = 1, 0
    trans = tuple(trans)
    cyc = tuple((i + 1) % n for i in range(n))
    cyc_inv = tuple((i - 1) % n for i in range(n))
    gens = [trans, cyc, cyc_inv]

    index = {ident: 0}
    order = [ident]
    from collections import deque

    dq = deque([ident])
    while dq:
        g = dq.popleft()
        for s in gens:
            h = compose(g, s)
            if h not in index:
                index[h] = len(order)
                order.append(h)
                dq.append(h)
    N = len(order)
    nb = np.empty(3 * N, dtype=np.int32)
    for gi, g in enumerate(order):
        for k, s in enumerate(gens):
            nb[3 * gi + k] = index[compose(g, s)]
    indptr, indices = _csr_from_degree3(N, nb)
    return indptr, indices, {"family": "cayley_perm", "n": N, "n_sym": n_sym}


# ---------------------------------------------------------------------------
# high-girth expander
# ---------------------------------------------------------------------------
def high_girth(n: int, seed: int = 0, moves: int = 40_000, girth_target: int = 9):
    """Random cubic graph pushed toward high girth by annealing.

    Returns the CSR plus the *achieved* girth so the caller can report honestly
    whether C4/C8 were actually eliminated at this size.
    """
    from ..core.cycles import girth_via_bfs
    from ..core.graph import random_cubic_graph
    from ..fitness import FitnessConfig
    from ..search.sa import girth_anneal

    cfg = FitnessConfig()
    g0 = random_cubic_graph(n, np.random.default_rng(seed))
    g, info = girth_anneal(
        g0, moves=moves, seed=seed, girth_target=girth_target, config=cfg
    )
    indptr, indices = _csr_from_degree3(n, g.nb)
    return indptr, indices, {
        "family": "high_girth",
        "n": n,
        "achieved_girth": info["exact_girth"],
        "short_cycles_after": info["short_cycle_counts_after"],
    }


GENERATORS = {
    "random_cubic": random_cubic,
    "cayley_dihedral": cayley_dihedral,
    "cayley_psl2": cayley_psl2,
    "cayley_perm": cayley_perm,
    "high_girth": high_girth,
}
