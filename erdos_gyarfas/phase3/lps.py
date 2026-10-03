"""Explicit Lubotzky–Phillips–Sarnak (LPS) Ramanujan graphs X^{p,q}.

For distinct primes p, q ≡ 1 (mod 4) the LPS construction yields a
(p+1)-regular Ramanujan graph: every non-trivial adjacency eigenvalue λ
satisfies |λ| ≤ 2√p (the Alon–Boppana optimum), and the girth is at least
(4/3)·log_q(|V|).  These are the canonical explicit expanders, so they are the
natural algebraic test-bed for the Erdős–Gyárfás conjecture: does high girth +
optimal spectral gap eliminate power-of-two cycles?

Construction (Hamilton quaternions):

* Fix r with r² ≡ −1 (mod q)  (exists because q ≡ 1 mod 4).  This gives a
  faithful representation of the Hamilton quaternions into M₂(F_q)::

      a + b i + c j + d k  ↦  [[a+br, c+dr], [−c+dr, a−br]]   (det = a²+b²+c²+d²)

* The p+1 generators are the images of the integer quaternions of norm p
  (a²+b²+c²+d² = p), taken projectively (M ~ scalar·M).
* Vertices are PGL(2, F_q) = GL(2, F_q) / F_q^× ; the graph is the Cayley graph
  w.r.t. the generator set, hence (p+1)-regular.

Note: the standard LPS family is (p+1)-regular, so a *cubic* LPS graph would
need p = 2, which is not ≡ 1 (mod 4); the smallest member is 6-regular (p=5).
The conjecture only needs δ ≥ 3, which every member satisfies.
"""

from __future__ import annotations

from itertools import product

import numpy as np


def _sqrt_minus_one_mod_q(q: int) -> int:
    for r in range(2, q):
        if (r * r) % q == q - 1:
            return r
    raise ValueError(f"-1 is not a quadratic residue mod {q} (need q ≡ 1 mod 4)")


def _legendre(a: int, q: int) -> int:
    if a % q == 0:
        return 0
    v = pow(a % q, (q - 1) // 2, q)
    return 1 if v == 1 else -1


def _quat_matrix(a: int, b: int, c: int, d: int, r: int, q: int):
    return ((a + b * r) % q, (c + d * r) % q, (-c + d * r) % q, (a - b * r) % q)


def _canonical(m: tuple, q: int) -> tuple:
    """Canonical representative of the scalar-class of an invertible matrix."""
    a, b, c, d = m
    for x in (a, b, c, d):
        if x % q:
            inv = pow(x, q - 2, q)
            return tuple((inv * y) % q for y in m)
    raise ValueError("zero matrix")


def _matmul(m1: tuple, m2: tuple, q: int) -> tuple:
    a, b, c, d = m1
    e, f, g, h = m2
    return ((a * e + b * g) % q, (a * f + b * h) % q,
            (c * e + d * g) % q, (c * f + d * h) % q)


def lps_generators(p: int, q: int, r: int) -> list[tuple]:
    """The p+1 projective generator matrices from norm-p integer quaternions.

    Of the r_4(p) = 8(p+1) integer solutions of a²+b²+c²+d² = p, the *primary*
    ones (a > 0 and a odd) number exactly p+1 and form a transversal modulo the
    Lipschitz units; their projective images are the LPS generators.  The set is
    closed under inversion (the conjugate a−bi−cj−dk is also primary), so the
    Cayley graph is undirected and (p+1)-regular.
    """
    lim = int(p ** 0.5) + 1
    mats = set()
    for a, b, c, d in product(range(-lim, lim + 1), repeat=4):
        if a * a + b * b + c * c + d * d != p:
            continue
        if a <= 0 or a % 2 == 0:        # primary: a > 0, a odd
            continue
        m = _quat_matrix(a, b, c, d, r, q)
        mats.add(_canonical(m, q))
    return sorted(mats)


def lps_graph(p: int, q: int):
    """Return (indptr, indices, meta) for the Cayley graph of PGL(2,q) on the
    symmetric set of projective images of norm-p integer Hamilton quaternions.

    This is the LPS generator set *before* quotienting by the maximal order's
    unit group, so the degree is |S| (e.g. 24 for p=5) rather than the classical
    p+1; the graph is still an explicit algebraic expander and we verify its
    spectral gap empirically against the Ramanujan bound 2*sqrt(deg-1).
    """
    if p == q:
        raise ValueError("p and q must be distinct")
    r = _sqrt_minus_one_mod_q(q)
    gens = lps_generators(p, q, r)

    # enumerate PGL(2,q): canonical forms of all invertible 2x2 matrices mod q
    index = {}
    verts = []
    for a in range(q):
        for b in range(q):
            for c in range(q):
                for d in range(q):
                    if (a * d - b * c) % q == 0:
                        continue
                    m = _canonical((a, b, c, d), q)
                    if m not in index:
                        index[m] = len(verts)
                        verts.append(m)
    n = len(verts)
    assert n == q * (q * q - 1), f"|PGL(2,{q})| should be {q*(q*q-1)}, got {n}"

    deg = len(gens)
    indptr = np.arange(0, (n + 1) * deg, deg, dtype=np.int64)
    indices = np.empty(n * deg, dtype=np.int32)
    for gi, g in enumerate(verts):
        base = gi * deg
        for k, s in enumerate(gens):
            nb = _canonical(_matmul(g, s, q), q)
            indices[base + k] = index[nb]

    # When Legendre(p,q) = -1 the generators lie in PSL(2,q) (index 2 in
    # PGL(2,q)) and the Cayley graph splits into two cosets; keep the connected
    # component of the identity, which is the proper LPS graph.
    identity = _canonical((1, 0, 0, 1), q)
    start = index[identity]
    seen = np.zeros(n, dtype=np.int8)
    stack = [start]
    seen[start] = 1
    while stack:
        v = stack.pop()
        for kk in range(indptr[v], indptr[v + 1]):
            w = int(indices[kk])
            if not seen[w]:
                seen[w] = 1
                stack.append(w)
    comp = np.nonzero(seen)[0]
    if len(comp) < n:
        remap = -np.ones(n, dtype=np.int64)
        remap[comp] = np.arange(len(comp))
        new_indptr = np.arange(0, (len(comp) + 1) * deg, deg, dtype=np.int64)
        new_indices = np.empty(len(comp) * deg, dtype=np.int32)
        for new_v, old_v in enumerate(comp):
            lo, hi = indptr[old_v], indptr[old_v + 1]
            new_indices[new_v * deg:(new_v + 1) * deg] = remap[indices[lo:hi]]
        indptr, indices, n = new_indptr, new_indices, len(comp)

    meta = {
        "family": "lps", "p": p, "q": q, "n": n, "degree": deg,
        "sqrt_minus1": r, "legendre_p_q": _legendre(p, q),
        "ramanujan_bound": float(2 * np.sqrt(deg - 1)),
        "girth_lower_bound": float((4.0 / 3.0) * np.log(n) / np.log(q)),
    }
    return indptr, indices, meta


def adjacency_lambda2(indptr, indices, n: int) -> float:
    """Largest |eigenvalue| other than the trivial ±degree (spectral check)."""
    import scipy.sparse as sp
    from scipy.sparse.linalg import eigsh

    A = sp.csr_matrix(
        (np.ones(len(indices), dtype=np.float64), indices, indptr), shape=(n, n)
    )
    deg = indptr[1] - indptr[0]
    # top eigenvalue is deg (all-ones); find the next largest in magnitude
    vals = eigsh(A.astype(float), k=min(6, n - 1), which="LA", return_eigenvectors=False)
    vals = sorted(float(v) for v in vals)
    nontrivial = [v for v in vals if abs(v - deg) > 1e-6]
    return max(abs(v) for v in nontrivial) if nontrivial else 0.0
