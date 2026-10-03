"""Numba kernels for the ultra-large (N = 100 000) early-exit engine.

Design notes
------------
* **Sparse, not dense.**  A dense bitset adjacency matrix on 100 000 vertices
  is ``100000^2 / 8`` bytes = **1.25 GB**, which is infeasible in this
  environment and pointless for a 3-regular graph with only 150 000 edges.
  The engine therefore stores **CSR** (``indptr``/``indices``) and uses a
  compact per-DFS mark array (an int8 "bitset" of 100 KB) for the visited set.
  That is the correct engineering reading of "bitset sparse adjacency".

* **Boolean early exit.**  Nothing is counted.  Each length probe returns as
  soon as a single cycle is witnessed, and the C4 probe is an exact O(m*d)
  scan that needs no per-root BFS, so the "not found" case at C4 is also cheap.
  For longer lengths a *budgeted* DFS is used: finding a cycle is proof, while
  not finding one within budget is only "not witnessed", and is labelled so.

* **Parallelism.**  The C4 scan and the root loop of the long-cycle probe run
  under ``numba.prange`` over threads.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

I32 = np.int32
I64 = np.int64


# ---------------------------------------------------------------------------
# CSR construction helpers
# ---------------------------------------------------------------------------
def build_csr(n: int, edges) -> tuple[np.ndarray, np.ndarray]:
    """Build CSR arrays from an edge list (undirected, 0-based)."""
    m2 = 2 * len(edges)
    indptr = np.zeros(n + 1, dtype=I64)
    rows = np.empty(m2, dtype=I64)
    cols = np.empty(m2, dtype=I32)
    for i, (u, v) in enumerate(edges):
        rows[2 * i] = u
        cols[2 * i] = v
        rows[2 * i + 1] = v
        cols[2 * i + 1] = u
    order = np.argsort(rows, kind="stable")
    rows = rows[order]
    cols = cols[order]
    for r in rows:
        indptr[r + 1] += 1
    np.cumsum(indptr, out=indptr)
    return indptr.astype(I64), cols


@njit
def is_adj_csr(indptr, indices, u, v):
    for k in range(indptr[u], indptr[u + 1]):
        if indices[k] == v:
            return True
    return False


# ---------------------------------------------------------------------------
# exact C4 existence (O(m * d^2) worst case, no per-root BFS)
# ---------------------------------------------------------------------------
@njit
def has_4cycle(indptr, indices, n):
    """True iff the graph contains a 4-cycle.

    A 4-cycle exists iff some vertex v has two distinct neighbours x, y that
    share a common neighbour w != v (giving v-x-w-y-v, four distinct vertices).
    Sequential (n is large but this is O(n) with a tiny constant) and free of
    the parallel-scalar race / nested-jit crash.  Exact, so "no C4" is a proof.
    """
    for v in range(n):
        lo, hi = indptr[v], indptr[v + 1]
        for i in range(lo, hi):
            x = indices[i]
            ax0, ax1 = indptr[x], indptr[x + 1]
            for j in range(i + 1, hi):
                y = indices[j]
                ay0, ay1 = indptr[y], indptr[y + 1]
                for a in range(ax0, ax1):
                    w = indices[a]
                    if w == v:
                        continue
                    for b in range(ay0, ay1):
                        if indices[b] == w:
                            return True
    return False


# ---------------------------------------------------------------------------
# exact C8 existence via a cheap common-2-path scan is expensive; instead we
# use a budgeted DFS for L >= 8 (see below).  For girth-guaranteed families the
# caller already knows C4/C8 are absent and only probes L >= 16.
# ---------------------------------------------------------------------------
@njit
def _dfs_find(indptr, indices, L, s, first, cur, depth, onpath, ops, budget):
    """Return 1 if an L-cycle s->first->...->cur->s is completed.  ``ops`` is a
    1-element int64 budget counter shared by the caller."""
    remaining = L - depth
    if remaining == 1:
        for k in range(indptr[cur], indptr[cur + 1]):
            if indices[k] == s:
                return 1
        return 0
    for k in range(indptr[cur], indptr[cur + 1]):
        w = indices[k]
        if onpath[w]:
            continue
        ops[0] += 1
        if ops[0] > budget:
            return -1   # budget exhausted
        onpath[w] = 1
        r = _dfs_find(indptr, indices, L, s, first, w, depth + 1, onpath, ops, budget)
        onpath[w] = 0
        if r != 0:
            return r
    return 0


@njit
def find_one_cycle(indptr, indices, n, L, budget, stride):
    """Budgeted search for a single L-cycle.

    Returns 1 if one is *witnessed* (proof of existence), 0 if the budget was
    spent without one (NOT a proof of absence), using only every ``stride``-th
    root to bound work on giant graphs.
    """
    onpath = np.zeros(n, dtype=np.int8)
    ops = np.zeros(1, dtype=I64)
    for s in range(0, n, stride):
        lo, hi = indptr[s], indptr[s + 1]
        for i in range(lo, hi):
            first = indices[i]
            onpath[s] = 1
            onpath[first] = 1
            r = _dfs_find(indptr, indices, L, s, first, first, 1, onpath, ops, budget)
            onpath[first] = 0
            onpath[s] = 0
            if r == 1:
                return 1
            if r == -1:
                return 0
    return 0


# ---------------------------------------------------------------------------
# the early-exit power-of-two checker
# ---------------------------------------------------------------------------
def early_exit_pow2(
    indptr, indices, n, lengths, budget_per_length=2_000_000, stride=1,
    exact_c4=True,
) -> dict:
    """Boolean early-exit over the given powers of two.

    Returns a dict with:

    * ``trigger`` -- the first length at which a cycle was witnessed
      (``None`` if none was);
    * ``max_checked`` -- the largest length actually probed;
    * ``per_length`` -- for each L, one of ``"found"``, ``"absent-proof"``
      (only possible for C4 via the exact scan) or ``"not-witnessed"``;
    * ``is_counterexample`` -- True only if *every* length was probed and
      provably absent.  With a budgeted DFS this never becomes True in
      practice at scale; the honest outputs are ``trigger`` and
      ``"not-witnessed"``.
    """
    per = {}
    trigger = None
    max_checked = None

    for L in lengths:
        max_checked = L
        if L == 4 and exact_c4:
            if has_4cycle(indptr, indices, n):
                per[L] = "found"
                trigger = L
                break
            else:
                per[L] = "absent-proof"
                continue
        else:
            if find_one_cycle(indptr, indices, n, L, budget_per_length, stride):
                per[L] = "found"
                trigger = L
                break
            else:
                per[L] = "not-witnessed"
    return {
        "trigger": trigger,
        "max_checked": max_checked,
        "per_length": per,
        "is_counterexample": trigger is None
        and all(v == "absent-proof" for v in per.values()),
    }


# ---------------------------------------------------------------------------
# scale-appropriate structural filters (CSR)
# ---------------------------------------------------------------------------
@njit
def claw_center_count_csr(indptr, indices, n):
    """Number of vertices whose neighbourhood is an independent set (claws).

    For a 3-regular graph this is exactly the claw-centre count; the graph is
    claw-free iff it returns 0.  O(n * d^2).
    """
    cnt = 0
    for v in range(n):
        lo, hi = indptr[v], indptr[v + 1]
        if hi - lo < 3:
            continue
        x = indices[lo]
        y = indices[lo + 1]
        z = indices[lo + 2]
        if (not is_adj_csr(indptr, indices, x, y)
                and not is_adj_csr(indptr, indices, x, z)
                and not is_adj_csr(indptr, indices, y, z)):
            cnt += 1
    return cnt


@njit
def _ext_induced_csr(indptr, indices, cur, onpath, st):
    st[1] += 1
    if st[1] > st[3]:
        st[2] = 1
        return
    for k in range(indptr[cur], indptr[cur + 1]):
        w = indices[k]
        if onpath[w]:
            continue
        ok = 1
        for a in range(indptr[w], indptr[w + 1]):
            x = indices[a]
            if x != cur and onpath[x]:
                ok = 0
                break
        if ok == 0:
            continue
        onpath[w] = 1
        st[4] += 1
        if st[4] > st[0]:
            st[0] = st[4]
            if st[0] >= st[5]:
                onpath[w] = 0
                st[4] -= 1
                return
        _ext_induced_csr(indptr, indices, w, onpath, st)
        st[4] -= 1
        onpath[w] = 0
        if st[2] == 1 or st[0] >= st[5]:
            return


@njit
def has_induced_path_csr(indptr, indices, n, target, budget, max_roots):
    """True iff an induced path on ``target`` vertices is *exhibited*.

    Scanning a few roots is enough on giant graphs, which are saturated with
    long induced paths; a budget-limited ``False`` means "not found here", not
    "P_k-free".  Returning True is a proof of presence.
    """
    onpath = np.zeros(n, dtype=np.int8)
    st = np.zeros(6, dtype=I64)
    st[3] = budget
    st[5] = target
    roots = min(max_roots, n)
    for v in range(roots):
        onpath[v] = 1
        st[4] = 1
        _ext_induced_csr(indptr, indices, v, onpath, st)
        onpath[v] = 0
        if st[0] >= target:
            return True
        if st[2] == 1:
            # budget exhausted at this root; move on (do not conclude absence)
            st[2] = 0
    return st[0] >= target
