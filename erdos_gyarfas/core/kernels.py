"""Numba kernels for the EG-Search engine.

Conventions
-----------
A cubic (3-regular) simple graph on ``n`` vertices is stored as a flat
``int32`` array ``nb`` of length ``3n`` such that the neighbours of vertex
``v`` are ``nb[3v]``, ``nb[3v + 1]`` and ``nb[3v + 2]``.  Equivalently, the
graph is a perfect matching on ``3n`` half-edges ("stubs"): stub ``p``
belongs to vertex ``p // 3`` and is matched to stub ``pairing[p]``; then
``nb[p] == pairing[p] // 3``.

Storing the adjacency as a fixed-stride array (instead of CSR with an
``indptr``) lets every kernel hard-code degree three, which removes an
indirection from the hottest loops and makes the DFS branching factor a
compile-time constant.

All counting kernels are *budgeted*: they return a ``(count, completed)``
pair, where ``completed == 0`` means the answer is a lower bound only
(the operation budget or the counting cap was exhausted).  Callers must
never report ``count == 0`` as a proof of absence unless ``completed == 1``.
"""

from __future__ import annotations

import numpy as np
from numba import njit

# NOTE: ``cache=True`` is deliberately NOT used anywhere in this module.  The
# cycle-counting kernels are recursive, and caching recursive njit functions
# produces a hard segfault in numba 0.68 (observed directly: a cached
# ``_extend`` crashed the interpreter, the identical uncached function did
# not).  Compilation of this module takes ~2 s once per process, which is
# negligible next to the search budgets.

I32 = np.int32
I64 = np.int64


# ---------------------------------------------------------------------------
# 2-opt edge swaps on the half-edge matching
# ---------------------------------------------------------------------------
@njit(inline="always")
def _is_adj(nb, u, v):
    """True iff u ~ v (degree 3)."""
    b = 3 * u
    return nb[b] == v or nb[b + 1] == v or nb[b + 2] == v


@njit
def swap_is_valid(nb, s1, s2):
    """Can stubs ``s1`` and ``s2`` exchange partners and keep the graph simple?

    The swap turns edges (v1, w1) and (v2, w2) into (v1, w2) and (v2, w1),
    where ``v_i = s_i // 3`` and ``w_i`` is the vertex currently matched to
    ``s_i``.  It is rejected when it would create a loop or a double edge,
    or when it is a no-op.
    """
    if s1 == s2:
        return False
    v1 = s1 // 3
    v2 = s2 // 3
    w1 = nb[s1]
    w2 = nb[s2]
    if w1 == w2:
        return False            # both stubs already share a partner
    if v1 == v2:
        return False            # would produce a loop at v1
    if v1 == w2 or v2 == w1:
        return False            # loop
    # A double edge appears as soon as *either* new edge duplicates a surviving
    # one.  Neither (v1,w2) nor (v2,w1) can coincide with a removed edge here,
    # because that would force w1 == w2 or v1 == v2, both rejected above.
    if _is_adj(nb, v1, w2) or _is_adj(nb, v2, w1):
        return False
    return True


@njit
def swap_apply(pairing, nb, s1, s2):
    """Apply a valid 2-opt swap.  Returns 1 on success, 0 if it was invalid."""
    p1 = pairing[s1]
    p2 = pairing[s2]
    if p1 == s2:
        return 0
    v1 = s1 // 3
    v2 = s2 // 3
    w1 = p1 // 3
    w2 = p2 // 3
    if w1 == w2 or v1 == v2 or v1 == w2 or v2 == w1:
        return 0
    if _is_adj(nb, v1, w2) or _is_adj(nb, v2, w1):
        return 0
    pairing[s1] = p2
    pairing[p2] = s1
    pairing[s2] = p1
    pairing[p1] = s2
    nb[s1] = w2
    nb[p2] = v1
    nb[s2] = w1
    nb[p1] = v2
    return 1


@njit
def swap_apply_unchecked(pairing, nb, s1, s2):
    """Apply a swap that :func:`swap_is_valid` already accepted."""
    p1 = pairing[s1]
    p2 = pairing[s2]
    pairing[s1] = p2
    pairing[p2] = s1
    pairing[s2] = p1
    pairing[p1] = s2
    nb[s1] = p2 // 3
    nb[p2] = s1 // 3
    nb[s2] = p1 // 3
    nb[p1] = s2 // 3


@njit
def random_swap_seed(nb, pairing, rng_state, attempts):
    """Draw up to ``attempts`` random valid swaps; return the pair or (-1, -1)."""
    n3 = nb.shape[0]
    for _ in range(attempts):
        s1 = np.int32(np.random.randint(0, n3))
        s2 = np.int32(np.random.randint(0, n3))
        if swap_is_valid(nb, s1, s2):
            return s1, s2
    return np.int32(-1), np.int32(-1)


# ---------------------------------------------------------------------------
# BFS distances, bipartiteness, girth
# ---------------------------------------------------------------------------
@njit
def bfs_from(nb, n, src, dist, queue):
    """BFS distances from ``src`` into ``dist`` (pre-filled with -1)."""
    dist[:] = -1
    dist[src] = 0
    head = 0
    tail = 0
    queue[tail] = src
    tail += 1
    while head < tail:
        u = queue[head]
        head += 1
        du = dist[u] + 1
        b = 3 * u
        for k in range(3):
            w = nb[b + k]
            if dist[w] == -1:
                dist[w] = du
                queue[tail] = w
                tail += 1


@njit
def component_bipartite_flags(nb, n, out):
    """``out[v] == 1`` iff the connected component of ``v`` is bipartite."""
    color = np.full(n, np.int8(-1))
    queue = np.empty(n, I32)
    out[:] = 1
    for s in range(n):
        if color[s] != -1:
            continue
        bip = np.int8(1)
        color[s] = 0
        head = 0
        tail = 0
        queue[tail] = s
        tail += 1
        while head < tail:
            u = queue[head]
            head += 1
            cu = color[u]
            base = 3 * u
            for k in range(3):
                w = nb[base + k]
                if color[w] == -1:
                    color[w] = np.int8(1 - cu)
                    queue[tail] = w
                    tail += 1
                elif color[w] == cu:
                    bip = np.int8(0)
        if bip == 0:
            for i in range(tail):
                out[queue[i]] = 0
    return out


@njit
def girth_via_bfs(nb, n, max_girth):
    """Exact girth (length of a shortest cycle), or ``max_girth + 1`` if none.

    Standard "BFS from every vertex, look for an edge closing an odd/even
    cycle" routine; runs in O(n m) which is fast for cubic graphs.
    """
    dist = np.empty(n, I32)
    parent = np.empty(n, I32)
    queue = np.empty(n, I32)
    best = max_girth + 1
    for s in range(n):
        dist[:] = -1
        parent[:] = -1
        dist[s] = 0
        head = 0
        tail = 0
        queue[tail] = s
        tail += 1
        while head < tail:
            u = queue[head]
            head += 1
            if dist[u] * 2 >= best:
                continue
            base = 3 * u
            for k in range(3):
                w = nb[base + k]
                if dist[w] == -1:
                    dist[w] = dist[u] + 1
                    parent[w] = u
                    queue[tail] = w
                    tail += 1
                elif w != parent[u]:
                    cand = dist[u] + dist[w] + 1
                    if cand < best:
                        best = cand
            if best == 3:
                return 3
    return best


@njit
def shortest_cycle_through_edge(nb, n, u, v, cap_len):
    """Length of a shortest cycle using edge uv, or ``cap_len + 1`` if longer.

    Computed as 1 + dist_{G - uv}(u, v) with a depth-limited BFS.
    """
    dist = np.full(n, np.int32(-1))
    queue = np.empty(n, I32)
    dist[u] = 0
    head = 0
    tail = 0
    queue[tail] = u
    tail += 1
    while head < tail:
        x = queue[head]
        head += 1
        if dist[x] >= cap_len:
            continue
        base = 3 * x
        for k in range(3):
            w = nb[base + k]
            if x == u and w == v:
                continue
            if dist[w] == -1:
                if w == v:
                    return dist[x] + 2
                dist[w] = dist[x] + 1
                queue[tail] = w
                tail += 1
    return cap_len + 1


# ---------------------------------------------------------------------------
# Exact simple-cycle counting of a prescribed length
# ---------------------------------------------------------------------------
@njit
def _extend(nb, n, L, cap, s, first, cur, depth, onpath, dist, parity_ok, st):
    """Depth-first extension of the path s -> first -> ... -> cur.

    ``depth`` counts edges used so far.  ``st`` is ``[count, ops, exhausted,
    budget, 0]``.  Cycles are canonicalised by requiring the minimum vertex
    ``s`` and, among its two neighbours on the cycle, the smaller one as
    ``first``; each cycle is therefore counted exactly once.
    """
    remaining = L - depth
    if remaining == 1:
        base = 3 * cur
        for k in range(3):
            if nb[base + k] == s:
                if cur > first:
                    st[0] += 1
                    if st[0] >= cap:
                        return
                break
        return
    base = 3 * cur
    for k in range(3):
        w = nb[base + k]
        if w < s:
            continue        # s must stay the minimum vertex of the cycle
        if onpath[w]:
            continue
        d = dist[w]
        if d < 0:
            continue
        if d > remaining - 1:
            continue
        if parity_ok and ((d - (remaining - 1)) % 2) != 0:
            continue
        st[1] += 1
        if st[1] > st[3]:
            st[2] = 1
            return
        onpath[w] = 1
        _extend(nb, n, L, cap, s, first, w, depth + 1, onpath, dist, parity_ok, st)
        onpath[w] = 0
        if st[2] == 1 or st[0] >= cap:
            return


@njit
def _extend_to_target(
    nb, n, L, cap, tgt, cur, depth, onpath, dist, parity_ok, st, bu, bv, ban
):
    """Extend the path tgt -> ... -> cur until it closes back on ``tgt``.

    Used to count cycles through a *fixed* anchoring edge, which fixes the
    orientation and makes canonicalisation unnecessary.  When ``ban == 1``
    the edge (bu, bv) may not be used.
    """
    remaining = L - depth
    if remaining == 1:
        if ban and ((cur == bu and tgt == bv) or (cur == bv and tgt == bu)):
            return
        base = 3 * cur
        for k in range(3):
            if nb[base + k] == tgt:
                st[0] += 1
                if st[0] >= cap:
                    return
                break
        return
    base = 3 * cur
    for k in range(3):
        w = nb[base + k]
        if onpath[w]:
            continue
        if ban and ((cur == bu and w == bv) or (cur == bv and w == bu)):
            continue
        d = dist[w]
        if d < 0:
            continue
        if d > remaining - 1:
            continue
        if parity_ok and ((d - (remaining - 1)) % 2) != 0:
            continue
        st[1] += 1
        if st[1] > st[3]:
            st[2] = 1
            return
        onpath[w] = 1
        _extend_to_target(
            nb, n, L, cap, tgt, w, depth + 1, onpath, dist, parity_ok, st, bu, bv, ban
        )
        onpath[w] = 0
        if st[2] == 1 or st[0] >= cap:
            return


@njit
def count_cycles_from_start_kernel(
    nb, n, L, cap, budget, s, dist, onpath, parity_ok, st
):
    """Count L-cycles whose *minimum* vertex is ``s``.

    ``dist`` must already hold BFS distances from ``s``.  Returns
    ``(count, completed)`` where ``completed == 0`` means the count is only a
    lower bound (cap hit or budget exhausted).  The caller owns the outer loop
    over start vertices and the BFS, so that one distance layer can be shared
    across several lengths.
    """
    if L < 3 or L > n:
        return (0, 1)
    st[0] = 0
    st[1] = 0
    st[2] = 0
    st[3] = budget
    base = 3 * s
    onpath[s] = 1
    for k in range(3):
        first = nb[base + k]
        if first < s:
            continue
        onpath[first] = 1
        _extend(nb, n, L, cap, s, first, first, 1, onpath, dist, parity_ok, st)
        onpath[first] = 0
        if st[2] == 1 or st[0] >= cap:
            break
    onpath[s] = 0
    completed = 1 if (st[2] == 0 and st[0] < cap) else 0
    return (st[0], completed)


@njit
def count_pow2_cycles_kernel(nb, n, lengths, caps, budgets, out_count, out_done):
    """Count cycles of every length in ``lengths`` in a single sweep.

    The BFS distance layer from each start vertex is shared across all
    lengths, so the O(n m) BFS cost is paid once instead of once per length.
    """
    dist = np.empty(n, I32)
    queue = np.empty(n, I32)
    onpath = np.zeros(n, np.int8)
    bip = np.zeros(n, np.int8)
    component_bipartite_flags(nb, n, bip)
    st = np.zeros(5, I64)
    nl = lengths.shape[0]
    for s in range(n):
        bfs_from(nb, n, s, dist, queue)
        parity_ok = bip[s]
        base = 3 * s
        for li in range(nl):
            if out_done[li] == 0:
                continue
            L = lengths[li]
            if L > n:
                continue
            st[0] = 0
            st[1] = 0
            st[2] = 0
            st[3] = budgets[li]
            cap = caps[li]
            onpath[s] = 1
            for k in range(3):
                first = nb[base + k]
                if first < s:
                    continue
                onpath[first] = 1
                _extend(
                    nb, n, L, cap, s, first, first, 1, onpath, dist, parity_ok, st
                )
                onpath[first] = 0
                if st[2] == 1 or st[0] >= cap:
                    break
            onpath[s] = 0
            out_count[li] += st[0]
            if st[2] == 1 or out_count[li] >= cap:
                out_done[li] = 0
    return


@njit
def count_through_edge(nb, n, L, u, v, cap, budget, dist, onpath, parity_ok,
                       bu, bv, ban, st):
    """Cycles of length ``L`` through the edge uv (optionally avoiding bu-bv)."""
    if L < 3:
        return 0
    st[0] = 0
    st[1] = 0
    st[2] = 0
    st[3] = budget
    onpath[u] = 1
    onpath[v] = 1
    _extend_to_target(
        nb, n, L, cap, u, v, 1, onpath, dist, parity_ok, st, bu, bv, ban
    )
    onpath[v] = 0
    onpath[u] = 0
    return st[0]


@njit
def count_using_two_edges(nb, n, L, u1, v1, u2, v2, cap, budget, dist, onpath,
                          parity_ok, st, queue):
    """``|{C : C is an L-cycle using edge (u1,v1) or edge (u2,v2)}|``.

    Each anchored DFS needs a BFS layer rooted at its own anchor, so the
    distance array is recomputed between the two calls -- pruning against a
    layer rooted elsewhere silently discards valid cycles.
    """
    bfs_from(nb, n, u1, dist, queue)
    a = count_through_edge(
        nb, n, L, u1, v1, cap, budget, dist, onpath, parity_ok, 0, 0, 0, st
    )
    if a >= cap or st[2] == 1:
        return cap
    bfs_from(nb, n, u2, dist, queue)
    b = count_through_edge(
        nb, n, L, u2, v2, cap - a, budget, dist, onpath, parity_ok, u1, v1, 1, st
    )
    return a + b


@njit
def count_cycles_through_edge(nb, n, L, u, v, cap, budget, dist, onpath, parity_ok, st):
    """Number of L-cycles containing edge uv (both endpoints already adjacent)."""
    return count_through_edge(
        nb, n, L, u, v, cap, budget, dist, onpath, parity_ok, 0, 0, 0, st
    )


@njit
def count_four_cycles_through_edge(nb, u, v):
    """Closed-form #C4 through edge uv for a cubic graph (no DFS needed)."""
    total = 0
    bv = 3 * v
    for i in range(3):
        p = nb[bv + i]
        if p == u:
            continue
        bp = 3 * p
        for j in range(3):
            q = nb[bp + j]
            if q == v or q == p or q == u:
                continue
            bq = 3 * q
            hit = 0
            for t in range(3):
                x = nb[bq + t]
                if x == u:
                    hit = 1
                    break
            if hit:
                total += 1
    return total


@njit
def count_four_cycles_using_two(nb, u1, v1, u2, v2, check_overlap):
    """#C4 using (u1,v1) or (u2,v2), with exact inclusion-exclusion."""
    a = count_four_cycles_through_edge(nb, u1, v1)
    b = count_four_cycles_through_edge(nb, u2, v2)
    overlap = 0
    if check_overlap:
        # The only 4-cycle using two disjoint edges is the alternating one.
        if _is_adj(nb, v1, u2) and _is_adj(nb, v2, u1):
            overlap += 1
        if _is_adj(nb, v1, v2) and _is_adj(nb, u2, u1):
            overlap += 1
    return a + b - overlap


# ---------------------------------------------------------------------------
# Structural filters
# ---------------------------------------------------------------------------
@njit
def claw_center_count(nb, n):
    """Number of vertices whose three neighbours are pairwise non-adjacent.

    For a cubic graph this is exactly the number of induced K_{1,3} centres,
    so the graph is claw-free iff this returns 0.
    """
    cnt = 0
    for v in range(n):
        b = 3 * v
        x = nb[b]
        y = nb[b + 1]
        z = nb[b + 2]
        if not _is_adj(nb, x, y) and not _is_adj(nb, x, z) and not _is_adj(nb, y, z):
            cnt += 1
    return cnt


@njit
def _ext_induced(nb, cur, onpath, st):
    """DFS for induced paths; ``st = [best, ops, exhausted, budget, curlen, target]``."""
    st[1] += 1
    if st[1] > st[3]:
        st[2] = 1
        return
    base = 3 * cur
    for k in range(3):
        w = nb[base + k]
        if onpath[w]:
            continue
        ok = 1
        bw = 3 * w
        for j in range(3):
            x = nb[bw + j]
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
        _ext_induced(nb, w, onpath, st)
        st[4] -= 1
        onpath[w] = 0
        if st[2] == 1 or st[0] >= st[5]:
            return


@njit
def longest_induced_path_kernel(nb, n, target, budget):
    """Longest induced path (in vertices) found within ``budget`` DFS nodes.

    Returns ``(best, completed, budget_exhausted)``.  ``best >= target`` is a
    *proof* that the graph contains an induced path on ``target`` vertices;
    ``completed == 1 and best < target`` is a *proof* that it does not.
    """
    onpath = np.zeros(n, np.int8)
    st = np.zeros(6, I64)
    st[3] = budget
    st[5] = target
    for v in range(n):
        onpath[v] = 1
        st[4] = 1
        _ext_induced(nb, v, onpath, st)
        onpath[v] = 0
        if st[2] == 1 or st[0] >= target:
            break
    exhausted = st[2]
    completed = 1 if st[2] == 0 else 0
    return (st[0], completed, exhausted)


@njit
def _bridge_dfs(nb, u, parent, disc, low, timer, is_bridge):
    disc[u] = timer[0]
    low[u] = timer[0]
    timer[0] += 1
    base = 3 * u
    for k in range(3):
        w = nb[base + k]
        if w == parent:
            continue
        if disc[w] == -1:
            _bridge_dfs(nb, w, u, disc, low, timer, is_bridge)
            if low[w] < low[u]:
                low[u] = low[w]
            if low[w] > disc[u]:
                is_bridge[0] += 1
        elif disc[w] < low[u]:
            low[u] = disc[w]


@njit
def bridge_count(nb, n):
    """Number of bridges (cut edges) in the graph."""
    disc = np.full(n, np.int32(-1))
    low = np.empty(n, I32)
    timer = np.zeros(1, I64)
    is_bridge = np.zeros(1, I64)
    for s in range(n):
        if disc[s] == -1:
            _bridge_dfs(nb, s, -1, disc, low, timer, is_bridge)
    return is_bridge[0]


@njit
def _artic_dfs(nb, u, parent, disc, low, timer, state):
    disc[u] = timer[0]
    low[u] = timer[0]
    timer[0] += 1
    children = 0
    base = 3 * u
    for k in range(3):
        w = nb[base + k]
        if w == parent:
            continue
        if disc[w] == -1:
            children += 1
            _artic_dfs(nb, w, u, disc, low, timer, state)
            if low[w] < low[u]:
                low[u] = low[w]
            if parent != -1 and low[w] >= disc[u]:
                state[0] = 1
        elif disc[w] < low[u]:
            low[u] = disc[w]
    if parent == -1 and children > 1:
        state[0] = 1


@njit
def is_biconnected_kernel(nb, n):
    """True iff the graph is connected and has no articulation point."""
    disc = np.full(n, np.int32(-1))
    low = np.empty(n, I32)
    timer = np.zeros(1, I64)
    state = np.zeros(1, I64)
    _artic_dfs(nb, 0, -1, disc, low, timer, state)
    for v in range(n):
        if disc[v] == -1:
            return False
    return state[0] == 0


@njit
def count_triangles(nb, n):
    cnt = 0
    for v in range(n):
        b = 3 * v
        x = nb[b]
        y = nb[b + 1]
        z = nb[b + 2]
        if _is_adj(nb, x, y):
            cnt += 1
        if _is_adj(nb, x, z):
            cnt += 1
        if _is_adj(nb, y, z):
            cnt += 1
    return cnt // 3


@njit
def is_connected_kernel(nb, n):
    seen = np.zeros(n, np.int8)
    queue = np.empty(n, I32)
    seen[0] = 1
    head = 0
    tail = 0
    queue[tail] = 0
    tail += 1
    while head < tail:
        u = queue[head]
        head += 1
        base = 3 * u
        for k in range(3):
            w = nb[base + k]
            if seen[w] == 0:
                seen[w] = 1
                queue[tail] = w
                tail += 1
    return tail == n


@njit
def delta_counts_kernel(
    nb_new, nb_old, n, a, b, c, d, lengths, caps, budgets, dist, onpath, st, out
):
    """Exact per-length cycle-count change for the 2-opt swap on stubs (a-side, c-side).

    ``(a, b)`` and ``(c, d)`` are the two *removed* edges (``b = nb_old[a]``,
    ``d = nb_old[c]``); the swap replaces them by ``(a, d)`` and ``(c, b)``.
    The parity (bipartite) prune is deliberately switched off here: a swap can
    change whether a component is bipartite, and the prune is only a speed-up,
    so dropping it keeps the delta exact.  Callers should pass short lengths
    (4 and 8 in the default configuration) where the un-pruned DFS is tiny.
    """
    queue = np.empty(n, I32)
    for i in range(lengths.shape[0]):
        L = lengths[i]
        if L < 3 or L > n:
            out[i] = 0
            continue
        if L == 4:
            add = count_four_cycles_using_two(nb_new, a, d, b, c, True)
            rem = count_four_cycles_using_two(nb_old, a, b, c, d, True)
            out[i] = add - rem
            continue
        add = count_using_two_edges(
            nb_new, n, L, a, d, b, c, caps[i], budgets[i], dist, onpath, 0, st, queue
        )
        rem = count_using_two_edges(
            nb_old, n, L, a, b, c, d, caps[i], budgets[i], dist, onpath, 0, st, queue
        )
        out[i] = add - rem
    return
