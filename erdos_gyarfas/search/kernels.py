"""Numba kernels for the search drivers.

The inner loop of the metaheuristics lives here so that a move costs a few
microseconds rather than the tens of microseconds a Python-level loop adds.

Two invariants make the loop cheap:

* **Exact incremental deltas.**  A 2-opt swap only changes two edges, so the
  change in the number of L-cycles is computed exactly by enumerating cycles
  through those four edges -- never by recounting the whole graph.
* **A rolling ``nb_old`` snapshot.**  ``nb_old`` is kept equal to ``nb``
  between moves; applying a swap therefore *creates* the before/after pair
  the delta kernel needs, and only four array entries ever have to be
  restored.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from ..core.kernels import (
    _is_adj,
    count_four_cycles_using_two,
    shortest_cycle_through_edge,
    swap_apply_unchecked,
    swap_is_valid,
)

I32 = np.int32
I64 = np.int64


# ---------------------------------------------------------------------------
# lean DFS for cycles through a fixed edge (no distance / parity pruning)
# ---------------------------------------------------------------------------
@njit
def _ext_t(nb, L, cap, tgt, cur, depth, onpath, st, bu, bv, ban):
    """Enumerate L-cycles tgt -> cur -> ... -> tgt, optionally avoiding bu-bv.

    No distance or parity prune is used: those require a BFS layer rooted at
    ``tgt``, which costs more than the whole DFS for the short lengths this
    kernel is meant for.
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
        st[1] += 1
        if st[1] > st[3]:
            st[2] = 1
            return
        onpath[w] = 1
        _ext_t(nb, L, cap, tgt, w, depth + 1, onpath, st, bu, bv, ban)
        onpath[w] = 0
        if st[2] == 1 or st[0] >= cap:
            return


@njit
def _through_edge_lean(nb, L, u, v, cap, budget, onpath, st, bu, bv, ban):
    st[0] = 0
    st[1] = 0
    st[2] = 0
    st[3] = budget
    onpath[u] = 1
    onpath[v] = 1
    _ext_t(nb, L, cap, u, v, 1, onpath, st, bu, bv, ban)
    onpath[v] = 0
    onpath[u] = 0
    return st[0]


@njit
def count_using_two_edges_lean(nb, L, u1, v1, u2, v2, cap, budget, onpath, st):
    """``|{L-cycles using (u1,v1) or (u2,v2)}|`` without distance pruning."""
    a = _through_edge_lean(nb, L, u1, v1, cap, budget, onpath, st, 0, 0, 0)
    if a >= cap or st[2] == 1:
        return cap
    b = _through_edge_lean(
        nb, L, u2, v2, cap - a, budget, onpath, st, u1, v1, 1
    )
    return a + b


@njit
def delta_fast(nb_new, nb_old, n, a, b, c, d, lengths, caps, budgets,
               onpath, st, out):
    """Exact per-length delta for the swap turning (a,b),(c,d) into (a,d),(c,b)."""
    for i in range(lengths.shape[0]):
        L = lengths[i]
        if L == 4:
            out[i] = count_four_cycles_using_two(nb_new, a, d, b, c, True) \
                - count_four_cycles_using_two(nb_old, a, b, c, d, True)
        else:
            add = count_using_two_edges_lean(
                nb_new, L, a, d, b, c, caps[i], budgets[i], onpath, st
            )
            rem = count_using_two_edges_lean(
                nb_old, L, a, b, c, d, caps[i], budgets[i], onpath, st
            )
            out[i] = add - rem
    return


@njit
def loss_from_counts(counts, caps, weights, n):
    total = 0.0
    for i in range(counts.shape[0]):
        c = counts[i]
        if c > caps[i]:
            c = caps[i]
        total += weights[i] * c
    return total / n


# ---------------------------------------------------------------------------
# move proposal
# ---------------------------------------------------------------------------
@njit
def propose_swap(nb, n3, attempts):
    for _ in range(attempts):
        s1 = np.random.randint(0, n3)
        s2 = np.random.randint(0, n3)
        if swap_is_valid(nb, np.int32(s1), np.int32(s2)):
            return np.int32(s1), np.int32(s2)
    return np.int32(-1), np.int32(-1)


@njit
def restore_four(nb, idx, vals):
    """Write the four saved neighbour entries back (used to roll a move back)."""
    for k in range(4):
        nb[idx[k]] = vals[k]


# ---------------------------------------------------------------------------
# simulated annealing driver (power-of-two objective)
# ---------------------------------------------------------------------------
@njit
def sa_search(
    nb, pairing, n, n_moves, T0, T1, seed, weights, lengths, caps, budgets,
    counts, onpath, st, log_every, out_log, out_counts, out_stats,
):
    """Simulated annealing over 2-opt swaps, minimising the weighted 2^k count.

    ``out_log`` is a ``(ceil(n_moves/log_every), 4)`` float array receiving
    ``[iteration, temperature, loss, best_loss]``.  ``out_stats`` receives
    ``[proposed, accepted, improved, final_loss, best_loss]``.
    Returns the best ``nb`` found (the caller keeps ``pairing`` in sync).
    """
    np.random.seed(seed)
    n3 = 3 * n
    nl = lengths.shape[0]
    nb_old = nb.copy()
    out_delta = np.zeros(nl, I64)
    idx = np.zeros(4, I64)
    vals = np.zeros(4, I64)

    cur_loss = loss_from_counts(counts, caps, weights, n)
    best_loss = cur_loss
    best_nb = nb.copy()
    best_pairing = pairing.copy()
    best_counts = counts.copy()

    proposed = 0
    accepted = 0
    improved = 0
    log_row = 0

    for it in range(n_moves):
        frac = it / max(n_moves - 1, 1)
        T = T0 * ((T1 / T0) ** frac) if T0 > 0 else 0.0
        s1, s2 = propose_swap(nb, n3, 8)
        if s1 < 0:
            continue
        proposed += 1
        a = s1 // 3
        b = nb[s1]
        c = s2 // 3
        d = nb[s2]
        p1 = pairing[s1]
        p2 = pairing[s2]

        swap_apply_unchecked(pairing, nb, s1, s2)
        delta_fast(nb, nb_old, n, a, b, c, d, lengths, caps, budgets,
                   onpath, st, out_delta)

        new_loss = cur_loss
        for i in range(nl):
            nc = counts[i] + out_delta[i]
            if nc < 0:
                nc = 0
            old_c = counts[i] if counts[i] < caps[i] else caps[i]
            new_c = nc if nc < caps[i] else caps[i]
            new_loss += weights[i] * (new_c - old_c) / n
        # keep `counts` in sync with the accepted state only
        take = False
        if new_loss <= cur_loss:
            take = True
        else:
            r = np.random.random()
            if T > 0 and r < np.exp(-(new_loss - cur_loss) / T):
                take = True

        idx[0] = s1
        idx[1] = s2
        idx[2] = p1
        idx[3] = p2
        if take:
            for i in range(nl):
                nc = counts[i] + out_delta[i]
                if nc < 0:
                    nc = 0
                counts[i] = nc
            cur_loss = new_loss
            accepted += 1
            # nb_old <- nb on the four touched stubs
            vals[0] = nb[s1]
            vals[1] = nb[s2]
            vals[2] = nb[p1]
            vals[3] = nb[p2]
            restore_four(nb_old, idx, vals)
            if cur_loss < best_loss - 1e-12:
                best_loss = cur_loss
                best_nb[:] = nb
                best_pairing[:] = pairing
                best_counts[:] = counts
                improved += 1
        else:
            # roll the move back: nb <- nb_old on the four touched stubs
            vals[0] = nb_old[s1]
            vals[1] = nb_old[s2]
            vals[2] = nb_old[p1]
            vals[3] = nb_old[p2]
            restore_four(nb, idx, vals)
            pairing[s1] = p1
            pairing[p1] = s1
            pairing[s2] = p2
            pairing[p2] = s2

        if log_every > 0 and (it + 1) % log_every == 0:
            if log_row < out_log.shape[0]:
                out_log[log_row, 0] = it + 1
                out_log[log_row, 1] = T
                out_log[log_row, 2] = cur_loss
                out_log[log_row, 3] = best_loss
                log_row += 1

    nb[:] = best_nb
    pairing[:] = best_pairing
    counts[:] = best_counts
    out_counts[:] = best_counts
    out_stats[0] = proposed
    out_stats[1] = accepted
    out_stats[2] = improved
    out_stats[3] = cur_loss
    out_stats[4] = best_loss
    return best_loss


@njit
def claw_delta(nb_new, nb_old, n, a, b, c, d):
    """Change in the number of claw centres caused by the same swap.

    Only the four endpoints a, b, c, d can change status, so this is O(1).
    """
    new_cnt = 0
    old_cnt = 0
    # only a, c, d, b can change status
    verts_new = (a, c, d, b)
    for k in range(4):
        v = verts_new[k]
        base = 3 * v
        x = nb_new[base]
        y = nb_new[base + 1]
        z = nb_new[base + 2]
        if not _is_adj(nb_new, x, y) and not _is_adj(nb_new, x, z) and not _is_adj(nb_new, y, z):
            new_cnt += 1
        x = nb_old[base]
        y = nb_old[base + 1]
        z = nb_old[base + 2]
        if not _is_adj(nb_old, x, y) and not _is_adj(nb_old, x, z) and not _is_adj(nb_old, y, z):
            old_cnt += 1
    return new_cnt - old_cnt
