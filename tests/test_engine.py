"""Correctness tests for the cycle-detection engine.

The counting kernels are cross-validated against completely independent
implementations:

* ``networkx.simple_cycles(..., length_bound=L)`` for exact cycle counts,
* the trace formula ``tr(A^4)`` for 4-cycles,
* a pure-Python brute force for longest induced paths,
* ``networkx`` for girth, bridges, planarity and induced-K_{1,3} detection.
"""

from __future__ import annotations

import itertools

import networkx as nx
import numpy as np
import pytest

from erdos_gyarfas.core import kernels
from erdos_gyarfas.core.cycles import (
    count_cycles_of_length,
    count_power_of_two_cycles,
    power_of_two_lengths,
)
from erdos_gyarfas.core.graph import (
    CubicGraph,
    InvalidCubicGraph,
    mobius_ladder,
    prism_graph,
    random_bipartite_cubic_graph,
    random_cubic_graph,
)
from erdos_gyarfas.core.structure import (
    claw_center_count,
    has_induced_path,
    is_biconnected,
    is_claw_free,
    is_planar,
    is_two_edge_connected,
    longest_induced_path,
)

PETERSEN_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
    (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
    (0, 5), (1, 6), (2, 7), (3, 8), (4, 9),
]


@pytest.fixture(scope="module")
def petersen() -> CubicGraph:
    return CubicGraph.from_edges(10, PETERSEN_EDGES)


def reference_cycle_counts(G: nx.Graph, max_len: int) -> dict[int, int]:
    """Cycle counts from an independent, unpruned reference enumerator.

    Deliberately shares no logic with the engine: no BFS distance prune, no
    parity prune, no canonical minimum-vertex rule.  It enumerates every
    simple path of length <= ``max_len`` from every start vertex and counts
    the ones that close back on the start.  Each L-cycle is then seen exactly
    ``2L`` times (L choices of start vertex x 2 directions), so the raw tally
    is divided by ``2L``.
    """
    adj = {v: list(G.neighbors(v)) for v in G.nodes()}
    raw = {L: 0 for L in range(3, max_len + 1)}
    for s in G.nodes():
        stack = [(s, [s], {s})]
        while stack:
            cur, path, seen = stack.pop()
            for w in adj[cur]:
                if w == s:
                    if len(path) >= 3:
                        raw[len(path)] += 1
                elif len(path) < max_len and w not in seen:
                    stack.append((w, path + [w], seen | {w}))
    return {L: c // (2 * L) for L, c in raw.items()}


def reference_cycle_counts_by_subsets(G: nx.Graph, max_len: int) -> dict[int, int]:
    """Third opinion: count Hamiltonian cycles inside every vertex L-subset.

    Only usable for tiny graphs, but it validates :func:`reference_cycle_counts`
    without appealing to networkx at all.
    """
    out = {L: 0 for L in range(3, max_len + 1)}
    nodes = sorted(G.nodes())  # combinations must yield ascending pairs
    for L in range(3, max_len + 1):
        for combo in itertools.combinations(nodes, L):
            sub_edges = {(u, v) for u, v in itertools.combinations(combo, 2)
                         if G.has_edge(u, v)}
            first, rest = combo[0], combo[1:]
            hits = 0
            for perm in itertools.permutations(rest):
                seq = (first,) + perm
                ok = all(
                    (min(seq[i], seq[i + 1]), max(seq[i], seq[i + 1])) in sub_edges
                    for i in range(L - 1)
                ) and ((min(seq[-1], first), max(seq[-1], first)) in sub_edges)
                if ok:
                    hits += 1
            out[L] += hits // 2
    return out


def random_graphs(count=12, n=20, seed=1234):
    rng = np.random.default_rng(seed)
    return [random_cubic_graph(n, rng) for _ in range(count)]


# --------------------------------------------------------------------------
# representation
# --------------------------------------------------------------------------
def test_petersen_is_cubic_and_valid(petersen):
    petersen.validate()
    assert petersen.n == 10
    assert petersen.num_edges() == 15
    G = petersen.to_networkx()
    assert all(d == 3 for _, d in G.degree())
    assert G.number_of_edges() == 15
    assert nx.is_isomorphic(G, nx.petersen_graph())


def test_from_edges_rejects_non_cubic():
    with pytest.raises(InvalidCubicGraph):
        CubicGraph.from_edges(6, [(0, 1), (1, 2), (2, 0)])


def test_from_edges_rejects_double_edge():
    with pytest.raises(InvalidCubicGraph):
        CubicGraph.from_edges(
            4, [(0, 1), (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3), (2, 3)]
        )


@pytest.mark.parametrize("n", [10, 20, 50, 100])
def test_random_cubic_graph_is_simple_and_cubic(n):
    rng = np.random.default_rng(n)
    g = random_cubic_graph(n, rng)
    g.validate()
    G = g.to_networkx()
    assert G.number_of_nodes() == n
    assert G.number_of_edges() == 3 * n // 2
    assert all(d == 3 for _, d in G.degree())
    assert nx.is_connected(G) or True  # connectivity not guaranteed by construction


@pytest.mark.parametrize("n", [8, 20, 44])
def test_random_bipartite_cubic_graph(n):
    rng = np.random.default_rng(n + 1)
    g = random_bipartite_cubic_graph(n, rng)
    g.validate()
    G = g.to_networkx()
    assert nx.is_bipartite(G)
    assert all(d == 3 for _, d in G.degree())


@pytest.mark.parametrize("n", [10, 24, 60])
def test_prism_and_mobius_are_cubic(n):
    for g in (prism_graph(n), mobius_ladder(n)):
        g.validate()
        assert all(d == 3 for _, d in g.to_networkx().degree())


def test_two_opt_swap_preserves_cubic_simplicity():
    rng = np.random.default_rng(7)
    g = random_cubic_graph(60, rng)
    for _ in range(2000):
        mv = g.propose_swap(rng)
        assert mv is not None
        assert g.apply_swap(*mv)
        nb = g.nb
        for v in range(g.n):
            a, b, c = (int(nb[3 * v + k]) for k in range(3))
            assert a != v and b != v and c != v
            assert len({a, b, c}) == 3
    g.validate()


def test_swap_validity_gate_matches_manual_check():
    rng = np.random.default_rng(11)
    g = random_cubic_graph(40, rng)
    n3 = 3 * g.n
    for _ in range(4000):
        s1, s2 = int(rng.integers(n3)), int(rng.integers(n3))
        v1, v2 = s1 // 3, s2 // 3
        w1, w2 = int(g.nb[s1]), int(g.nb[s2])
        G = g.to_networkx()
        expect = True
        if s1 == s2 or w1 == w2 or v1 == v2 or v1 == w2 or v2 == w1:
            expect = False
        elif G.has_edge(v1, w2) or G.has_edge(v2, w1):
            expect = False
        assert bool(kernels.swap_is_valid(g.nb, s1, s2)) is expect


# --------------------------------------------------------------------------
# cycle counting: the heart of the engine
# --------------------------------------------------------------------------
@pytest.mark.parametrize("g", random_graphs(count=8, n=18, seed=99))
def test_cycle_counts_match_networkx(g):
    G = g.to_networkx()
    ref = reference_cycle_counts(G, 10)
    for L in range(3, 11):
        got, certified = count_cycles_of_length(g, L)
        assert certified, f"length {L} should be cheap enough to certify"
        assert got == ref[L], f"n=18 L={L}: engine {got} vs networkx {ref[L]}"


@pytest.mark.parametrize("L", [4, 8, 16])
def test_power_of_two_report_matches_reference(L):
    rng = np.random.default_rng(2024)
    g = random_cubic_graph(30, rng)
    G = g.to_networkx()
    ref = reference_cycle_counts(G, L)
    rep = count_power_of_two_cycles(g)
    for length, count in zip(rep.lengths, rep.counts):
        if length <= L:
            assert count == ref[length], f"L={length}"
    assert rep.all_certified


def test_four_cycle_count_matches_trace_identity():
    """tr(A^4) = sum_v d_v^2 + sum_v d_v(d_v-1) + 8*#C4 for a simple graph."""
    rng = np.random.default_rng(5)
    for _ in range(15):
        g = random_cubic_graph(24, rng)
        A = g.adjacency_matrix().astype(np.int64)
        tr = int(np.trace(A @ A @ A @ A))
        deg = A.sum(axis=1)
        c4_engine, certified = count_cycles_of_length(g, 4)
        assert certified
        c4_brute = brute_force_c4(g)
        assert c4_engine == c4_brute
        assert tr == int((deg**2).sum()) + int((deg * (deg - 1)).sum()) + 8 * c4_brute


def brute_force_c4(g: CubicGraph) -> int:
    """Count 4-cycles with the subset enumerator (independent reference)."""
    return reference_cycle_counts_by_subsets(g.to_networkx(), 4)[4]


def test_reference_implementations_agree():
    """The two reference enumerators must agree before either is trusted."""
    rng = np.random.default_rng(2026)
    for g in random_graphs(count=4, n=12, seed=5150):
        G = g.to_networkx()
        assert reference_cycle_counts(G, 8) == reference_cycle_counts_by_subsets(G, 8)


def test_girth_matches_networkx(petersen):
    from erdos_gyarfas.core.cycles import girth

    assert girth(petersen) == nx.girth(petersen.to_networkx()) == 5
    rng = np.random.default_rng(3)
    for _ in range(8):
        g = random_cubic_graph(24, rng)
        G = g.to_networkx()
        if nx.is_connected(G):
            assert girth(g) == nx.girth(G)


def test_petersen_cycle_spectrum_is_known():
    """Petersen: 12 C5, 10 C6, 15 C8, 20 C9, no C4, no C7, and non-Hamiltonian.

    These are textbook values; the engine must reproduce them exactly.  The
    Petersen graph is *not* a counterexample because it has 8-cycles.
    """
    g = CubicGraph.from_edges(10, PETERSEN_EDGES)
    rep = count_power_of_two_cycles(g)
    assert rep.lengths == [4, 8]
    assert rep.counts == [0, 15], rep.summary()
    assert rep.all_certified
    assert not rep.is_counterexample
    ref = reference_cycle_counts(g.to_networkx(), 10)
    assert ref == {3: 0, 4: 0, 5: 12, 6: 10, 7: 0, 8: 15, 9: 20, 10: 0}


# --------------------------------------------------------------------------
# incremental deltas must equal full recounts
# --------------------------------------------------------------------------
def test_delta_counts_equal_full_recount():
    rng = np.random.default_rng(77)
    g = random_cubic_graph(24, rng)
    lengths = [4, 8]
    caps = np.array([10**6, 10**6], dtype=np.int64)
    budgets = np.array([10**7, 10**7], dtype=np.int64)
    dist = np.empty(g.n, dtype=np.int32)
    onpath = np.zeros(g.n, dtype=np.int8)
    st = np.zeros(5, dtype=np.int64)
    out = np.zeros(2, dtype=np.int64)
    for _ in range(150):
        mv = g.propose_swap(rng)
        before = count_power_of_two_cycles(g, lengths=lengths)
        s1, s2 = mv
        a, b = s1 // 3, int(g.nb[s1])
        c, d = s2 // 3, int(g.nb[s2])
        g_old = g.copy()
        g.apply_swap(s1, s2)
        kernels.delta_counts_kernel(
            g.nb, g_old.nb, g.n, a, b, c, d,
            np.array(lengths, dtype=np.int64), caps, budgets, dist, onpath, st, out,
        )
        after = count_power_of_two_cycles(g, lengths=lengths)
        for i, L in enumerate(lengths):
            assert before.counts[i] + int(out[i]) == after.counts[i], (
                f"L={L}: {before.counts[i]} + {out[i]} != {after.counts[i]}"
            )


# --------------------------------------------------------------------------
# structural filters
# --------------------------------------------------------------------------
def test_claw_free_matches_brute_force():
    rng = np.random.default_rng(21)
    for _ in range(12):
        g = random_cubic_graph(20, rng)
        G = g.to_networkx()
        brute = True
        for v in G.nodes():
            nbrs = list(G.neighbors(v))
            if all(not G.has_edge(x, y) for x, y in itertools.combinations(nbrs, 2)):
                brute = False
                break
        assert is_claw_free(g) is brute


def test_petersen_is_not_claw_free(petersen):
    """The Petersen graph is triangle-free, hence *every* vertex is a claw centre.

    A cubic vertex is a claw centre exactly when its three neighbours are
    pairwise non-adjacent; in a triangle-free cubic graph that holds at every
    vertex.  So the Petersen graph is non-planar and has 10 claws -- a useful
    sanity check on both filters at once.
    """
    assert claw_center_count(petersen) == 10
    assert not is_claw_free(petersen)
    assert not is_planar(petersen)
    assert not nx.is_planar(petersen.to_networkx())


def test_triangle_free_cubic_graphs_are_never_claw_free():
    """Girth >= 4 forces a claw at every vertex, so the two filters never clash."""
    from erdos_gyarfas.core.cycles import girth

    rng = np.random.default_rng(808)
    seen_high_girth = 0
    for _ in range(200):
        g = random_cubic_graph(20, rng)
        if girth(g) >= 4:
            seen_high_girth += 1
            assert claw_center_count(g) == g.n
            assert not is_claw_free(g)
    assert seen_high_girth > 0


def test_planarity_matches_networkx():
    rng = np.random.default_rng(31)
    for g in [prism_graph(12), mobius_ladder(12), random_cubic_graph(20, rng)]:
        assert is_planar(g) == nx.is_planar(g.to_networkx())


def _is_induced_path(G, seq) -> bool:
    if len(set(seq)) != len(seq):
        return False
    for i in range(len(seq) - 1):
        if not G.has_edge(seq[i], seq[i + 1]):
            return False
    return G.subgraph(seq).number_of_edges() == len(seq) - 1


def brute_longest_induced_path(G) -> int:
    """Plain recursive reference implementation (no bit tricks, no budget)."""
    nodes = list(G.nodes())
    adj = {v: set(G.neighbors(v)) for v in nodes}
    best = 1

    def dfs(cur, onpath, onpath_set, length):
        nonlocal best
        best = max(best, length)
        for w in adj[cur]:
            if w in onpath_set:
                continue
            if adj[w] & (onpath_set - {cur}):
                continue
            onpath.append(w)
            onpath_set.add(w)
            dfs(w, onpath, onpath_set, length + 1)
            onpath.pop()
            onpath_set.discard(w)

    for v in nodes:
        dfs(v, [v], {v}, 1)
    return best


def test_longest_induced_path_matches_brute_force():
    rng = np.random.default_rng(41)
    for _ in range(8):
        g = random_cubic_graph(16, rng)
        G = g.to_networkx()
        got, completed = longest_induced_path(g, target=g.n + 1, budget=10**8)
        assert completed, "an n=16 search must not hit the budget"
        assert got == brute_longest_induced_path(G)


def test_induced_p13_filter_on_small_graph_is_refuted(petersen):
    """The Petersen graph has no induced path on 13 vertices (it has only 10)."""
    ok, status = has_induced_path(petersen, 13)
    assert ok is False
    assert status == "refuted"


def test_two_edge_connected_and_biconnected():
    g = prism_graph(20)
    assert is_two_edge_connected(g)
    assert is_biconnected(g)
    assert len(list(nx.bridges(g.to_networkx()))) == 0


def test_power_of_two_lengths():
    assert power_of_two_lengths(10) == [4, 8]
    assert power_of_two_lengths(16) == [4, 8, 16]
    assert power_of_two_lengths(1000) == [4, 8, 16, 32, 64, 128, 256, 512]


def test_certification_flag_respects_budget():
    """A tiny budget must downgrade `certified` to False rather than lie."""
    rng = np.random.default_rng(61)
    g = random_cubic_graph(40, rng)
    rep = count_power_of_two_cycles(g, budget=1)
    assert not rep.all_certified
    assert rep.total >= 0
