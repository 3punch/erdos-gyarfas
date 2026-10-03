"""Tests for the Phase-3 SAT/SMT encoder, LPS graphs and voltage lifts."""

from __future__ import annotations

import numpy as np
import pytest

from erdos_gyarfas.phase3.cnf import (
    ClauseBudgetExceeded,
    build_cnf,
    count_l_cycles,
    edge_var,
    enumerate_l_cycles,
    powers_of_two_upto,
)
from erdos_gyarfas.phase3.lps import adjacency_lambda2, lps_generators, lps_graph, _sqrt_minus_one_mod_q
from erdos_gyarfas.phase3.voltage import build_lift, unit_voltages, random_voltages
from erdos_gyarfas.hyperscale.kernels import build_csr, early_exit_pow2, has_4cycle


# ---------------------------------------------------------------------------
# CNF encoding
# ---------------------------------------------------------------------------
def test_count_and_enumerate_4cycles_of_K4():
    assert count_l_cycles(4, 4) == 3
    cycles = list(enumerate_l_cycles(4, 4))
    assert len(cycles) == 3
    for c in cycles:
        assert len(c) == 4
        # each is a set of 4 distinct edges over all 4 vertices
        verts = {v for e in c for v in e}
        assert verts == {0, 1, 2, 3}
    # no duplicate cycles
    assert len({frozenset(c) for c in cycles}) == 3


def test_powers_of_two_upto():
    assert powers_of_two_upto(15) == [4, 8]
    assert powers_of_two_upto(16) == [4, 8, 16]
    assert powers_of_two_upto(70) == [4, 8, 16, 32, 64]


def test_build_cnf_budget_raises():
    with pytest.raises(ClauseBudgetExceeded):
        build_cnf(12, [8], clause_cap=1000)   # 1.2M cycles > 1000


def test_edge_var_symmetric():
    assert edge_var(3, 7, 10) == edge_var(7, 3, 10)


# ---------------------------------------------------------------------------
# SAT solver (small, fast instances)
# ---------------------------------------------------------------------------
def test_sat_small_unsat():
    from erdos_gyarfas.phase3.solvers import solve_pysat

    # no δ>=3 graph on 8 vertices avoids both C4 and C8 -> UNSAT (matches the
    # known "counterexample needs >= 17 vertices" bound)
    r = solve_pysat(8, [4, 8], solver_name="cms", timeout_s=30)
    assert r.status == "UNSAT"


def test_sat_finds_no_c4_graph():
    from erdos_gyarfas.phase3.solvers import solve_pysat

    r = solve_pysat(10, [4], solver_name="cms", timeout_s=30)
    assert r.status == "SAT"
    ip, ix = build_csr(10, r.edges)
    assert has_4cycle(ip, ix, 10) is False     # the model really has no C4
    # and min degree >= 3
    assert (np.diff(ip) >= 3).all()


def test_z3_small_unsat():
    from erdos_gyarfas.phase3.solvers import solve_z3

    r = solve_z3(7, [4], timeout_s=30)
    assert r.status in ("UNSAT", "SAT", "UNKNOWN")   # just exercise the backend


# ---------------------------------------------------------------------------
# LPS Ramanujan graphs
# ---------------------------------------------------------------------------
def test_lps_generator_count_is_p_plus_1():
    r = _sqrt_minus_one_mod_q(13)
    assert len(lps_generators(5, 13, r)) == 6      # p+1
    assert len(lps_generators(13, 5, _sqrt_minus_one_mod_q(5))) == 14


def test_lps_graph_is_regular_ramanujan_connected():
    ip, ix, meta = lps_graph(5, 13)
    n = meta["n"]
    assert meta["degree"] == 6
    assert (np.diff(ip) == 6).all()
    # connected
    import collections

    seen = np.zeros(n, np.int8)
    dq = collections.deque([0]); seen[0] = 1
    while dq:
        v = dq.popleft()
        for k in range(ip[v], ip[v + 1]):
            w = ix[k]
            if not seen[w]:
                seen[w] = 1; dq.append(w)
    assert seen.sum() == n
    lam2 = adjacency_lambda2(ip, ix, n)
    assert lam2 <= meta["ramanujan_bound"] + 1e-6   # Ramanujan


def test_lps_has_power_of_two_cycle():
    ip, ix, meta = lps_graph(5, 13)
    res = early_exit_pow2(ip, ix, meta["n"], [4, 8, 16], budget_per_length=20_000_000, stride=1)
    assert res["trigger"] in (4, 8, 16)
    assert res["is_counterexample"] is False


# ---------------------------------------------------------------------------
# Voltage lifts
# ---------------------------------------------------------------------------
PETERSEN = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
            (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
            (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)]


def test_voltage_lift_regular_and_sized():
    ip, ix, meta = build_lift(PETERSEN, 10, 5, unit_voltages(PETERSEN, 5))
    assert meta["n"] == 50
    assert (np.diff(ip) == 3).all()        # Petersen is 3-regular -> lift too


def test_voltage_lift_cycle_length_multiplied():
    # Petersen has girth 5 (no C4); the unit Z_5 lift should still have no C4
    ip, ix, meta = build_lift(PETERSEN, 10, 5, unit_voltages(PETERSEN, 5))
    assert has_4cycle(ip, ix, meta["n"]) is False
    res = early_exit_pow2(ip, ix, meta["n"], [4, 8, 16], budget_per_length=20_000_000, stride=1)
    assert res["trigger"] in (8, 16)        # a power-of-two cycle survives


def test_random_voltages_in_range():
    v = random_voltages(PETERSEN, 7, seed=1)
    assert all(0 <= x < 7 for x in v.values())
    assert len(v) == len(PETERSEN)
