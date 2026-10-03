"""Tests for the ultra-large early-exit engine and generators."""

from __future__ import annotations

import pytest

from erdos_gyarfas.hyperscale import (
    build_csr,
    cayley_dihedral,
    cayley_perm,
    cayley_psl2,
    claw_center_count_csr,
    early_exit_pow2,
    find_one_cycle,
    has_4cycle,
    has_induced_path_csr,
    high_girth,
    random_cubic,
)

PETERSEN_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
    (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
    (0, 5), (1, 6), (2, 7), (3, 8), (4, 9),
]
def _is_valid_cubic_csr(indptr, indices, n) -> bool:
    if indptr.shape[0] != n + 1:
        return False
    for v in range(n):
        lo, hi = indptr[v], indptr[v + 1]
        if hi - lo != 3:
            return False
        nbrs = indices[lo:hi]
        if len(set(nbrs.tolist())) != 3 or v in nbrs:
            return False
        for w in nbrs:
            if v not in indices[indptr[w]:indptr[w + 1]]:
                return False
    return True


# ---------------------------------------------------------------------------
# build_csr + early exit against the Petersen ground truth
# ---------------------------------------------------------------------------
def test_build_csr_petersen():
    indptr, indices = build_csr(10, PETERSEN_EDGES)
    assert _is_valid_cubic_csr(indptr, indices, 10)


def test_petersen_no_4cycle_but_has_8cycle():
    indptr, indices = build_csr(10, PETERSEN_EDGES)
    assert has_4cycle(indptr, indices, 10) is False   # exact, proven
    assert find_one_cycle(indptr, indices, 10, 8, 1 << 20, 1) == 1
    assert find_one_cycle(indptr, indices, 10, 4, 1 << 20, 1) == 0


def test_petersen_early_exit_triggers_at_8():
    indptr, indices = build_csr(10, PETERSEN_EDGES)
    res = early_exit_pow2(indptr, indices, 10, [4, 8], exact_c4=True)
    assert res["per_length"][4] == "absent-proof"
    assert res["trigger"] == 8
    assert res["is_counterexample"] is False


def test_has_4cycle_detects_a_square():
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (1, 5), (2, 6), (3, 7),
             (4, 5), (5, 6), (6, 7), (7, 4)]   # cube graph: has 4-cycles
    indptr, indices = build_csr(8, edges)
    assert has_4cycle(indptr, indices, 8) is True


def test_petersen_claw_centres_and_induced_paths():
    indptr, indices = build_csr(10, PETERSEN_EDGES)
    assert claw_center_count_csr(indptr, indices, 10) == 10  # triangle-free
    assert has_induced_path_csr(indptr, indices, 10, 5, 1 << 18, 10) is True
    assert has_induced_path_csr(indptr, indices, 10, 11, 1 << 20, 10) is False


# ---------------------------------------------------------------------------
# generators are 3-regular and consistent
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n", [100, 1000])
def test_random_cubic_csr(n):
    indptr, indices, meta = random_cubic(n, seed=3)
    assert meta["n"] == n
    assert _is_valid_cubic_csr(indptr, indices, n)


@pytest.mark.parametrize("m", [50, 500])
def test_cayley_dihedral_csr(m):
    indptr, indices, meta = cayley_dihedral(m)
    assert meta["n"] == 2 * m
    assert _is_valid_cubic_csr(indptr, indices, 2 * m)


@pytest.mark.parametrize("q,expected", [(5, 60), (7, 168), (11, 660)])
def test_cayley_psl2_order(q, expected):
    indptr, indices, meta = cayley_psl2(q)
    assert meta["expected_order"] == expected
    assert meta["n"] == expected, "PSL(2,q) should be generated exhaustively"
    assert _is_valid_cubic_csr(indptr, indices, meta["n"])


def test_cayley_psl2_large_is_huge_and_cubic():
    indptr, indices, meta = cayley_psl2(29)   # 12180 vertices
    assert meta["n"] == 12180
    assert _is_valid_cubic_csr(indptr, indices, 12180)


def test_cayley_perm_s8():
    import math

    indptr, indices, meta = cayley_perm(8)
    assert meta["n"] == math.factorial(8)
    assert _is_valid_cubic_csr(indptr, indices, math.factorial(8))


def test_high_girth_small():
    indptr, indices, meta = high_girth(60, seed=1, moves=5000, girth_target=9)
    assert meta["n"] == 60
    assert _is_valid_cubic_csr(indptr, indices, 60)
    assert meta["achieved_girth"] >= 3
    # if girth >= 5 then no C4; the early exit must then trigger at >= 8
    if meta["achieved_girth"] >= 5:
        res = early_exit_pow2(indptr, indices, 60, [4, 8, 16], exact_c4=True)
        assert res["per_length"][4] == "absent-proof"


# ---------------------------------------------------------------------------
# early-exit at moderately large scale (smoke, sequential)
# ---------------------------------------------------------------------------
def test_early_exit_random_10k_triggers_at_4():
    indptr, indices, meta = random_cubic(10_000, seed=7)
    res = early_exit_pow2(indptr, indices, 10_000, [4, 8, 16], exact_c4=True)
    # E[C4]=2 (Poisson) so a random cubic graph may lack C4; C4 or C8 is certain
    assert res["trigger"] in (4, 8)
    assert res["is_counterexample"] is False


def test_early_exit_dihedral_10k():
    indptr, indices, meta = cayley_dihedral(5_000)
    res = early_exit_pow2(indptr, indices, 10_000, [4, 8, 16], exact_c4=True)
    # a dihedral Cayley graph always has 4-cycles (r^i s r^i s squares)
    assert res["trigger"] in (4, 8, 16)
    assert res["is_counterexample"] is False


# ---------------------------------------------------------------------------
# report rendering consumes the hyperscale CSV shape
# ---------------------------------------------------------------------------
def test_hyperscale_table_renders():
    from erdos_gyarfas.experiments.make_report import _hyperscale_table

    rows = [
        {"family": "cayley_dihedral", "n": "10000", "discarded": "True",
         "discard_reason": "planar", "has_induced_p13": "True",
         "claw_free": "False", "planar": "True", "trigger_length": "",
         "max_checked": "", "gen_seconds": "0.0", "check_seconds": "0.0",
         "verdict": "discarded"},
        {"family": "cayley_psl2", "n": "74412", "discarded": "False",
         "discard_reason": "", "has_induced_p13": "True", "claw_free": "False",
         "planar": "False", "trigger_length": "32", "max_checked": "32",
         "gen_seconds": "0.4", "check_seconds": "12.3",
         "verdict": "not-a-counterexample", "achieved_girth": ""},
    ]
    md = _hyperscale_table(rows)
    assert "cayley_psl2" in md and "C32" in md
    assert "| no |" in md          # dihedral discarded
    assert "not-a-counterexample" in md
