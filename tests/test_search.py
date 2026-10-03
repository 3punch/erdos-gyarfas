"""Tests for the search drivers, fitness function, IO and reference graphs.

The load-bearing invariant is the one everything else depends on: after an
arbitrary sequence of 2-opt swaps, the *incrementally maintained* cycle counts
must still equal a full independent recount.  That is asserted here for every
driver.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

from erdos_gyarfas.core.cycles import (
    count_power_of_two_cycles,
    girth_via_bfs,
    power_of_two_lengths,
)
from erdos_gyarfas.core.graph import random_cubic_graph
from erdos_gyarfas.core.structure import longest_induced_path
from erdos_gyarfas.fitness import FitnessConfig, FitnessEvaluator, default_config_for
from erdos_gyarfas.io_utils import load_graph_json, save_candidate
from erdos_gyarfas.reference import generalized_petersen, reference_library
from erdos_gyarfas.search import (
    ActorCriticAgent,
    DQNAgent,
    MoveStats,
    RLSearch,
    SimulatedAnnealer,
    TabuSearch,
    girth_anneal,
)
from erdos_gyarfas.search.moves import MoveEvaluator


def recount(graph, lengths=(4, 8)):
    rep = count_power_of_two_cycles(graph, lengths=list(lengths), cap=4096)
    return dict(zip(rep.lengths, rep.counts))


@pytest.mark.parametrize("n", [40, 100])
def test_simulated_annealing_keeps_graph_valid_and_counts_exact(n):
    cfg = default_config_for(n)
    g = random_cubic_graph(n, np.random.default_rng(3))
    res = SimulatedAnnealer(cfg).run(g, moves=20_000, seed=1)
    res.graph.validate()
    tracked = {int(k): int(v) for k, v in res.meta["verified_counts"].items()}
    assert tracked == recount(res.graph), (tracked, recount(res.graph))
    assert res.meta["counts_match_incremental"] is True
    assert res.best_loss <= res.final_loss + 1e-9
    assert res.stats.moves_per_second > 0


@pytest.mark.parametrize("driver", ["sa", "tabu", "dqn", "actor_critic"])
def test_every_driver_reduces_or_holds_the_loss(driver):
    cfg = default_config_for(60)
    g = random_cubic_graph(60, np.random.default_rng(11))
    start = FitnessEvaluator(cfg).fast_loss(recount(g), 60)
    if driver == "sa":
        res = SimulatedAnnealer(cfg).run(g, moves=20_000, seed=2)
    elif driver == "tabu":
        res = TabuSearch(cfg, neighbourhood=12, tenure=20).run(g, moves=300, seed=2)
    else:
        agent = (DQNAgent if driver == "dqn" else ActorCriticAgent)(config=cfg, seed=2)
        res = RLSearch(cfg, agent=agent, slate_size=8).run(g, moves=200, seed=2)
    res.graph.validate()
    assert res.best_loss <= start + 1e-9
    reported = res.meta.get("verified_counts") or res.meta.get("tracked_counts")
    if reported:
        assert {int(k): int(v) for k, v in reported.items()} == recount(res.graph)


def test_girth_anneal_never_overstates_the_girth():
    """The regression this guards against: a girth bound initialised to the cap."""
    cfg = default_config_for(80)
    g = random_cubic_graph(80, np.random.default_rng(21))
    before = girth_via_bfs(g)
    g2, info = girth_anneal(g, moves=30_000, seed=1, girth_target=9, config=cfg)
    g2.validate()
    exact = girth_via_bfs(g2)
    assert info["exact_girth"] == exact
    # the reported short-cycle counts must pin the girth down exactly
    short = info["short_cycle_counts_after"]
    nonzero = [L for L in range(3, 9) if short.get(L, 0) > 0]
    assert exact == (min(nonzero) if nonzero else 9)
    # annealing must not make things worse than where it started
    assert sum(short.values()) <= sum(info["short_cycle_counts_before"].values())


def test_move_evaluator_peek_is_non_destructive():
    cfg = default_config_for(40)
    g = random_cubic_graph(40, np.random.default_rng(31))
    ev = MoveEvaluator(g.copy(), cfg)
    rng = np.random.default_rng(1)
    snapshot = ev.graph.nb.copy()
    pairing = ev.graph.pairing.copy()
    for _ in range(200):
        mv = ev.propose(rng)
        assert mv is not None
        ev.peek(mv, with_features=True)
    assert np.array_equal(ev.graph.nb, snapshot)
    assert np.array_equal(ev.graph.pairing, pairing)


def test_move_evaluator_commit_matches_recount():
    cfg = default_config_for(60)
    g = random_cubic_graph(60, np.random.default_rng(41))
    ev = MoveEvaluator(g.copy(), cfg)
    rng = np.random.default_rng(2)
    for _ in range(500):
        mv = ev.propose(rng)
        d = ev.peek(mv)
        ev.commit(mv, d)
    assert recount(ev.graph) == dict(zip(map(int, ev.lengths), map(int, ev.counts)))


def test_fitness_marks_small_graphs_as_provably_not_counterexamples():
    """n <= 48 cubic graphs are exhaustively verified, so the filter must fire."""
    cfg = default_config_for(30)
    g = random_cubic_graph(30, np.random.default_rng(51))
    ev = FitnessEvaluator(cfg).evaluate(g)
    assert ev.provably_not_counterexample
    assert not ev.is_jackpot
    assert any("48" in r for r in ev.provable_reasons)


def test_fitness_hard_filters_are_reported():
    cfg = FitnessConfig()
    ev = FitnessEvaluator(cfg)
    # a generalized Petersen graph G(5,2) is the Petersen graph: non-planar,
    # triangle-free (so claw-free == False), and only 10 vertices long
    g = generalized_petersen(5, 2)
    e = ev.evaluate(g)
    assert e.claw_free is False
    assert e.planar is False
    assert e.two_edge_connected is True
    assert e.report.total > 0


def test_io_round_trip(tmp_path):
    g = random_cubic_graph(40, np.random.default_rng(61))
    rep = count_power_of_two_cycles(g)
    paths = save_candidate(str(tmp_path), "cand", g, rep, extra={"loss": 1.25})
    assert set(paths) == {"json", "graphml"}
    back = load_graph_json(paths["json"])
    back.validate()
    assert sorted(back.edges()) == sorted(g.edges())
    with open(paths["json"]) as fh:
        rec = json.load(fh)
    assert rec["power_of_two"]["total"] == rep.total
    with open(paths["graphml"]) as fh:
        text = fh.read()
    assert "<graphml" in text and "pow2_total" in text


def test_reference_library_is_cubic():
    for name, g in reference_library().items():
        g.validate()
        assert all(d == 3 for _, d in g.to_networkx().degree()), name


def test_petersen_via_generalized_constructor():
    g = generalized_petersen(5, 2)
    import networkx as nx

    assert nx.is_isomorphic(g.to_networkx(), nx.petersen_graph())


def test_power_of_two_lengths_cover_every_relevant_power():
    for n in (50, 100, 300, 1000):
        lengths = power_of_two_lengths(n)
        assert all(4 <= L <= n for L in lengths)
        assert lengths[-1] <= n < 2 * lengths[-1]


def test_longest_induced_path_lower_bound_is_honest():
    g = random_cubic_graph(200, np.random.default_rng(71))
    lb, completed = longest_induced_path(g, target=200, budget=1 << 22)
    assert lb >= 13
    assert lb <= g.n


def test_move_stats_throughput():
    s = MoveStats(proposed=1000, seconds=2.0)
    assert s.moves_per_second == 500.0


def test_report_generator_runs_on_empty_results(tmp_path):
    """`make-report` must produce a log even before any stage has run."""
    from erdos_gyarfas.experiments.make_report import make_report

    out = tmp_path / "LOG.md"
    path = make_report(str(tmp_path / "results"), str(out))
    text = pathlib.Path(path).read_text()
    assert "Experiment log" in text
    assert "0 certified counterexamples" in text


def test_report_generator_renders_real_rows(tmp_path):
    import csv

    from erdos_gyarfas.experiments.make_report import make_report

    results = tmp_path / "results" / "warmup"
    results.mkdir(parents=True)
    with open(results / "results.csv", "w", newline="") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=["n", "algorithm", "best_algorithm", "best_loss", "c4",
                        "c4_certified", "c8", "c8_certified", "c16",
                        "c16_certified", "c32", "c32_certified", "girth",
                        "longest_induced_path_lb", "two_edge_connected",
                        "planar", "claw_free", "lambda_2",
                        "certified_all_lengths", "is_counterexample",
                        "moves_per_second", "proposed_moves",
                        "search_seconds", "total_seconds"],
        )
        w.writeheader()
        w.writerow({
            "n": 100, "algorithm": "best", "best_algorithm": "tabu",
            "best_loss": 0.0, "c4": 0, "c4_certified": True,
            "c8": 0, "c8_certified": True, "c16": 1644,
            "c16_certified": True, "c32": 4096, "c32_certified": False,
            "girth": 3, "longest_induced_path_lb": 66,
            "two_edge_connected": True, "planar": False, "claw_free": False,
            "lambda_2": 2.732, "certified_all_lengths": False,
            "is_counterexample": False, "moves_per_second": 744,
            "proposed_moves": 1500, "search_seconds": 2.0, "total_seconds": 6.0,
        })
    out = tmp_path / "LOG.md"
    text = pathlib.Path(make_report(str(tmp_path / "results"), str(out))).read_text()
    assert "\u22654096" in text          # capped counts render as lower bounds
    assert "-0.00000" not in text          # no negative zero
    assert "1644" in text
