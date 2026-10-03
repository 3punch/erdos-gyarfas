"""Multi-stage experiment runner.

================  ==========================================================
Stage             Purpose
================  ==========================================================
``warmup``        n = 50..100.  Validates the engine end-to-end, measures
                  throughput, checks that incremental counts match a full
                  recount, and produces the first logs.
``medium``        n = 100..300.  Runs simulated annealing, tabu search, DQN
                  and actor-critic on the *same* instances so the drivers can
                  be compared move for move.
``large``         n = 400..1000.  Parallel workers, sparse adjacency, full
                  certification sweep on the best candidates.
================  ==========================================================

Every stage writes ``results/<stage>/results.csv``, the best candidate per
size as GraphML + JSON, the figures in ``analysis/plotting.py``, and a
JSON-lines run log.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import platform
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Iterable, Sequence

import numpy as np

from ..analysis import metrics as M
from ..analysis import plotting as P
from ..core.cycles import (
    count_power_of_two_cycles,
    girth_via_bfs,
    power_of_two_lengths,
)
from ..core.graph import CubicGraph, random_cubic_graph
from ..fitness import FitnessConfig, default_config_for
from ..io_utils import append_jsonl, save_candidate, save_table
from ..search import (
    ActorCriticAgent,
    DQNAgent,
    RLSearch,
    SimulatedAnnealer,
    TabuSearch,
    girth_anneal,
)

__all__ = [
    "StageConfig",
    "STAGES",
    "run_task",
    "run_stage",
    "run_validation",
    "run_benchmark",
    "run_pareto",
    "run_all",
]


@dataclass
class StageConfig:
    name: str
    sizes: Sequence[int]
    girth_moves: int = 60_000
    sa_moves: int = 150_000
    tabu_steps: int = 2_000
    rl_steps: int = 1_500
    seeds: int = 2
    algorithms: Sequence[str] = ("sa", "tabu", "dqn", "actor_critic")
    cert_budget: int = 2_000_000
    note: str = ""


STAGES: dict[str, StageConfig] = {
    "warmup": StageConfig(
        name="warmup",
        sizes=(50, 76, 100),   # n must be even for a 3-regular graph to exist
        girth_moves=40_000,
        sa_moves=100_000,
        tabu_steps=1_500,
        rl_steps=1_000,
        seeds=2,
        note="Stage 1: engine validation, throughput and logging.",
    ),
    "medium": StageConfig(
        name="medium",
        sizes=(100, 150, 200, 300),
        girth_moves=100_000,
        sa_moves=250_000,
        tabu_steps=3_000,
        rl_steps=2_000,
        seeds=2,
        note="Stage 2: SA / tabu / DQN / actor-critic on identical instances.",
    ),
    "large": StageConfig(
        name="large",
        sizes=(400, 600, 800, 1000),
        girth_moves=150_000,
        sa_moves=400_000,
        tabu_steps=2_000,
        rl_steps=1_000,
        seeds=2,
        algorithms=("sa", "tabu"),
        note="Stage 3: scaling to 1000 vertices with parallel workers.",
    ),
}


# ---------------------------------------------------------------------------
# single task (picklable, so it can run in a worker process)
# ---------------------------------------------------------------------------
def run_task(task: dict) -> dict:
    """Run one (size, seed, algorithm) triple and return a flat record."""
    n = int(task["n"])
    seed = int(task["seed"])
    algo = task["algorithm"]
    t_start = time.perf_counter()

    cfg = default_config_for(n)
    rng = np.random.default_rng(seed)
    g0 = random_cubic_graph(n, rng)
    baseline = count_power_of_two_cycles(g0, cap=cfg.count_cap)

    g_warm, girth_info = girth_anneal(
        g0, moves=int(task.get("girth_moves", 60_000)), seed=seed,
        girth_target=9, config=cfg,
    )

    t0 = time.perf_counter()
    curves: dict[str, np.ndarray] = {}
    if algo == "sa":
        res = SimulatedAnnealer(cfg, T0=0.5, T1=0.01).run(
            g_warm, moves=int(task["sa_moves"]), seed=seed
        )
    elif algo == "tabu":
        res = TabuSearch(cfg, neighbourhood=24, tenure=40).run(
            g_warm, moves=int(task["tabu_steps"]), seed=seed
        )
    elif algo in ("dqn", "actor_critic"):
        agent = (DQNAgent if algo == "dqn" else ActorCriticAgent)(
            config=cfg, seed=seed
        )
        res = RLSearch(cfg, agent=agent, slate_size=16).run(
            g_warm, moves=int(task["rl_steps"]), seed=seed
        )
        res.meta.pop("weights", None)  # keep the CSV small
    else:
        raise ValueError(f"unknown algorithm {algo!r}")
    search_seconds = time.perf_counter() - t0

    # full certification sweep on the winner, with an explicit budget
    certified = count_power_of_two_cycles(
        res.graph, cap=cfg.count_cap, budget=int(task.get("cert_budget", 2_000_000))
    )
    meas = M.measure(res.graph, report=certified, extra={
        "algorithm": algo,
        "seed": seed,
        "girth_after_warmstart": girth_info["exact_girth"],
    })
    rec = meas.as_dict()
    rec.update({
        "stage": task.get("stage", ""),
        "algorithm": algo,
        "seed": seed,
        "search_seconds": search_seconds,
        "total_seconds": time.perf_counter() - t_start,
        "moves_per_second": res.stats.moves_per_second,
        "proposed_moves": res.stats.proposed,
        "best_loss": res.best_loss,
        "counts_match_incremental": res.meta.get("counts_match_incremental"),
        "certified_all_lengths": certified.all_certified,
        "is_counterexample": certified.is_counterexample,
        "baseline_c4": baseline.counts[0] if baseline.lengths else None,
        "baseline_c8": baseline.counts[1] if len(baseline.counts) > 1 else None,
        "baseline_c16": baseline.counts[2] if len(baseline.counts) > 2 else None,
        "baseline_total": baseline.total,
    })
    rec["_graph"] = res.graph
    rec["_history"] = res.history.as_array()
    return rec


def _strip(rec: dict) -> dict:
    return {k: v for k, v in rec.items() if not k.startswith("_")}


def _worker(task: dict) -> dict:
    try:
        return run_task(task)
    except Exception as exc:  # pragma: no cover - keep the stage alive
        return {
            "stage": task.get("stage", ""),
            "n": task.get("n"),
            "algorithm": task.get("algorithm"),
            "seed": task.get("seed"),
            "error": f"{type(exc).__name__}: {exc}",
        }


# ---------------------------------------------------------------------------
# stage driver
# ---------------------------------------------------------------------------
def run_stage(
    stage: str,
    outdir: str = "results",
    workers: int = 0,
    sizes: Sequence[int] | None = None,
    seeds: int | None = None,
    algorithms: Sequence[str] | None = None,
    scale: float = 1.0,
) -> list[dict]:
    """Run one stage and write all of its artefacts."""
    if stage not in STAGES:
        raise KeyError(f"unknown stage {stage!r}; expected one of {list(STAGES)}")
    cfg = STAGES[stage]
    root = os.path.join(outdir, stage)
    figdir = os.path.join(root, "figures")
    os.makedirs(figdir, exist_ok=True)
    graphs_dir = os.path.join(root, "graphs")

    sizes = list(sizes) if sizes else list(cfg.sizes)
    seeds_n = seeds if seeds is not None else cfg.seeds
    algos = list(algorithms) if algorithms else list(cfg.algorithms)

    tasks = []
    for n in sizes:
        for s in range(seeds_n):
            for a in algos:
                tasks.append({
                    "stage": stage,
                    "n": n,
                    "seed": 1000 + s,
                    "algorithm": a,
                    "girth_moves": int(cfg.girth_moves * scale),
                    "sa_moves": int(cfg.sa_moves * scale),
                    "tabu_steps": int(cfg.tabu_steps * scale),
                    "rl_steps": int(cfg.rl_steps * scale),
                    "cert_budget": cfg.cert_budget,
                })

    nproc = workers or min(len(tasks), max(1, (os.cpu_count() or 1)))
    print(f"[{stage}] {len(tasks)} tasks on {nproc} worker(s); sizes={sizes}")
    t0 = time.perf_counter()
    if nproc > 1:
        ctx = mp.get_context("fork")
        with ctx.Pool(nproc) as pool:
            records = pool.map(_worker, tasks)
    else:
        records = [_worker(t) for t in tasks]
    wall = time.perf_counter() - t0

    ok = [r for r in records if "error" not in r]
    for r in records:
        if "error" in r:
            print(f"  !! task failed: {r}")

    # ---- persist the best candidate per size ------------------------------
    best_rows = []
    for n in sizes:
        subset = [r for r in ok if r["n"] == n]
        if not subset:
            continue
        best = min(subset, key=lambda r: r["best_loss"])
        g = best.pop("_graph", None)
        if g is not None:
            save_candidate(
                graphs_dir, f"best_n{n}_{best['algorithm']}_s{best['seed']}", g,
                count_power_of_two_cycles(g, cap=4096, budget=cfg.cert_budget),
                extra={k: best[k] for k in best if k != "_history"},
            )
        row = _strip(best)
        row["algorithm"] = "best"
        row["best_algorithm"] = best["algorithm"]
        best_rows.append(row)
        print(f"  n={n:5d} best={best['algorithm']:12s} loss={best['best_loss']:.5f} "
              f"C4={row.get('c4')} C8={row.get('c8')} C16={row.get('c16')} "
              f"girth={row['girth']} inducedP={row['longest_induced_path_lb']}")

    all_rows = [_strip(r) for r in ok] + best_rows
    save_table(os.path.join(root, "results.csv"), all_rows)
    for r in ok:
        append_jsonl(os.path.join(root, "run_log.jsonl"), _strip(r))

    # ---- figures -----------------------------------------------------------
    if best_rows:
        P.plot_pow2_counts_vs_n(best_rows, figdir)
        P.plot_induced_path_vs_n(best_rows, figdir)
        P.plot_spectral(best_rows, figdir)
        P.plot_theory_vs_measured(best_rows, figdir)
    P.plot_scaling_timing(all_rows, figdir)

    curves = {
        f"{r['algorithm']}_n{r['n']}_s{r['seed']}": r.pop("_history", np.zeros((0, 3)))
        for r in ok if r.get("_history") is not None
    }
    if curves:
        P.plot_search_curves(curves, figdir, title=f"{stage}: best loss traces")

    summary = {
        "stage": stage,
        "note": cfg.note,
        "sizes": sizes,
        "seeds": seeds_n,
        "algorithms": algos,
        "tasks": len(tasks),
        "failures": len(records) - len(ok),
        "workers": nproc,
        "wall_seconds": wall,
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "best": best_rows,
    }
    with open(os.path.join(root, "stage_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(f"[{stage}] done in {wall:.1f}s -> {root}")
    return all_rows


# ---------------------------------------------------------------------------
# validation: does the engine reproduce the random-regular theory?
# ---------------------------------------------------------------------------
def run_validation(
    outdir: str = "results/validation",
    sizes: Sequence[int] = (50, 100, 200, 400, 1000),
    trials: int = 8,
) -> list[dict]:
    """Compare measured 2^k counts with the limiting Poisson means.

    For a random 3-regular graph the number of L-cycles converges in
    distribution to Poisson with mean ``(d-1)^L / (2L)``, i.e. ``2^L/(2L) =
    2^(L-1)/L`` for d = 3 -- so 2 for C4, 16 for C8 and 2048 for C16.  A cycle
    counter that reproduces those means across five orders of magnitude is
    almost certainly correct -- and the same numbers show why the search cannot
    win: the mean number of 16-cycles alone is already 2048.
    """
    os.makedirs(outdir, exist_ok=True)
    rows = []
    for n in sizes:
        per: dict[int, list[int]] = {}
        cert: dict[int, list[bool]] = {}
        for t in range(trials):
            g = random_cubic_graph(n, np.random.default_rng(9000 + t))
            rep = count_power_of_two_cycles(g, cap=200_000, budget=20_000_000)
            for L, c, ok in zip(rep.lengths, rep.counts, rep.certified):
                per.setdefault(L, []).append(c)
                cert.setdefault(L, []).append(ok)
        for L, vals in sorted(per.items()):
            mean = 2 ** (L - 1) / L          # (d-1)^L / (2L) with d = 3
            n_cert = sum(cert[L])
            rows.append({
                "n": n,
                "length": L,
                "measured_mean": float(np.mean(vals)),
                "measured_std": float(np.std(vals)),
                "theory_mean": mean,
                "ratio": float(np.mean(vals) / mean) if mean else float("nan"),
                "trials": len(vals),
                # below 1.0 the "measured" figure is a lower bound only: the
                # budget ran out before the enumeration completed
                "certified_trials": n_cert,
            })
            tag = "" if n_cert == len(vals) else f"  [lower bound: {n_cert}/{len(vals)} certified]"
            print(f"  n={n:5d} C{L:<4d} measured={np.mean(vals):12.2f} "
                  f"theory={mean:12.2f} ratio={np.mean(vals)/mean:6.3f}{tag}")
    save_table(os.path.join(outdir, "poisson_validation.csv"), rows)
    return rows


# ---------------------------------------------------------------------------
# literature benchmark
# ---------------------------------------------------------------------------
def run_benchmark(outdir: str = "results/benchmark",
                  sizes: Sequence[int] = (14, 20, 24, 30, 40)) -> list[dict]:
    """Reproduce Markstrom's 'no C4 and no C8' cubic graphs at several orders."""
    from ..reference import benchmark_no_c4_c8

    os.makedirs(outdir, exist_ok=True)
    rows = []
    for n in sizes:
        r = benchmark_no_c4_c8(n=n, moves=200_000, restarts=4, seed=17)
        rows.append(r.as_dict())
        print(f"  n={n:4d} success={r.success} achieved={r.achieved} "
              f"girth={r.girth} planar={r.planar} ({r.seconds:.1f}s)")
    save_table(os.path.join(outdir, "no_c4_c8_benchmark.csv"), rows)
    return rows


# ---------------------------------------------------------------------------
# the C4 / C8 frontier
# ---------------------------------------------------------------------------
def run_pareto(
    outdir: str = "results/pareto",
    sizes: Sequence[int] = (100, 200, 400),
    ratios: Sequence[float] = (16.0, 8.0, 4.0, 2.0, 1.0, 0.5, 0.25),
    moves: int = 200_000,
) -> list[dict]:
    """Trace the achievable (C4, C8) pairs as the length weights are retuned.

    Driving C4 to zero and driving C8 to zero pull against each other, and the
    shape of the trade-off is the most informative single measurement the
    search produces.
    """
    os.makedirs(outdir, exist_ok=True)
    rows = []
    for n in sizes:
        for ratio in ratios:
            cfg = FitnessConfig(length_weights={4: ratio, 8: 1.0})
            g = random_cubic_graph(n, np.random.default_rng(4242))
            g, _ = girth_anneal(g, moves=max(moves // 3, 1), seed=5,
                                girth_target=9, config=cfg)
            res = SimulatedAnnealer(cfg, T0=0.6, T1=0.01).run(
                g, moves=moves, seed=5
            )
            rep = count_power_of_two_cycles(res.graph, lengths=[4, 8, 16],
                                            cap=200_000, budget=20_000_000)
            rows.append({
                "n": n,
                "w4_over_w8": ratio,
                "c4": rep.counts[0],
                "c8": rep.counts[1],
                "c16": rep.counts[2],
                "girth": girth_via_bfs(res.graph),
            })
            print(f"  n={n:5d} w4/w8={ratio:6.2f} -> C4={rep.counts[0]:5d} "
                  f"C8={rep.counts[1]:5d} C16={rep.counts[2]:6d} girth={girth_via_bfs(res.graph)}")
    save_table(os.path.join(outdir, "pareto.csv"), rows)
    return rows


def run_all(outdir: str = "results", workers: int = 0, scale: float = 1.0) -> dict:
    """Run every stage plus the validation and benchmark suites."""
    out: dict[str, Any] = {}
    for stage in ("warmup", "medium", "large"):
        out[stage] = run_stage(stage, outdir=outdir, workers=workers, scale=scale)
    out["validation"] = run_validation(os.path.join(outdir, "validation"))
    out["benchmark"] = run_benchmark(os.path.join(outdir, "benchmark"))
    out["pareto"] = run_pareto(os.path.join(outdir, "pareto"))
    return out
