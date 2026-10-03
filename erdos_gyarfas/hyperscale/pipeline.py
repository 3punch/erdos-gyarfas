"""Ultra-large (N = 100 000) scaling pipeline.

For each (family, size) task the pipeline:

1. generates a cubic graph as CSR;
2. applies the exclusion filters (planar, claw-free, induced-P13) and records
   whether the graph is kept or discarded and why;
3. runs the boolean early-exit power-of-two checker and records the length at
   which it triggered, the largest length probed, and wall-clock time;
4. appends a flat row to ``results/hyperscale/hyperscale.csv``.

The report section in ``docs/EXPERIMENT_LOG.md`` is then regenerated from the
CSV, so every number below is reproducible.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time
from typing import Any, Sequence

import numpy as np

from ..core.cycles import power_of_two_lengths
from ..io_utils import save_table
from . import generators as gen
from .kernels import (
    claw_center_count_csr,
    early_exit_pow2,
    has_induced_path_csr,
)

#: the exact LR planarity test costs ~4s at n=100k, so we run it for every
#: family we generate rather than assuming non-planarity; above this order it
#: would be skipped and the verdict flagged ``planar_exact=False``
PLANARITY_EXACT_CAP = 200_000


def _task_list(sizes: Sequence[int]) -> list[dict]:
    tasks = []
    for n in sizes:
        if n % 2 == 0:
            tasks.append({"family": "random_cubic", "param": n})
            tasks.append({"family": "cayley_dihedral", "param": n // 2})
            tasks.append({"family": "high_girth", "param": n})
    tasks.append({"family": "cayley_perm", "param": 8})       # 40320 vertices
    for q in (29, 41, 53):                                    # 12 180 / 34 440 / 74 412
        tasks.append({"family": "cayley_psl2", "param": q})
    return tasks


def _generate(task: dict):
    fam = task["family"]
    p = task["param"]
    if fam == "random_cubic":
        return gen.random_cubic(p)
    if fam == "cayley_dihedral":
        return gen.cayley_dihedral(p)
    if fam == "cayley_psl2":
        return gen.cayley_psl2(p)
    if fam == "cayley_perm":
        return gen.cayley_perm(p)
    if fam == "high_girth":
        moves = 30_000 if p <= 20_000 else 20_000
        return gen.high_girth(p, moves=moves)
    raise ValueError(fam)


def run_task(task: dict, max_length: int = 2048) -> dict:
    t0 = time.perf_counter()
    indptr, indices, meta = _generate(task)
    n = int(meta["n"])
    gen_seconds = time.perf_counter() - t0

    claw_centers = int(claw_center_count_csr(indptr, indices, n))
    claw_free = claw_centers == 0
    has_p13 = bool(has_induced_path_csr(indptr, indices, n, 13, 1 << 20, 64))

    if n <= PLANARITY_EXACT_CAP:
        from ..core.graph import CubicGraph
        from ..core.structure import is_planar

        g = CubicGraph.from_edges(n, _edges_from_csr(indptr, indices, n))
        planar = bool(is_planar(g))
        planar_exact = True
    else:
        # members of these families at this order are non-planar expanders;
        # the exact LR test is skipped purely for wall-clock
        planar = False
        planar_exact = False

    discarded_by = []
    if planar:
        discarded_by.append("planar")
    if claw_free:
        discarded_by.append("claw_free")
    if not has_p13:
        discarded_by.append("no_P13_found")

    row: dict[str, Any] = {
        "family": meta["family"],
        "n": n,
        "claw_centers": claw_centers,
        "claw_free": claw_free,
        "has_induced_p13": has_p13,
        "planar": planar,
        "planar_exact": planar_exact,
        "discarded": bool(discarded_by),
        "discard_reason": ",".join(discarded_by),
        "gen_seconds": gen_seconds,
    }
    row.update({k: v for k, v in meta.items() if k not in ("n", "family")})

    if discarded_by:
        row.update({"trigger_length": None, "max_checked": None,
                    "check_seconds": 0.0, "verdict": "discarded"})
        row["total_seconds"] = time.perf_counter() - t0
        return row

    lengths = [L for L in power_of_two_lengths(n, max_length=max_length)]
    # stride=1: try every root (early exit makes this cheap once a cycle is
    # witnessed).  The per-length op budget is the real cost cap; it is sized so
    # the DFS can cross the girth gap of the high-girth Cayley expanders (PSL(2,q)
    # has girth > 16, first power-of-two cycle at C32), which a 2M budget could
    # not reach and would have mislabelled as "unresolved".
    stride = 1
    t1 = time.perf_counter()
    res = early_exit_pow2(
        indptr, indices, n, lengths,
        budget_per_length=80_000_000, stride=stride,
    )
    check_seconds = time.perf_counter() - t1

    row.update({
        "trigger_length": res["trigger"],
        "max_checked": res["max_checked"],
        "per_length": {str(k): v for k, v in res["per_length"].items()},
        "check_seconds": check_seconds,
        "total_seconds": time.perf_counter() - t0,
        "verdict": "not-a-counterexample" if res["trigger"] else "unresolved",
    })
    return row


def _edges_from_csr(indptr, indices, n):
    edges = []
    for v in range(n):
        for k in range(indptr[v], indptr[v + 1]):
            w = int(indices[k])
            if v < w:
                edges.append((v, w))
    return edges


def _worker(t: dict) -> dict:
    try:
        return run_task(t)
    except Exception as exc:  # pragma: no cover
        return {"family": t.get("family"), "n": t.get("param"),
                "error": f"{type(exc).__name__}: {exc}"}


def run_hyperscale(
    outdir: str = "results/hyperscale",
    sizes: Sequence[int] = (10_000, 25_000, 50_000, 100_000),
    workers: int = 0,
    max_length: int = 2048,
) -> list[dict]:
    os.makedirs(outdir, exist_ok=True)
    tasks = _task_list(sizes)
    nproc = workers or min(len(tasks), max(1, os.cpu_count() or 1))
    print(f"[hyperscale] {len(tasks)} tasks on {nproc} worker(s)")
    t0 = time.perf_counter()
    if nproc > 1:
        ctx = mp.get_context("fork")
        with ctx.Pool(nproc) as pool:
            rows = pool.map(_worker, tasks)
    else:
        rows = [_worker(t) for t in tasks]
    wall = time.perf_counter() - t0

    rows = [r for r in rows]
    for r in rows:
        if "error" in r:
            print(f"  !! {r}")
    ok = [r for r in rows if "error" not in r]
    ok.sort(key=lambda r: (str(r.get("family")), r.get("n") or 0))
    for r in ok:
        trig = r.get("trigger_length")
        print(
            f"  {r['family']:16s} n={r['n']:7d} verdict={r['verdict']:20s} "
            f"trigger={'C'+str(trig) if trig else '-':>6s} "
            f"maxC={r.get('max_checked')} p13={r['has_induced_p13']} "
            f"clawfree={r['claw_free']} planar={r['planar']} "
            f"gen={r['gen_seconds']:.1f}s check={r['check_seconds']:.2f}s"
        )
    save_table(os.path.join(outdir, "hyperscale.csv"), ok)
    print(f"[hyperscale] done in {wall:.1f}s -> {outdir}")
    return ok
