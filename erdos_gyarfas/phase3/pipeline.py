"""Phase 3 driver: SAT sweep + LPS Ramanujan graphs + voltage lifts.

Writes three CSVs into ``results/phase3/`` and returns the rows:

* ``sat_sweep.csv``  -- per (n, lengths): solver status, time, clause count,
  and for SAT models the early-exit verdict of the extracted graph;
* ``lps.csv``        -- per (p,q): order, degree, spectral λ₂ vs the 2√p
  Ramanujan bound, girth bound, and the early-exit trigger length;
* ``voltage.csv``    -- per lift: base, k, voltage scheme, order, trigger.
"""

from __future__ import annotations

import os
import time

import networkx as nx
import numpy as np

from ..hyperscale.kernels import build_csr, early_exit_pow2
from .cnf import powers_of_two_upto
from .lps import adjacency_lambda2, lps_graph
from .solvers import solve_pysat, solve_z3
from .voltage import build_lift, random_voltages, unit_voltages

CHECK_LENGTHS = [4, 8, 16, 32, 64]


def _filters(n: int, edges: list) -> dict:
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges)
    degs = [d for _, d in G.degree()]
    # claw-free: no vertex has 3 pairwise non-adjacent neighbours
    claw_free = True
    for v in G.nodes():
        nb = list(G.neighbors(v))
        if len(nb) >= 3:
            found_indep = False
            for i in range(len(nb)):
                for j in range(i + 1, len(nb)):
                    for kk in range(j + 1, len(nb)):
                        a, b, c = nb[i], nb[j], nb[kk]
                        if (not G.has_edge(a, b) and not G.has_edge(a, c)
                                and not G.has_edge(b, c)):
                            found_indep = True
                            break
                    if found_indep:
                        break
                if found_indep:
                    break
            if found_indep:
                claw_free = False
                break
    return {
        "min_degree": min(degs) if degs else 0,
        "planar": bool(nx.is_planar(G)),
        "claw_free": claw_free,
        "connected": bool(nx.is_connected(G)),
    }


def _check(n: int, edges: list) -> dict:
    indptr, indices = build_csr(n, edges)
    res = early_exit_pow2(indptr, indices, n, CHECK_LENGTHS,
                          budget_per_length=20_000_000, stride=1)
    return {"trigger": res["trigger"], "max_checked": res["max_checked"],
            "per_length": {str(k): v for k, v in res["per_length"].items()},
            "is_counterexample": res["is_counterexample"]}


# ---------------------------------------------------------------------------
# 1. SAT / SMT sweep
# ---------------------------------------------------------------------------
def sat_sweep(outdir, small_ns=range(5, 11), large_ns=(30, 40, 50, 60, 70),
              solver="cd15", timeout_s=120.0, clause_cap=2_000_000):
    rows = []
    # (a) exact UNSAT proofs for small n (forbid every 2^k <= n)
    for n in small_ns:
        L = powers_of_two_upto(n)
        r = solve_pysat(n, L, solver_name=solver, timeout_s=timeout_s,
                        clause_cap=clause_cap)
        rows.append({"n": n, "lengths": ",".join(map(str, L)), "backend": r.backend,
                     "status": r.status, "clauses": r.n_clauses,
                     "seconds": round(r.seconds, 2), "trigger": "",
                     "verdict": ("no-counterexample(proven)" if r.status == "UNSAT"
                                 else r.status.lower()), "message": r.message})
        print(f"  [sat] n={n:3d} L={L} -> {r.status} {r.seconds:.1f}s")
    # (b) large n: only C4 is enumerable; extract model, check longer 2^k
    for n in large_ns:
        r = solve_pysat(n, [4], solver_name=solver, timeout_s=timeout_s,
                        clause_cap=5_000_000)
        if r.status == "SAT" and r.edges:
            f = _filters(n, r.edges)
            c = _check(n, r.edges)
            verdict = ("counterexample" if c["is_counterexample"]
                       else "not-a-counterexample")
            rows.append({"n": n, "lengths": "4", "backend": r.backend,
                         "status": "SAT", "clauses": r.n_clauses,
                         "seconds": round(r.seconds, 2),
                         "trigger": c["trigger"], "verdict": verdict,
                         "message": f"mindeg={f['min_degree']} planar={f['planar']} "
                                    f"clawfree={f['claw_free']} per={c['per_length']}"})
            print(f"  [sat] n={n:3d} no-C4 model -> trigger C{c['trigger']} "
                  f"mindeg={f['min_degree']} planar={f['planar']}")
        else:
            rows.append({"n": n, "lengths": "4", "backend": r.backend,
                         "status": r.status, "clauses": r.n_clauses,
                         "seconds": round(r.seconds, 2), "trigger": "",
                         "verdict": r.status.lower(), "message": r.message})
            print(f"  [sat] n={n:3d} -> {r.status}")
    _save(os.path.join(outdir, "sat_sweep.csv"), rows)
    return rows


# ---------------------------------------------------------------------------
# 2. LPS Ramanujan graphs
# ---------------------------------------------------------------------------
def lps_run(outdir, pairs=((5, 13), (5, 17), (13, 5), (5, 29), (13, 17))):
    rows = []
    for (p, q) in pairs:
        t0 = time.perf_counter()
        indptr, indices, meta = lps_graph(p, q)
        n = meta["n"]
        lam2 = adjacency_lambda2(indptr, indices, n)
        edges = _csr_edges(indptr, indices, n)
        c = _check(n, edges)
        rows.append({
            "p": p, "q": q, "n": n, "degree": meta["degree"],
            "legendre_p_q": meta["legendre_p_q"],
            "lambda2": round(lam2, 4),
            "ramanujan_bound": round(meta["ramanujan_bound"], 4),
            "ramanujan_ok": bool(lam2 <= meta["ramanujan_bound"] + 1e-6),
            "girth_lower_bound": round(meta["girth_lower_bound"], 3),
            "trigger": c["trigger"], "max_checked": c["max_checked"],
            "per_length": c["per_length"],
            "is_counterexample": c["is_counterexample"],
            "seconds": round(time.perf_counter() - t0, 2),
        })
        print(f"  [lps] X^{{{p},{q}}} n={n} deg={meta['degree']} "
              f"λ2={lam2:.3f}≤{meta['ramanujan_bound']:.3f} "
              f"trigger=C{c['trigger']}")
    _save(os.path.join(outdir, "lps.csv"), rows)
    return rows


# ---------------------------------------------------------------------------
# 3. Voltage lifts
# ---------------------------------------------------------------------------
def voltage_run(outdir, ks=(3, 5, 7)):
    base = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
            (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
            (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)]  # Petersen graph
    n_base = 10
    rows = []
    for k in ks:
        for scheme, volt in (("unit", unit_voltages(base, k)),
                             ("random", random_voltages(base, k, seed=k))):
            indptr, indices, meta = build_lift(base, n_base, k, volt)
            n = meta["n"]
            edges = _csr_edges(indptr, indices, n)
            c = _check(n, edges)
            rows.append({"base": "petersen", "k": k, "scheme": scheme, "n": n,
                         "trigger": c["trigger"], "max_checked": c["max_checked"],
                         "per_length": c["per_length"],
                         "is_counterexample": c["is_counterexample"]})
            print(f"  [volt] Petersen×Z_{k} ({scheme}) n={n} -> trigger C{c['trigger']}")
    _save(os.path.join(outdir, "voltage.csv"), rows)
    return rows


def _csr_edges(indptr, indices, n):
    edges = []
    for v in range(n):
        for kk in range(indptr[v], indptr[v + 1]):
            w = int(indices[kk])
            if v < w:
                edges.append((v, w))
    return edges


def _save(path, rows):
    import csv
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        return
    keys = sorted({k for r in rows for k in r})
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    print(f"  -> {path}")


def run_phase3(outdir="results/phase3", small_ns=range(5, 11),
               large_ns=(30, 40, 50, 60, 70), solver="cd15", timeout_s=120.0):
    os.makedirs(outdir, exist_ok=True)
    print("[phase3] SAT/SMT sweep")
    sat = sat_sweep(outdir, small_ns=small_ns, large_ns=large_ns,
                    solver=solver, timeout_s=timeout_s)
    print("[phase3] LPS Ramanujan graphs")
    lps = lps_run(outdir)
    print("[phase3] voltage lifts")
    volt = voltage_run(outdir)
    return {"sat": sat, "lps": lps, "voltage": volt}
