"""SAT/SMT backends for the Erdős–Gyárfás encoding.

Two interchangeable backends:

* :func:`solve_pysat` -- python-sat, defaulting to CryptoMiniSat (``cryptosat``)
  with CaDiCaL / Glucose / Kissat available;
* :func:`solve_z3` -- the Z3 SMT solver (``PbGe`` cardinalities + ``Or`` of
  negated edge atoms per forbidden cycle).

Both return a :class:`SatResult` with a status in ``{SAT, UNSAT, UNKNOWN,
BUDGET}`` and, when SAT, the extracted edge set.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Sequence

from .cnf import ClauseBudgetExceeded, build_cnf, edge_var


@dataclass
class SatResult:
    status: str                      # SAT | UNSAT | UNKNOWN | BUDGET
    n: int
    lengths: tuple
    edges: list = field(default_factory=list)
    n_clauses: int = 0
    seconds: float = 0.0
    backend: str = ""
    message: str = ""

    @property
    def sat(self) -> bool:
        return self.status == "SAT"


def _edges_from_model(n: int, true_vars: set) -> list:
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if edge_var(u, v, n) in true_vars:
                edges.append((u, v))
    return edges


# ---------------------------------------------------------------------------
# python-sat backend
# ---------------------------------------------------------------------------
def _pysat_worker(n, lengths, solver_name, clause_cap, enforce_delta3, q):
    """Run in a child process so any solver can be hard-killed on timeout
    (neither CryptoMiniSat nor CaDiCaL supports interrupt() in this build)."""
    try:
        from pysat.card import CardEnc, EncType
        from pysat.formula import CNF
        from pysat.solvers import Solver

        t0 = time.perf_counter()
        try:
            clauses, cards, info = build_cnf(n, lengths, clause_cap, enforce_delta3)
        except ClauseBudgetExceeded as exc:
            q.put(("BUDGET", [], 0, time.perf_counter() - t0, str(exc)))
            return
        cnf = CNF()
        for cl in clauses:
            cnf.append(cl)
        top_id = n * n + 2
        for lits, bound in cards:
            for cl in CardEnc.atleast(lits=lits, bound=bound, top_id=top_id,
                                      encoding=EncType.seqcounter).clauses:
                cnf.append(cl)
                top_id = max(top_id, max(abs(x) for x in cl))
        with Solver(name=solver_name, bootstrap_with=cnf) as s:
            got = s.solve()
            dt = time.perf_counter() - t0
            if got:
                true_vars = {v for v in s.get_model() if v > 0}
                q.put(("SAT", _edges_from_model(n, true_vars), cnf.nv, dt, ""))
            else:
                q.put(("UNSAT", [], cnf.nv, dt, ""))
    except Exception as exc:  # pragma: no cover
        q.put(("ERROR", [], 0, 0.0, f"{type(exc).__name__}: {exc}"))


def solve_pysat(
    n: int,
    lengths: Sequence[int],
    solver_name: str = "cms",
    timeout_s: float = 60.0,
    clause_cap: int = 2_000_000,
    enforce_delta3: bool = True,
) -> SatResult:
    import multiprocessing as mp

    ctx = mp.get_context("fork")
    q = ctx.Queue()
    p = ctx.Process(target=_pysat_worker,
                    args=(n, list(lengths), solver_name, clause_cap, enforce_delta3, q))
    t0 = time.perf_counter()
    p.start()
    p.join(timeout_s)
    dt = time.perf_counter() - t0
    backend = f"pysat:{solver_name}"
    if p.is_alive():
        p.terminate()
        p.join()
        return SatResult("UNKNOWN", n, tuple(lengths), seconds=dt, backend=backend,
                         message=f"timeout>{timeout_s:.0f}s")
    if q.empty():
        return SatResult("UNKNOWN", n, tuple(lengths), seconds=dt, backend=backend,
                         message="no result")
    status, edges, ncl, cdt, msg = q.get()
    return SatResult(status, n, tuple(lengths), edges=edges, n_clauses=ncl,
                     seconds=cdt or dt, backend=backend, message=msg)



# ---------------------------------------------------------------------------
# Z3 backend
# ---------------------------------------------------------------------------
def solve_z3(
    n: int,
    lengths: Sequence[int],
    timeout_s: float = 60.0,
    clause_cap: int = 5_000_000,
    enforce_delta3: bool = True,
) -> SatResult:
    import z3

    t0 = time.perf_counter()
    try:
        clauses, cards, info = build_cnf(n, lengths, clause_cap, enforce_delta3)
    except ClauseBudgetExceeded as exc:
        return SatResult("BUDGET", n, tuple(lengths), backend="z3",
                         seconds=time.perf_counter() - t0, message=str(exc))

    e = {}
    for u in range(n):
        for v in range(u + 1, n):
            e[(u, v)] = z3.Bool(f"e_{u}_{v}")

    s = z3.Solver()
    s.set("timeout", int(timeout_s * 1000))

    if enforce_delta3:
        for v in range(n):
            lits = [e[(min(v, u), max(v, u))] for u in range(n) if u != v]
            s.add(z3.PbGe([(lit, 1) for lit in lits], 3))

    inv = {edge_var(u, v, n): e[(u, v)]
           for u in range(n) for v in range(u + 1, n)}
    for cl in clauses:
        # cl is a list of negative edge-var ids -> Or of Not(edge bool)
        s.add(z3.Or([z3.Not(inv[-lit]) for lit in cl]))

    r = s.check()
    dt = time.perf_counter() - t0
    if r == z3.sat:
        m = s.model()
        true_vars = set()
        for (u, v), b in e.items():
            if z3.is_true(m.eval(b, model_completion=True)):
                true_vars.add(edge_var(u, v, n))
        return SatResult("SAT", n, tuple(lengths), edges=_edges_from_model(n, true_vars),
                         n_clauses=len(clauses), seconds=dt, backend="z3")
    if r == z3.unsat:
        return SatResult("UNSAT", n, tuple(lengths), n_clauses=len(clauses),
                         seconds=dt, backend="z3")
    return SatResult("UNKNOWN", n, tuple(lengths), n_clauses=len(clauses),
                     seconds=dt, backend="z3", message="timeout/unknown")
