"""Phase 3 — SAT/SMT decision + explicit algebraic expanders.

Moves away from local search / random generation toward

* :mod:`.cnf` / :mod:`.solvers` -- a CNF/SMT encoding of "δ≥3 and no 2^k
  cycle" solved with CryptoMiniSat / CaDiCaL (python-sat) and Z3;
* :mod:`.lps` -- explicit Lubotzky–Phillips–Sarnak Ramanujan graphs X^{p,q};
* :mod:`.voltage` -- Z_k voltage-graph lifts engineered to cancel 2^k cycles;
* :mod:`.pipeline` -- runs all three and writes ``results/phase3/*.csv``.
"""

from .cnf import build_cnf, count_l_cycles, powers_of_two_upto, ClauseBudgetExceeded
from .solvers import solve_pysat, solve_z3, SatResult
from .lps import lps_graph, lps_generators, adjacency_lambda2
from .voltage import build_lift, random_voltages, unit_voltages
from .pipeline import run_phase3

__all__ = [
    "build_cnf", "count_l_cycles", "powers_of_two_upto", "ClauseBudgetExceeded",
    "solve_pysat", "solve_z3", "SatResult",
    "lps_graph", "lps_generators", "adjacency_lambda2",
    "build_lift", "random_voltages", "unit_voltages",
    "run_phase3",
]
