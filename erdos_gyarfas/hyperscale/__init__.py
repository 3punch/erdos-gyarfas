"""Ultra-large (N up to 100 000) early-exit search for Erdos-Gyarfas.

Submodules
----------
* :mod:`.kernels`     -- CSR + bitset-mark early-exit cycle checker (Numba,
                         parallel) and scale-appropriate structural filters;
* :mod:`.generators`  -- Cayley (PSL(2,q), dihedral, S_n), random cubic and
                         budgeted high-girth expanders up to 100k vertices;
* :mod:`.pipeline`    -- the scaling experiment that writes
                         ``results/hyperscale/hyperscale.csv``.
"""

from .kernels import (
    build_csr,
    early_exit_pow2,
    find_one_cycle,
    has_4cycle,
    claw_center_count_csr,
    has_induced_path_csr,
)
from .generators import (
    random_cubic,
    cayley_dihedral,
    cayley_psl2,
    cayley_perm,
    high_girth,
)
from .pipeline import run_hyperscale

__all__ = [
    "build_csr",
    "early_exit_pow2",
    "find_one_cycle",
    "has_4cycle",
    "claw_center_count_csr",
    "has_induced_path_csr",
    "random_cubic",
    "cayley_dihedral",
    "cayley_psl2",
    "cayley_perm",
    "high_girth",
    "run_hyperscale",
]
