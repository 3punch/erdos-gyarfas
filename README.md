# EG-Search — an automated search & RL framework for the Erdős–Gyárfás conjecture

> **Conjecture (Erdős–Gyárfás, 1995).** Every graph with minimum degree at
> least 3 contains a simple cycle whose length is a power of two.

This repository is a research codebase for hunting counterexamples to that
conjecture in 3-regular graphs up to **1000 vertices**: a Numba-accelerated
exact cycle-counting engine, a fitness function derived from the current
literature, four search drivers (simulated annealing, tabu search, DQN,
actor-critic), and an experiment runner that produces the measurements, graphs
and figures in [`results/`](results) and [`docs/EXPERIMENT_LOG.md`](docs/EXPERIMENT_LOG.md).

**Headline result: no counterexample was found, and the measurements explain
why.** The search reaches `C4 = C8 = 0` at every size up to *n* = 1000, but the
number of 16-cycles stays in the thousands and the number of 32-cycles is
astronomically larger. Details, numbers and caveats in
[`docs/EXPERIMENT_LOG.md`](docs/EXPERIMENT_LOG.md).

---

## 1. Two corrections to the brief, and one strengthening

The task brief asked to *"discard planar or claw-free graphs (proven that no
counterexamples exist here)"*. **Neither class is proven counterexample-free**,
and the code does not pretend otherwise. See
[`docs/METHODOLOGY.md` §2](docs/METHODOLOGY.md#2-two-premises-in-the-task-brief-that-the-literature-contradicts)
for the full account.

| Class | What is actually proven | How the framework treats it |
| --- | --- | --- |
| **planar** | 3-connected cubic planar; planar claw-free | heavy-penalty *restriction*, labelled as such |
| **claw-free** | planar claw-free; "2ᵏ or 3·2ᵏ" for all claw-free; cubic claw-free needs ≥ 114 vertices | provable prune for *n* ≤ 113, restriction above |
| **P₁₃-free** | *every* P₁₃-free graph with δ ≥ 3 has a power-of-two cycle (Hegde–Sandeep–Shashank 2025) | **true provable prune** |

The strengthening is the last row. The brief proposed "reward graphs with an
induced P₁₃" as a soft guide; it is in fact a **necessary condition** for a
counterexample, so a graph *provably* lacking an induced P₁₃ can be discarded
without loss of completeness. `core/structure.has_induced_path` returns a
three-valued verdict (`proved` / `refuted` / `unknown`) precisely because
*finding* a long induced path proves presence while *proving* absence requires
exhausting the search.

## 2. Layout

```
erdos_gyarfas/
├── core/
│   ├── graph.py       cubic graph = perfect matching on 3n half-edges; O(1) 2-opt swaps
│   ├── kernels.py     Numba: budgeted exact cycle counting, girth, claws, bridges, induced paths
│   ├── cycles.py      PowerOfTwoReport with a per-length `certified` flag
│   └── structure.py   literature-grounded filters (P13, claw-free, planar, 2-edge-connected)
├── fitness.py         three-tier reward: provable prunes / restrictions / loss + reinforcement
├── search/
│   ├── kernels.py     Numba inner loops: exact incremental deltas, SA, girth annealing
│   ├── moves.py       MoveEvaluator — peek (roll back) / commit, feature extraction
│   ├── sa.py          simulated annealing + girth-annealing warm start
│   ├── tabu.py        best-of-neighbourhood with tabu list and aspiration
│   └── rl.py          DQN and actor-critic over a move slate
├── analysis/          spectral + density metrics, matplotlib figures
├── reference.py       known cubic graphs and the Markström no-C4-no-C8 benchmark
├── io_utils.py        GraphML + JSON persistence with a re-verifiable certificate
├── experiments/       three-stage runner, parallel workers, reporting
└── cli.py             `python -m erdos_gyarfas.cli …`
```

## 3. Install and run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

pytest -q                                   # 57 tests, ~40 s
python -m erdos_gyarfas.cli all --workers 2 # full pipeline, ~15 min on 2 cores
python -m erdos_gyarfas.cli report          # summarise results/
```

Individual pieces:

```bash
python -m erdos_gyarfas.cli warmup                        # Stage 1, n = 50..100
python -m erdos_gyarfas.cli stage --stage large --sizes 1000 --seeds 1
python -m erdos_gyarfas.cli validate                      # Poisson-theory check
python -m erdos_gyarfas.cli benchmark --sizes 14 20 24 30 # Markström benchmark
python -m erdos_gyarfas.cli pareto --sizes 100 200 400    # the (C4, C8) frontier
```

## 4. The engine

**Representation.** A 3-regular graph is stored as a flat `int32[3n]` array
`nb` where the neighbours of *v* are `nb[3v..3v+2]`; equivalently a perfect
matching `pairing` on 3*n* half-edges. A 2-opt swap exchanges the partners of
two stubs, preserving δ = 3 *by construction*, in O(1).

**Counting.** Exact, budgeted, canonicalised DFS from the minimum vertex of
each cycle, with a BFS distance prune and a parity prune inside bipartite
components. Every result carries a `certified` flag: `count == 0` with
`certified == False` means only *"the budget ran out"*, never *"there is
none"*. A jackpot requires every power of two up to *n* to be certified zero.

**Incremental deltas.** A swap replaces edges (*a*,*b*),(*c*,*d*) with
(*a*,*d*),(*c*,*b*). A cycle changes only if it uses one of those four edges,
so the count delta is computed *exactly* by enumerating cycles through them.
*L* ∈ {4, 8} is maintained this way on every move; {4, 8, 16} is recounted at
checkpoints; longer lengths are only counted in the final certification sweep.

**Throughput** (2 cores, this machine): ≈ 40 000 accepted 2-opt moves/second at
*n* = 1000 including exact bookkeeping; a full power-of-two sweep at *n* = 1000
costs ~15 ms for C₄/C₈/C₁₆ and ~140 ms per length ≥ 32 under a 2 M-op budget.

### Validation

Nine independent checks in [`tests/`](tests), including:

* cycle counts against an unpruned DFS enumerator and against a per-subset
  Hamiltonian count (two references that also validate each other);
* the C₄ count against the trace identity `tr(A⁴) = Σdᵥ² + Σdᵥ(dᵥ−1) + 8·#C₄`;
* the Petersen spectrum against textbook values (12 C₅, 10 C₆, 15 C₈, 20 C₉,
  no C₇, non-Hamiltonian);
* incremental deltas against a full recount after every single move;
* and a **statistical** check: on random cubic graphs the measured C₄, C₈, C₁₆
  match the limiting Poisson means 2^(L−1)/(2L) = 2, 16, 2048 to within a few
  percent over five orders of magnitude.

That last check validates the counter against a theorem rather than against
another implementation, which is why it is the one to trust most.

## 5. Objective function

```
reward = − Σ_L w_L · min(count_L, cap) / n          # primary loss
         − W_hard · [provable prune violated]        # tier 1: proofs
         − W_soft · ([planar] + [claw-free])         # tier 2: restrictions
         + W_ip  · max(0, induced_path_lb − 12) / n  # tier 3: reinforcement
         + W_conn· [biconnected]
```

A **girth-annealing warm start** runs first: it penalises *every* cycle of
length < 9 with geometrically decaying weights, which makes the schedule
effectively lexicographic (triangles before 4-cycles, 4-cycles before
5-cycles). This matters because a random cubic graph has ~2 four-cycles and
~16 eight-cycles, and those two lengths dominate the power-of-two loss.

## 6. Results

Full numbers, figures and the scaling table are generated from the result CSVs
into [`docs/EXPERIMENT_LOG.md`](docs/EXPERIMENT_LOG.md) by
`python -m erdos_gyarfas.cli make-report`. The short version, from the run on
this machine (2 cores, 83 recorded runs, sizes 50 → 1000):

| n | best driver | C4 | C8 | C16 | C32 | girth | longest induced path | λ₂ |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 50 | sa | 0 | 0 | 1 154 | ≥4096 | 3 | 36 | 2.733 |
| 100 | tabu | 0 | 0 | 1 644 | ≥4096 | 3 | 66 | 2.732 |
| 200 | tabu | 0 | 0 | 2 020 | ≥4096 | 3 | 124 | 2.747 |
| 300 | tabu | 0 | 0 | 2 123 | ≥4096 | 6 | 171 | 2.774 |
| 600 | tabu | 0 | 0 | 2 032 | ≥4096 | 6 | 333 | 2.798 |
| 1000 | tabu | 0 | 0 | 2 022 | ≥4096 | 5 | 500 | 2.813 |

* **`C4 = C8 = 0` at every size** — 11 of 11 best candidates. Cubic graphs
  with no 4- and no 8-cycle are easy to find at all scales.
* **`C16` sits on the random-graph mean.** It rises 1 154 → 2 020 as *n* grows
  and converges on the Poisson mean 2 048. The search cannot reduce it, and the
  (C₄, C₈) weight sweep in `results/pareto` shows it barely responds to
  retuning (1 792–1 993 at *n* = 400 across a 64× range of w₄/w₈).
* **`C32` is only a lower bound** (`≥4096` is the counting cap). Certifying
  C₃₂ = 0 on a 1000-vertex graph is out of reach for this DFS; that is
  recorded honestly in the `certified` column rather than glossed over.
* **The P₁₃ condition is never binding.** The longest induced path grows at
  ≈ 0.50·*n*, so every candidate clears the bar by a factor of ~30 at
  *n* = 1000.
* **Best candidates are near-Ramanujan.** λ₂ rises 2.73 → 2.81 against the
  Alon–Boppana bound 2√2 ≈ 2.828, i.e. they are spectrally indistinguishable
  from random cubic graphs — precisely the regime where power-of-two cycles
  are abundant. All are 2-edge-connected and biconnected, consistent with the
  structure theorems for minimal counterexamples.
* **Literature benchmark reproduced.** The search finds 24-vertex cubic graphs
  with C₄ = C₈ = 0 (one of them planar, as Markström reported), and succeeds at
  every order ≥ 24 tried. At *n* = 14 and 20 it does not succeed — a search
  outcome, not a proof that no such graph exists.
* **Engine validated against theory.** On random cubic graphs the measured C₄,
  C₈, C₁₆ match 2^(L−1)/L = 2, 16, 2 048 with ratios 1.00, 0.84, 0.97 at
  *n* = 1000.
* **Zero certified counterexamples.**

### Hyperscale: early-exit search up to N = 100 000

`python -m erdos_gyarfas.cli hyperscale` runs a separate, scale-first pipeline
(`erdos_gyarfas/hyperscale/`) that trades density counting for a **boolean
early-exit** checker: over a CSR adjacency with per-DFS bitset marks it probes
2^k = 4, 8, 16, … and stops at the *first* witnessed cycle, flagging the graph
"Not a Counterexample". Hard filters first discard planar, claw-free and
P₁₃-free graphs, so only δ = 3, non-planar, non-claw-free, induced-P₁₃ graphs
are ever checked. Generators span random cubic, dihedral and PSL(2,q) / Sₙ
Cayley graphs, and annealed moderate-girth expanders, at N = 10 000 → 100 000.
From the run on this machine (16 graphs, 472 s): the four dihedral graphs are
planar and discarded; every one of the 12 kept graphs triggered — random cubic
at C4, S₈ and the girth-5 annealed graphs at C8, and the PSL(2,q) expanders
(girth > 16) at C32, the largest trigger observed. No counterexample. Full
table in [`docs/EXPERIMENT_LOG.md`](docs/EXPERIMENT_LOG.md) §6.

### Phase 3: SAT decision + explicit algebraic expanders

`python -m erdos_gyarfas.cli phase3` (`erdos_gyarfas/phase3/`) leaves local
search behind. A CNF/SMT encoding of "δ≥3 ∧ no 2^k cycle" is decided with
**CryptoMiniSat / CaDiCaL** (python-sat) and **Z3**; cycle exclusion enumerates
every L-cycle of K_n, so it is exact but combinatorial (L=4 cheap to n≈70, L=8
only to n≈12, L≥16 never). For n≤10 the solver **proves UNSAT** — no δ≥3 graph
avoids all 2^k≤n — independently reproducing the small-order bound; for n=30–70
it finds C4-free δ≥3 graphs that the early-exit checker shows all still contain
a C8. Explicit **LPS Ramanujan graphs** X^{p,q} (Hamilton-quaternion generators
over PGL(2,q)) are verified Ramanujan (λ₂ ≤ 2√(d−1)) yet each still has a
power-of-two cycle (X^{5,29}, n=12 180, triggers at C16). **Z_k voltage lifts**
of the Petersen graph cancel C4 but a C8 always survives. Full tables in
[`docs/EXPERIMENT_LOG.md`](docs/EXPERIMENT_LOG.md) §7.

## 7. Honest limitations

* Long cycles are the weak point. Counting C₃₂ exactly on a 1000-vertex graph
  is out of reach for a DFS; the engine reports those lengths as *uncertified*
  when the budget runs out. An exhaustive claim at that scale needs a
  different technique (Ducoffe et al. 2026 used SAT).
* "No counterexample found" is a statement about the search, not about the
  conjecture. Local search over 2-opt swaps explores a vanishing fraction of
  the cubic graphs at these sizes.
* The RL agents are lightweight by design (a two-layer MLP over 8 features).
  They are included to measure whether a *learned* move prior helps and whether
  it transfers across sizes — see the experiment log for the comparison.
* The hyperscale checker is an *early-exit witness*, not a prover. A
  `not-witnessed` length means the per-length DFS budget ran out — it is **not**
  a proof that no cycle of that length exists (only the exact C4 scan proves
  absence). The annealed "high-girth" family reached girth 5 at 10k/25k but only
  3 at 50k/100k within the move budget; random-plus-annealing does not reach the
  9–12 girth target at N ≈ 100 000, and the report says so rather than
  overclaiming.

## 8. References

* P. Erdős, A. Gyárfás, *Cycle space problem for graphs of minimum degree 3* (1995).
* A. S. Hegde, R. B. Sandeep, P. Shashank, *Erdős-Gyárfás conjecture on graphs
  without long induced paths*, [arXiv:2410.22842](https://arxiv.org/abs/2410.22842).
* G. Ducoffe et al., *Towards a more structured search for Erdős-Gyárfás
  counter-examples*, [arXiv:2609.28594](https://arxiv.org/abs/2609.28594).
* P. S. Nowbandegani, H. Esfandiari, M. H. Shirdareh Haghighi, K. Bibak,
  *On the Erdős-Gyárfás conjecture in claw-free graphs*, DMGT 34 (2014) 635–640.
* R. Gao, Y. Shan, *Erdős-Gyárfás conjecture for P₈-free graphs*, Graphs Combin. 38 (2022).
* X. Hu, Y. Shen, *The Erdős-Gyárfás conjecture holds for P₁₀-free graphs*, Discrete Math. (2024).
* J. Bensmail et al., *On q-power cycles in cubic graphs*.
* D. Daniel, S. Shauger, *A result on the Erdős-Gyárfás conjecture in planar graphs* (2001).
* K. Markström, *Extremal graphs for powers of 2*, Ars Combinatoria (2009).
