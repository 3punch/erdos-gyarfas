# Methodology

This document records the mathematics behind the search: what is *proved*,
what is merely *plausible*, and how each enters the objective function. Where
the task brief and the literature disagree, the literature wins and the
disagreement is stated explicitly.

---

## 1. The conjecture and its status

> **Erdős–Gyárfás (1995).** Every graph with minimum degree at least 3
> contains a simple cycle whose length is a power of two.

The conjecture is open. Erdős offered \$100 for a proof and \$50 for a
counterexample. What is actually known:

| Result | Source |
| --- | --- |
| Every **P₁₃-free** graph with δ ≥ 3 has a power-of-two cycle | Hegde, Sandeep & Shashank, [arXiv:2410.22842](https://arxiv.org/abs/2410.22842) (2025) |
| Every **P₁₀-free** graph with δ ≥ 3 has a C₄ or C₈ | Hu & Shen, *Discrete Math.* (2024) |
| Every **P₈-free** graph with δ ≥ 3 has a C₄ or C₈ | Gao & Shan, *Graphs Combin.* (2022) |
| Every graph of average degree ≥ *c* (some absolute *c*) has a power-of-two cycle | Liu & Montgomery (2023) |
| Holds for **3-connected cubic planar** graphs | (discharging + computer verification) |
| Holds for **planar claw-free** graphs | Daniel & Shauger (2001) |
| Holds for Cayley graphs on dihedral, semidihedral, generalised quaternion and order-*p³* groups | (2018) |
| Every **claw-free** graph with δ ≥ 3 has a cycle of length 2ᵏ **or 3·2ᵏ** | Nowbandegani et al., *DMGT* 34 (2014) |
| Verified for **all** graphs of order ≤ 40, bipartite ≤ 66, **cubic ≤ 48** | Ducoffe et al., [arXiv:2609.28594](https://arxiv.org/abs/2609.28594) (2026) |
| Cubic counterexample needs ≥ 30 vertices | Markström |
| Cubic **claw-free** counterexample needs ≥ 114 vertices | Nowbandegani & Esfandiari |
| Minimal counterexample: 2-edge-connected, biconnected or a 1-clique-sum of two biconnected graphs, *m* ≤ 2*n*−2, at least 2*n*/3+1 vertices of degree 3 | Ducoffe et al. (2026) |
| Arbitrarily large cubic graphs whose only 2-power cycles have length 4 only, or 8 only | Bensmail et al., *On q-power cycles in cubic graphs* |

---

## 2. Two premises in the task brief that the literature contradicts

The brief instructed: *"Hard Filter / Heavy Penalty: Discard Planar or
Claw-free graphs (proven that no counterexamples exist here)."* **Neither of
those classes is proven counterexample-free.**

* **Planar graphs.** What is proven is *3-connected cubic planar* and
  *planar claw-free*. Planarity alone is not enough: Markström's 24-vertex
  examples with no C₄ or C₈ include a planar graph (it does have 16-cycles, so
  it is not a counterexample — but planarity clearly does not exclude
  near-misses). We therefore keep the planarity filter as a *search-space
  restriction* with a heavy penalty, and label it as such everywhere in the
  code (`core/structure.py` documents this next to the function).

* **Claw-free graphs.** What is proven is *planar* claw-free, plus the weaker
  "2ᵏ or 3·2ᵏ" statement for all claw-free graphs, plus the ≥ 114-vertex lower
  bound for *cubic* claw-free counterexamples. So for *n* ≤ 113 the filter is
  a genuine proof of absence; above that it is a heuristic restriction. The
  code applies exactly this split (`CUBIC_CLAW_FREE_VERIFIED_ORDER = 113`).

There is a useful structural reason the claw-free filter costs almost nothing
here: **a cubic vertex is a claw centre exactly when its three neighbours are
pairwise non-adjacent**, so any *triangle-free* cubic graph has a claw at
every vertex. High-girth search and claw-freeness are therefore close to
mutually exclusive, and restricting to non-claw-free graphs does not remove
the high-girth region the search targets. (`tests/test_search.py` and
`tests/test_engine.py::test_triangle_free_cubic_graphs_are_never_claw_free`
assert this.)

## 3. The premise the brief got right — and it is stronger than stated

The brief asked to *"reward graphs that contain long induced paths
(specifically induced P₁₃ or longer)"*, presented as a soft guide. It is
actually a **hard necessary condition**.

Hegde, Sandeep & Shashank prove: *every P₁₃-free graph with minimum degree at
least 3 has a cycle of length a power of two*. Contrapositive:

> **Any counterexample must contain an induced path on 13 vertices.**

So a graph that is *provably* P₁₃-free is *provably* not a counterexample, and
can be discarded without loss of completeness. The same paper's Theorem 0.2
(*every P₁₂-free graph with δ ≥ 3 has a C₄ or C₈*) gives a second, weaker
filter that applies even before the full power-of-two sweep.

`core/structure.has_induced_path` therefore returns a three-valued answer:

| return | meaning | use |
| --- | --- | --- |
| `(True, "proved")` | an induced P₁₃ was exhibited | passes the necessary condition |
| `(False, "refuted")` | the search completed exhaustively and found none | **provable prune** |
| `(False, "unknown")` | budget exhausted | soft penalty only |

This is the one filter in the framework that is a true proof of absence, and
the asymmetry matters: *finding* a long induced path proves presence, while
*proving* absence requires exhausting the search.

## 4. Search space: why cubic graphs

Every regular minimal counterexample is cubic, and at least 2*n*/3+1 vertices
of any minimal counterexample have degree exactly 3 (Ducoffe et al. 2026,
building on Markström's observation). Searching cubic graphs is therefore both
the natural restriction and the one that makes the neighbourhood cheap:

* **Representation.** A cubic graph is a perfect matching on 3*n* half-edges.
  Vertex *v* owns stubs 3*v*, 3*v*+1, 3*v*+2.
* **Move.** A 2-opt swap exchanges the partners of two stubs. It preserves
  δ = 3 *by construction* — no degree repair, no rejection of "illegal degree"
  states. Validity (no loop, no double edge) is a constant-time check.
* **Cost.** O(1) per move. Measured: ≈ 40 000 accepted 2-opt moves/second at
  *n* = 1000 including exact cycle-count bookkeeping.

## 5. The cycle-counting engine

Counting simple cycles of a prescribed length is #P-hard in general, so the
engine is built around *budgeted exactness*: every count is returned as
`(count, certified)`, and `certified == False` means "this is a lower bound;
the budget or the cap ran out". A counterexample claim requires every length
to be certified zero.

Pruning, in the order it is applied:

1. **Canonicalisation.** Each cycle is enumerated once, from its minimum
   vertex *s*, taking the smaller of *s*'s two cycle-neighbours first. Vertices
   below *s* are never entered.
2. **BFS distance layer.** From *s*, a branch is cut when `dist(w) > remaining`.
3. **Parity.** Inside a bipartite component, `dist(w) ≡ remaining (mod 2)` must
   hold.
4. **Budget and cap.** Hard limits, flagged in the result.

Two performance decisions matter at scale:

* **One BFS shared across all lengths.** `count_pow2_cycles_kernel` puts the
  start vertex in the outer loop and the lengths in the inner loop, so the
  O(*nm*) BFS cost is paid once rather than once per length.
* **Exact incremental deltas.** A 2-opt swap replaces edges (*a*,*b*),(*c*,*d*)
  with (*a*,*d*),(*c*,*b*). A cycle is created or destroyed only if it uses one
  of those four edges, so
  Δ_L = |{L-cycles using (*a*,*d*) or (*c*,*b*)}|_new − |{L-cycles using (*a*,*b*) or (*c*,*d*)}|_old,
  computed exactly. The engine tracks *L* ∈ {4, 8} this way on every move and
  recounts {4, 8, 16} at checkpoints.

### 5.1 How the engine was validated

| Check | Reference | Result |
| --- | --- | --- |
| Cycle counts, all lengths 3–10, eight random cubic graphs at *n* = 18 | independent unpruned DFS enumerator | exact match |
| The two reference enumerators | unpruned DFS vs. per-subset Hamiltonian count | agree |
| C₄ count | trace identity `tr(A⁴) = Σdᵥ² + Σdᵥ(dᵥ−1) + 8·#C₄` | exact match |
| Petersen spectrum | textbook values (12 C₅, 10 C₆, 15 C₈, 20 C₉, no C₇, non-Hamiltonian) | exact match |
| Incremental deltas | full recount after every move, 150 swaps | exact match |
| Tracked counts after a whole search | independent recount of the returned graph | exact match, every driver |
| Longest induced path | independent recursive reference at *n* = 16 | exact match |
| Girth, bridges, planarity, claw centres | `networkx` | exact match |
| **Statistical**: C₄, C₈, C₁₆ on random cubic graphs | limiting Poisson means 2^(L−1)/L = 2, 16, 2048 | see `results/validation` |

The statistical check is the strongest one, because it validates the counter
against a theorem rather than against another implementation. See
`docs/EXPERIMENT_LOG.md` for the measured ratios.

## 6. Objective function

```
reward = − Σ_L w_L · min(count_L, cap) / n          # primary loss
         − W_hard · [provable prune violated]        # tier 1, proofs
         − W_soft · ([planar] + [claw-free])         # tier 2, restrictions
         + W_ip  · max(0, induced_path_lb − 12) / n  # tier 3, reinforcement
         + W_conn· [biconnected]
```

with `w_L` a geometric profile over the powers of two up to *n*. Three
curriculum variants are provided (`default_config_for(n, focus=…)`):
`small` (emphasise C₄/C₈), `large`, and `balanced`.

A graph is reported as a **jackpot** only when `report.total == 0`,
`report.all_certified`, and no provable prune fired.

## 7. Why the search cannot simply win

For a random 3-regular graph the number of *L*-cycles converges to a Poisson
law with mean (d−1)^L/(2L), which for d = 3 is 2^L/(2L) = 2^(L−1)/L. That is

| L | 4 | 8 | 16 | 32 | 64 |
| --- | --- | --- | --- | --- | --- |
| E[#C_L] | 2 | 16 | 2 048 | 6.7 × 10⁷ | 3.4 × 10¹⁷ |

Local search readily achieves C₄ = C₈ = 0 (that is exactly "girth ≥ 9"), and
the measurements in this repository reproduce it at every size up to *n* =
1000. But C₁₆ is already in the thousands for *any* cubic graph of that size,
and C₃₂ is astronomically larger. The conjecture asks a graph to have **none**
of them simultaneously. The measured scaling is the honest result of this
project, and it is reported as such rather than dressed up as a near-miss.
