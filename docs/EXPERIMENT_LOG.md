# Experiment log

_Generated from `results/` on 2026-10-03 17:27 UTC by `python -m erdos_gyarfas.cli make-report`. Every number below is read from the CSVs the runner wrote; none is transcribed by hand._

See [METHODOLOGY.md](METHODOLOGY.md) for the mathematics and the literature each filter is grounded in.

## 0. Headline

* **0 certified counterexamples** found across 83 recorded runs.
* Sizes covered: 50, 76, 100, 150, 200, 300, 400, 600, 800, 1000.
* `C4 = C8 = 0` reached at 11 of 11 (size, seed) best-candidates.
* `C16` across those candidates: min 1,154, median 2,020, max 2,123 (random-cubic mean 2048).
* Longest induced path grows ≈ 0.50·n (least-squares over the best candidates); the P₁₃ necessary condition is never binding.

## 1. Method, in one paragraph

Cubic graphs are stored as perfect matchings on 3n half-edges; a move is a 2-opt exchange of two stub partners, which preserves δ = 3 by construction and costs O(1). A search begins with a girth-annealing warm start (penalising every cycle of length < 9 with geometrically decaying weights) and then minimises the length-weighted count of power-of-two cycles. Counts for C₄ and C₈ are maintained *exactly* by incremental deltas; C₁₆ is recounted at checkpoints; longer lengths are counted only in a final budgeted certification sweep that reports, per length, whether the zero is certified or merely unexhausted. Filters from the literature (P₁₃-free, claw-free, planar, 2-edge-connected) are applied on top, with proofs and heuristics kept strictly separate.

## 2.1 Stage `warmup`

| n | best driver | loss | C4 | C8 | C16 | C32 | girth | longest induced path | 2-edge-conn | planar | claw-free | λ₂ | certified | counterexample |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|---:|---|---|
| 50 | sa | -0.00000 | 0 | 0 | 1154 | ≥4096 | 3 | 36 | True | False | False | 2.733 | False | False |
| 76 | tabu | 0.00000 | 0 | 0 | 1576 | ≥4096 | 3 | 53 | True | False | False | 2.695 | False | False |
| 100 | sa | -0.00000 | 0 | 0 | 1691 | ≥4096 | 3 | 65 | True | False | False | 2.710 | False | False |

**Driver comparison on identical instances**

| driver | runs | median best loss | median C4 | median C8 | median moves/s | median search s |
|---|---:|---:|---:|---:|---:|---:|
| sa | 6 | 0.00465 | 0 | 0 | 26,869 | 3.7 |
| tabu | 6 | 0.00000 | 0 | 0 | 744 | 2.0 |
| dqn | 6 | 0.04964 | 0 | 3 | 170 | 5.9 |
| actor_critic | 6 | 0.05812 | 1 | 5 | 310 | 3.2 |

_`moves/s` is not comparable across drivers: for SA it counts accepted 2-opt swaps inside a compiled loop, while for tabu and the RL agents it counts *steps*, each of which scores a whole slate (24 and 16 candidate moves respectively). The loss column is the comparable quantity._

## 2.2 Stage `medium`

| n | best driver | loss | C4 | C8 | C16 | C32 | girth | longest induced path | 2-edge-conn | planar | claw-free | λ₂ | certified | counterexample |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|---:|---|---|
| 100 | tabu | 0.00000 | 0 | 0 | 1644 | ≥4096 | 3 | 66 | True | False | False | 2.732 | False | False |
| 150 | tabu | 0.00000 | 0 | 0 | 1957 | ≥4096 | 5 | 91 | True | False | False | 2.742 | False | False |
| 200 | tabu | 0.00000 | 0 | 0 | 2020 | ≥4096 | 3 | 124 | True | False | False | 2.747 | False | False |
| 300 | tabu | 0.00000 | 0 | 0 | 2123 | ≥4096 | 6 | 171 | True | False | False | 2.774 | False | False |

**Driver comparison on identical instances**

| driver | runs | median best loss | median C4 | median C8 | median moves/s | median search s |
|---|---:|---:|---:|---:|---:|---:|
| sa | 8 | 0.00780 | 0 | 2 | 25,591 | 9.8 |
| tabu | 8 | 0.00000 | 0 | 0 | 746 | 4.0 |
| dqn | 8 | 0.00000 | 0 | 0 | 154 | 13.0 |
| actor_critic | 8 | 0.02337 | 0 | 4 | 298 | 6.7 |

_`moves/s` is not comparable across drivers: for SA it counts accepted 2-opt swaps inside a compiled loop, while for tabu and the RL agents it counts *steps*, each of which scores a whole slate (24 and 16 candidate moves respectively). The loss column is the comparable quantity._

## 2.3 Stage `large`

| n | best driver | loss | C4 | C8 | C16 | C32 | girth | longest induced path | 2-edge-conn | planar | claw-free | λ₂ | certified | counterexample |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|---:|---|---|
| 400 | tabu | 0.00000 | 0 | 0 | 2112 | ≥4096 | 6 | 235 | True | False | False | 2.799 | False | False |
| 600 | tabu | 0.00000 | 0 | 0 | 2032 | ≥4096 | 6 | 333 | True | False | False | 2.798 | False | False |
| 800 | tabu | 0.00000 | 0 | 0 | 2091 | ≥4096 | 6 | 425 | True | False | False | 2.808 | False | False |
| 1000 | tabu | 0.00000 | 0 | 0 | 2022 | ≥4096 | 5 | 500 | True | False | False | 2.813 | False | False |

**Driver comparison on identical instances**

| driver | runs | median best loss | median C4 | median C8 | median moves/s | median search s |
|---|---:|---:|---:|---:|---:|---:|
| sa | 8 | 0.00253 | 0 | 2 | 25,572 | 15.7 |
| tabu | 8 | 0.00000 | 0 | 0 | 675 | 3.0 |

_`moves/s` is not comparable across drivers: for SA it counts accepted 2-opt swaps inside a compiled loop, while for tabu and the RL agents it counts *steps*, each of which scores a whole slate (24 and 16 candidate moves respectively). The loss column is the comparable quantity._

## 2.4 Scaling

| n | driver | proposed moves | search s | moves/s | total s (incl. warm start + certification sweep) |
|---:|---|---:|---:|---:|---:|
| 50 | sa | 100,000 | 3.9 | 25,495 | 26.0 |
| 50 | sa | 100,000 | 3.6 | 27,519 | 8.2 |
| 50 | tabu | 1,500 | 4.4 | 341 | 7.5 |
| 50 | tabu | 1,500 | 2.0 | 760 | 6.4 |
| 76 | sa | 100,000 | 3.7 | 26,894 | 7.3 |
| 76 | sa | 100,000 | 3.4 | 29,114 | 7.5 |
| 76 | tabu | 1,500 | 2.0 | 767 | 5.6 |
| 76 | tabu | 1,500 | 2.1 | 728 | 6.0 |
| 100 | sa | 100,000 | 3.8 | 26,608 | 14.1 |
| 100 | sa | 100,000 | 3.7 | 26,845 | 14.9 |
| 100 | tabu | 1,500 | 2.1 | 723 | 11.8 |
| 100 | tabu | 1,500 | 1.9 | 803 | 12.8 |
| 100 | sa | 250,000 | 9.9 | 25,362 | 41.5 |
| 100 | sa | 250,000 | 9.7 | 25,708 | 43.2 |
| 100 | tabu | 3,000 | 8.1 | 372 | 24.2 |
| 100 | tabu | 3,000 | 7.6 | 393 | 25.3 |
| 150 | sa | 250,000 | 9.9 | 25,365 | 27.1 |
| 150 | sa | 250,000 | 10.5 | 23,856 | 41.2 |
| 150 | tabu | 3,000 | 4.0 | 747 | 19.4 |
| 150 | tabu | 3,000 | 4.0 | 757 | 34.3 |
| 200 | sa | 250,000 | 9.8 | 25,474 | 18.6 |
| 200 | sa | 250,000 | 9.7 | 25,891 | 31.0 |
| 200 | tabu | 3,000 | 4.5 | 670 | 13.3 |
| 200 | tabu | 3,000 | 4.0 | 748 | 27.0 |
| 300 | sa | 250,000 | 9.3 | 26,880 | 26.4 |
| 300 | sa | 250,000 | 7.8 | 32,128 | 21.9 |
| 300 | tabu | 3,000 | 4.0 | 745 | 18.3 |
| 300 | tabu | 3,000 | 2.7 | 1,096 | 12.6 |
| 400 | sa | 400,000 | 15.7 | 25,561 | 56.9 |
| 400 | sa | 400,000 | 15.3 | 26,215 | 42.6 |
| 400 | tabu | 2,000 | 6.4 | 313 | 33.6 |
| 400 | tabu | 2,000 | 5.5 | 364 | 18.0 |
| 600 | sa | 400,000 | 16.5 | 24,213 | 35.9 |
| 600 | sa | 400,000 | 15.9 | 25,213 | 34.6 |
| 600 | tabu | 2,000 | 2.4 | 829 | 23.4 |
| 600 | tabu | 2,000 | 3.2 | 635 | 21.8 |
| 800 | sa | 400,000 | 15.7 | 25,583 | 35.5 |
| 800 | sa | 400,000 | 15.8 | 25,361 | 60.8 |
| 800 | tabu | 2,000 | 3.0 | 658 | 21.3 |
| 800 | tabu | 2,000 | 2.7 | 743 | 44.4 |
| 1000 | sa | 400,000 | 15.2 | 26,347 | 31.0 |
| 1000 | sa | 400,000 | 10.6 | 37,965 | 47.9 |
| 1000 | tabu | 2,000 | 2.9 | 692 | 15.4 |
| 1000 | tabu | 2,000 | 1.8 | 1,113 | 30.2 |

The SA inner loop is compiled and runs flat at ~25k–40k swaps/second from n = 50 to n = 1000; the per-move cost is dominated by the exact C4/C8 deltas, which do not grow with n. The certification sweep is what gets expensive, because lengths >= 32 need a budgeted DFS.

## 3. Engine validation against Poisson theory

For a random 3-regular graph the number of L-cycles converges to a Poisson law with mean (d−1)^L/(2L), i.e. 2^(L−1)/L for d = 3. Reproducing those means is an independent check on the counter that does not depend on any second implementation of cycle enumeration.

| n | L | measured mean | theory 2^(L−1)/L | ratio | certified trials |
|---:|---:|---:|---:|---:|---:|
| 50 | 4 | 2.38 | 2.00 | 1.188 | 8/8 |
| 50 | 8 | 15.88 | 16.00 | 0.992 | 8/8 |
| 50 | 16 | 1,119.25 | 2,048.00 | 0.547 | 8/8 |
| 50 | 32 | 194,811.38 | 67,108,864.00 | 0.003 | 0/8 ⚠ lower bound |
| 100 | 4 | 2.12 | 2.00 | 1.062 | 8/8 |
| 100 | 8 | 15.88 | 16.00 | 0.992 | 8/8 |
| 100 | 16 | 1,532.38 | 2,048.00 | 0.748 | 8/8 |
| 100 | 32 | 125,000.00 | 67,108,864.00 | 0.002 | 0/8 ⚠ lower bound |
| 100 | 64 | 96,396.38 | 144,115,188,075,855,872.00 | 0.000 | 0/8 ⚠ lower bound |
| 200 | 4 | 2.12 | 2.00 | 1.062 | 8/8 |
| 200 | 8 | 15.75 | 16.00 | 0.984 | 8/8 |
| 200 | 16 | 1,825.00 | 2,048.00 | 0.891 | 8/8 |
| 200 | 32 | 150,000.00 | 67,108,864.00 | 0.002 | 0/8 ⚠ lower bound |
| 200 | 64 | 125,000.00 | 144,115,188,075,855,872.00 | 0.000 | 0/8 ⚠ lower bound |
| 200 | 128 | 85,484.00 | 1,329,227,995,784,915,872,903,807,060,280,344,576.00 | 0.000 | 0/8 ⚠ lower bound |
| 400 | 4 | 3.00 | 2.00 | 1.500 | 8/8 |
| 400 | 8 | 15.62 | 16.00 | 0.977 | 8/8 |
| 400 | 16 | 1,910.38 | 2,048.00 | 0.933 | 8/8 |
| 400 | 32 | 125,000.00 | 67,108,864.00 | 0.002 | 0/8 ⚠ lower bound |
| 400 | 64 | 125,000.00 | 144,115,188,075,855,872.00 | 0.000 | 0/8 ⚠ lower bound |
| 400 | 128 | 125,000.00 | 1,329,227,995,784,915,872,903,807,060,280,344,576.00 | 0.000 | 0/8 ⚠ lower bound |
| 400 | 256 | 51,994.00 | 226,156,424,291,633,194,186,662,080,095,093,570,025,917,938,800,079,226,639,565,593,765,455,331,328.00 | 0.000 | 0/8 ⚠ lower bound |
| 1000 | 4 | 2.00 | 2.00 | 1.000 | 8/8 |
| 1000 | 8 | 13.38 | 16.00 | 0.836 | 8/8 |
| 1000 | 16 | 1,985.62 | 2,048.00 | 0.970 | 8/8 |
| 1000 | 32 | 150,000.00 | 67,108,864.00 | 0.002 | 0/8 ⚠ lower bound |
| 1000 | 64 | 150,000.00 | 144,115,188,075,855,872.00 | 0.000 | 0/8 ⚠ lower bound |
| 1000 | 128 | 150,000.00 | 1,329,227,995,784,915,872,903,807,060,280,344,576.00 | 0.000 | 0/8 ⚠ lower bound |
| 1000 | 256 | 125,000.00 | 226,156,424,291,633,194,186,662,080,095,093,570,025,917,938,800,079,226,639,565,593,765,455,331,328.00 | 0.000 | 0/8 ⚠ lower bound |
| 1000 | 512 | 94,323.75 | 13,093,562,431,584,567,480,052,758,787,310,396,608,866,568,184,172,259,157,933,165,472,384,535,185,618,698,219,533,080,369,303,616,628,603,546,736,510,240,284,036,869,026,183,541,572,213,314,110,357,504.00 | 0.000 | 0/8 ⚠ lower bound |

## 4. Literature benchmark: cubic graphs with no C₄ and no C₈

Markström's search produced 24-vertex cubic graphs whose only power-of-two cycles are 16-cycles (one of them planar). Reproducing `C4 = C8 = 0` is the smallest non-trivial target the engine can be calibrated against.

| n | target | success | achieved | girth | planar | seconds |
|---:|---|---|---|---:|---|---:|
| 14 | cubic graph with C4 = C8 = 0 | False | {4: 0, 8: 7} | 3 | True | 29.2 |
| 20 | cubic graph with C4 = C8 = 0 | False | {4: 0, 8: 2, 16: 144} | 3 | False | 25.7 |
| 24 | cubic graph with C4 = C8 = 0 | True | {4: 0, 8: 0, 16: 228} | 3 | True | 6.8 |
| 30 | cubic graph with C4 = C8 = 0 | True | {4: 0, 8: 0, 16: 440} | 3 | False | 7.6 |
| 40 | cubic graph with C4 = C8 = 0 | True | {4: 0, 8: 0, 16: 681, 32: 55974} | 3 | False | 15.4 |

## 5. The (C₄, C₈) frontier

Retuning the ratio w₄/w₈ traces the achievable trade-off. Driving C₄ to zero and driving C₈ to zero pull against each other, and C₁₆ barely responds to either — which is the clearest single piece of evidence that the conjecture is out of reach for this kind of search.

| n | w4/w8 | C4 | C8 | C16 | girth |
|---:|---:|---:|---:|---:|---:|
| 100 | 16.00 | 0 | 0 | 1792 | 5 |
| 100 | 8.00 | 0 | 0 | 1794 | 5 |
| 100 | 4.00 | 0 | 0 | 1287 | 3 |
| 100 | 2.00 | 0 | 1 | 1716 | 3 |
| 100 | 1.00 | 0 | 1 | 1495 | 3 |
| 100 | 0.50 | 0 | 0 | 1767 | 5 |
| 100 | 0.25 | 3 | 0 | 1642 | 3 |
| 200 | 16.00 | 0 | 1 | 1915 | 3 |
| 200 | 8.00 | 0 | 0 | 1900 | 3 |
| 200 | 4.00 | 0 | 2 | 1765 | 3 |
| 200 | 2.00 | 0 | 2 | 1728 | 3 |
| 200 | 1.00 | 0 | 1 | 1844 | 3 |
| 200 | 0.50 | 0 | 2 | 1853 | 3 |
| 200 | 0.25 | 1 | 1 | 2013 | 3 |
| 400 | 16.00 | 0 | 3 | 1907 | 3 |
| 400 | 8.00 | 0 | 3 | 1988 | 5 |
| 400 | 4.00 | 0 | 2 | 1927 | 3 |
| 400 | 2.00 | 0 | 2 | 1919 | 3 |
| 400 | 1.00 | 0 | 3 | 1993 | 5 |
| 400 | 0.50 | 0 | 1 | 1990 | 5 |
| 400 | 0.25 | 3 | 1 | 1923 | 3 |

## 6. Hyperscale early-exit search (N up to 100 000)

Boolean early-exit checker over CSR + bitset marks: nothing is counted, each length probe stops at the first witnessed cycle. `trigger` is the first power-of-two length at which a cycle was witnessed (proof the graph is **not** a counterexample); `largest length probed` is the maximum 2^k the checker attempted before stopping. A `not-witnessed` entry at some length is *not* a proof of absence at that length.

* kept candidates: 12 of 16; early-exit triggered for 12 of 12.
* highest trigger length observed: C32.

| family | n | kept? | why discarded | has P13 | claw-free | planar | achieved girth | early-exit trigger | largest length probed | gen s | check s | verdict |
|---|---:|---|---|---|---|---|---:|---|---:|---:|---:|---|
| cayley_dihedral | 10000 | no | planar | True | False | True |  | - | - | 0.0 | 0.00 | discarded |
| cayley_dihedral | 25000 | no | planar | True | False | True |  | - | - | 0.0 | 0.00 | discarded |
| cayley_dihedral | 50000 | no | planar | True | False | True |  | - | - | 0.0 | 0.00 | discarded |
| cayley_dihedral | 100000 | no | planar | True | False | True |  | - | - | 0.0 | 0.00 | discarded |
| cayley_perm | 40320 | yes | - | True | False | False |  | C8 | 8 | 0.4 | 0.01 | not-a-counterexample |
| cayley_psl2 | 12180 | yes | - | True | False | False |  | C32 | 32 | 0.1 | 6.79 | not-a-counterexample |
| cayley_psl2 | 34440 | yes | - | True | False | False |  | C32 | 32 | 0.3 | 7.13 | not-a-counterexample |
| cayley_psl2 | 74412 | yes | - | True | False | False |  | C32 | 32 | 0.4 | 12.33 | not-a-counterexample |
| high_girth | 10000 | yes | - | True | False | False | 5 | C8 | 8 | 21.2 | 1.00 | not-a-counterexample |
| high_girth | 25000 | yes | - | True | False | False | 5 | C8 | 8 | 43.5 | 1.04 | not-a-counterexample |
| high_girth | 50000 | yes | - | True | False | False | 3 | C4 | 4 | 123.5 | 0.00 | not-a-counterexample |
| high_girth | 100000 | yes | - | True | False | False | 3 | C4 | 4 | 406.1 | 0.00 | not-a-counterexample |
| random_cubic | 10000 | yes | - | True | False | False |  | C4 | 4 | 0.0 | 0.16 | not-a-counterexample |
| random_cubic | 25000 | yes | - | True | False | False |  | C4 | 4 | 0.2 | 0.00 | not-a-counterexample |
| random_cubic | 50000 | yes | - | True | False | False |  | C4 | 4 | 0.1 | 0.00 | not-a-counterexample |
| random_cubic | 100000 | yes | - | True | False | False |  | C4 | 4 | 1.8 | 0.01 | not-a-counterexample |

**Reading the table.** The dihedral Cayley graphs $D_m$ with $S=\{r,r^{-1},s\}$ are Möbius/prism ladders and therefore *planar*; the filter discards all of them, which is exactly the brief's requirement (3-connected cubic planar graphs are a proven case, so they cannot be counterexamples). The `high_girth` family is a random cubic graph annealed toward large girth; the `achieved girth` column reports the girth *actually reached* — 5 at n=10k/25k but only 3 at n=50k/100k within the move budget. Random-plus-annealing does **not** reach the 9–12 girth target at N≈100k, so these are moderate-girth graphs and are reported as such rather than overclaimed. The `cayley_psl2` expanders have girth > 16: C8 and C16 come back `not-witnessed` (the per-length DFS budget was exhausted, which is *not* a proof of absence) and the first power-of-two cycle is witnessed at C32 — the largest trigger length in the study. Every kept graph triggered, so none is a counterexample, consistent with the conjecture; the checker never had to prove absence at scale.

## 7. Figures

* `results/warmup/figures/induced_path_vs_n.png`
* `results/warmup/figures/pow2_vs_n.png`
* `results/warmup/figures/scaling_timing.png`
* `results/warmup/figures/search_curves.png`
* `results/warmup/figures/spectral.png`
* `results/warmup/figures/theory_vs_measured.png`
* `results/medium/figures/induced_path_vs_n.png`
* `results/medium/figures/pow2_vs_n.png`
* `results/medium/figures/scaling_timing.png`
* `results/medium/figures/search_curves.png`
* `results/medium/figures/spectral.png`
* `results/medium/figures/theory_vs_measured.png`
* `results/large/figures/induced_path_vs_n.png`
* `results/large/figures/pow2_vs_n.png`
* `results/large/figures/scaling_timing.png`
* `results/large/figures/search_curves.png`
* `results/large/figures/spectral.png`
* `results/large/figures/theory_vs_measured.png`

## 8. Reproduce

```bash
pytest -q
python -m erdos_gyarfas.cli all --workers 2
python -m erdos_gyarfas.cli make-report
```
