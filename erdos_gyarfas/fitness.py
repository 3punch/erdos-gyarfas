"""Guided fitness / reward function for the Erdos-Gyarfas search.

The objective has three tiers, deliberately ordered by how much is *known*:

1. **Provable prunes (hard filter).**  These are conditions a counterexample
   is *proved* not to satisfy, so a graph failing them can be discarded
   without any loss of completeness:

   * no induced ``P_13``  -- every P13-free graph with delta >= 3 has a
     power-of-two cycle (Hegde, Sandeep & Shashank 2025);
   * not 2-edge-connected -- every minimal counterexample is 2-edge-connected
     (Ducoffe et al. 2026);
   * ``n < 49`` for cubic graphs -- exhaustively verified (Ducoffe et al. 2026).

2. **Literature-motivated restrictions (heavy penalty).**  These are *not*
   proofs of absence, but they concentrate the search where a counterexample
   would have to live:

   * planar  (proved for 3-connected cubic planar and planar claw-free);
   * claw-free (proved for planar claw-free; cubic claw-free needs >= 114
     vertices, and claw-free delta >= 3 always has a 2^k or 3*2^k cycle).

3. **Primary loss and positive reinforcement.**

   * loss: the number of power-of-two cycles, length-weighted;
   * guide: a bonus for long induced paths, which a counterexample is forced
     to contain anyway (see tier 1).

The evaluator separates a *fast* path (exact incremental deltas for the short
lengths that dominate the loss) from a *checkpoint* path (full recount plus
the structural filters), so the inner loop of the search stays cheap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from .core import kernels
from .core.cycles import (
    DEFAULT_BUDGET,
    DEFAULT_CAP,
    PowerOfTwoReport,
    count_power_of_two_cycles,
    power_of_two_lengths,
)
from .core.graph import CubicGraph
from .core.structure import (
    REQUIRED_INDUCED_PATH_VERTICES,
    claw_center_count,
    has_induced_path,
    is_biconnected,
    is_planar,
    is_two_edge_connected,
    longest_induced_path,
)

#: Below this order a cubic counterexample is excluded by exhaustive search.
EXHAUSTIVELY_VERIFIED_CUBIC_ORDER = 48

#: Below this order a cubic claw-free counterexample is excluded.
CUBIC_CLAW_FREE_VERIFIED_ORDER = 113


@dataclass
class FitnessConfig:
    """Knobs for the reward function."""

    #: lengths whose counts are maintained exactly after every accepted move
    track_lengths: tuple[int, ...] = (4, 8)
    #: per-length weights; missing lengths get ``default_weight``
    length_weights: dict[int, float] = field(default_factory=dict)
    default_weight: float = 1.0
    count_cap: int = 4096
    budget: int = DEFAULT_BUDGET
    w_planar: float = 50.0
    w_claw_free: float = 50.0
    w_induced_path: float = 30.0
    w_not_two_edge_connected: float = 100.0
    w_biconnected_bonus: float = 2.0
    target_induced_path: int = REQUIRED_INDUCED_PATH_VERTICES
    induced_path_budget: int = 1 << 20
    #: reward per extra induced-path vertex beyond the required threshold
    induced_path_scale: float = 1.0
    enforce_p13: bool = True
    enforce_planar_free: bool = True
    enforce_claw_free: bool = True

    def weight(self, length: int) -> float:
        return self.length_weights.get(length, self.default_weight)

    def lengths_for(self, n: int) -> list[int]:
        return power_of_two_lengths(n)


@dataclass
class Evaluation:
    """Full evaluation of one candidate graph."""

    n: int
    report: PowerOfTwoReport
    loss: float
    penalties: dict[str, float]
    bonus: float
    reward: float
    provably_not_counterexample: bool
    provable_reasons: list[str]
    restricted_reasons: list[str]
    planar: bool
    claw_free: bool
    claw_centers: int
    two_edge_connected: bool
    biconnected: bool
    longest_induced_path_lb: int
    has_induced_p13: bool
    induced_p13_status: str

    @property
    def is_jackpot(self) -> bool:
        """delta >= 3, no power-of-two cycle, and nothing disproves it."""
        return (
            self.report.total == 0
            and self.report.all_certified
            and not self.provably_not_counterexample
        )

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            "pow2_counts": dict(zip(self.report.lengths, self.report.counts)),
            "pow2_certified": dict(zip(self.report.lengths, self.report.certified)),
            "pow2_total": self.report.total,
            "pow2_distinct_lengths": self.report.distinct_lengths,
            "smallest_pow2_present": self.report.smallest_present,
            "loss": self.loss,
            "penalties": self.penalties,
            "bonus": self.bonus,
            "reward": self.reward,
            "planar": self.planar,
            "claw_free": self.claw_free,
            "claw_centers": self.claw_centers,
            "two_edge_connected": self.two_edge_connected,
            "biconnected": self.biconnected,
            "longest_induced_path_lb": self.longest_induced_path_lb,
            "has_induced_p13": self.has_induced_p13,
            "induced_p13_status": self.induced_p13_status,
            "provably_not_counterexample": self.provably_not_counterexample,
            "provable_reasons": self.provable_reasons,
            "restricted_reasons": self.restricted_reasons,
            "is_jackpot": self.is_jackpot,
            "certified": self.report.all_certified,
        }


class FitnessEvaluator:
    """Evaluates cubic graphs against the Erdos-Gyarfas objective.

    Two entry points:

    :meth:`fast_counts`
        exact power-of-two cycle counts for ``config.track_lengths`` only;
        cheap enough to call on every accepted move.
    :meth:`evaluate`
        the full reward, including every length up to ``n`` and all
        structural filters.
    """

    def __init__(self, config: FitnessConfig | None = None):
        self.config = config or FitnessConfig()
        self._scratch: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}

    # ------------------------------------------------------------- scratch
    def scratch(self, n: int):
        if n not in self._scratch:
            self._scratch[n] = (
                np.empty(n, dtype=np.int32),          # dist
                np.zeros(n, dtype=np.int8),           # onpath
                np.zeros(5, dtype=np.int64),          # st
                np.empty(n, dtype=np.int32),          # queue
            )
        return self._scratch[n]

    # ----------------------------------------------------------- fast path
    def fast_counts(self, graph: CubicGraph) -> dict[int, int]:
        """Exact counts for the tracked (short) lengths."""
        lengths = [L for L in self.config.track_lengths if 3 <= L <= graph.n]
        if not lengths:
            return {}
        rep = count_power_of_two_cycles(
            graph, lengths=lengths, cap=self.config.count_cap,
            budget=self.config.budget,
        )
        return dict(zip(rep.lengths, rep.counts))

    def fast_loss(self, counts: Mapping[int, int], n: int) -> float:
        cfg = self.config
        total = 0.0
        for L, c in counts.items():
            total += cfg.weight(L) * min(c, cfg.count_cap)
        return total / max(n, 1)

    def delta_counts(self, graph: CubicGraph, swap: tuple[int, int],
                     lengths: Sequence[int] | None = None) -> np.ndarray:
        """Exact change in the tracked counts caused by ``swap``.

        ``graph`` must already be in the post-swap state; the pre-swap
        adjacency is reconstructed internally by applying the (involutive)
        swap to a copy.
        """
        cfg = self.config
        lengths = list(lengths) if lengths else [
            L for L in cfg.track_lengths if 3 <= L <= graph.n
        ]
        s1, s2 = int(swap[0]), int(swap[1])
        a, b = s1 // 3, int(graph.nb[s1])
        c, d = s2 // 3, int(graph.nb[s2])
        g_old = graph.copy()
        g_old.apply_swap(s1, s2)
        dist, onpath, st, _queue = self.scratch(graph.n)
        out = np.zeros(len(lengths), dtype=np.int64)
        kernels.delta_counts_kernel(
            graph.nb, g_old.nb, graph.n, a, b, c, d,
            np.array(lengths, dtype=np.int64),
            np.full(len(lengths), cfg.count_cap, dtype=np.int64),
            np.full(len(lengths), cfg.budget, dtype=np.int64),
            dist, onpath, st, out,
        )
        return out

    # ---------------------------------------------------------- short girth
    def shortest_new_cycle(self, graph: CubicGraph, swap: tuple[int, int],
                           cap_len: int = 32) -> int:
        """Length of the shortest cycle created by ``swap`` (post-swap state)."""
        s1, s2 = int(swap[0]), int(swap[1])
        a, d = s1 // 3, int(graph.nb[s1])
        c, b = s2 // 3, int(graph.nb[s2])
        x = kernels.shortest_cycle_through_edge(graph.nb, graph.n, a, d, cap_len)
        y = kernels.shortest_cycle_through_edge(graph.nb, graph.n, c, b, cap_len)
        return int(min(x, y))

    # ---------------------------------------------------------- full path
    def evaluate(self, graph: CubicGraph, with_structure: bool = True) -> Evaluation:
        cfg = self.config
        n = graph.n
        report = count_power_of_two_cycles(
            graph, cap=cfg.count_cap, budget=cfg.budget
        )
        loss = 0.0
        for L, c in zip(report.lengths, report.counts):
            loss += cfg.weight(L) * min(c, cfg.count_cap)
        loss /= max(n, 1)

        provable: list[str] = []
        restricted: list[str] = []
        penalties: dict[str, float] = {}
        bonus = 0.0

        planar = False
        claw_free = False
        claw_centers = -1
        two_ec = False
        biconn = False
        ip_lb = -1
        has_p13 = True
        p13_status = "assumed"

        if with_structure:
            two_ec = is_two_edge_connected(graph)
            biconn = is_biconnected(graph)
            planar = is_planar(graph)
            claw_centers = claw_center_count(graph)
            claw_free = claw_centers == 0
            ip_lb, _completed = longest_induced_path(
                graph, target=max(n, cfg.target_induced_path),
                budget=cfg.induced_path_budget,
            )
            has_p13, p13_status = has_induced_path(
                graph, cfg.target_induced_path, cfg.induced_path_budget
            )

            # ---- tier 1: provable prunes -----------------------------------
            if n <= EXHAUSTIVELY_VERIFIED_CUBIC_ORDER:
                provable.append(
                    f"cubic graphs of order <= {EXHAUSTIVELY_VERIFIED_CUBIC_ORDER} "
                    "are exhaustively verified (Ducoffe et al. 2026)"
                )
            if cfg.enforce_p13 and p13_status == "refuted":
                provable.append(
                    f"P{cfg.target_induced_path}-free graphs with delta>=3 always have "
                    "a power-of-two cycle (Hegde-Sandeep-Shashank 2025)"
                )
            if not two_ec:
                provable.append(
                    "not 2-edge-connected; every minimal counterexample is "
                    "2-edge-connected (Ducoffe et al. 2026)"
                )

            # ---- tier 2: literature-motivated restrictions -----------------
            if cfg.enforce_planar_free and planar:
                restricted.append("planar")
                penalties["planar"] = cfg.w_planar
            if cfg.enforce_claw_free and claw_free:
                restricted.append("claw-free")
                penalties["claw_free"] = cfg.w_claw_free
                if n <= CUBIC_CLAW_FREE_VERIFIED_ORDER:
                    provable.append(
                        f"cubic claw-free counterexamples need >= "
                        f"{CUBIC_CLAW_FREE_VERIFIED_ORDER + 1} vertices "
                        "(Nowbandegani et al. 2014)"
                    )

            # ---- tier 3: reinforcement -------------------------------------
            if not two_ec:
                penalties["not_two_edge_connected"] = cfg.w_not_two_edge_connected
            if biconn:
                bonus += cfg.w_biconnected_bonus
            extra = max(0, ip_lb - (cfg.target_induced_path - 1))
            bonus += cfg.w_induced_path * cfg.induced_path_scale * extra / max(n, 1)

        reward = -loss - sum(penalties.values()) + bonus
        return Evaluation(
            n=n,
            report=report,
            loss=loss,
            penalties=penalties,
            bonus=bonus,
            reward=reward,
            provably_not_counterexample=bool(provable),
            provable_reasons=provable,
            restricted_reasons=restricted,
            planar=planar,
            claw_free=claw_free,
            claw_centers=claw_centers,
            two_edge_connected=two_ec,
            biconnected=biconn,
            longest_induced_path_lb=ip_lb,
            has_induced_p13=has_p13,
            induced_p13_status=p13_status,
        )

    # ------------------------------------------------------------- summary
    def objective_breakdown(self, ev: Evaluation) -> str:
        return (
            f"loss={ev.loss:.4f} penalties={ev.penalties} bonus={ev.bonus:.4f} "
            f"reward={ev.reward:.4f}"
        )


def default_config_for(n: int, focus: str = "balanced") -> FitnessConfig:
    """A sensible configuration for a given graph order.

    ``focus`` biases the length weights:

    * ``"small"``    -- emphasise C4/C8, useful early in the search;
    * ``"large"``    -- emphasise the long powers of two;
    * ``"balanced"`` -- geometric decay, length L weighted 2^{-k/2}.
    """
    lengths = power_of_two_lengths(n)
    weights: dict[int, float] = {}
    for L in lengths:
        k = L.bit_length() - 1
        if focus == "small":
            weights[L] = 2.0 ** (-(k - 2))
        elif focus == "large":
            weights[L] = 2.0 ** (k - 2)
        else:
            weights[L] = 2.0 ** (-(k - 2) / 2.0)
    return FitnessConfig(length_weights=weights)
