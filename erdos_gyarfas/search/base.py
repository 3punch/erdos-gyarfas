"""Shared scaffolding for the search drivers."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from ..core.graph import CubicGraph


@dataclass
class MoveStats:
    """Move-level counters reported by every driver."""

    proposed: int = 0
    accepted: int = 0
    improved: int = 0
    rejected: int = 0
    seconds: float = 0.0

    @property
    def moves_per_second(self) -> float:
        return self.proposed / self.seconds if self.seconds > 0 else 0.0

    def as_dict(self) -> dict:
        return {
            "proposed": self.proposed,
            "accepted": self.accepted,
            "improved": self.improved,
            "rejected": self.rejected,
            "seconds": self.seconds,
            "moves_per_second": self.moves_per_second,
        }


@dataclass
class SearchHistory:
    """A downsampled trace of the run, for plots and the experiment log."""

    iteration: list[int] = field(default_factory=list)
    loss: list[float] = field(default_factory=list)
    best_loss: list[float] = field(default_factory=list)
    extra: dict[str, list[float]] = field(default_factory=dict)

    def add(self, it: int, loss: float, best_loss: float, **extra: float) -> None:
        self.iteration.append(it)
        self.loss.append(loss)
        self.best_loss.append(best_loss)
        for k, v in extra.items():
            self.extra.setdefault(k, []).append(v)

    def as_array(self) -> np.ndarray:
        return np.column_stack(
            [self.iteration, self.loss, self.best_loss]
        ) if self.iteration else np.zeros((0, 3))


@dataclass
class SearchResult:
    """Everything a driver hands back to the experiment runner."""

    algorithm: str
    graph: CubicGraph
    best_loss: float
    final_loss: float
    stats: MoveStats
    history: SearchHistory
    meta: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "algorithm": self.algorithm,
            "n": self.graph.n,
            "best_loss": self.best_loss,
            "final_loss": self.final_loss,
            "stats": self.stats.as_dict(),
            "meta": self.meta,
        }


def timed() -> float:
    return time.perf_counter()


OnBest = Callable[[CubicGraph, float, int], None]
