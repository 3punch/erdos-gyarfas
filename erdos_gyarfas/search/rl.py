"""Reinforcement-learning drivers.

The action space of a graph search is not a fixed vector, so both agents work
on a **slate**: at every step the evaluator draws ``k`` candidate 2-opt swaps,
each is turned into a feature vector (exact cycle-count deltas, the length of
the shortest cycle the move creates, the change in the number of claws, ...),
and the network scores the whole slate.  The agent then picks one candidate.

* :class:`DQNAgent` -- tabular-style Q-learning over the slate with a replay
  buffer and a periodically synchronised target network.
* :class:`ActorCriticAgent` -- REINFORCE with a learned value baseline and an
  entropy bonus.

Both are deliberately tiny (a two-layer MLP over 8 features).  The point is not
to out-scale the metaheuristics but to give the search a *learned* prior over
which edge swaps are worth making, and to measure whether that prior transfers
across graph sizes.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

try:  # torch is a hard requirement for the RL drivers only
    import torch
    import torch.nn as nn
    from torch.distributions import Categorical

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    TORCH_AVAILABLE = False

from ..core.graph import CubicGraph
from ..fitness import FitnessConfig
from .base import MoveStats, SearchHistory, SearchResult
from .moves import FEATURE_NAMES, MoveEvaluator

N_FEATURES = len(FEATURE_NAMES)


def _require_torch() -> None:
    if not TORCH_AVAILABLE:
        raise RuntimeError(
            "PyTorch is required for the RL drivers; install it with "
            "`pip install torch` or run the metaheuristic drivers instead."
        )


class SlateNet(nn.Module if TORCH_AVAILABLE else object):  # type: ignore[misc]
    """Scores every candidate in a slate and estimates the state value."""

    def __init__(self, n_features: int = N_FEATURES, hidden: int = 64):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(n_features, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.score = nn.Linear(hidden, 1)
        self.value = nn.Linear(hidden, 1)

    def forward(self, slate: "torch.Tensor"):
        """``slate`` is ``(k, n_features)`` -> ``(k,)`` scores, scalar value."""
        h = self.trunk(slate)
        return self.score(h).squeeze(-1), self.value(h.mean(dim=0)).squeeze()


@dataclass
class _RLBase:
    config: FitnessConfig
    slate_size: int = 16
    hidden: int = 64
    lr: float = 1e-3
    gamma: float = 0.99
    log_every: int = 100
    track_lengths: tuple[int, ...] = (4, 8)
    reward_scale: float = 1.0
    seed: int = 0
    device: str = "cpu"

    def __post_init__(self) -> None:
        _require_torch()
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        self.net = SlateNet(N_FEATURES, self.hidden).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)

    # ---------------------------------------------------------------- reward
    def shaped_reward(self, delta_loss: float, n: int, short_cycle: int,
                      girth_cap: int) -> float:
        """Reward = cycle reduction, plus a small bonus for avoiding short cycles."""
        r = -delta_loss * n * self.reward_scale
        r += 0.05 * (min(short_cycle, girth_cap) / girth_cap - 0.5)
        return float(r)

    def state_dict(self) -> dict:
        return {k: v.detach().cpu().numpy().tolist() for k, v in self.net.state_dict().items()}

    def load_state(self, weights: dict) -> None:
        self.net.load_state_dict(
            {k: torch.as_tensor(np.asarray(v), dtype=torch.float32)
             for k, v in weights.items()}
        )


@dataclass
class DQNAgent(_RLBase):
    """Q-learning over a move slate with experience replay."""

    buffer_size: int = 20_000
    batch_size: int = 64
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay: int = 5_000
    target_sync: int = 500
    warmup: int = 200

    def __post_init__(self) -> None:
        super().__post_init__()
        self.target = SlateNet(N_FEATURES, self.hidden).to(self.device)
        self.target.load_state_dict(self.net.state_dict())
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.buffer: list[tuple] = []
        self.pos = 0
        self.steps = 0

    def _epsilon(self) -> float:
        f = min(1.0, self.steps / max(self.eps_decay, 1))
        return self.eps_start + (self.eps_end - self.eps_start) * f

    @torch.no_grad()
    def act(self, slate: np.ndarray) -> tuple[int, np.ndarray]:
        s = torch.as_tensor(slate, dtype=torch.float32, device=self.device)
        q, _ = self.net(s)
        qv = q.detach().cpu().numpy()
        if np.random.random() < self._epsilon():
            return int(np.random.randint(len(slate))), qv
        return int(np.argmax(qv)), qv

    def remember(self, slate, action, reward, next_slate, done) -> None:
        item = (slate, action, reward, next_slate, done)
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(item)
        else:
            self.buffer[self.pos % self.buffer_size] = item
        self.pos += 1

    def learn(self) -> float:
        if len(self.buffer) < max(self.batch_size, self.warmup):
            return 0.0
        idx = np.random.randint(0, len(self.buffer), size=self.batch_size)
        batch = [self.buffer[i] for i in idx]
        # slates may have different widths if fewer candidates were available
        width = min(b[0].shape[0] for b in batch)
        S = torch.as_tensor(np.stack([b[0][:width] for b in batch]),
                            dtype=torch.float32, device=self.device)
        A = torch.as_tensor(np.array([b[1] if b[1] < width else 0 for b in batch]),
                            dtype=torch.long, device=self.device)
        R = torch.as_tensor(np.array([b[2] for b in batch], dtype=np.float32),
                            device=self.device)
        S2 = torch.as_tensor(np.stack([b[3][:width] for b in batch]),
                             dtype=torch.float32, device=self.device)
        D = torch.as_tensor(np.array([b[4] for b in batch], dtype=np.float32),
                            device=self.device)

        q, _ = self.net(S)
        qa = q.gather(1, A.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q2, _ = self.target(S2)
            tgt = R + self.gamma * q2.max(dim=1).values * (1.0 - D)
        loss = torch.nn.functional.smooth_l1_loss(qa, tgt)
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
        self.opt.step()
        self.steps += 1
        if self.steps % self.target_sync == 0:
            self.target.load_state_dict(self.net.state_dict())
        return float(loss.item())


@dataclass
class ActorCriticAgent(_RLBase):
    """REINFORCE with a value baseline and an entropy bonus."""

    batch_size: int = 32
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 5.0

    def __post_init__(self) -> None:
        super().__post_init__()
        self.traj: list[tuple] = []
        self.steps = 0

    def act(self, slate: np.ndarray) -> tuple[int, float]:
        s = torch.as_tensor(slate, dtype=torch.float32, device=self.device)
        scores, value = self.net(s)
        dist = Categorical(logits=scores)
        a = dist.sample()
        return int(a.item()), float(value.item()), dist.log_prob(a), value

    def learn(self) -> float:
        if len(self.traj) < self.batch_size:
            return 0.0
        batch = self.traj[-self.batch_size:]
        self.traj = []
        returns = []
        R = 0.0
        for step in reversed(batch):
            R = step[1] + self.gamma * R
            returns.insert(0, R)
        ret = torch.as_tensor(np.array(returns, dtype=np.float32), device=self.device)
        logp = torch.stack([b[2] for b in batch])
        values = torch.stack([b[3] for b in batch])
        adv = (ret - values).detach()
        if adv.numel() > 1 and adv.std() > 1e-8:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        scores_all = torch.stack([b[4] for b in batch])
        entropy = -(torch.softmax(scores_all, dim=-1)
                    * torch.log_softmax(scores_all, dim=-1)).sum(-1).mean()
        policy_loss = -(logp * adv).mean()
        value_loss = torch.nn.functional.mse_loss(values, ret)
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
        self.opt.step()
        self.steps += 1
        return float(loss.item())


@dataclass
class RLSearch:
    """Driver that lets either agent choose the edge swaps."""

    config: FitnessConfig
    agent: object = None
    slate_size: int = 16
    log_every: int = 50
    track_lengths: tuple[int, ...] = (4, 8)

    def run(self, graph: CubicGraph, moves: int = 10_000, seed: int = 0) -> SearchResult:
        _require_torch()
        rng = np.random.default_rng(seed)
        g = graph.copy()
        ev = MoveEvaluator(g, self.config, track_lengths=self.track_lengths)
        agent = self.agent
        if agent is None:
            agent = ActorCriticAgent(config=self.config, slate_size=self.slate_size,
                                     seed=seed)
        agent.slate_size = self.slate_size
        hist = SearchHistory()
        t0 = time.perf_counter()
        cur_loss = ev.loss()
        best_loss = cur_loss
        best_graph = g.copy()
        best_counts = ev.counts.copy()
        losses: list[float] = []
        rewards: list[float] = []

        for it in range(1, moves + 1):
            ev.proposed += 1
            cands = ev.propose_k(rng, self.slate_size)
            if len(cands) < 2:
                continue
            slates, deltas, feats = [], [], []
            for mv in cands:
                d, f = ev.peek(mv, with_features=True)
                slates.append(ev.feature_vector(d, f))
                deltas.append(d)
                feats.append(f)
            slate = np.stack(slates)

            if isinstance(agent, DQNAgent):
                action, _q = agent.act(slate)
            else:
                action, _value, logp, vtensor = agent.act(slate)

            mv = cands[action]
            d = deltas[action]
            loss_after = ev.loss_after(d)
            delta_loss = loss_after - cur_loss
            # a learned policy proposes; the physical objective still decides
            ev.commit(mv, d)
            cur_loss = loss_after
            reward = agent.shaped_reward(delta_loss, ev.n, feats[action][0],
                                         ev.girth_cap)
            rewards.append(reward)

            if it < moves:
                nxt_cands = ev.propose_k(rng, min(4, self.slate_size))
                nxt = np.stack([
                    ev.feature_vector(*_peek_pair(ev, c)) for c in nxt_cands
                ])
            else:
                nxt = slate
            done = 1.0 if it == moves else 0.0

            if isinstance(agent, DQNAgent):
                agent.remember(slate, action, reward, nxt, done)
                losses.append(agent.learn())
            else:
                agent.traj.append((slate, reward, logp, vtensor,
                                   torch.as_tensor(slate, dtype=torch.float32)))
                losses.append(agent.learn())

            if cur_loss < best_loss - 1e-12:
                best_loss = cur_loss
                best_graph = g.copy()
                best_counts = ev.counts.copy()

            if it % self.log_every == 0:
                recent_loss = [l for l in losses[-self.log_every:] if l]
                hist.add(
                    it, cur_loss, best_loss,
                    mean_reward=float(np.mean(rewards[-self.log_every:]))
                    if rewards else 0.0,
                    agent_loss=float(np.mean(recent_loss)) if recent_loss else 0.0,
                )
                losses = losses[-self.log_every:]

        elapsed = time.perf_counter() - t0
        name = "dqn" if isinstance(agent, DQNAgent) else "actor_critic"
        return SearchResult(
            algorithm=f"rl_{name}",
            graph=best_graph,
            best_loss=best_loss,
            final_loss=cur_loss,
            stats=MoveStats(
                proposed=ev.proposed, accepted=ev.accepted,
                improved=0, rejected=ev.proposed - ev.accepted,
                seconds=elapsed,
            ),
            history=hist,
            meta={
                "agent": name,
                "slate_size": self.slate_size,
                "steps": getattr(agent, "steps", None),
                "mean_reward_last": float(np.mean(rewards[-100:])) if rewards else 0.0,
                "tracked_counts": dict(zip(map(int, ev.lengths), map(int, best_counts))),
                "final_counts": dict(zip(map(int, ev.lengths), map(int, ev.counts))),
                "weights": agent.state_dict(),
            },
        )


def _peek_pair(ev: MoveEvaluator, mv):
    d, f = ev.peek(mv, with_features=True)
    return d, f
