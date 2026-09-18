"""On-policy rollout buffer with Generalized Advantage Estimation (GAE).

Used by REINFORCE / A2C / PPO.  Each graph in the batch is an independent
trajectory that ends at its own terminal step; we record per-env
``(selected-state, action, log-prob, value, reward)`` and, at episode end,
compute per-trajectory advantages/returns.  States are stored as ``selected``
masks (like the replay buffer) so PPO epochs can rebuild observations and
recompute masked log-probs/values consistently with the live env.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterator, List

import torch
from torch import Tensor

from graco.buffers.base import SingleGraph


@dataclass
class RolloutBuffer:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    state_spec: Dict[str, str] = field(default_factory=lambda: {"selected": "node"})

    def start(self, single_graphs: List[SingleGraph]) -> None:
        self.graphs = single_graphs
        b = len(single_graphs)
        self.state: List[List[Dict[str, Tensor]]] = [[] for _ in range(b)]
        self.action: List[List[int]] = [[] for _ in range(b)]
        self.logp: List[List[float]] = [[] for _ in range(b)]
        self.value: List[List[float]] = [[] for _ in range(b)]
        self.reward: List[List[float]] = [[] for _ in range(b)]
        # flattened samples produced by finalize()
        self._samples: List[dict] = []

    def record_action(
        self,
        env_idx: int,
        state: Dict[str, Tensor],
        action_local: int,
        logp: float,
        value: float,
    ) -> None:
        self.state[env_idx].append(state)
        self.action[env_idx].append(int(action_local))
        self.logp[env_idx].append(float(logp))
        self.value[env_idx].append(float(value))

    def record_reward(self, env_idx: int, reward: float) -> None:
        self.reward[env_idx].append(float(reward))

    def finalize(self) -> None:
        """Compute GAE advantages/returns per trajectory and flatten samples."""
        self._samples.clear()
        for b, graph in enumerate(self.graphs):
            values = self.value[b]
            rewards = self.reward[b]
            T = min(len(values), len(rewards))
            if T == 0:
                continue
            adv = 0.0
            advantages = [0.0] * T
            returns = [0.0] * T
            for t in reversed(range(T)):
                v_next = values[t + 1] if t + 1 < T else 0.0  # terminal bootstrap = 0
                delta = rewards[t] + self.gamma * v_next - values[t]
                adv = delta + self.gamma * self.gae_lambda * adv
                advantages[t] = adv
                returns[t] = adv + values[t]
            for t in range(T):
                self._samples.append(
                    {
                        "graph": graph,
                        "state": self.state[b][t],
                        "action": self.action[b][t],
                        "logp": self.logp[b][t],
                        "value": values[t],
                        "advantage": advantages[t],
                        "ret": returns[t],
                    }
                )

    def __len__(self) -> int:
        return len(self._samples)

    def iter_minibatches(
        self, batch_size: int, device, shuffle: bool = True, normalize_adv: bool = True
    ) -> Iterator[dict]:
        n = len(self._samples)
        order = torch.randperm(n) if shuffle else torch.arange(n)
        all_adv = torch.tensor([s["advantage"] for s in self._samples], dtype=torch.float32)
        if normalize_adv and n > 1:
            all_adv = (all_adv - all_adv.mean()) / (all_adv.std() + 1e-8)
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            batch = [self._samples[int(i)] for i in idx]
            yield self._collate(batch, all_adv[idx], device)

    def _collate(self, batch: List[dict], adv: Tensor, device) -> dict:
        from graco.buffers.base import collate_single_graphs

        graph, offsets = collate_single_graphs([s["graph"] for s in batch], device)
        out_state: Dict[str, Tensor] = {}
        for name, scope in self.state_spec.items():
            vals = [s["state"][name] for s in batch]
            if scope == "graph":
                out_state[name] = torch.stack([v.reshape(()) for v in vals]).to(device)
            else:
                out_state[name] = torch.cat([v.reshape(-1) for v in vals]).to(device)
        action = torch.tensor([s["action"] for s in batch], device=device) + offsets
        return {
            "graph": graph,
            "state": out_state,
            "action": action,
            "logp_old": torch.tensor([s["logp"] for s in batch], dtype=torch.float32, device=device),
            "value_old": torch.tensor([s["value"] for s in batch], dtype=torch.float32, device=device),
            "advantage": adv.to(device),
            "ret": torch.tensor([s["ret"] for s in batch], dtype=torch.float32, device=device),
        }
