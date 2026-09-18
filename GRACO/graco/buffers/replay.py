"""n-step transition construction (DIRAC/FINDER-faithful) + uniform replay.

``make_nstep_transitions`` turns one episode's (action, reward) sequence for a
single graph into n-step transitions, storing states as ``selected`` masks
reconstructed from the action prefix — exactly the "store action lists, rebuild
later" scheme of the reference code, so replay is compact.
"""

from __future__ import annotations

import random
from collections import deque
from typing import Dict, List, Optional

import torch
from torch import Tensor

from graco.buffers.base import Buffer, SampledBatch, SingleGraph, Transition, collate
from graco.registries import BUFFERS


def make_nstep_transitions(
    graph: SingleGraph,
    actions: List[int],
    rewards: List[float],
    n_step: int,
    gamma: float,
) -> List[Transition]:
    """Build n-step transitions for one episode of one graph."""
    L = len(actions)
    if L == 0:
        return []
    n = graph.num_nodes
    # prefixes[j] = boolean 'selected' mask after the first j actions
    prefixes: List[Tensor] = [torch.zeros(n, dtype=torch.bool)]
    cur = prefixes[0]
    for a in actions:
        cur = cur.clone()
        cur[a] = True
        prefixes.append(cur)

    transitions: List[Transition] = []
    for i in range(L):
        end = min(i + n_step, L)
        r_n, disc = 0.0, 1.0
        for k in range(i, end):
            r_n += disc * rewards[k]
            disc *= gamma
        terminal = (i + n_step) >= L
        transitions.append(
            Transition(
                graph=graph,
                state={"selected": prefixes[i]},
                action=int(actions[i]),
                reward=float(r_n),
                next_state={"selected": prefixes[end]},
                terminal=bool(terminal),
            )
        )
    return transitions


def make_nstep_from_snapshots(graph, states, actions, rewards, final_state, n_step, gamma):
    """Build n-step transitions from EXPLICIT per-step state snapshots.

    Unlike :func:`make_nstep_transitions` (which reconstructs the ``selected``
    mask from the monotone action prefix), this uses the actual state at each
    step, so it is exact for *non-monotone* envs too (revisitable flips in the
    ECO-DQN improvement env). ``states[t]`` is s_t; ``final_state`` is s_L.
    """
    L = len(actions)
    if L == 0:
        return []
    trans: List[Transition] = []
    for t in range(L):
        end = min(t + n_step, L)
        r, disc = 0.0, 1.0
        for k in range(t, end):
            r += disc * rewards[k]
            disc *= gamma
        terminal = (t + n_step) >= L
        next_sel = states[end] if end < L else final_state
        trans.append(
            Transition(
                graph=graph,
                state={"selected": states[t]},
                action=int(actions[t]),
                reward=float(r),
                next_state={"selected": next_sel},
                terminal=bool(terminal),
            )
        )
    return trans


@BUFFERS.register("nstep_replay", aliases=["uniform", "uniform_replay", "replay"])
class UniformNStepReplay(Buffer):
    """Uniform n-step replay over shared-graph transitions (DIRAC/FINDER default)."""

    def __init__(self, capacity: int = 50000, state_spec: Optional[Dict[str, str]] = None):
        self.capacity = int(capacity)
        self._store: deque = deque(maxlen=self.capacity)
        if state_spec is not None:
            self.state_spec = state_spec

    def add(self, transition: Transition) -> None:
        self._store.append(transition)

    def add_many(self, transitions: List[Transition]) -> None:
        self._store.extend(transitions)

    def sample(self, batch_size: int, device) -> SampledBatch:
        idx = [random.randrange(len(self._store)) for _ in range(batch_size)]
        batch = [self._store[i] for i in idx]
        graph, state, next_state, action, reward, terminal = collate(batch, self.state_spec, device)
        return SampledBatch(
            graph=graph,
            state=state,
            next_state=next_state,
            action=action,
            reward=reward,
            terminal=terminal,
            is_weights=torch.ones(batch_size, device=device),
        )

    def __len__(self) -> int:
        return len(self._store)
