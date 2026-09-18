"""Central registries for every pluggable family in GRACO.

Importing all registries from one module keeps registration import-order simple
and avoids circular imports: sub-packages import the registry object from here
and decorate their classes onto it.
"""

from __future__ import annotations

from graco.utils.registry import Registry

#: graph instance generators (ER, BA, WS, lattice, hypergraph, hetero, ...)
GENERATORS = Registry("generator")
#: vectorized combinatorial-optimization environments (maxcut, mvc, dismantling, ...)
ENVS = Registry("env")
#: GNN encoders (gcn, gin, gat, s2v, gatedgcn, pna, transformer, ...)
ENCODERS = Registry("encoder")
#: output heads (q_node, dueling_q, actor_critic, ...)
HEADS = Registry("head")
#: RL algorithms (dqn, ppo, a2c, reinforce, ...)
ALGOS = Registry("algo")
#: experience buffers (nstep_replay, prioritized_replay, rollout)
BUFFERS = Registry("buffer")

__all__ = ["GENERATORS", "ENVS", "ENCODERS", "HEADS", "ALGOS", "BUFFERS"]
