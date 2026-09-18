"""Heterogeneous **combat networks** (SHATTER / HDGED).

A combat network has three functional node roles — Sensor (S), Decision (D) and
Influence (I) — wired into ``S → D → I`` *combat chains* (a target is sensed,
a decision is made, an effect is delivered).  This generator lays down the three
role sub-populations and the directed wiring between them, storing:

* ``node_type`` — ``0`` = S, ``1`` = D, ``2`` = I (the ids read by the combat
  operational-capability metric and by the ``rgcn`` / ``han`` encoders).
* ``edge_type`` — relation id per undirected edge: ``0`` = S–D, ``1`` = D–I,
  ``2`` = D–D (lateral command links).
* ``meta`` — ``num_node_types=3``, ``num_edge_types=3`` and the S/D/I ids, so the
  heterogeneous encoders build the right relation/type tables automatically.

Edges are stored undirected (symmetrized by :func:`build_batch`); the chain
*direction* is recovered from the endpoint node types, so no extra bookkeeping is
needed by the environment.
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch
from torch import Tensor

from graco.data.batch import BatchedGraph
from graco.generators._common import build_batch
from graco.generators.base import GraphGenerator
from graco.registries import GENERATORS

_S, _D, _I = 0, 1, 2


@GENERATORS.register("combat", aliases=["combat_network", "sdi_network"])
class CombatNetwork(GraphGenerator):
    """Sensor/Decision/Influence combat network with ``S→D→I`` chains.

    ``n`` nodes are split into S/D/I sub-populations by ``role_fracs`` (normalized).
    Each S links to each D with prob ``p_sd``, each D to each I with prob ``p_di``,
    and D–D lateral links with prob ``p_dd`` — guaranteeing at least one outgoing
    link per S and D so isolated command roles do not dominate.
    """

    def __init__(
        self,
        role_fracs=(0.4, 0.3, 0.3),
        p_sd: float = 0.25,
        p_di: float = 0.25,
        p_dd: float = 0.1,
        **base_kwargs,
    ):
        super().__init__(**base_kwargs)
        f = np.asarray(role_fracs, dtype=float)
        self.role_fracs = f / f.sum()
        self.p_sd = float(p_sd)
        self.p_di = float(p_di)
        self.p_dd = float(p_dd)

    def _split(self, n: int) -> tuple[int, int, int]:
        nS = max(1, int(round(n * self.role_fracs[0])))
        nD = max(1, int(round(n * self.role_fracs[1])))
        nI = max(1, n - nS - nD)
        return nS, nD, nI

    def _bipartite(self, srcs, dsts, p: float, rng) -> List[Tensor]:
        """Directed links src->dst with prob ``p``, each src forced ≥1 outgoing."""
        edges = []
        for s in srcs:
            mask = rng.random(len(dsts)) < p
            if not mask.any():
                mask[rng.integers(0, len(dsts))] = True
            for d in np.asarray(dsts)[mask]:
                edges.append((s, int(d)))
        return edges

    def _build_one(self, n: int, rng):
        nS, nD, nI = self._split(int(n))
        n = nS + nD + nI
        S = list(range(nS))
        D = list(range(nS, nS + nD))
        Inf = list(range(nS + nD, n))
        node_type = torch.empty(n, dtype=torch.long)
        node_type[S], node_type[D], node_type[Inf] = _S, _D, _I

        pairs, etypes = [], []
        for (u, v) in self._bipartite(S, D, self.p_sd, rng):  # S–D (relation 0)
            pairs.append((u, v))
            etypes.append(0)
        for (u, v) in self._bipartite(D, Inf, self.p_di, rng):  # D–I (relation 1)
            pairs.append((u, v))
            etypes.append(1)
        if self.p_dd > 0 and nD > 1:                          # D–D lateral (relation 2)
            for i in range(nD):
                for j in range(i + 1, nD):
                    if rng.random() < self.p_dd:
                        pairs.append((D[i], D[j]))
                        etypes.append(2)

        if pairs:
            und = torch.tensor(pairs, dtype=torch.long).t().contiguous()
            et = torch.tensor(etypes, dtype=torch.long)
        else:
            und = torch.zeros(2, 0, dtype=torch.long)
            et = torch.zeros(0, dtype=torch.long)
        return und, node_type, et

    def _edges(self, n: int, rng) -> Tensor:
        return self._build_one(int(n), rng)[0]

    def sample(self, batch_size, device="cpu", generator=None, rng=None) -> BatchedGraph:
        if rng is None:
            rng = np.random.default_rng(self._np_seed)
        sizes = self._sizes(batch_size, generator)
        und_list, nt_list, et_list, actual = [], [], [], []
        for n in sizes:
            und, nt, et = self._build_one(int(n), rng)
            und_list.append(und)
            nt_list.append(nt)
            et_list.append(et)
            actual.append(nt.numel())

        bg = build_batch(
            self,
            und_list,
            actual,
            device=device,
            generator=generator,
            node_type_list=nt_list,
            edge_type_und_list=et_list,
        )
        bg.meta.update(
            num_node_types=3,
            num_edge_types=3,
            sensor_type=_S,
            decision_type=_D,
            influence_type=_I,
        )
        return bg
