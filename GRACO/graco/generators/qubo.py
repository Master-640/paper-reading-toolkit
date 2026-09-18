"""QUBO instance generator.

Produces random QUBO instances as graphs — a base topology (ER / BA / complete)
whose **edge weights are the pairwise couplings** and whose per-node **bias**
``Q_ii`` is attached as ``node_attr['q_diag']`` — plus helpers to wrap a
*user-supplied* ``Q`` matrix (:meth:`from_matrix` / :meth:`from_matrices`), so you
can solve your own QUBO with the ``qubo`` environment / a trained policy.
"""

from __future__ import annotations

from typing import List, Sequence

import networkx as nx
import numpy as np
import torch
from torch import Tensor

from graco.data.batch import BatchedGraph
from graco.generators.base import GraphGenerator
from graco.registries import GENERATORS


@GENERATORS.register("qubo")
class QUBOGenerator(GraphGenerator):
    def __init__(
        self,
        num_nodes=(20, 40),
        base: str = "er",
        p: float = 0.15,
        m: int = 4,
        coupling_dist: str = "uniform",
        coupling_params: dict | None = None,
        bias_scale: float = 1.0,
        seed=None,
        **kw,
    ):
        for k in ("weighted", "weight_dist", "weight_params", "m"):
            kw.pop(k, None)
        super().__init__(
            num_nodes=num_nodes,
            weighted=True,
            weight_dist=coupling_dist,
            weight_params=coupling_params or {"low": -1.0, "high": 1.0},
            seed=seed,
            **kw,
        )
        self.base = str(base)
        self.p = float(p)
        self.m = int(m)
        self.bias_scale = float(bias_scale)

    def _edges(self, n: int, rng) -> Tensor:  # undirected edge list [2, e]
        seed = int(rng.integers(0, 2**31 - 1))
        if self.base == "ba":
            g = nx.barabasi_albert_graph(n, max(1, min(self.m, n - 1)), seed=seed)
        elif self.base in ("complete", "full"):
            g = nx.complete_graph(n)
        else:
            g = nx.erdos_renyi_graph(n, self.p, seed=seed)
        if g.number_of_edges() == 0:
            return torch.zeros(2, 0, dtype=torch.long)
        return torch.tensor(list(g.edges()), dtype=torch.long).t().contiguous()

    def sample(self, batch_size, device="cpu", generator=None, rng=None) -> BatchedGraph:
        if rng is None:
            rng = np.random.default_rng(self._np_seed)
        bg = super().sample(batch_size, device=device, generator=generator, rng=rng)
        # per-node bias b_i = Q_ii ~ U(-bias_scale, bias_scale)
        bg.node_attr["q_diag"] = torch.empty(bg.num_nodes, device=device).uniform_(
            -self.bias_scale, self.bias_scale
        )
        return bg

    # ------------------------------------------------- user-supplied Q matrices
    @staticmethod
    def from_matrix(Q, device="cpu") -> BatchedGraph:
        """Build a single-instance :class:`BatchedGraph` from a QUBO matrix ``Q``.

        ``Q`` is any square ``[n, n]`` array/tensor; the diagonal becomes the node
        bias and the symmetrized off-diagonal ``Q_ij + Q_ji`` the edge couplings.
        """
        return QUBOGenerator.from_matrices([Q], device=device)

    @staticmethod
    def from_matrices(mats: Sequence, device="cpu") -> BatchedGraph:
        """Batch several QUBO matrices into one :class:`BatchedGraph`."""
        ei_list: List[Tensor] = []
        ew_list: List[Tensor] = []
        sizes: List[int] = []
        biases: List[Tensor] = []
        for Q in mats:
            Q = torch.as_tensor(Q, dtype=torch.float32)
            if Q.dim() != 2 or Q.shape[0] != Q.shape[1]:
                raise ValueError(f"each Q must be a square [n,n] matrix, got {tuple(Q.shape)}")
            n = int(Q.shape[0])
            biases.append(torch.diagonal(Q).clone())
            C = Q + Q.t()
            C.fill_diagonal_(0.0)
            iu = torch.triu_indices(n, n, offset=1)
            w = C[iu[0], iu[1]]
            keep = w != 0
            ii, jj, w = iu[0][keep], iu[1][keep], w[keep]
            if ii.numel():
                src = torch.cat([ii, jj])
                dst = torch.cat([jj, ii])
                ei_list.append(torch.stack([src, dst]))
                ew_list.append(torch.cat([w, w]))
            else:
                ei_list.append(torch.zeros(2, 0, dtype=torch.long))
                ew_list.append(torch.zeros(0))
            sizes.append(n)
        bg = BatchedGraph.from_graph_list(ei_list, sizes, ew_list, device=device)
        bg.node_attr["q_diag"] = torch.cat([b.to(device) for b in biases])
        return bg
