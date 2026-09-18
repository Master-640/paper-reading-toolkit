"""Faithful DIRAC baseline (Fan et al., Nature Communications 2023).

DIRAC = structure2vec (node<->edge alternation) + n-step DQN on 3D periodic
lattices with N(0,1) couplings, maximizing the Ising energy H = sum w_ij s_i s_j
(reward = ΔH / |E|, bootstrap clipped at 0).  Almost everything is reused from
the framework; the two DIRAC-specific pieces provided here are:

* :class:`DiracLattice` — a lattice generator that also records integer node
  **coordinates** (``meta['node_pos']``), which survive replay collation.
* :class:`DiracSpinGlassEnv` — a spin-glass env whose node features are the
  normalized lattice coordinates (DIRAC's ``node_coords / n``), exactly as in the
  reference; the 4-d edge features are inherited unchanged.

Run it with ``graco train -c dirac``.  The DQN hyper-parameters (n_step=3, no
double-DQN, bootstrap clipped at 0, s2v encoder, 3D lattice) live in that config.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
from torch import Tensor

from graco.data.batch import BatchedGraph
from graco.envs.base import State
from graco.envs.maxcut import SpinGlassEnv, _spin
from graco.generators._common import build_batch
from graco.generators.base import GraphGenerator
from graco.registries import ENVS, GENERATORS


@GENERATORS.register("dirac_lattice")
class DiracLattice(GraphGenerator):
    """d-dimensional (optionally periodic) lattice with node coordinates."""

    def __init__(self, dim: int = 3, periodic: bool = True, side: Optional[int] = None, **base_kwargs):
        base_kwargs.setdefault("weighted", True)
        base_kwargs.setdefault("weight_dist", "normal")
        super().__init__(**base_kwargs)
        self.dim = int(dim)
        self.periodic = bool(periodic)
        self.side = side

    def _grid(self, n: int):
        side = int(self.side or max(2, round(n ** (1.0 / self.dim))))
        shape = [side] * self.dim
        N = side ** self.dim
        idx = np.arange(N).reshape(shape)
        coords = np.stack(np.unravel_index(np.arange(N), shape), axis=1)  # [N, dim]
        edges = set()
        for d in range(self.dim):
            nb = np.roll(idx, -1, axis=d)
            a = idx.reshape(-1)
            b = nb.reshape(-1)
            keep = np.ones(N, dtype=bool) if self.periodic else (coords[:, d] != side - 1)
            for u, v in zip(a[keep].tolist(), b[keep].tolist()):
                edges.add((min(u, v), max(u, v)))
        if edges:
            und = torch.tensor(sorted(edges), dtype=torch.long).t().contiguous()
        else:
            und = torch.zeros(2, 0, dtype=torch.long)
        return und, torch.tensor(coords, dtype=torch.float32), N

    def _edges(self, n: int, rng) -> Tensor:  # ABC satisfaction; sample() is overridden
        return self._grid(int(n))[0]

    def sample(self, batch_size, device="cpu", generator=None, rng=None) -> BatchedGraph:
        if rng is None:
            rng = np.random.default_rng(self._np_seed)
        sizes_req = self._sizes(batch_size, generator)
        und_list: List[Tensor] = []
        actual: List[int] = []
        coords_list: List[Tensor] = []
        for n in sizes_req:
            und, coords, N = self._grid(int(n))
            und_list.append(und)
            actual.append(N)
            coords_list.append(coords)
        bg = build_batch(self, und_list, actual, device=device, generator=generator)
        bg.meta["node_pos"] = torch.cat(coords_list, dim=0).to(bg.device)
        return bg


@ENVS.register("dirac_spinglass", aliases=["dirac"])
class DiracSpinGlassEnv(SpinGlassEnv):
    """Spin-glass env with DIRAC's lattice-coordinate node features (dim 4)."""

    node_feature_dim = 4  # coords padded to 4 (covers 2D/3D/4D lattices)
    edge_feature_dim = 4  # inherited [weight, sel_src, cut_indicator, 1]

    @staticmethod
    def node_features(graph: BatchedGraph, state: State) -> Tensor:
        pos = graph.meta.get("node_pos") if graph.meta else None
        if pos is not None:
            n = graph.broadcast_to_nodes(graph.graph_num_nodes.float().unsqueeze(-1)).clamp_min(1)
            feat = pos.to(n.dtype) / n  # DIRAC: coords / num_nodes
            d = feat.shape[1]
            if d < 4:
                feat = torch.cat([feat, feat.new_zeros(feat.shape[0], 4 - d)], dim=1)
            return feat[:, :4]
        # fallback (non-lattice graph): spin + normalized degree + constants
        spin = _spin(state["selected"]).unsqueeze(-1)
        deg = graph.degree().unsqueeze(-1)
        deg_max = graph.broadcast_to_nodes(graph.pool(deg, reduce="max")).clamp_min(1)
        return torch.cat([spin, deg / deg_max, torch.ones_like(spin), torch.zeros_like(spin)], dim=1)
