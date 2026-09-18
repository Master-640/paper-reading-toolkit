"""Sentinel-node-selection data generator (see why.md).

Two dynamics modes, both attaching per-node trajectories and a global steady
target to the batch so the same replay plumbing serves selection *and* the
learned predictor:

``dynamics='glv'`` (**Generalized Lotka-Volterra**, the observability task).
Each condition ``m`` uses a per-graph interaction strength (coupling) ``c`` drawn
from ``coupling_range`` and Euler-integrates, with row-normalised neighbour
coupling for stability,

    x_i(t+1) = clamp_{>=0}( x_i + dt * x_i * ( r_i - self_decay*x_i + c * mean_{j~i} x_j ) ) + noise

The consensus equilibrium is ``x* = r/(self_decay - c)``, so **the whole coupling
range must stay on the stable side ``c < self_decay``** (no transcritical
bifurcation / extinction — the disjoint train/val ranges in ``sentinel_glv.yaml``
both satisfy this).  With ``fixed_graph=True`` the *same* topology and growth
rates ``r`` are reused every ``sample()`` (seeded by ``graph_seed``), so training
and the disjoint-range validation set share one system — the deployment setting
of "same graph, brand-new coupling".

``dynamics='map'`` (legacy) is the bounded tanh map ``x(t+1)=tanh(theta*mean_nbr+h)``.

Attached tensors:

* ``node_attr['dyn_mean']`` = ``a``   ``[N, M]``  robust steady value per node/condition
* ``node_attr['traj']``     = full trajectory ``[N, M, T]`` (for the Transformer)
* ``graph_attr['y']``       = ``y``   ``[B, M]``  global steady mean per condition
* ``graph_attr['coupling']``= ``c``   ``[B, M]``  (GLV only; bookkeeping/eval, **never** fed to the predictor)
* ``graph_attr['converged']``=``[B, M]`` bool (GLV only; non-converged conditions are masked out downstream)
"""

from __future__ import annotations

from typing import List

import networkx as nx
import numpy as np
import torch
from torch import Tensor

from graco.data.batch import BatchedGraph
from graco.generators.base import GraphGenerator
from graco.registries import GENERATORS
from graco.utils.scatter import scatter_mean, scatter_sum


@GENERATORS.register("sentinel", aliases=["sentinel_dynamics"])
class SentinelDataGenerator(GraphGenerator):
    def __init__(
        self,
        num_nodes=(30, 50),
        base: str = "ba",
        m: int = 4,
        p: float = 0.15,
        num_conditions: int = 5,
        timesteps: int = 40,
        steady_window: int = 10,
        # --- map (tanh) dynamics ---
        theta_range=(0.5, 2.0),
        h_range=(-1.0, 1.0),
        # --- glv dynamics ---
        dynamics: str = "map",
        coupling_range=(0.2, 0.5),
        r_range=(0.8, 1.2),
        dt: float = 0.1,
        self_decay: float = 1.0,
        steady_tol: float = 0.05,
        row_normalize: bool = True,
        x_init: float = 0.1,
        x_clamp: float = 1e3,
        # --- shared ---
        noise: float = 0.02,
        store_traj: bool = True,
        fixed_graph: bool = False,
        graph_seed: int = 0,
        **base_kwargs,
    ):
        base_kwargs.pop("m", None)
        super().__init__(num_nodes=num_nodes, **base_kwargs)
        self.base = base
        self.m = int(m)
        self.p = float(p)
        self.M = int(num_conditions)
        self.T = int(timesteps)
        self.T_ss = int(min(steady_window, timesteps))
        self.theta_range = theta_range
        self.h_range = h_range
        self.dynamics = str(dynamics).lower()
        self.coupling_range = coupling_range
        self.r_range = r_range
        self.dt = float(dt)
        self.self_decay = float(self_decay)
        self.steady_tol = float(steady_tol)
        self.row_normalize = bool(row_normalize)
        self.x_init = float(x_init)
        self.x_clamp = float(x_clamp)
        self.noise = float(noise)
        self.store_traj = bool(store_traj)
        self.fixed_graph = bool(fixed_graph)
        self.graph_seed = int(graph_seed)
        self.eps = 1e-6
        self._fixed = None  # cached (undirected edges, n, r) for fixed_graph mode

    # ------------------------------------------------------------- topology
    def _base_graph(self, n: int, seed: int) -> nx.Graph:
        if self.base == "er":
            return nx.erdos_renyi_graph(n, self.p, seed=seed)
        return nx.barabasi_albert_graph(n, max(1, min(self.m, n - 1)), seed=seed)

    def _und_edges(self, g: nx.Graph) -> Tensor:
        if g.number_of_edges() == 0:
            return torch.zeros(2, 0, dtype=torch.long)
        return torch.tensor(list(g.edges()), dtype=torch.long).t().contiguous()

    def _edges(self, n: int, rng) -> Tensor:  # ABC satisfaction; sample() overridden
        seed = int(rng.integers(0, 2**31 - 1))
        return self._und_edges(self._base_graph(int(n), seed))

    def _fixed_topology(self):
        """One cached system (topology + growth rates ``r``), seeded by ``graph_seed``."""
        if self._fixed is None:
            n = self.num_nodes_range[0]
            und = self._und_edges(self._base_graph(int(n), self.graph_seed))
            r_rng = np.random.default_rng(self.graph_seed + 1)
            r = torch.tensor(r_rng.uniform(*self.r_range, size=int(n)), dtype=torch.float32)
            self._fixed = (und, int(n), r)
        return self._fixed

    @staticmethod
    def _directed(und: Tensor) -> Tensor:
        if und.numel() == 0:
            return und
        src = torch.cat([und[0], und[1]])
        dst = torch.cat([und[1], und[0]])
        return torch.stack([src, dst])

    # --------------------------------------------------------------- sampling
    def sample(self, batch_size, device="cpu", generator=None, rng=None) -> BatchedGraph:
        if rng is None:
            rng = np.random.default_rng(self._np_seed)

        if self.fixed_graph:
            und, n, r_fixed = self._fixed_topology()
            sizes = [n] * batch_size
            und_list = [und] * batch_size
            r_list = [r_fixed] * batch_size
        else:
            sizes = self._sizes(batch_size, generator)
            und_list, r_list = [], []
            for nn in sizes:
                seed = int(rng.integers(0, 2**31 - 1))
                und_list.append(self._und_edges(self._base_graph(int(nn), seed)))
                r_list.append(torch.tensor(rng.uniform(*self.r_range, size=int(nn)), dtype=torch.float32))

        ei_list = [self._directed(u) for u in und_list]
        bg = BatchedGraph.from_graph_list(ei_list, sizes, device=device)
        B, M = bg.num_graphs, self.M

        if self.dynamics == "glv":
            a, traj, extra = self._simulate_glv(bg, r_list, rng, device)
        else:
            a, traj, extra = self._simulate_map(bg, rng, device)

        y = scatter_mean(a, bg.batch, B)  # [B, M] global steady mean per condition
        bg.node_attr["dyn_mean"] = a
        if traj is not None:
            bg.node_attr["traj"] = traj
        bg.graph_attr["y"] = y
        for k, v in extra.items():
            bg.graph_attr[k] = v
        bg.meta["num_conditions"] = M
        bg.meta["dynamics"] = self.dynamics
        return bg

    # ------------------------------------------------------------- dynamics
    def _simulate_map(self, bg: BatchedGraph, rng, device):
        """Legacy bounded tanh map (unchanged behaviour)."""
        N, M, T = bg.num_nodes, self.M, self.T
        theta = torch.tensor(rng.uniform(*self.theta_range, size=M), dtype=torch.float32, device=device)
        h = torch.tensor(rng.uniform(*self.h_range, size=M), dtype=torch.float32, device=device)
        src, dst = bg.edge_index[0], bg.edge_index[1]
        deg = bg.degree().clamp_min(1.0).unsqueeze(-1)
        x = torch.zeros(N, M, device=device)
        traj = torch.empty(N, M, T, device=device) if self.store_traj else None
        for t in range(T):
            nbr = scatter_sum(x.index_select(0, src), dst, N) / deg
            x = torch.tanh(theta * nbr + h)
            if self.noise:
                x = x + self.noise * torch.randn(N, M, device=device)
            if traj is not None:
                traj[:, :, t] = x
        a = traj[:, :, -self.T_ss:].mean(dim=-1) if traj is not None else x
        return a, traj, {}

    def _simulate_glv(self, bg: BatchedGraph, r_list: List[Tensor], rng, device):
        """Generalized Lotka-Volterra. ``row_normalize`` picks the coupling operator:
        neighbour **mean** (near-consensus, hub-agnostic) or raw neighbour **sum**
        (degree-weighted — hubs dominate, so a hub sentinel set is a biased,
        coupling-dependent sample of the global mean)."""
        N, B, M, T = bg.num_nodes, bg.num_graphs, self.M, self.T
        src, dst = bg.edge_index[0], bg.edge_index[1]
        deg = bg.degree().clamp_min(1.0).unsqueeze(-1)  # [N,1]
        r = torch.cat([rr.to(device) for rr in r_list]).unsqueeze(-1)  # [N,1]
        c = torch.tensor(
            rng.uniform(*self.coupling_range, size=(B, M)), dtype=torch.float32, device=device
        )  # [B, M] per-graph, per-condition coupling
        c_nodes = c.index_select(0, bg.batch)  # [N, M]

        x = self.x_init + 0.01 * torch.rand(N, M, device=device)
        traj = torch.empty(N, M, T, device=device)
        for t in range(T):
            agg = scatter_sum(x.index_select(0, src), dst, N)  # [N,M] neighbour sum
            nbr = agg / deg if self.row_normalize else agg
            growth = r - self.self_decay * x + c_nodes * nbr
            x = (x + self.dt * x * growth).clamp(min=0.0, max=self.x_clamp)
            if self.noise:
                x = (x + self.noise * torch.randn(N, M, device=device)).clamp_min(0.0)
            traj[:, :, t] = x
        if not torch.isfinite(traj).all():
            raise FloatingPointError(
                "GLV trajectory diverged (non-finite). Reduce dt / coupling_range or "
                "keep coupling < self_decay (stable side of the bifurcation)."
            )
        a, converged = self._steady_state(traj, bg.batch, B)
        extra = {"coupling": c, "converged": converged}
        return a, (traj if self.store_traj else None), extra

    def _steady_state(self, traj: Tensor, batch: Tensor, B: int):
        """Robust steady value (median over a window) + plateau-convergence flag.

        Fixes the "last step is not robust under noise" critique: the steady value
        is the per-node/per-condition **median** over the last ``T_ss`` steps, and a
        condition is marked converged only if the window has plateaued under a
        mixed absolute-OR-relative tolerance (so a flat line at ~0 also counts).
        """
        win = traj[..., -self.T_ss:]  # [N, M, Tss]
        med = win.median(dim=-1).values  # [N, M]
        half = max(1, self.T_ss // 2)
        drift = (win[..., half:].mean(-1) - win[..., :half].mean(-1)).abs()  # [N, M]
        abs_floor = 3.0 * self.noise + 1e-3
        node_conv = (drift < abs_floor) | (drift / (med.abs() + self.eps) < self.steady_tol)
        frac = scatter_mean(node_conv.float(), batch, B)  # [B, M] fraction of converged nodes
        converged = frac >= 0.9
        return med, converged
