"""Noisy networks for exploration (Fortunato et al., 2018).

:class:`NoisyLinear` is a drop-in replacement for ``nn.Linear`` whose weights and
biases carry learnable Gaussian noise (factorized variant), so exploration comes
from *parameter* noise rather than an epsilon schedule.  :class:`NoisyQHead`
mirrors :class:`~graco.models.heads.QHead` but builds its scoring MLP out of
:class:`NoisyLinear` layers; it is usable with ``epsilon=0``.  Self-contained:
this module does not modify ``heads.py`` (it only reuses the node-input plumbing
of ``_NodeScoreHead``).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from graco.data.batch import BatchedGraph
from graco.models.base import get_activation
from graco.models.heads import _NodeScoreHead
from graco.registries import HEADS


class NoisyLinear(nn.Module):
    """Factorized Gaussian NoisyNet linear layer (drop-in for ``nn.Linear``).

    ``y = (μ_w + σ_w ⊙ ε_w) x + (μ_b + σ_b ⊙ ε_b)`` in training mode, where the
    factorized noise is ``ε_w = f(ε_out) f(ε_in)^T`` and ``ε_b = f(ε_out)`` with
    ``f(x) = sign(x) √|x|``.  In eval mode only the means ``μ`` are used.
    """

    def __init__(self, in_features: int, out_features: int, sigma0: float = 0.5):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.sigma0 = float(sigma0)

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self) -> None:
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma0 / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma0 / math.sqrt(self.out_features))

    def _scale_noise(self, size: int) -> Tensor:
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign() * x.abs().sqrt()

    def reset_noise(self) -> None:
        eps_in = self._scale_noise(self.in_features)  # [in]
        eps_out = self._scale_noise(self.out_features)  # [out]
        self.weight_epsilon.copy_(eps_out.unsqueeze(1) * eps_in.unsqueeze(0))
        self.bias_epsilon.copy_(eps_out)

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight, bias = self.weight_mu, self.bias_mu
        return F.linear(x, weight, bias)


class NoisyMLP(nn.Module):
    """MLP mirroring :class:`~graco.models.base.MLP` but with NoisyLinear layers."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 2,
        act: str = "relu",
        sigma0: float = 0.5,
    ):
        super().__init__()
        self.act = get_activation(act)
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        self.layers = nn.ModuleList(
            NoisyLinear(dims[i], dims[i + 1], sigma0) for i in range(len(dims) - 1)
        )

    def forward(self, x: Tensor) -> Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.act(x)
        return x

    def reset_noise(self) -> None:
        for layer in self.layers:
            layer.reset_noise()


@HEADS.register("noisy_q")
class NoisyQHead(_NodeScoreHead):
    """Scalar Q-value per node ``[N]`` with NoisyNet parameter-noise exploration."""

    def __init__(
        self,
        embed_dim: int,
        sigma0: float = 0.5,
        num_layers: int = 2,
        hidden_dim: Optional[int] = None,
        **kw,
    ):
        super().__init__(embed_dim, out_dim=1, hidden_dim=hidden_dim, num_layers=num_layers, **kw)
        in_dim = self.mlp.layers[0].in_features
        hid = hidden_dim or embed_dim
        # replace the plain MLP with a noisy one of matching dimensions
        self.mlp = NoisyMLP(in_dim, hid, 1, num_layers=num_layers, sigma0=sigma0)

    def forward(self, h: Tensor, graph: BatchedGraph, aux: Optional[Tensor] = None) -> Tensor:
        return self.mlp(self.node_input(h, graph, aux)).squeeze(-1)

    def reset_noise(self) -> None:
        self.mlp.reset_noise()
