"""Deep Q-Network agents: DQN (+ Double, Dueling, n-step, PER) and QR-DQN.

Faithful to the DIRAC/FINDER lineage: S2V-style encoder, target network,
n-step returns built at episode end from stored action sequences, uniform or
prioritized replay, epsilon-greedy exploration with a linear schedule.  All
per-node scoring and action selection is vectorized via segment ops; a learning
batch is K single-graph transitions collated into one disjoint-union graph.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from graco.algos.base import Algorithm
from graco.buffers.base import to_single_graphs
from graco.buffers.replay import UniformNStepReplay, make_nstep_from_snapshots
from graco.envs.base import Observation, StepResult, VectorizedEnv
from graco.models.build import build_q_policy
from graco.registries import ALGOS, BUFFERS
from graco.utils.accel import build_optimizer
from graco.utils.scatter import scatter_max, scatter_sum
from graco.utils.segment_ops import NEG_INF, segment_argmax, segment_gumbel_sample


@ALGOS.register("dqn", aliases=["doubledqn", "dueling_dqn", "s2v_dqn"])
class DQNAlgorithm(Algorithm):
    learns_online = True

    def __init__(
        self,
        env: VectorizedEnv,
        model: Optional[dict] = None,
        buffer: Optional[Any] = None,
        device: Any = "cpu",
        gamma: float = 1.0,
        n_step: int = 3,
        batch_size: int = 64,
        lr: float = 1e-4,
        double: bool = True,
        dueling: bool = False,
        huber: bool = True,
        target_update_period: int = 1000,
        learn_start: int = 2000,
        learn_freq: int = 1,
        updates_per_step: int = 1,
        grad_clip: Optional[float] = 10.0,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay_steps: int = 50000,
        bootstrap_clip_min: Optional[float] = None,
        weight_decay: float = 0.0,
        **kwargs,
    ):
        super().__init__(env, device=device, gamma=gamma)
        if kwargs:
            import warnings

            warnings.warn(f"{type(self).__name__} ignoring config keys: {sorted(kwargs)}")
        self.env_class = type(env)
        self.state_spec = env.state_spec
        self.n_step = n_step
        self.batch_size = batch_size
        self.double = double
        self.huber = huber
        self.target_update_period = target_update_period
        self.learn_start = learn_start
        self.learn_freq = learn_freq
        self.updates_per_step = updates_per_step
        self.grad_clip = grad_clip
        self.eps_start, self.eps_end, self.eps_decay_steps = eps_start, eps_end, eps_decay_steps
        self.bootstrap_clip_min = bootstrap_clip_min

        model = dict(model or {})
        head_default = "dueling_q" if dueling else "q_node"
        self.policy = build_q_policy(model, env, head_default=head_default).to(self.device)
        self.target = copy.deepcopy(self.policy).to(self.device)
        self.target.eval()
        self.optimizer = build_optimizer(
            self.policy.parameters(), lr=lr, weight_decay=weight_decay
        )

        if buffer is None:
            buffer = UniformNStepReplay(capacity=50000)
        elif isinstance(buffer, dict):
            buffer = BUFFERS.build(dict(buffer))
        self.buffer = buffer
        self.buffer.state_spec = self.state_spec

        self._learn_steps = 0
        self._env_steps = 0

    # ----------------------------------------------------------- exploration
    def epsilon(self) -> float:
        # driven by GRADIENT-UPDATE count (like FINDER's Fit iterations), so the
        # schedule is invariant to num_envs / learn_freq / episode length.
        frac = max(0.0, (self.eps_decay_steps - self._learn_steps) / max(1, self.eps_decay_steps))
        return self.eps_end + (self.eps_start - self.eps_end) * frac

    # ------------------------------------------------------------- lifecycle
    def on_episode_start(self, env: VectorizedEnv) -> None:
        self._graphs = to_single_graphs(env.graph)
        self._ptr = env.graph.ptr.detach().cpu()
        b = env.num_envs
        self._traj_states = [[] for _ in range(b)]  # actual s_t snapshots (exact for non-monotone)
        self._traj_actions = [[] for _ in range(b)]
        self._traj_rewards = [[] for _ in range(b)]

    # --------------------------------------------------------------- scoring
    def _node_scores(self, policy, graph, mask) -> Tensor:
        out = policy(graph, mask=mask) if policy.needs_mask else policy(graph)
        return out  # scalar-per-node [N] for plain/dueling

    @torch.no_grad()
    def act(self, obs: Observation, explore: bool = True) -> Tensor:
        self.policy.eval()
        # snapshot the pre-step selected mask (s_t) for exact replay reconstruction
        if self.env is not None and self.env.state:
            self._pre_sel = self.env.state["selected"].detach().cpu().clone()
        g = obs.graph
        q = self._node_scores(self.policy, g, obs.action_mask)
        greedy = segment_argmax(q, g.batch, g.num_graphs, obs.action_mask)
        if not explore:
            return greedy
        eps = self.epsilon()
        rand_a = segment_gumbel_sample(
            torch.zeros_like(q), g.batch, g.num_graphs, obs.action_mask
        )
        use_rand = torch.rand(g.num_graphs, device=self.device) < eps
        return torch.where(use_rand, rand_a, greedy)

    @torch.no_grad()
    def score(self, obs: Observation) -> Tensor:
        """Per-node scores ``[N]`` for decoding (Q-values; softmax = a policy)."""
        self.policy.eval()
        return self._node_scores(self.policy, obs.graph, obs.action_mask)

    def observe(
        self, env: VectorizedEnv, obs: Observation, action: Tensor, step: StepResult
    ) -> None:
        active = (~obs.done) & (action >= 0)
        act_cpu = action.detach().cpu()
        rew_cpu = step.reward.detach().cpu()
        ptr = self._ptr
        pre_sel = self._pre_sel  # s_t snapshot from act()
        for i in torch.nonzero(active.cpu(), as_tuple=False).flatten().tolist():
            lo, hi = int(ptr[i]), int(ptr[i + 1])
            self._traj_states[i].append(pre_sel[lo:hi].clone())
            self._traj_actions[i].append(int(act_cpu[i] - ptr[i]))
            self._traj_rewards[i].append(float(rew_cpu[i]))

    def after_step(self) -> Optional[Dict[str, float]]:
        self._env_steps += 1
        if len(self.buffer) < self.learn_start or self._env_steps % self.learn_freq != 0:
            return None
        metrics = None
        for _ in range(self.updates_per_step):
            metrics = self._learn()
        return metrics

    def after_episode(self, env: VectorizedEnv) -> Optional[Dict[str, float]]:
        final_sel = env.state["selected"].detach().cpu()
        ptr = self._ptr
        for i in range(len(self._graphs)):
            lo, hi = int(ptr[i]), int(ptr[i + 1])
            trans = make_nstep_from_snapshots(
                self._graphs[i], self._traj_states[i], self._traj_actions[i],
                self._traj_rewards[i], final_sel[lo:hi], self.n_step, self.gamma,
            )
            if trans:
                self.buffer.add_many(trans)
        return {"buffer_size": float(len(self.buffer))}

    # ------------------------------------------------------------- learning
    def _learn(self) -> Dict[str, float]:
        self.policy.train()
        if hasattr(self.buffer, "anneal_beta"):
            self.buffer.anneal_beta(self.progress)
        batch = self.buffer.sample(self.batch_size, self.device)
        graph, state, next_state = batch.graph, batch.state, batch.next_state

        obs_t = self.env_class.obs_graph(graph, state)
        mask_t = self.env_class.valid_mask(graph, state)
        q_all = self._node_scores(self.policy, obs_t, mask_t)  # [N]
        qa = q_all.index_select(0, batch.action)  # [K]

        with torch.no_grad():
            q_next = self._target_value(graph, next_state)  # [K]
            disc = self.gamma ** self.n_step
            bootstrap = disc * q_next * (~batch.terminal).float()
            if self.bootstrap_clip_min is not None:
                bootstrap = bootstrap.clamp_min(self.bootstrap_clip_min)
            target_q = batch.reward + bootstrap

        td = target_q - qa
        if self.huber:
            per = F.smooth_l1_loss(qa, target_q, reduction="none")
        else:
            per = td.pow(2)
        weights = batch.is_weights if batch.is_weights is not None else 1.0
        loss = (weights * per).mean()
        loss = loss + self._aux_loss(obs_t, mask_t)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip)
        self.optimizer.step()

        if batch.indices is not None:
            self.buffer.update_priorities(batch.indices, td)

        self._learn_steps += 1
        if self._learn_steps % self.target_update_period == 0:
            self.target.load_state_dict(self.policy.state_dict())

        return {
            "loss": float(loss.detach()),
            "q_mean": float(qa.mean().detach()),
            "epsilon": self.epsilon(),
            "target_mean": float(target_q.mean().detach()),
        }

    def _aux_loss(self, obs_t, mask_t) -> Tensor:
        """Auxiliary loss added to the DQN loss (0 by default).

        Override to add, e.g., FINDER's graph-reconstruction regularizer.
        """
        return torch.zeros((), device=self.device)

    def _target_value(self, graph, next_state) -> Tensor:
        obs_tn = self.env_class.obs_graph(graph, next_state)
        mask_tn = self.env_class.valid_mask(graph, next_state)
        b = graph.num_graphs
        has_valid = scatter_sum(mask_tn.float(), graph.batch, b) > 0
        q_target = self._node_scores(self.target, obs_tn, mask_tn)  # [N]
        if self.double:
            q_online = self._node_scores(self.policy, obs_tn, mask_tn)
            a_star = segment_argmax(q_online, graph.batch, b, mask_tn)  # [B] global (-1 if none)
            safe = a_star.clamp_min(0)
            q_next = q_target.index_select(0, safe)
        else:
            masked = q_target.masked_fill(~mask_tn, NEG_INF)
            q_next = scatter_max(masked, graph.batch, b, fill_value=0.0)
        return torch.where(has_valid, q_next, torch.zeros_like(q_next))

    # -------------------------------------------------------- checkpointing
    def state_dict(self) -> Dict[str, Any]:
        return {
            "policy": self.policy.state_dict(),
            "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "learn_steps": self._learn_steps,
        }

    def load_state_dict(self, sd: Dict[str, Any]) -> None:
        self.policy.load_state_dict(sd["policy"])
        self.target.load_state_dict(sd["target"])
        self.optimizer.load_state_dict(sd["optimizer"])
        self._learn_steps = sd.get("learn_steps", 0)

    def parameters(self):
        return self.policy.parameters()


@ALGOS.register("qrdqn", aliases=["qr_dqn", "quantile_dqn"])
class QRDQNAlgorithm(DQNAlgorithm):
    """Distributional QR-DQN: quantile regression over returns."""

    def __init__(self, env, num_quantiles: int = 51, kappa: float = 1.0, model=None, **kw):
        model = dict(model or {})
        model.setdefault("head", {})
        model["head"] = dict(model["head"])
        model["head"].setdefault("type", "quantile_q")
        model["head"]["num_quantiles"] = num_quantiles
        self.num_quantiles = num_quantiles
        self.kappa = kappa
        super().__init__(env, model=model, **kw)
        taus = (torch.arange(num_quantiles, device=self.device) + 0.5) / num_quantiles
        self.register_taus = taus  # [Q]

    def _node_scores(self, policy, graph, mask) -> Tensor:
        # expected value over quantiles -> scalar per node (used for acting/argmax)
        q = policy(graph)  # [N, Q]
        return q.mean(dim=-1)

    def _quantiles(self, policy, graph) -> Tensor:
        return policy(graph)  # [N, Q]

    def _learn(self) -> Dict[str, float]:
        self.policy.train()
        if hasattr(self.buffer, "anneal_beta"):
            self.buffer.anneal_beta(self.progress)
        batch = self.buffer.sample(self.batch_size, self.device)
        graph, state, next_state = batch.graph, batch.state, batch.next_state
        b = graph.num_graphs
        Q = self.num_quantiles

        obs_t = self.env_class.obs_graph(graph, state)
        theta_all = self._quantiles(self.policy, obs_t)  # [N, Q]
        theta_sa = theta_all.index_select(0, batch.action)  # [K, Q]

        with torch.no_grad():
            obs_tn = self.env_class.obs_graph(graph, next_state)
            mask_tn = self.env_class.valid_mask(graph, next_state)
            has_valid = scatter_sum(mask_tn.float(), graph.batch, b) > 0
            # greedy next action by expected value (double if enabled)
            scorer = self.policy if self.double else self.target
            q_next_mean = self._node_scores(scorer, obs_tn, mask_tn)
            a_star = segment_argmax(q_next_mean, graph.batch, b, mask_tn).clamp_min(0)
            theta_next = self._quantiles(self.target, obs_tn).index_select(0, a_star)  # [K, Q]
            theta_next = theta_next * has_valid.float().unsqueeze(-1)
            disc = self.gamma ** self.n_step
            target = batch.reward.unsqueeze(-1) + disc * (~batch.terminal).float().unsqueeze(-1) * theta_next
            # [K, Q]

        # quantile Huber loss over the (Q x Q) pairwise TD matrix
        taus = self.register_taus.view(1, Q, 1)
        td = target.unsqueeze(1) - theta_sa.unsqueeze(2)  # [K, Q(theta), Q(target)]
        huber = torch.where(
            td.abs() <= self.kappa,
            0.5 * td.pow(2),
            self.kappa * (td.abs() - 0.5 * self.kappa),
        )
        rho = (taus - (td.detach() < 0).float()).abs() * huber / self.kappa
        loss_per = rho.sum(dim=2).mean(dim=1)  # [K]
        weights = batch.is_weights if batch.is_weights is not None else 1.0
        loss = (weights * loss_per).mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip)
        self.optimizer.step()
        if batch.indices is not None:
            self.buffer.update_priorities(batch.indices, loss_per.detach())
        self._learn_steps += 1
        if self._learn_steps % self.target_update_period == 0:
            self.target.load_state_dict(self.policy.state_dict())
        return {"loss": float(loss.detach()), "epsilon": self.epsilon()}
