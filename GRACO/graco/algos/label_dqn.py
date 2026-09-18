"""Label-DQN: value-based RL over a joint (node, colour) action space.

Drives the k-label constructive-labeling envs (:mod:`graco.envs.label` — max-k-cut,
graph coloring).  The Q head emits ``[N, k]``; this algorithm flattens it to a
per-graph ``N·k`` action space (``flat = node·k + colour``) so the existing
``segment_argmax`` / ``segment_gumbel_sample`` machinery selects a joint
``(node, colour)`` action.  Everything else — epsilon schedule, n-step returns,
target network, PER — is inherited from :class:`~graco.algos.dqn.DQNAlgorithm`.

Because labeling is monotone (each node coloured once), the chosen colour is
recovered exactly from ``next_state['label']`` at learning time, so the replay
transition only stores the (local) node id — the binary path's replay format and
collation are reused verbatim.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from graco.algos.dqn import DQNAlgorithm
from graco.buffers.base import Transition
from graco.envs.base import Observation, StepResult, VectorizedEnv
from graco.registries import ALGOS
from graco.utils.scatter import scatter_max, scatter_sum
from graco.utils.segment_ops import NEG_INF, segment_argmax, segment_gumbel_sample


def make_nstep_label_from_snapshots(graph, states, actions, rewards, final_state, n_step, gamma):
    """n-step transitions with a per-node ``label`` state (mirror of the binary one)."""
    L = len(actions)
    if L == 0:
        return []
    trans: List[Transition] = []
    for t in range(L):
        end = min(t + n_step, L)
        r, disc = 0.0, 1.0
        for j in range(t, end):
            r += disc * rewards[j]
            disc *= gamma
        terminal = (t + n_step) >= L
        next_lab = states[end] if end < L else final_state
        trans.append(
            Transition(
                graph=graph,
                state={"label": states[t]},
                action=int(actions[t]),
                reward=float(r),
                next_state={"label": next_lab},
                terminal=bool(terminal),
            )
        )
    return trans


@ALGOS.register("label_dqn", aliases=["labeldqn", "kcut_dqn"])
class LabelDQNAlgorithm(DQNAlgorithm):
    def __init__(self, env: VectorizedEnv, model: Optional[dict] = None, **kw):
        model = dict(model or {})
        model["head"] = dict(model.get("head") or {})
        model["head"].setdefault("type", "q_label")  # force the [N,k] head
        self.k = int(env.num_labels)
        kw.setdefault("dueling", False)  # dueling head emits [N], incompatible with labels
        super().__init__(env, model=model, **kw)

    # -------------------------------------------------- flattened [N*k] scoring
    def _flat_scores(self, policy, graph, node_mask: Tensor):
        q = policy(graph)  # [N, k]
        k = q.shape[1]
        q_flat = q.reshape(-1)  # [N*k], node-major: flat = node*k + colour
        batch_flat = graph.batch.repeat_interleave(k)  # [N*k]
        mask_flat = node_mask.repeat_interleave(k)  # every colour of an unassigned node is legal
        return q_flat, batch_flat, mask_flat, k

    @torch.no_grad()
    def act(self, obs: Observation, explore: bool = True) -> Tensor:
        self.policy.eval()
        if self.env is not None and self.env.state:
            self._pre_label = self.env.state["label"].detach().cpu().clone()  # exact s_t snapshot
        g = obs.graph
        qf, bf, mf, _ = self._flat_scores(self.policy, g, obs.action_mask)
        greedy = segment_argmax(qf, bf, g.num_graphs, mf)  # [B] flat global index
        if not explore:
            return greedy
        eps = self.epsilon()
        rand_a = segment_gumbel_sample(torch.zeros_like(qf), bf, g.num_graphs, mf)
        use_rand = torch.rand(g.num_graphs, device=self.device) < eps
        return torch.where(use_rand, rand_a, greedy)

    @torch.no_grad()
    def score(self, obs: Observation) -> Tensor:
        """Per-node best-colour Q ``[N]`` (for decoders/compat; acting uses the flat head)."""
        self.policy.eval()
        return self.policy(obs.graph).max(dim=-1).values

    def observe(self, env, obs: Observation, action: Tensor, step: StepResult) -> None:
        active = (~obs.done) & (action >= 0)
        act_cpu = action.detach().cpu()
        rew_cpu = step.reward.detach().cpu()
        ptr, pre, k = self._ptr, self._pre_label, self.k
        for i in torch.nonzero(active.cpu(), as_tuple=False).flatten().tolist():
            lo, hi = int(ptr[i]), int(ptr[i + 1])
            self._traj_states[i].append(pre[lo:hi].clone())
            node_global = int(act_cpu[i]) // k
            self._traj_actions[i].append(node_global - lo)  # LOCAL node id (colour recovered at learn)
            self._traj_rewards[i].append(float(rew_cpu[i]))

    def after_episode(self, env) -> Optional[Dict[str, float]]:
        final = env.state["label"].detach().cpu()
        ptr = self._ptr
        for i in range(len(self._graphs)):
            lo, hi = int(ptr[i]), int(ptr[i + 1])
            trans = make_nstep_label_from_snapshots(
                self._graphs[i], self._traj_states[i], self._traj_actions[i],
                self._traj_rewards[i], final[lo:hi], self.n_step, self.gamma,
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
        qf, _, _, k = self._flat_scores(self.policy, obs_t, mask_t)
        node_global = batch.action  # global node id (collate added the node offset)
        color = next_state["label"].index_select(0, node_global)
        assert bool((color >= 0).all()), "chosen node must be coloured in next_state (monotone)"
        qa = qf.index_select(0, node_global * k + color)  # [K]

        with torch.no_grad():
            q_next = self._target_value(graph, next_state)
            disc = self.gamma ** self.n_step
            bootstrap = disc * q_next * (~batch.terminal).float()
            if self.bootstrap_clip_min is not None:
                bootstrap = bootstrap.clamp_min(self.bootstrap_clip_min)
            target_q = batch.reward + bootstrap

        td = target_q - qa
        per = F.smooth_l1_loss(qa, target_q, reduction="none") if self.huber else td.pow(2)
        weights = batch.is_weights if batch.is_weights is not None else 1.0
        loss = (weights * per).mean()

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
        return {"loss": float(loss.detach()), "q_mean": float(qa.mean().detach()),
                "epsilon": self.epsilon(), "target_mean": float(target_q.mean().detach())}

    def _target_value(self, graph, next_state) -> Tensor:
        obs_tn = self.env_class.obs_graph(graph, next_state)
        mask_tn = self.env_class.valid_mask(graph, next_state)
        b = graph.num_graphs
        has_valid = scatter_sum(mask_tn.float(), graph.batch, b) > 0
        qf, bf, mf, _ = self._flat_scores(self.target, obs_tn, mask_tn)
        if self.double:
            qo, _, _, _ = self._flat_scores(self.policy, obs_tn, mask_tn)
            a_star = segment_argmax(qo, bf, b, mf).clamp_min(0)
            q_next = qf.index_select(0, a_star)
        else:
            masked = qf.masked_fill(~mf, NEG_INF)
            q_next = scatter_max(masked, bf, b, fill_value=0.0)
        return torch.where(has_valid, q_next, torch.zeros_like(q_next))
