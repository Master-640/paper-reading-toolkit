"""Build models (encoder + heads + policy) from config, wired to an env.

Input/edge feature dimensions are read from the environment, so YAML configs
need only name the encoder/head and their hyper-parameters — never the input
dims.  Custom classes drop in via the registry's dotted-path support (``type:
my_pkg.MyEncoder``).
"""

from __future__ import annotations

from typing import Any

from graco.envs.base import VectorizedEnv
from graco.models.heads import DuelingQHead
from graco.models.policy import ActorCriticPolicy, QPolicy
from graco.registries import ENCODERS, HEADS


def _as_dict(cfg: Any) -> dict:
    if cfg is None:
        return {}
    try:
        from omegaconf import DictConfig, OmegaConf

        if isinstance(cfg, DictConfig):
            return dict(OmegaConf.to_container(cfg, resolve=True))
    except Exception:
        pass
    return dict(cfg)


def build_encoder(cfg: Any, in_dim: int, edge_dim: int, default_type: str = "s2v"):
    cfg = _as_dict(cfg)
    cfg.setdefault("type", default_type)
    return ENCODERS.build(cfg, in_dim=in_dim, edge_dim=edge_dim)


def build_head(cfg: Any, embed_dim: int, default_type: str = "q_node", **extra):
    cfg = _as_dict(cfg)
    cfg.setdefault("type", default_type)
    return HEADS.build(cfg, embed_dim=embed_dim, **extra)


def build_q_policy(model_cfg: Any, env: VectorizedEnv, head_default: str = "q_node") -> QPolicy:
    model_cfg = _as_dict(model_cfg)
    enc = build_encoder(model_cfg.get("encoder"), env.node_feature_dim, env.edge_feature_dim)
    # forward the label count to k-label heads (additive: absent on binary envs)
    extra = {}
    num_labels = getattr(env, "num_labels", None)
    if num_labels is not None:
        extra["num_labels"] = int(num_labels)
    head = build_head(model_cfg.get("head"), enc.out_dim, default_type=head_default, **extra)
    return QPolicy(enc, head, needs_mask=isinstance(head, DuelingQHead))


def build_actor_critic_policy(
    model_cfg: Any, env: VectorizedEnv, shared_encoder: bool = True
) -> ActorCriticPolicy:
    model_cfg = _as_dict(model_cfg)
    enc = build_encoder(model_cfg.get("encoder"), env.node_feature_dim, env.edge_feature_dim)
    actor = build_head(model_cfg.get("actor"), enc.out_dim, default_type="actor")
    critic = build_head(model_cfg.get("critic"), enc.out_dim, default_type="critic")
    critic_enc = (
        None
        if shared_encoder
        else build_encoder(model_cfg.get("encoder"), env.node_feature_dim, env.edge_feature_dim)
    )
    return ActorCriticPolicy(enc, actor, critic, critic_enc)
