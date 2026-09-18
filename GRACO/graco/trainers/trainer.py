"""The training loop.

The :class:`Trainer` builds the environment, generator, and algorithm from a
single resolved config, then runs the unified batched-rollout loop that works
for both value-based and policy-gradient algorithms (see
:mod:`graco.algos.base`).  It owns logging, periodic evaluation against
heuristic baselines, and checkpointing (weights + optimizer + RNG + resolved
config, for exact resume).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict

import torch
from omegaconf import DictConfig

from graco.registries import ALGOS, ENVS, GENERATORS
from graco.trainers.evaluator import Evaluator
from graco.utils.accel import configure
from graco.utils.config import import_user_modules, to_container
from graco.utils.logging import Logger
from graco.utils.seeding import resolve_device, seed_everything


def _ensure_registered() -> None:
    # importing the packages runs the @register decorators
    import graco.algos  # noqa: F401
    import graco.baselines  # noqa: F401
    import graco.buffers  # noqa: F401
    import graco.envs  # noqa: F401
    import graco.generators  # noqa: F401
    import graco.models  # noqa: F401


class Trainer:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        import_user_modules(cfg.get("imports"))
        _ensure_registered()

        self.seed = int(cfg.get("seed", 0))
        seed_everything(self.seed, deterministic=bool(cfg.get("deterministic", False)))
        self.device = resolve_device(cfg.get("device", "auto"))

        # global GPU speedups (TF32 / matmul precision / cuDNN autotune); no-op on CPU
        # and skipped in deterministic mode. Override via the `accel:` config block.
        accel_cfg = to_container(cfg.get("accel", {})) or {}
        configure(deterministic=bool(cfg.get("deterministic", False)),
                  **{k: accel_cfg[k] for k in
                     ("tf32", "matmul_precision", "cudnn_benchmark", "compile", "spmm")
                     if k in accel_cfg})

        self.env = ENVS.build(to_container(cfg.env))
        self.generator = GENERATORS.build(to_container(cfg.generator))
        vg = cfg.get("val_generator", None)
        self.val_generator = GENERATORS.build(to_container(vg)) if vg else self.generator

        algo_cfg = to_container(cfg.algo)
        self.algo = ALGOS.build(algo_cfg, env=self.env, device=str(self.device))

        tcfg = cfg.trainer
        self.num_envs = int(tcfg.get("num_envs", 32))
        self.iterations = int(tcfg.get("iterations", 1000))
        self.eval_every = int(tcfg.get("eval_every", 100))
        self.log_every = int(tcfg.get("log_every", 10))
        self.checkpoint_every = int(tcfg.get("checkpoint_every", 500))
        self.grad_from_env_steps = 0
        self.log_dir = Path(tcfg.get("log_dir", "runs/default"))

        # fixed validation set (generated once for stable comparison)
        n_valid = int(tcfg.get("n_valid", 64))
        rng = __import__("numpy").random.default_rng(self.seed + 12345)
        val_graph = self.val_generator.sample(n_valid, device=self.device, rng=rng)
        self.evaluator = Evaluator(
            self.env, [val_graph], baselines=list(tcfg.get("baselines", []))
        )

        lcfg = cfg.get("logger", {}) or {}
        self.logger = Logger(
            log_dir=str(self.log_dir),
            use_tensorboard=bool(lcfg.get("tensorboard", True)),
            use_wandb=bool(lcfg.get("wandb", False)),
            wandb_project=lcfg.get("wandb_project", "graco"),
            run_name=cfg.get("run_name", None),
            config=to_container(cfg),
        )
        self.best_metric = -float("inf")

    # ------------------------------------------------------------------ train
    def train(self) -> None:
        rng = __import__("numpy").random.default_rng(self.seed)
        for it in range(self.iterations):
            t0 = time.time()
            graph = self.generator.sample(self.num_envs, device=self.device, rng=rng)
            graph = self.algo.on_sample(graph)  # POMO multi-start / augmentation hook
            obs = self.env.reset(graph)
            self.algo.on_episode_start(self.env)
            self.algo.set_progress(it / max(1, self.iterations), self.grad_from_env_steps)

            step_metrics: Dict[str, float] = {}
            steps = 0
            if self.algo.single_step:
                # one-shot algos (GRLND): select a whole removal set in a single pass
                step_metrics = self.algo.learn_step(self.env, obs) or {}
                steps = 1
                self.grad_from_env_steps += 1
                ep_metrics: Dict[str, float] = {}
            else:
                while not obs.done.all():
                    self.algo.set_progress(it / max(1, self.iterations), self.grad_from_env_steps)
                    action = self.algo.act(obs, explore=True)
                    step = self.env.step(action)
                    self.algo.observe(self.env, obs, action, step)
                    m = self.algo.after_step()
                    if m:
                        step_metrics = m
                    obs = step.obs
                    steps += 1
                    self.grad_from_env_steps += 1

                ep_metrics = self.algo.after_episode(self.env) or {}
            iter_time = time.time() - t0

            if it % self.log_every == 0:
                train_obj = self.env.objective(self.env.graph, self.env.state)
                metrics = {
                    **step_metrics,
                    **ep_metrics,
                    "train/objective": float(train_obj.float().mean()),
                    "iter_time_s": iter_time,
                    "episode_len": steps,
                }
                self.logger.log(metrics, step=it)
                self.logger.console_line(it)

            if self.eval_every and it % self.eval_every == 0 and it > 0:
                eval_metrics = self.evaluator.evaluate(self.algo)
                self.logger.log(eval_metrics, step=it)
                self.logger.console_line(it, keys=list(eval_metrics))
                # rank checkpoints by the env's primary metric (dismantling ranks
                # by return = -ANC, since its terminal objective is trivially 1)
                key = f"eval/{getattr(self.env, 'eval_metric', 'objective')}"
                score = eval_metrics.get(key, eval_metrics.get("eval/objective", 0.0))
                signed = score * (1 if self.env.maximize else -1)
                if signed > self.best_metric:
                    self.best_metric = signed
                    self.save_checkpoint("best.pt", it)

            if self.checkpoint_every and it % self.checkpoint_every == 0 and it > 0:
                self.save_checkpoint("last.pt", it)

        self.save_checkpoint("last.pt", self.iterations)
        self.logger.close()

    # ------------------------------------------------------------- checkpoint
    def save_checkpoint(self, name: str, iteration: int) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        path = self.log_dir / name
        ckpt = {
            "iteration": iteration,
            "algo": self.algo.state_dict(),
            "config": to_container(self.cfg),
            "rng": {
                "torch": torch.get_rng_state(),
                "numpy": __import__("numpy").random.get_state(),
            },
            "graco_version": __import__("graco").__version__,
        }
        tmp = path.with_suffix(".tmp")
        torch.save(ckpt, tmp)
        tmp.rename(path)

    def load_checkpoint(self, path: str) -> int:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.algo.load_state_dict(ckpt["algo"])
        return int(ckpt.get("iteration", 0))

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        return self.evaluator.evaluate(self.algo)


def run_training(cfg: DictConfig) -> Trainer:
    trainer = Trainer(cfg)
    trainer.train()
    return trainer
