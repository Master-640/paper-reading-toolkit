"""Micro + end-to-end speed benchmark for GRACO.

Times the hot scatter reductions and a full train iteration, comparing the
default (eager) path against the accelerated one (TF32 / fused Adam, and — when
requested — ``torch.compile``).  Prints a small table so speedups are measured,
not assumed.

    python benchmarks/bench.py                 # ops + train-iter, cpu/gpu auto
    python benchmarks/bench.py --device cuda --compile
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

import graco
import graco.algos  # noqa: F401  (register)
import graco.envs  # noqa: F401
import graco.generators  # noqa: F401
import graco.models  # noqa: F401
from graco.registries import ALGOS, ENVS, GENERATORS
from graco.utils import accel
from graco.utils.scatter import scatter_max, scatter_sum


def _time(fn, iters, warmup, sync):
    for _ in range(warmup):
        fn()
    if sync:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if sync:
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3  # ms/iter


def bench_ops(device, n=200_000, e=2_000_000, f=64, iters=50):
    dev = torch.device(device)
    sync = dev.type == "cuda"
    src = torch.randn(e, f, device=dev)
    idx = torch.randint(0, n, (e,), device=dev)
    print(f"\n[scatter ops] N={n} E={e} F={f}")
    for name, fn in [("scatter_sum", lambda: scatter_sum(src, idx, n)),
                     ("scatter_max", lambda: scatter_max(src, idx, n))]:
        print(f"  {name:14s} {_time(fn, iters, 5, sync):8.3f} ms/call")


def bench_train_iter(device, compile_policy=False, iters=20):
    dev = torch.device(device)
    sync = dev.type == "cuda"
    env = ENVS.build({"type": "maxcut"})
    gen = GENERATORS.build({"type": "barabasi_albert", "num_nodes": [200, 200], "m": 4})
    algo = ALGOS.build(
        {"type": "dqn", "batch_size": 64, "learn_start": 64, "n_step": 3, "dueling": True,
         "model": {"encoder": {"type": "graphsage", "hidden_dim": 128, "num_layers": 3},
                   "head": {"type": "dueling_q"}}},
        env=env, device=device,
    )
    if compile_policy:
        # compile the ENCODER submodule (not the policy, which flips train/eval each step)
        algo.policy.encoder = accel.maybe_compile(algo.policy.encoder, enabled=True)
    rng = np.random.default_rng(0)

    def one_iter():
        g = gen.sample(32, device=dev, rng=rng)
        obs = env.reset(g)
        algo.on_episode_start(env)
        while not obs.done.all():
            a = algo.act(obs)
            step = env.step(a)
            algo.observe(env, obs, a, step)
            algo.after_step()
            obs = step.obs
        algo.after_episode(env)

    print(f"\n[train iter] maxcut BA-200 x32  compile={compile_policy}")
    print(f"  {_time(one_iter, iters, 3, sync):8.1f} ms/iter")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--accelerate", action="store_true", help="enable TF32/matmul precision")
    ap.add_argument("--compile", action="store_true", help="torch.compile the policy")
    args = ap.parse_args()
    if args.accelerate:
        graco.accelerate()
        print("acceleration: TF32 / matmul precision enabled")
    print(f"device={args.device}  torch={torch.__version__}")
    bench_ops(args.device, n=20_000, e=200_000) if args.device == "cpu" else bench_ops(args.device)
    bench_train_iter(args.device, compile_policy=args.compile)


if __name__ == "__main__":
    main()
