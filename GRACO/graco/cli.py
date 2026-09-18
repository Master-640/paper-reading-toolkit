"""GRACO command-line interface.

Usage::

    graco train  --config finder [key=value ...]   # or a path: --config path/to.yaml
    graco eval   --config <cfg> --ckpt runs/exp/best.pt
    graco generate --config <cfg> --out data/graphs.pt --num 100
    graco benchmark --config <cfg> [--ckpt runs/exp/best.pt] [--num 128] [--data <path|name>]
    graco datasets  [--download <name>]
    graco list    [envs|encoders|algos|generators|heads|buffers|heuristics|configs]

``--config`` accepts a path **or** a short name resolved under ``configs/``
(e.g. ``graco train -c maxcut_ppo``).  Any trailing ``key=value`` arguments are
dotted OmegaConf overrides, e.g. ``algo.n_step=5 trainer.iterations=5000
model.encoder.type=gat``.

``--data`` points training/benchmarking at real graphs: a local path (file, glob
or directory) or a built-in dataset name (see ``graco datasets``); it overrides
the config's ``generator`` with the file-dataset loader.
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from graco.utils.config import load_config, to_container


def _split_overrides(extra: List[str]) -> List[str]:
    return [a for a in extra if "=" in a]


def _apply_data(cfg, args, default_mode: str):
    """If ``--data`` was given, replace ``cfg.generator`` with the file loader."""
    data = getattr(args, "data", None)
    if not data:
        return cfg
    from omegaconf import OmegaConf

    from graco.data.benchmarks import DATASETS, fetch

    if data in DATASETS:
        path, fmt = fetch(data), DATASETS[data]["fmt"]
    else:
        path, fmt = data, (getattr(args, "data_fmt", None) or "auto")
    mode = getattr(args, "data_mode", None) or default_mode
    cfg.generator = OmegaConf.create({"type": "dataset", "path": path, "fmt": fmt, "mode": mode})
    return cfg


def cmd_train(args, overrides: List[str]) -> None:
    cfg = load_config(args.config, overrides=overrides)
    cfg = _apply_data(cfg, args, default_mode="random")
    from graco.trainers.trainer import run_training

    run_training(cfg)


def cmd_eval(args, overrides: List[str]) -> None:
    cfg = load_config(args.config, overrides=overrides)
    from graco.trainers.trainer import Trainer

    trainer = Trainer(cfg)
    if args.ckpt:
        trainer.load_checkpoint(args.ckpt)
    metrics = trainer.evaluate()
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}")


def cmd_generate(args, overrides: List[str]) -> None:
    import torch

    cfg = load_config(args.config, overrides=overrides)
    import graco.generators  # noqa: F401
    from graco.registries import GENERATORS

    gen = GENERATORS.build(to_container(cfg.generator))
    bg = gen.sample(args.num, device="cpu")
    torch.save(
        {"edge_index": bg.edge_index, "edge_weight": bg.edge_weight, "ptr": bg.ptr, "batch": bg.batch},
        args.out,
    )
    print(f"saved {bg.num_graphs} graphs ({bg.num_nodes} nodes, {bg.num_edges} edges) -> {args.out}")


def cmd_benchmark(args, overrides: List[str]) -> None:
    cfg = load_config(args.config, overrides=overrides)
    cfg = _apply_data(cfg, args, default_mode="all")
    from graco.eval.benchmark import run_benchmark

    run_benchmark(
        cfg, ckpt=args.ckpt, num=args.num, methods=args.methods,
        decode_strategy=args.decode, samples=args.samples, temperature=args.temperature,
    )


def cmd_datasets(args, overrides: List[str]) -> None:
    from graco.data.benchmarks import DATASETS, fetch

    if args.download:
        print(f"downloaded {args.download} -> {fetch(args.download)}")
        return
    for name, d in DATASETS.items():
        print(f"{name:22s} [{d.get('problem', '-'):11s}] {d.get('note', '')}  ({d['url']})")


def cmd_list(args, overrides: List[str]) -> None:
    if args.kind == "configs":
        from graco.utils.config import list_configs

        print("[configs] " + ", ".join(list_configs()))
        return
    import graco.algos  # noqa: F401
    import graco.baselines  # noqa: F401
    import graco.buffers  # noqa: F401
    import graco.envs  # noqa: F401
    import graco.generators  # noqa: F401
    import graco.heuristics  # noqa: F401
    import graco.models  # noqa: F401
    from graco.heuristics.base import HEURISTICS
    from graco.registries import ALGOS, BUFFERS, ENCODERS, ENVS, GENERATORS, HEADS

    regs = {
        "envs": ENVS,
        "encoders": ENCODERS,
        "algos": ALGOS,
        "generators": GENERATORS,
        "heads": HEADS,
        "buffers": BUFFERS,
        "heuristics": HEURISTICS,
    }
    which = [args.kind] if args.kind else list(regs)
    for k in which:
        print(f"[{k}] {', '.join(regs[k].keys())}")


def build_parser() -> argparse.ArgumentParser:
    from graco import __version__

    p = argparse.ArgumentParser(prog="graco", description="Deep RL for graph combinatorial optimization")
    p.add_argument("--version", "-V", action="version", version=f"graco {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    pt = sub.add_parser("train", help="train a model")
    pt.add_argument("--config", "-c", required=True)
    pt.add_argument("--data", default=None, help="real-graph path / glob / dir or built-in dataset name")
    pt.add_argument("--data-fmt", default=None)
    pt.add_argument("--data-mode", default=None, choices=[None, "random", "all"])
    pt.set_defaults(func=cmd_train)

    pe = sub.add_parser("eval", help="evaluate a checkpoint")
    pe.add_argument("--config", "-c", required=True)
    pe.add_argument("--ckpt", default=None)
    pe.set_defaults(func=cmd_eval)

    pg = sub.add_parser("generate", help="generate a dataset of graphs")
    pg.add_argument("--config", "-c", required=True)
    pg.add_argument("--out", "-o", required=True)
    pg.add_argument("--num", "-n", type=int, default=100)
    pg.set_defaults(func=cmd_generate)

    pb = sub.add_parser("benchmark", help="benchmark heuristics (and optionally a checkpoint)")
    pb.add_argument("--config", "-c", required=True)
    pb.add_argument("--ckpt", default=None)
    pb.add_argument("--num", "-n", type=int, default=None)
    pb.add_argument("--methods", nargs="*", default=None)
    pb.add_argument("--decode", default="greedy", choices=["greedy", "sample", "multistart"],
                    help="inference decoding strategy for the checkpoint policy")
    pb.add_argument("--samples", type=int, default=8, help="rollouts per graph for sample/multistart")
    pb.add_argument("--temperature", type=float, default=1.0)
    pb.add_argument("--data", default=None, help="real-graph path / glob / dir or built-in dataset name")
    pb.add_argument("--data-fmt", default=None)
    pb.add_argument("--data-mode", default=None, choices=[None, "random", "all"])
    pb.set_defaults(func=cmd_benchmark)

    pd = sub.add_parser("datasets", help="list / download built-in real datasets")
    pd.add_argument("--download", default=None, help="dataset name to download")
    pd.set_defaults(func=cmd_datasets)

    pl = sub.add_parser("list", help="list registered components (or configs)")
    pl.add_argument("kind", nargs="?", default=None,
                    help="envs|encoders|algos|generators|heads|buffers|heuristics|configs")
    pl.set_defaults(func=cmd_list)
    return p


def main(argv: Optional[List[str]] = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args, extra = parser.parse_known_args(argv)
    overrides = _split_overrides(extra)
    args.func(args, overrides)


if __name__ == "__main__":
    main()
