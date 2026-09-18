"""Controlled experiment: does the learned sentinel predictor generalize across
interaction-strength (coupling) regimes better than the transductive sentinel-mean?

The GLV equilibrium with heterogeneous growth rates is x = r + c·L·x  (L the
coupling operator), i.e. x = (I − c·L)^{-1} r — NOT a consensus.  Whether a fixed
sentinel-average tracks the global mean across couplings depends on the operator:

* ``--regime consensus`` (row-normalised, neighbour **mean**): near-consensus, hubs
  are not special, so even a biased sentinel set's mean already tracks the global
  mean across c — the analytic baseline is hard to beat (an honest negative control).
* ``--regime hub`` (un-normalised, neighbour **sum**): hubs dominate and their
  steady values scale nonlinearly with c, so a degree-top-K sentinel mean is a
  biased, coupling-dependent estimate of the global mean — it degrades out-of-range,
  and a Transformer that reads the sentinels' trajectories (inferring c from shape)
  can correct for it.

Design — a 2×2 with the sentinel set held FIXED (degree top-K) so ONLY the
estimator varies, crossed with the coupling regime (in-range vs a DISJOINT range
the predictor never trains on).  Metric = scale-aware relative error of the
estimated global steady mean over converged conditions.

Run (8×H20 available — full training):
    python3 experiments/sentinel_glv_generalization.py --regime hub       --steps 8000 --lr 1e-4
    python3 experiments/sentinel_glv_generalization.py --regime consensus  --steps 8000 --lr 1e-4
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

import graco.envs  # noqa: F401  (register sentinel env)
import graco.generators  # noqa: F401  (register sentinel generator)
from graco.models.predictors import SentinelPredictor
from graco.registries import ENVS, GENERATORS

FLOOR = 0.1  # scale-aware relative-error denominator (matches env.err_floor)

REGIMES = {
    # row_normalize, (train range), (disjoint range), timesteps, steady_window, dt
    "consensus": (True, (0.2, 0.5), (0.5, 0.8), 120, 40, 0.2),
    "hub": (False, (0.02, 0.05), (0.05, 0.08), 160, 50, 0.1),
}


def make_gen(coupling, nodes, M, seed, row_norm, T, W, dt):
    return GENERATORS.build(
        {"type": "sentinel", "dynamics": "glv", "fixed_graph": True, "graph_seed": seed,
         "num_nodes": [nodes, nodes], "base": "ba", "m": 4, "num_conditions": M,
         "timesteps": T, "steady_window": W, "coupling_range": list(coupling),
         "r_range": [0.5, 1.5], "self_decay": 1.0, "dt": dt, "noise": 0.02,
         "row_normalize": row_norm, "store_traj": True}
    )


def degree_topk_mask(bg, K):
    """Fixed sentinel set: the K highest-degree nodes in each graph."""
    mask = torch.zeros(bg.num_nodes, dtype=torch.bool, device=bg.device)
    deg = bg.degree()
    for b in range(bg.num_graphs):
        lo, hi = int(bg.ptr[b]), int(bg.ptr[b + 1])
        top = torch.topk(deg[lo:hi], min(K, hi - lo)).indices + lo
        mask[top] = True
    return mask


def random_k_mask(bg, K):
    """Vectorised random-K set per graph (assumes a fixed common size — fixed_graph)."""
    B, n = bg.num_graphs, int(bg.ptr[1] - bg.ptr[0])
    idx = torch.rand(B, n, device=bg.device).argsort(dim=1)[:, :K]  # K distinct local ids/graph
    gidx = (idx + (torch.arange(B, device=bg.device) * n).unsqueeze(1)).reshape(-1)
    mask = torch.zeros(bg.num_nodes, dtype=torch.bool, device=bg.device)
    mask[gidx] = True
    return mask


def rel_error(yhat, bg):
    return float(_rel_error_t(yhat, bg))


def abs_error(yhat, bg):
    y, conv = bg.graph_attr["y"], bg.graph_attr["converged"].to(bg.graph_attr["y"].dtype)
    return float(((yhat - y).abs() * conv).sum() / conv.sum().clamp_min(1.0))


def _rel_error_t(yhat, bg):
    y = bg.graph_attr["y"]
    conv = bg.graph_attr["converged"].to(y.dtype)
    err = (yhat - y).abs() / y.abs().clamp_min(FLOOR)  # [B, M]
    return (err * conv).sum() / conv.sum().clamp_min(1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", choices=list(REGIMES), default="hub")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--nodes", type=int, default=40)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--M", type=int, default=5)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--pool", type=int, default=128, help="pre-simulated train batches")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = torch.device(args.device)

    row_norm, TRAIN, DISJOINT, T, W, dt = REGIMES[args.regime]
    g = lambda cp: make_gen(cp, args.nodes, args.M, args.seed, row_norm, T, W, dt)  # noqa: E731
    train_gen, test_in, test_out = g(TRAIN), g(TRAIN), g(DISJOINT)
    env = ENVS.build({"type": "sentinel", "num_conditions": args.M, "budget": args.K,
                      "err_floor": FLOOR})

    # Pre-simulate the training pool ONCE (dynamics is the bottleneck), then train the
    # predictor over it — a fixed TRAIN set of couplings drawn from the train range,
    # with fresh sentinel sets each step; the disjoint TEST set is held out entirely.
    print(f"[{args.regime}] device={dev} | pre-simulating {args.pool} train batches "
          f"(B={args.batch}) coupling∈{TRAIN} …", flush=True)
    pool = [train_gen.sample(args.batch, device=dev, rng=rng) for _ in range(args.pool)]
    test_in_bg = test_in.sample(256, device=dev, rng=np.random.default_rng(args.seed + 777))
    test_out_bg = test_out.sample(256, device=dev, rng=np.random.default_rng(args.seed + 778))
    S_deg = degree_topk_mask(pool[0], args.K)  # fixed_graph => same top-K nodes every batch

    pred = SentinelPredictor(num_conditions=args.M, d=64, n_heads=4, tf_layers=2).to(dev)
    opt = torch.optim.Adam(pred.parameters(), lr=args.lr)

    print(f"[{args.regime}] training predictor: {args.steps} steps, lr={args.lr}, "
          f"disjoint∈{DISJOINT} (never trained on)", flush=True)
    for step in range(args.steps):
        bg = pool[int(rng.integers(len(pool)))]
        # mix random-K sets with the fixed degree-top-K eval set so out-of-range eval
        # measures COUPLING extrapolation, not set-generalization.
        S = S_deg if rng.random() < 0.3 else random_k_mask(bg, args.K)
        pred.train()
        loss = _rel_error_t(pred(bg, S), bg)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(pred.parameters(), 10.0)
        opt.step()
        if step % 1000 == 0 or step == args.steps - 1:
            print(f"  step {step:5d}  train_loss={float(loss):.4f}", flush=True)

    pred.eval()
    print(f"\n[{args.regime}] 2×2 controlled comparison — global steady-mean error  rel (abs)")
    print(f"  (fixed degree-top-{args.K} sentinels; predictor NEVER saw the disjoint range)\n")
    header = f"{'REGIME':<26}{'analytic mean':>20}{'learned pred':>20}"
    print(header + "\n" + "-" * len(header))
    res = {}
    for label, bg in [(f"in-range  {TRAIN}", test_in_bg), (f"disjoint  {DISJOINT}", test_out_bg)]:
        with torch.no_grad():
            S = degree_topk_mask(bg, args.K)
            ya, yl = env._estimate_analytic(bg, S), pred(bg, S)
            res[label] = (rel_error(ya, bg), rel_error(yl, bg))
            aa, al = abs_error(ya, bg), abs_error(yl, bg)
        print(f"{label:<26}{res[label][0]:>11.4f} ({aa:.3f}){res[label][1]:>11.4f} ({al:.3f})")

    (a_in, l_in), (a_out, l_out) = list(res.values())
    print("\nVerdict:")
    print(f"  analytic in→out : {a_in:.4f} → {a_out:.4f}  (×{a_out / max(a_in, 1e-9):.2f})")
    print(f"  learned  in→out : {l_in:.4f} → {l_out:.4f}  (×{l_out / max(l_in, 1e-9):.2f})")
    if l_out < a_out:
        print(f"  ✓ On the DISJOINT range the learned predictor beats the sentinel-mean "
              f"({l_out:.4f} < {a_out:.4f}) — it extrapolates across coupling regimes.")
    else:
        print(f"  ✗ No extrapolation gain here ({l_out:.4f} ≥ {a_out:.4f}): the transductive "
              f"mean already suffices in this regime.")


if __name__ == "__main__":
    main()
