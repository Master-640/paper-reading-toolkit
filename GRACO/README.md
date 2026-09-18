# GRACO

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**GRACO** — **GRA**ph **C**ombinatorial **O**ptimization — is a fully-vectorized,
YAML-configurable Deep-RL framework for combinatorial optimization on graphs
(network dismantling, spin-glass / Ising ground states, MaxCut, MVC, MIS, graph
coloring, QUBO, …).

You choose the **problem**, the **GNN encoder**, the **RL algorithm**, and the
**graph distribution** from **one YAML file** (or one line of Python) and press go.
The whole environment is a batched tensor program, so a step over hundreds of
graphs at once is a handful of GPU kernels. It reimplements the ideas of
[FINDER](https://www.nature.com/articles/s42256-020-0177-2) and
[DIRAC](https://www.nature.com/articles/s41467-023-36578-x) as a modern, GPU-native,
config-driven framework — and both ship as first-class baselines.

---

## Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.1 (CPU or CUDA build)
- NumPy, NetworkX, SciPy, OmegaConf (installed automatically)

GRACO ships its own scatter-based message passing, so `torch-geometric` is **not**
required.

## Set up the environment & install

Create an isolated environment (either works):

```bash
# --- option A: venv + pip ---
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install --upgrade pip

# --- option B: conda ---
conda create -n graco python=3.11 -y && conda activate graco
```

Then install (editable, so your edits take effect immediately):

```bash
git clone https://github.com/graco/graco.git && cd graco
pip install -e .            # core dependencies
# pip install -e ".[all]"   # optional: tensorboard, wandb, ILP solver, docs, dev tools
```

For GPU support, install a CUDA build of PyTorch first (see
<https://pytorch.org/get-started/locally/>); GRACO runs on CPU otherwise.

Verify the install:

```bash
graco --version
graco list configs          # list the ready-to-run experiment configs
graco train -c debug        # tiny CPU smoke test (a few seconds)
```

## Quickstart: train FINDER in three commands

[FINDER](https://www.nature.com/articles/s42256-020-0177-2) learns to *dismantle* a
network — remove the fewest nodes to break it apart. Train it, evaluate the saved
checkpoint, then benchmark it against classic heuristics, all on a shipped config:

```bash
graco train     -c finder                              # 1. train  -> writes runs/finder/best.pt
graco eval      -c finder --ckpt runs/finder/best.pt   # 2. evaluate the best checkpoint
graco benchmark -c finder --ckpt runs/finder/best.pt   # 3. rank the policy vs heuristics
```

`benchmark` prints a table comparing the learned policy against degree,
adaptive-degree, betweenness, collective-influence and random baselines on the same
graphs, ranked by the dismantling score.

Train **DIRAC** (spin-glass / Ising ground states on a 3-D lattice) the same way:

```bash
graco train -c dirac
```

> Add `device=cuda` for GPU, and any `key=value` to override the config, e.g.
> `graco train -c finder trainer.num_envs=128 algo.n_step=3 device=cuda`.

## Use it your way

**Command line** — a config short-name (or path) plus dotted overrides:

```bash
graco train -c dirac trainer.iterations=20000 model.encoder.type=gatv2 device=cuda
```

**Python — one line:**

```python
import graco

trainer = graco.train("finder", trainer__iterations=10000, device="cuda")
print(graco.evaluate("finder", ckpt="runs/finder/best.pt"))

graco.available("encoders")          # discover what's registered
env = graco.make_env("maxcut")       # build components straight from the registry
```

**Bring your own QUBO** — minimize `xᵀQx` over `x ∈ {0,1}ⁿ` for any matrix `Q`:

```python
import numpy as np, graco
Q = np.array([[-5., 2., 0.], [2., -3., 1.], [0., 1., -2.]])
env = graco.make_env("qubo"); env.reset(graco.qubo(Q))   # then roll out a policy
```

## How a config maps to code

A single YAML wires the whole experiment; `type:` names a registered component (or
a dotted path `type: my_pkg.MyEncoder` to your own class). Input/edge feature
dimensions are read from the environment, so you never specify them:

```yaml
env:        {type: dismantling, cost_mode: unit}                 # -> ENVS
generator:  {type: barabasi_albert, num_nodes: [30, 50], m: 4}   # -> GENERATORS
algo:
  type: dqn                                                      # -> ALGOS
  n_step: 5
  double: true
  model:
    encoder: {type: graphsage, hidden_dim: 64, num_layers: 3}    # -> ENCODERS
    head:    {type: q_node}                                      # -> HEADS
  buffer:   {type: prioritized_replay, capacity: 100000}         # -> BUFFERS
trainer:    {num_envs: 32, iterations: 3000, baselines: [degree, random]}
```

## Supported components

| Family | `type:` values |
|---|---|
| **Problems** (`env`) | `dismantling` (+ `dismantling_cost`), `maxcut`, `spinglass`, `mvc`, `mis`, `mwis`, `maxclique`, `mds`, `set_cover`, `influence_max`, `qubo`, `modularity` (community detection), `balanced_partition`/`min_bisection`, `maxkcut`/`graph_coloring` (k-label), `sentinel` (observability), `eco_maxcut`/`eco_spinglass` (improvement) |
| **Generators** | `erdos_renyi`/`er`, `barabasi_albert`/`ba`, `watts_strogatz`/`ws`, `powerlaw_cluster`, `complete`, `lattice`, `regular`, `rgg`, `sbm`, `tree`, `caveman`, `heterogeneous`, `multiplex`, `hypergraph`, `set_cover` (bipartite), `qubo`, `sentinel`, `dataset` (real files) |
| **Encoders** | `gcn`, `graphsage`, `gin`, `gine`, `gat`, `gatv2`, `s2v`, `gatedgcn`, `pna`, `mpnn`, `nnconv`, `graph_transformer`, `edgeconv`, `gcnii`, `appnp`, `sgc`, `chebnet`, `arma`, `tagcn`, `rgcn`, `hgt`, `han`, `typed_mpnn`, `multiplex_gcn`, `hgnn`, `hypergcn`, `hnhn`, `sentinel_encoder`, `pe` |
| **Algorithms** | `dqn` (+ `double`/`dueling`/`n_step`/PER), `qrdqn`, `c51`, `iqn`, `munchausen_dqn`, `ppo`, `a2c`, `reinforce`, `reinforce_rollout`, `pomo`, `sac`, `finder`, `label_dqn`, `sentinel_dqn` |
| **Heads** | `q_node`, `dueling_q`, `quantile_q`, `distributional_q`, `implicit_quantile`, `noisy_q`, `q_label`, `finder_q`, `actor`, `critic` |
| **Buffers** | `nstep_replay` (uniform), `prioritized_replay` (PER) |

`graco list <family>` prints the live registry for any of them.

## Reproducing FINDER & DIRAC

Both reference systems are faithful, first-class baselines (`graco/baselines/`) that
reuse everything that fits and override only what is paper-specific:

- **FINDER** (`graco/configs/finder.yaml`, `type: finder`) — dismantling env with constant
  `ones[n,2]` node inputs + a graph `aux_feat` readout, GraphSAGE (**sum**
  aggregation), a bilinear cross-product Q-head (`finder_q`), and a graph-
  reconstruction loss.
- **DIRAC** (`graco/configs/dirac.yaml`) — spin-glass env (reward `ΔH/|E|`), structure2vec
  node↔edge encoder, lattice-coordinate node features, bootstrap clipped at ≥0.

Also runnable from Python — see `examples/run_finder.py` and `examples/run_dirac.py`.

## More graph types & problems

Swap the generator + a structure-aware encoder; node/edge/layer *types* flow through
the whole RL loop (including replay):

```bash
graco train -c miner                    # multilayer/multiplex dismantling (GMCC)
graco train -c hitter                   # hypergraph dismantling (HNHN)
graco train -c setcover_dqn             # bipartite   + RGCN
graco train -c maxcut_ppo               # MaxCut      + PPO
graco train -c community_detection      # modularity / community detection (SBM)
graco train -c maxkcut_dqn              # max-k-cut   (k-label action head)
```

**Real datasets** — the `dataset` generator loads edgelist / GML / GraphML / MTX /
Pajek / Gset / DIMACS (`.gz` ok) from a file, glob, or directory:

```bash
graco datasets                                       # list built-in downloadable sets
graco benchmark -c dismantling_real --data snap-facebook
graco train     -c dismantling_real --data 'data/nets/*.edges' --data-mode random
```

## Benchmarking

`graco benchmark` runs the learned policy (from a checkpoint) and every applicable
heuristic on the same graphs and prints a ranked table with a `gap%` column. On
small instances, exact solvers (a vectorized brute-force MaxCut/QUBO solver and an
ILP solver via PuLP→CBC for MaxCut/MVC/MIS/MDS/Set-Cover) join automatically, so the
`gap%` becomes a true optimality gap.

```bash
graco benchmark -c qubo generator.num_nodes=[14,14] \
    --methods brute_force_qubo qubo_local_search random
```

## Performance

Turn on the standard, portable GPU speedups with one call — no-op on CPU, graceful
fallback, auto-disabled in deterministic mode, no YAML changes:

```python
import graco
graco.accelerate()                # TF32 + matmul precision + cuDNN autotune + fused Adam
graco.accelerate(compile=True)    # + torch.compile the encoder (kernel fusion)
graco.accelerate(spmm=True)       # + sparse-matmul aggregation (best on dense/large graphs)
```

Benchmark on your own hardware with `python benchmarks/bench.py --device cuda`. To
raise throughput, scale `trainer.num_envs`, `algo.learn_freq` and use `device=cuda`.

## Extending without forking

```python
from graco.registries import ENCODERS
from graco.models.base import GNNEncoder

@ENCODERS.register("my_gnn")
class MyGNN(GNNEncoder):
    def forward(self, graph):
        x = self.input_x(graph)
        ...                       # scatter over graph.edge_index
        return x                  # [N, self.out_dim]
```

Then `encoder: {type: my_gnn}` — or `type: my_project.MyGNN` for an out-of-tree
class (add `imports: [my_project]` to register short names).

## Development

```bash
pip install -e ".[dev]"      # dev extras (pytest, ruff)
pytest -q                    # run the test suite
ruff check graco tests       # lint
```

## License

MIT — see [LICENSE](LICENSE). Builds on the problem formulations of **FINDER**
(Fan et al., *Nature Machine Intelligence* 2020) and **DIRAC** (Fan et al.,
*Nature Communications* 2023), reimagined as a vectorized, config-driven PyTorch
framework.
