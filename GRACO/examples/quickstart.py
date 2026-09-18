"""Quickstart: train an agent from Python in one line, then evaluate it.

Run:  python examples/quickstart.py
"""

import graco

# Train from a shipped config (resolved by short name). `key__sub=value` is sugar
# for the OmegaConf dotted override `key.sub=value`.
trainer = graco.train(
    "dismantling_dqn",
    device="auto",
    trainer__num_envs=32,
    trainer__iterations=300,
    trainer__eval_every=100,
)

metrics = trainer.evaluate()
print("\n=== final evaluation ===")
for k, v in metrics.items():
    print(f"{k:32s} {v: .4f}")

# Discover what else is available:
print("\nalgorithms:", graco.available("algos"))
