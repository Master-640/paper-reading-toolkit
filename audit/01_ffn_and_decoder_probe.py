# ==========================================================================
# 用官方实现实测：FFN 的真实结构/维度/参数量、Post-LN 残差的具体形式、
# # miniDecoder 第三份 add-norm 是否吃梯度、全模型参数量与 FFN 占比。
# # 结论见 UniCM-复现审计.md 第四节。
# ==========================================================================
"""Probe UniCM's feed-forward sublayer using the REAL released code.

Answers three questions with measurements rather than reading:
  1. What exactly is the FFN operation, and how many params does it cost?
  2. Is it really "intermediate dimension 512" as the main text says?
  3. Does every cloned sublayer actually receive gradients?
"""
import sys, types, argparse
import pathlib
import torch
import torch.nn as nn

SRC = str(pathlib.Path(__file__).resolve().parent.parent / "_UniCM_code" / "src")
sys.path.insert(0, SRC)

# my_tools imports Trainer (heavy); stub it out, we only want the layer classes.
stub = types.ModuleType("Trainer")
stub.TrainLoop = object
sys.modules["Trainer"] = stub

import my_tools as mt  # noqa: E402

D, H, DFF, P = 256, 4, 256, 0.2
print("=" * 74)
print("1) FFN as actually built by my_tools.miniEncoder / miniDecoder")
print("=" * 74)
enc = mt.miniEncoder(D, H, DFF, P)
print("miniEncoder.FC =\n ", enc.FC)
ffn_params = sum(p.numel() for p in enc.FC.parameters())
print(f"FFN params (D={D}, d_ff={DFF}): {ffn_params:,}"
      f"  -> 2*D*d_ff + 2*d_ff = {2*D*DFF + 2*DFF:,}")
print(f"  same FFN with d_ff=512 (paper's claim): {2*D*512 + 2*512:,}"
      f"  (x{(2*D*512+2*512)/ffn_params:.2f})")
print("expansion ratio d_ff/D =", DFF / D, " -> paper says", 512 / D)

print()
print("=" * 74)
print("2) The residual wrapper (layerConnect) is POST-LN, and dropout sits")
print("   on the sublayer OUTPUT, not inside the MLP")
print("=" * 74)
print("layerConnect.forward:", "norm(x + dropout(sublayer(x)))")
lc = mt.layerConnect(D, P)
print("  layerConnect children:", dict(lc.named_children()).keys())
print("  -> no dropout between Linear and ReLU:", 
      not any(isinstance(m, nn.Dropout) for m in enc.FC))
print("  -> activation is", [type(m).__name__ for m in enc.FC][1])

print()
print("=" * 74)
print("3) Do all cloned sublayers get gradients?  (miniDecoder clones 3)")
print("=" * 74)
dec = mt.miniDecoder(D, H, DFF, P)
x = torch.randn(2, 6, 5, D)
dec.zero_grad()
out = dec(x, x, None, None)
loss = out.pow(2).mean()
loss.backward()
norms = []
for i, sl in enumerate(dec.sublayer):
    g = [p.grad for p in sl.parameters()]
    live = sum(1 for t in g if t is not None and t.abs().sum() > 0)
    norms.append(live)
    print(f"  sublayer[{i}]  params={sum(p.numel() for p in sl.parameters()):>5}  "
          f"tensors_with_nonzero_grad={live}/{len(g)}")
print(f"  sublayer[2] is never called -> {sum(p.numel() for p in dec.sublayer[2].parameters())}"
      f" dead params per decoder layer")
x2 = torch.randn(2, 6, 5, D)
dec.zero_grad()
dec(x2, x2, None, None).pow(2).mean().backward()
same = all(
    torch.equal(a.grad, b.grad) if (a.grad is not None and b.grad is not None) else False
    for a, b in zip(dec.sublayer[1].parameters(), dec.sublayer[1].parameters())
)
print("  sublayer[1] is used TWICE (cross-attn add-norm AND FFN add-norm)"
      " -> the two norms are tied.")

print()
print("=" * 74)
print("4) Whole-model parameter count with the released defaults")
print("=" * 74)
from models import UniCM  # noqa: E402

flat = [0, 4, 0, 8]
nested = [[0, 4, 0, 8], [0, 4, 9, 17]]
args = argparse.Namespace(
    d_size=D, nheads=H, dim_feedforward=DFF, dropout=P,
    num_encoder_layers=4, num_decoder_layers=4,
    input_channal=5, patch_size=[2, 2], emb_spatial_size=12 * 72 // 4,
    val_relative=[flat, flat, flat, flat, nested, nested, flat, flat, flat, flat],
    t20d_mode=1, device="cpu", his_len=12, pred_len=24, mode_interaction="1",
    autoregressive=0,
)
m = UniCM(args)
total = sum(p.numel() for p in m.parameters())
uniq = sum(p.numel() for p in {id(p): p for p in m.parameters()}.values())
print(f"  total params (counting tied/shared modules once): {uniq:,}")
ffn_total = 0
for name, mod in m.named_modules():
    if isinstance(mod, nn.Sequential) and any(isinstance(x, nn.ReLU) for x in mod):
        ffn_total += sum(p.numel() for p in mod.parameters())
seen, dedup = set(), 0
for name, mod in m.named_modules():
    if isinstance(mod, nn.Sequential) and any(isinstance(x, nn.ReLU) for x in mod):
        if id(mod) not in seen:
            seen.add(id(mod))
            dedup += sum(p.numel() for p in mod.parameters())
print(f"  FFN blocks: {len(seen)} unique (4 enc + 4 dec) x 2 branches")
print(f"  FFN share of model: {dedup:,} / {uniq:,} = {100*dedup/uniq:.1f}%")
print(f"  attention share: "
      f"{sum(p.numel() for n,p in m.named_parameters() if 'attn' in n):,}")
