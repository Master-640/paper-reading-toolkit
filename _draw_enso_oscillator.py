# -*- coding: utf-8 -*-
"""Draw the ENSO recharge-oscillator / skewness logic chain as a PNG flowchart."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ---- CJK font ----
cands = ["Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC",
         "Source Han Sans SC", "DengXian", "KaiTi"]
avail = {f.name for f in font_manager.fontManager.ttflist}
pick = next((c for c in cands if c in avail), None)
print("CJK font used:", pick)
if pick:
    plt.rcParams["font.sans-serif"] = [pick]
plt.rcParams["axes.unicode_minus"] = False

C_BLUE = ("#dbeafe", "#2563eb")
C_RED = ("#fee2e2", "#dc2626")
C_GREEN = ("#dcfce7", "#16a34a")
C_YELL = ("#fef3c7", "#d97706")
C_GREY = ("#f1f5f9", "#64748b")

fig, ax = plt.subplots(figsize=(13, 16.5))
ax.set_xlim(0, 13)
ax.set_ylim(0, 16.5)
ax.axis("off")


def box(x, y, w, h, text, colors, fs=12.5, bold=False, dashed=False):
    fc, ec = colors
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.12,rounding_size=0.18",
                                fc=fc, ec=ec, lw=2.0,
                                linestyle="--" if dashed else "-", zorder=2))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, zorder=3,
            fontweight="bold" if bold else "normal", linespacing=1.55)


def arrow(p1, p2, label="", fs=11, style="-|>", ls="-", color="#334155", rad=0.0):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=18,
                                 lw=1.9, color=color, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}", zorder=1))
    if label:
        ax.text((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2, label,
                ha="center", va="center", fontsize=fs, color="#0f172a",
                bbox=dict(fc="white", ec="none", alpha=0.85, pad=2.0), zorder=4)


CX = 4.85          # spine centre
NX = 10.9          # right note centre

# ---------- title ----------
ax.text(6.5, 16.05, "ENSO 充放电振子 → 正偏态 → 两种机制：一条逻辑链",
        ha="center", va="center", fontsize=17, fontweight="bold", color="#0f172a")

# ---------- 1. oscillator ----------
box(CX, 14.85, 7.2, 1.75,
    "① 充放电振子（Recharge Oscillator）\n"
    "两个状态变量：Te（赤道东太平洋 SST） +  h（赤道平均温跃层深度 = 赤道热含量）\n"
    "dTe/dt = I_BJ·Te + F·h      dh/dt = −ε·h − α·Te\n"
    "F（温跃层反馈，快）与 α（充放电速率，慢）构成延迟负反馈 → 自我振荡",
    C_BLUE, fs=12, bold=False)

box(NX, 14.85, 5.6, 1.75,
    "旁支：它带来了什么可预报性\n"
    "西热带太平洋暖池热含量 T300\n"
    "= ENSO 9–15 个月长提前期先兆\n"
    "但只是“必要条件”：\n"
    "2012 / 2014 / 2017 热含量有利，\n"
    "却因缺 WWE 而未发展成事件",
    C_BLUE, fs=11, dashed=True)
arrow((CX + 3.65, 14.85), (NX - 2.85, 14.85), style="-|>", ls="--", color="#2563eb")

# ---------- 2. observation ----------
box(CX, 12.75, 7.2, 1.5,
    "② 观测事实（Fig. 2）\n"
    "SSTA 分布【正偏】：右尾长\n"
    "El Niño 振幅 > La Niña 振幅；El Niño ≈ 1 年，La Niña 可 2–3 年",
    C_RED, fs=12)

# ---------- 3. judgement ----------
box(CX, 10.6, 7.2, 1.7,
    "③ 判决性推理：线性振子 + 对称噪声\n"
    "能给出正偏吗？\n"
    "不能 —— 线性系统 + 对称噪声在数学上必然是高斯分布，偏度 = 0",
    C_RED, fs=12, bold=False)

# ---------- 4. conclusion ----------
box(CX, 8.5, 7.2, 1.25,
    "④ 结论：真实 ENSO 必然含\n非线性反馈  或  状态依赖（乘性）噪声",
    C_RED, fs=13, bold=True)

# ---------- 5/6. two routes ----------
AX, BX = 2.55, 7.15
box(AX, 6.55, 4.3, 1.6,
    "路线 A：非线性 Bjerknes 反馈\n（确定性来源）\n"
    "大气深对流的阈值/饱和非线性；\n海洋热平流与上升流的非线性\n→ 反馈强度随状态变化",
    C_GREEN, fs=11.5)
box(BX, 6.55, 4.3, 1.6,
    "路线 B：乘性随机噪声\n（随机性来源）\n"
    "WWE 频次随西/中太平洋背景 SST\n变暖而增多（= state-dependent noise）\n→ 噪声强度随状态变化",
    C_GREEN, fs=11.5)

box(AX, 4.75, 4.3, 1.25,
    "Te > 0 时增长更快\nTe < 0 时加深受限",
    C_GREEN, fs=12)
box(BX, 4.75, 4.3, 1.25,
    "暖态噪声大 → 极值尾部拉长\n冷态噪声小 → 拉尼娜被“锁住”",
    C_GREEN, fs=12)

# ---------- 7. common result ----------
box(CX, 3.15, 7.2, 1.1, "⑤ 共同结果：分布右偏（正偏态）", C_YELL, fs=13.5, bold=True)

# ---------- 8. consequences ----------
box(CX, 1.55, 11.4, 1.7,
    "⑥ 三个后果\n"
    "El Niño 约 1 年（快起快止）   |   La Niña 可维持 2–3 年，且可预报性系统性更低   |   "
    "事件间差异大 = ENSO complexity",
    C_YELL, fs=12.5)

box(CX, 0.30, 11.4, 0.85,
    "一句话：海洋记忆给了 ENSO 预报的可能性，状态依赖的非线性与噪声给了它的上限。",
    C_GREY, fs=13, bold=True)

# ---------- arrows ----------
arrow((CX, 13.97), (CX, 13.52))                       # 1 -> 2
arrow((CX, 12.00), (CX, 11.47))                       # 2 -> 3
arrow((CX, 9.75), (CX, 9.14))                         # 3 -> 4
arrow((CX, 7.87), (AX, 7.37), rad=-0.12)              # 4 -> A
arrow((CX, 7.87), (BX, 7.37), rad=0.12)               # 4 -> B
arrow((AX, 5.75), (AX, 5.39))                         # A -> A1
arrow((BX, 5.75), (BX, 5.39))                         # B -> B1
arrow((AX, 4.12), (CX - 1.3, 3.72), rad=-0.15)        # A1 -> 5
arrow((BX, 4.12), (CX + 1.3, 3.72), rad=0.15)         # B1 -> 5
arrow((CX, 2.60), (CX, 2.42))                         # 5 -> 6
arrow((CX, 0.70), (CX, 0.74), style="-")

plt.tight_layout()
out = r"D:\collections2026\硕博申请\中国科学院计算所\学习材料\刘睿涵\刘睿涵\汇报合集\week2汇报\ENSO-振子与偏态逻辑链.png"
plt.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
print("saved:", out)
