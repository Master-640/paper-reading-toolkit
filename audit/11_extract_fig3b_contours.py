# ==========================================================================
# 从 Fig.3b 的 PDF 矢量数据里抽出 0.5 等值线的实际几何，并按 y 轴刻度换算成预报提前期，
# # 得到逐模态的 0.5 阈值穿越位置。结论：TNA 的 0.5 线被切成 4 段，说明其技巧在阈值附近抖动。
# ==========================================================================
"""Extract the actual 0.5 / -0.5 contour geometry from Fig. 3b's vector data,
and build a pixel->(lead) mapping from the y-axis tick labels, so the skill
threshold crossing per mode can be read off numerically instead of by eye.
"""
import fitz, sys, re
from _paths import MAIN_PDF
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PDF = MAIN_PDF
doc = fitz.open(PDF)
pg = doc[4]

# ---- y-axis tick labels of panel b (left of x=62, above the label row y=257)
ticks = []
for w in pg.get_text("words"):
    x0, y0, x1, y1, txt = w[:5]
    if 46 <= x0 <= 62 and 165 <= y0 <= 262 and re.fullmatch(r"\d+", txt):
        ticks.append((int(txt), (y0 + y1) / 2))
ticks.sort()
print("y tick labels (lead -> page y):", [(t, round(y, 1)) for t, y in ticks])

if len(ticks) >= 2:
    (l1, y1v), (l2, y2v) = ticks[0], ticks[-1]
    k = (y2v - y1v) / (l2 - l1)
    def lead_of(y):  return l1 + (y - y1v) / k
else:
    def lead_of(y):  return float("nan")

# ---- mode strips
strips = {"ENSO": 62.4, "NPMM": 96.5, "SPMM": 130.6, "IOB": 164.8,
          "IOD": 198.9, "SIOD": 233.0, "TNA": 267.1}
SW = 31.28

# ---- the 0.5 / -0.5 contour labels inside panel b
labels = []
for w in pg.get_text("words"):
    x0, y0, x1, y1, txt = w[:5]
    if 60 <= x0 <= 300 and 170 <= y0 <= 258 and txt.strip() in {"0.5", "-0.5", "−0.5"}:
        labels.append((txt.strip(), (x0 + x1) / 2, (y0 + y1) / 2))
print("\ncontour labels inside panel b:")
for t, cx, cy in labels:
    strip = min(strips.items(), key=lambda kv: abs(kv[1] + SW / 2 - cx))
    print(f"  {t:>5}  page=({cx:.1f},{cy:.1f})  lead≈{lead_of(cy):5.1f}  strip={strip[0]}")

# ---- stroke paths (the contour lines)
print("\nstroke paths per strip (lead range spanned):")
strokes = []
for d in pg.get_drawings():
    if d["type"] not in ("s", "fs"):
        continue
    r = d["rect"]
    if not (60 <= r.x0 and r.x1 <= 300 and 170 <= r.y0 and r.y1 <= 258):
        continue
    pts = []
    for it in d["items"]:
        if it[0] == "l":
            pts += [it[1], it[2]]
        elif it[0] == "c":
            pts += [it[1], it[2], it[3], it[4]]
    if not pts:
        continue
    ys = [p.y for p in pts]
    strokes.append((r.x0, min(ys), max(ys)))

for name, sx in strips.items():
    sel = [s for s in strokes if abs(s[0] - sx) < 2]
    if not sel:
        print(f"  {name:<5} no stroke found")
        continue
    tops = [lead_of(s[1]) for s in sel]
    bots = [lead_of(s[2]) for s in sel]
    print(f"  {name:<5} n_paths={len(sel)}  lead spans "
          f"{min(bots):5.1f} .. {max(tops):5.1f}   "
          f"(per-path top leads: {[round(t,1) for t in sorted(tops, reverse=True)]})")
doc.close()
