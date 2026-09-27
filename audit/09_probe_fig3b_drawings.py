# ==========================================================================
# 第一版：统计 Fig.3b 内的填充矩形与内嵌位图（结果：无内嵌位图，说明色场是矢量绘制的）。
# # 完整分析见 10_probe_fig3b_cell_geometry.py。
# ==========================================================================
"""Decide what Fig. 3b's hidden horizontal axis is, by inspecting the vector
drawing primitives (pcolormesh cells are drawn as thousands of small rects).

If panel b is 7 strips x 25 lead cells  -> each strip is a 1-D ACC(lead) curve
                                           and the caption is complete.
If it is 7 x 12 x 25                    -> there is an unlabelled target-month
                                           axis and the caption is incomplete.
"""
import fitz, sys
from _paths import MAIN_PDF
from collections import Counter
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PDF = MAIN_PDF
doc = fitz.open(PDF)
pg = doc[4]

BOX = fitz.Rect(38, 165, 395, 262)      # panel b only
print("panel b box:", BOX)

rects, others = [], Counter()
for d in pg.get_drawings():
    r = d["rect"]
    if not (r.x0 >= BOX.x0 - 2 and r.x1 <= BOX.x1 + 2
            and r.y0 >= BOX.y0 - 2 and r.y1 <= BOX.y1 + 2):
        continue
    if d.get("fill") is not None and abs(r.width * r.height) > 0.02:
        rects.append(r)
    else:
        others[d["type"]] += 1

print("filled rect count in panel b:", len(rects))
if rects:
    ws = Counter(round(r.width, 2) for r in rects)
    hs = Counter(round(r.height, 2) for r in rects)
    print("distinct widths  (top 8):", ws.most_common(8))
    print("distinct heights (top 8):", hs.most_common(8))
    xs = sorted({round(r.x0, 2) for r in rects})
    ys = sorted({round(r.y0, 2) for r in rects})
    print("distinct x0 count:", len(xs))
    print("distinct y0 count:", len(ys))
    print("x0 list:", xs)

print("\nembedded images on the page:")
for im in pg.get_images(full=True):
    print("  xref", im[0], "w x h =", im[2], "x", im[3], "bpc", im[4], "cs", im[5])
doc.close()
