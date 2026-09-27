# ==========================================================================
# 渲染 SI Fig.37-41（SI 第 45-50 页）为 PNG。这些图的数值只存在于栅格/矢量图形里，
# # 页面内嵌文本只有图注，因此必须渲染后才能读数。
# # 注意页码偏移：PDF 共 51 页，SI 页码从 1 编到 50，PDF 第 N 页 = SI 第 N-1 页。
# ==========================================================================
"""Render UniCM Supplementary Figs. 37-41 to PNG.

Note the offset: the PDF has 51 pages but SI page labels run 1..50, because the
title page is unnumbered. PDF page N = SI page N-1. Supplementary Fig. 36 sits on
PDF p45 ("44/50"), so Fig. 37 is PDF p46 and Fig. 41 is PDF p50.
"""
import fitz, os, sys
from _paths import SI_PDF, SI_FIGS

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PDF = SI_PDF
OUT = SI_FIGS
os.makedirs(OUT, exist_ok=True)

doc = fitz.open(PDF)
targets = [(45, "SI-Fig37-arch-ablation"),
           (46, "SI-Fig38-hyperparam"),
           (47, "SI-Fig39-data-requirement"),
           (48, "SI-Fig40-grid-resolution"),
           (49, "SI-Fig41-pretrain-vs-finetune")]

for idx, name in targets:
    page = doc[idx]
    label = page.get_text().strip().split("\n")[0][:70]
    pix = page.get_pixmap(dpi=220)
    path = f"{OUT}/{name}_pdfp{idx+1}.png"
    pix.save(path)
    print(f"pdf p{idx+1:>3} -> {path:<52} {pix.width}x{pix.height}  first-line={label!r}")

print()
print("embedded text length per page (0 => numbers live only in the raster):")
for idx, name in targets:
    print(f"  pdf p{idx+1}: {len(doc[idx].get_text())} chars")
doc.close()
