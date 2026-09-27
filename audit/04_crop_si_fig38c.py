# ==========================================================================
# 单独放大 SI Fig.38c（模型规模 S-1..S-4）。结论：最小的 S-1 在 6/7 模态最好，
# # 与 SI 正文 “increasing model capacity generally correlates with improved skill” 反向。
# ==========================================================================
import fitz, os, sys
from _paths import SI_PDF, SI_FIGS
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
doc = fitz.open(SI_PDF)
p = doc[46]                      # Fig 38
r = p.rect
# panel c sits below panel b on the same page
pix = p.get_pixmap(dpi=460, clip=fitz.Rect(r.x0 + 30, r.y0 + 400, r.x1 - 20, r.y1 - 60))
pix.save(SI_FIGS + "/crop/Fig38c_model_size.png")
print("Fig38c:", pix.width, "x", pix.height)
doc.close()
