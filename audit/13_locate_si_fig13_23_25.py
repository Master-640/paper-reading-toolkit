# ==========================================================================
# 渲染 SI Fig.13 / 23 / 25 并打印图注。这是**定论性证据**：SI Fig.13 是同一批图的 XRO 版，
# # 横轴明确标注 “Target season” 并列出 12 个季节，色标同为 −1.00…+1.00 —— 由此确证
# # 主文 Fig.3b 被删掉的横轴就是目标季节。同时得到 SI Fig.23 对显著性点的定义。
# ==========================================================================
"""Decisive check for Fig. 3b's missing horizontal axis.

Main text Fig. 3b has no x tick labels, yet the body text reads "predictability
windows" (IOD peaks in SON, IOB in FMA, TNA in JAS) off it -- which is only
possible if the horizontal axis is the target calendar month / season.

Supplementary Fig. 13 is described in its own caption as the XRO analogue with
"horizontal axis = target season, vertical axis = forecast lead time". If its
layout matches Fig. 3b, that settles what Fig. 3b's hidden x axis is.
"""
import fitz, sys
from _paths import SI_PDF, FIG3
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SI = SI_PDF
doc = fitz.open(SI)
print("SI pages:", doc.page_count)

targets = {}
for i in range(doc.page_count):
    t = doc[i].get_text()
    for tag in ["Supplementary Figure 13.", "Supplementary Figure 25.",
                "Supplementary Figure 23."]:
        if tag in t:
            targets.setdefault(tag, i)

for tag, i in sorted(targets.items(), key=lambda kv: kv[1]):
    print(f"{tag}  -> pdf page index {i} (1-based {i+1})")

for tag, i in targets.items():
    pix = doc[i].get_pixmap(dpi=300)
    p = f"{FIG3}/{tag.split()[2].rstrip('.')}_pdfp{i+1}.png"
    pix.save(p)
    t = doc[i].get_text()
    print("\n" + "=" * 70)
    print(p, pix.width, "x", pix.height)
    print(t.strip()[:900])
doc.close()
