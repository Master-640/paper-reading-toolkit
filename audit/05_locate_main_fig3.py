# ==========================================================================
# 在主文里定位 Fig.3 所在页并渲染整页（主文共 12 页，Fig.3 在第 5 页）。
# ==========================================================================
"""Locate the page of UniCM main text that carries Fig. 3 and render it at high
resolution, plus crop panel b (the mode x lead-time skill heatmap)."""
import fitz, os, sys
from _paths import MAIN_PDF, FIG3
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PDF = MAIN_PDF
OUT = FIG3
os.makedirs(OUT, exist_ok=True)

doc = fitz.open(PDF)
print("pages:", doc.page_count)
hit = []
for i in range(doc.page_count):
    t = doc[i].get_text()
    if "Fig. 3 | UniCM" in t or "Fig. 3 |UniCM" in t:
        hit.append(i)
    if "Monthly forecast skill" in t:
        hit.append(i)
print("pages mentioning Fig.3 caption / 'Monthly forecast skill':", sorted(set(hit)))

for i in sorted(set(hit)):
    txt = doc[i].get_text()
    print(f"\n=== page index {i} (1-based {i+1}) : "
          f"len={len(txt)}  hasMonthly={'Monthly forecast skill' in txt}")
    for kw in ["Monthly forecast skill", "Forecast lead (months)",
               "Fig. 3 | UniCM", "target season", "Target season"]:
        if kw in txt:
            print("   contains:", kw)

# render the caption-bearing page(s) at high dpi
for i in sorted(set(hit)):
    pix = doc[i].get_pixmap(dpi=300)
    p = f"{OUT}/UniCM_main_p{i+1}.png"
    pix.save(p)
    print("saved", p, pix.width, "x", pix.height)
doc.close()
