# ==========================================================================
# 把 SI Fig.37 的 a/b 面板与 Fig.38 放大到 500 dpi，使柱高可以目视读数。
# ==========================================================================
"""Crop + upscale the two panels of Supplementary Fig. 37 (PDF p46) and the
three panels of Fig. 38 (PDF p47) so the bar heights can be read accurately.

Fig 37 page geometry (portrait A4-ish, 595x842 pt): panel a occupies roughly the
top half, panel b the lower half. Fig 38 has three side-by-side panels.
"""
import fitz, os, sys
from _paths import SI_PDF, SI_FIGS

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.makedirs(SI_FIGS + "/crop", exist_ok=True)
doc = fitz.open(SI_PDF)

# --- Fig 37 (pdf p46 = index 45) ---
p37 = doc[45]
r = p37.rect
print("Fig37 page rect:", r)
crops37 = {
    "Fig37a_additive_vs_replace": fitz.Rect(r.x0 + 30, r.y0 + 90, r.x1 - 30, r.y0 + 420),
    "Fig37b_unicm_vs_single":     fitz.Rect(r.x0 + 30, r.y0 + 430, r.x1 - 30, r.y0 + 700),
}
for name, rect in crops37.items():
    pix = p37.get_pixmap(dpi=500, clip=rect)
    p = f"{SI_FIGS}/crop/{name}.png"
    pix.save(p)
    print(f"  {p}  {pix.width}x{pix.height}")

# --- Fig 38 (pdf p47 = index 46) ---
p38 = doc[46]
r = p38.rect
print("Fig38 page rect:", r)
pix = p38.get_pixmap(dpi=420, clip=fitz.Rect(r.x0, r.y0 + 60, r.x1, r.y0 + 460))
pix.save(SI_FIGS + "/crop/Fig38_all.png")
print(f"  Fig38_all.png {pix.width}x{pix.height}")
doc.close()
