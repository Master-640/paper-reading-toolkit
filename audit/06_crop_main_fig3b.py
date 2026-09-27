# ==========================================================================
# 裁出主文 Fig.3b（7 个模态并排的技巧条带）与图注，并打印完整图注原文。
# # 用于确认图注只写了 “as a function of lead time”、未提横轴、也未定义菱形标记。
# ==========================================================================
"""Inspect page 5 of UniCM main text: locate the Fig. 3 caption and the panel-b
heatmap precisely, then crop both at high resolution."""
import fitz, sys
from _paths import MAIN_PDF, FIG3
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PDF = MAIN_PDF
doc = fitz.open(PDF)
pg = doc[4]
W, H = pg.rect.width, pg.rect.height
print(f"page size: {W} x {H}")

print("\n--- text blocks containing key strings (x0,y0,x1,y1) ---")
for b in pg.get_text("blocks"):
    x0, y0, x1, y1, txt = b[0], b[1], b[2], b[3], b[4]
    flat = txt.replace("\n", " ")
    for kw in ["Monthly forecast skill", "Fig. 3 |", "Forecast lead (months)",
               "target season", "Contour lines"]:
        if kw in flat:
            print(f"  [{kw}] bbox=({x0:.0f},{y0:.0f},{x1:.0f},{y1:.0f})  {flat[:160]!r}")
            break

# find the heatmap panel: "Monthly forecast skill" title marks its top
top = None
for b in pg.get_text("blocks"):
    if "Monthly forecast skill" in b[4]:
        top = b[1]
        title_bbox = b[:4]
print("\ntitle bbox:", title_bbox if top else None)

# panel b occupies full width between the title and the caption
cap_y = None
for b in pg.get_text("blocks"):
    if "Fig. 3 |" in b[4]:
        cap_y = b[1]
print("caption y0:", cap_y)

if top and cap_y:
    # crop generously around panel b
    rect = fitz.Rect(60, title_bbox[1] - 34, W - 40, cap_y - 4)
    pix = pg.get_pixmap(dpi=600, clip=rect)
    pix.save(FIG3 + "/Fig3b_heatmap_600dpi.png")
    print("saved Fig3b_heatmap_600dpi.png", pix.width, "x", pix.height)

    cap = fitz.Rect(60, cap_y - 4, W - 40, cap_y + 86)
    pix2 = pg.get_pixmap(dpi=600, clip=cap)
    pix2.save(FIG3 + "/Fig3_caption_600dpi.png")
    print("saved Fig3_caption_600dpi.png", pix2.width, "x", pix2.height)

print("\n--- full caption text from PDF ---")
for b in pg.get_text("blocks"):
    if "Fig. 3 |" in b[4]:
        print(b[4])
doc.close()
