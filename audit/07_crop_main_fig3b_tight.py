# ==========================================================================
# 更紧地裁出 Fig.3b，并打印 y 轴刻度位置，用于建立「像素 y -> 预报提前期」的映射。
# ==========================================================================
"""Tight high-DPI crop of Fig. 3b plus the complete caption text."""
import fitz, sys
from _paths import MAIN_PDF, FIG3
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PDF = MAIN_PDF
doc = fitz.open(PDF)
pg = doc[4]

# panel b only, including its y-axis and the colorbar
rect = fitz.Rect(34, 148, 402, 262)
pix = pg.get_pixmap(dpi=900, clip=rect)
pix.save(FIG3 + "/Fig3b_only_900dpi.png")
print("Fig3b_only:", pix.width, "x", pix.height)

print("\n=== complete caption (all blocks with y0 in 375..470) ===")
for b in sorted(pg.get_text("blocks"), key=lambda b: b[1]):
    if 373 <= b[1] <= 470:
        print(f"[y={b[1]:.0f}] {b[4].strip()}")
doc.close()
