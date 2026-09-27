# ==========================================================================
# 极致放大 Fig.3b 的 ENSO / NPMM / IOD 三条条带，用于判断条带内部是否存在水平结构，
# # 并打印条带下方文字（结果：只有 7 个模态名，没有任何横轴刻度标签）。
# ==========================================================================
"""Ultra-zoom on two columns of Fig. 3b to test whether colour varies
HORIZONTALLY inside a strip. If it does, there is an unlabelled x axis
(most plausibly target calendar month); if not, each strip is a pure
1-D ACC(lead) curve and the caption is complete."""
import fitz, sys
from _paths import MAIN_PDF, FIG3
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PDF = MAIN_PDF
doc = fitz.open(PDF)
pg = doc[4]

crops = {
    "Fig3b_ENSO_zoom":  fitz.Rect(62, 168, 100, 259),
    "Fig3b_NPMM_zoom":  fitz.Rect(98, 168, 135, 259),
    "Fig3b_IOD_zoom":   fitz.Rect(240, 168, 277, 259),
}
for name, rect in crops.items():
    pix = pg.get_pixmap(dpi=1300, clip=rect)
    pix.save(f"{FIG3}/{name}.png")
    print(f"{name}: {pix.width} x {pix.height}")

# does the page contain any 3-letter month labels or numbers under the strips?
print("\n--- text under the strips (y 258..275) ---")
for b in sorted(pg.get_text("blocks"), key=lambda b: b[1]):
    if 255 <= b[1] <= 280:
        print(f"[y={b[1]:.0f} x={b[0]:.0f}] {b[4].strip()[:120]!r}")
doc.close()
