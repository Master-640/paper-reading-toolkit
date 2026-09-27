# ==========================================================================
# 第二版：量出 7 条条带的精确位置与宽度（x=62.4/96.5/130.6/164.8/198.9/233.0/267.1，
# # 宽 31.28 pt，间距 34.1 pt），并确认色场由多顶点填充多边形构成 —— 即 2 维场而非 1 维曲线。
# # 这是判定 Fig.3b 存在“未标注横轴”的关键证据。
# ==========================================================================
import fitz, sys
from _paths import MAIN_PDF
from collections import Counter
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PDF = MAIN_PDF
doc = fitz.open(PDF)
pg = doc[4]
BOX = fitz.Rect(38, 165, 395, 262)

print("=== get_image_info (finds XObject images too) ===")
for im in pg.get_image_info(xrefs=True):
    bb = im["bbox"]
    print(f"  bbox=({bb[0]:.0f},{bb[1]:.0f},{bb[2]:.0f},{bb[3]:.0f}) "
          f"{im['width']}x{im['height']} xref={im.get('xref')}")

print("\n=== all drawings inside panel b, sorted by area ===")
items = []
for d in pg.get_drawings():
    r = d["rect"]
    if not (r.x0 >= BOX.x0 - 2 and r.x1 <= BOX.x1 + 2
            and r.y0 >= BOX.y0 - 2 and r.y1 <= BOX.y1 + 2):
        continue
    items.append((r.width * r.height, d["type"], len(d["items"]),
                  round(r.x0, 1), round(r.y0, 1), round(r.width, 2), round(r.height, 2),
                  d.get("fill")))
items.sort(reverse=True)
for a, t, n, x0, y0, w, h, fill in items[:24]:
    print(f"  area={a:8.1f} type={t} nitems={n:5d} x0={x0:6.1f} y0={y0:6.1f} "
          f"w={w:7.2f} h={h:7.2f} fill={fill}")

print("\n=== count of drawings by type in panel b ===")
print(Counter(d["type"] for d in pg.get_drawings()
              if BOX.x0 - 2 <= d["rect"].x0 and d["rect"].x1 <= BOX.x1 + 2
              and BOX.y0 - 2 <= d["rect"].y0 and d["rect"].y1 <= BOX.y1 + 2))

# how wide is one mode strip? check the top axis ticks / bottom labels
print("\n=== bottom labels of panel b (mode names) positions ===")
for b in pg.get_text("words"):
    if b[4] in {"ENSO", "NPMM", "SPMM", "IOB", "IOD", "SIOD", "TNA"} and 255 < b[1] < 268:
        print(f"  {b[4]:<5} x0={b[0]:6.1f} x1={b[2]:6.1f}  center={(b[0]+b[2])/2:6.1f}  y={b[1]:.1f}")
doc.close()
