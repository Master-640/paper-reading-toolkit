# ==========================================================================
# 定位主文 Fig.2 并列出其季节刻度标签，确认 Fig.2b 用的是 12 个三个月滚动季节
# # （DJF…NDJ，间距 8.1 pt）—— 与 Fig.3b 的横轴一致。
# ==========================================================================
"""Decisive test for Fig. 3b's hidden horizontal axis.

Fig. 2b is the ENSO skill heatmap with (per its caption) x = forecast lead and
y = target season. Fig. 3b's ENSO strip plots the SAME quantity (ENSO ACC) but
with y = forecast lead. If Fig. 3b's ENSO strip is Fig. 2b transposed, then
Fig. 3b's horizontal axis is the target season -- which is the only way the body
text ("IOD peaks in boreal autumn", "IOB in early spring") can be read off it.
"""
import fitz, sys
from _paths import MAIN_PDF, FIG3
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PDF = MAIN_PDF
doc = fitz.open(PDF)

for i in range(doc.page_count):
    t = doc[i].get_text()
    if "Fig. 2 |" in t:
        print("Fig. 2 caption on page index", i, "(1-based", i + 1, ")")
        pg = doc[i]
        for b in sorted(pg.get_text("blocks"), key=lambda b: b[1]):
            if "Fig. 2 |" in b[4]:
                print("\n--- caption ---")
                print(b[4].strip())
        # find the seasonal heatmap: look for month abbreviations in the figure
        words = pg.get_text("words")
        months = [w for w in words if w[4] in {"MAM", "AMJ", "JJA", "JAS", "SON", "OND",
                                              "DJF", "JFM", "FMA", "Jan", "Feb", "Mar"}]
        print("\nmonth-like labels found:", [(w[4], round(w[0]), round(w[1])) for w in months][:24])
        print("\n'Forecast lead (months)' occurrences:",
              [(round(w[0]), round(w[1])) for w in words if w[4].startswith("Forecast")])
        pix = pg.get_pixmap(dpi=300)
        pix.save(f"{FIG3}/UniCM_main_p{i+1}_Fig2.png")
        print("saved page render")
doc.close()
