# -*- coding: utf-8 -*-
"""
pdf-tool —— PDF 自动抓取（分栏感知 + 页眉页脚清理 + 断词还原）

解决 pdftotext 的三个通病：
  1. 双栏论文按“行”交错，左右栏内容串在一起  → 找中缝 gutter，按 (横带, 栏, y) 排序
  2. 页眉页脚（期刊名 / DOI / 页码）混进正文    → 跨页统计“上下边缘重复行”，拉黑
  3. 行尾连字符断词 cli-\\nmate                → 还原为 climate

用法
----
python extract.py <pdf> [选项]

  -o, --outdir DIR    输出目录（默认 ./out/<pdf名>）
  --no-clean          关闭页眉页脚清理与断词还原
  --pages             额外输出逐页文本 pages/pNNN.txt
  --render DPI        每页渲染 PNG（看图 / 表用），例如 --render 150
  --outline           输出候选章节标题（按字号推断）
  --maxpages N        只处理前 N 页
  --no-txt            不输出全文 txt

输出
----
<outdir>/<name>.txt          清理后的全文
<outdir>/<name>.meta.json    元数据 + 每页栏数 + 被拉黑的页眉页脚
<outdir>/<name>.outline.txt  候选章节标题（--outline）
<outdir>/pages/pNNN.txt      逐页文本（--pages）
<outdir>/png/pNNN.png        页面图片（--render）
"""
import argparse
import json
import os
import re
import sys
from collections import Counter

try:
    import fitz  # PyMuPDF
except ImportError:
    sys.exit("需要 PyMuPDF： pip install pymupdf")

LINE = "=" * 72


def clean(t: str) -> str:
    t = t.replace("\u00ad", "")
    t = re.sub(r"[ \t]+", " ", t)
    return t.strip()


# ------------------------------------------------------------ 1. 分栏
def find_gutter(blocks, width, lo=0.28, hi=0.72, max_cross=0.15):
    """在页面中段找一条尽量不穿过任何文本块的竖线；找不到返回 None。"""
    if len(blocks) < 6:
        return None
    n = len(blocks)
    best = None
    for k in range(1, 240):
        xm = width * (lo + (hi - lo) * k / 240.0)
        left = sum(1 for b in blocks if b[2] <= xm)
        right = sum(1 for b in blocks if b[0] >= xm)
        cross = n - left - right
        if left >= 4 and right >= 4 and cross <= max(2, int(n * max_cross)):
            score = min(left, right) - 3 * cross
            if best is None or score > best[0]:
                best = (score, xm)
    return None if best is None else best[1]


def page_blocks(page):
    raw = page.get_text("blocks")
    return [(b[0], b[1], b[2], b[3], clean(b[4]))
            for b in raw if b[6] == 0 and clean(b[4])]


def layout(page, blocks):
    """返回 (ordered_blocks, ncols, gutter_x)"""
    if not blocks:
        return [], 0, None
    W = page.rect.width
    xm = find_gutter(blocks, W)
    if xm is None:
        return sorted(blocks, key=lambda b: (round(b[1], 1), b[0])), 1, None

    full_y = sorted(b[1] for b in blocks if not (b[2] <= xm or b[0] >= xm))

    def band(y):
        return sum(1 for fy in full_y if fy < y - 2)

    def key(b):
        col = 0 if b[2] <= xm else (1 if b[0] >= xm else 0)
        return (band(b[1]), col, round(b[1], 1), b[0])

    return sorted(blocks, key=key), 2, round(xm, 1)


# ------------------------------------------------------ 2. 页眉页脚
def edge_lines(page, top=0.085, bottom=0.915):
    """收集页面上下边缘的行（归一化后用于跨页比对）。"""
    H = page.rect.height
    out = []
    for blk in page.get_text("dict").get("blocks", []):
        if blk.get("type") != 0:
            continue
        for ln in blk.get("lines", []):
            txt = clean("".join(s["text"] for s in ln.get("spans", [])))
            if not txt or len(txt) > 120:
                continue
            y0 = ln["bbox"][1] / H
            y1 = ln["bbox"][3] / H
            if y0 < top or y1 > bottom:
                out.append(txt)
    return out


def norm_key(t: str) -> str:
    """归一化：去页码、压缩空白、小写。"""
    t = re.sub(r"\d+", "#", t)
    t = re.sub(r"\s+", " ", t).strip().lower()
    return t


def find_furniture(docs_pages, min_ratio=0.34, min_count=3, min_len=4):
    cnt = Counter()
    for pg in docs_pages:
        for t in set(edge_lines(pg)):
            k = norm_key(t)
            if len(k) < min_len:          # 太短（如单个图注字母 "a"）不当作页眉页脚
                continue
            cnt[k] += 1
    n = len(docs_pages)
    thr = max(min_count, int(n * min_ratio))
    return {k for k, v in cnt.items() if v >= thr}


def strip_furniture(text: str, black: set) -> str:
    keep = []
    for ln in text.split("\n"):
        if norm_key(ln) in black:
            continue
        keep.append(ln)
    return "\n".join(keep)


# -------------------------------------------------------- 3. 断词还原
def dehyphenate(t: str) -> str:
    t = re.sub(r"([A-Za-z])-\n([a-z])", r"\1\2", t)
    t = re.sub(r"([A-Za-z])-\s*\n\s*([a-z])", r"\1\2", t)
    return t


# ----------------------------------------------------------- 4. 目录
def outline(page, min_size=10.5, max_chars=90):
    hits = []
    for blk in page.get_text("dict").get("blocks", []):
        if blk.get("type") != 0:
            continue
        for ln in blk.get("lines", []):
            spans = ln.get("spans", [])
            if not spans:
                continue
            txt = clean("".join(s["text"] for s in spans))
            if not txt or len(txt) > max_chars:
                continue
            size = max(s["size"] for s in spans)
            bold = any("Bold" in s.get("font", "") for s in spans)
            if size >= min_size or (bold and 8.5 <= size and len(txt) < 60):
                hits.append((round(size, 1), bold, txt))
    return hits


# ------------------------------------------------------------ 主流程
def main():
    ap = argparse.ArgumentParser(description="分栏感知的 PDF 文本抓取")
    ap.add_argument("pdf")
    ap.add_argument("-o", "--outdir", default=None)
    ap.add_argument("--no-clean", action="store_true")
    ap.add_argument("--pages", action="store_true")
    ap.add_argument("--render", type=int, default=0, help="渲染 PNG 的 DPI，0=不渲染")
    ap.add_argument("--outline", action="store_true")
    ap.add_argument("--maxpages", type=int, default=0)
    ap.add_argument("--no-txt", action="store_true")
    a = ap.parse_args()

    src = os.path.abspath(a.pdf)
    if not os.path.isfile(src):
        sys.exit("找不到文件：" + src)
    name = os.path.splitext(os.path.basename(src))[0]
    outdir = os.path.abspath(a.outdir or os.path.join(".", "out", name))
    os.makedirs(outdir, exist_ok=True)

    doc = fitz.open(src)
    n_total = doc.page_count
    n = min(n_total, a.maxpages) if a.maxpages else n_total
    pages = [doc[i] for i in range(n)]

    black = set() if a.no_clean else find_furniture(pages)

    chunks, outlines, per_page = [], [], []
    multi = 0
    sparse = []

    for i, pg in enumerate(pages):
        blocks = page_blocks(pg)
        ordered, ncol, xm = layout(pg, blocks)
        raw = "\n".join(b[4] for b in ordered)

        if black:
            raw = strip_furniture(raw, black)
        if not a.no_clean:
            raw = dehyphenate(raw)
        raw = re.sub(r"\n{3,}", "\n\n", raw).strip()

        if ncol == 2:
            multi += 1
        if len(raw) < 40:
            sparse.append(i + 1)

        per_page.append({"page": i + 1, "cols": ncol, "gutter_x": xm, "chars": len(raw)})
        chunks.append(f"\n\n{LINE}\n[page {i+1} / {n}]  cols={ncol}\n{LINE}\n\n{raw}")

        if a.pages:
            pd = os.path.join(outdir, "pages")
            os.makedirs(pd, exist_ok=True)
            with open(os.path.join(pd, f"p{i+1:03d}.txt"), "w", encoding="utf-8") as f:
                f.write(raw)
        if a.outline:
            for size, bold, t in outline(pg):
                outlines.append(f"p{i+1:>3}  {size:>5}  {'B' if bold else ' '}  {t}")
        if a.render:
            rd = os.path.join(outdir, "png")
            os.makedirs(rd, exist_ok=True)
            pg.get_pixmap(dpi=a.render).save(os.path.join(rd, f"p{i+1:03d}.png"))

    full = "".join(chunks)
    if not a.no_txt:
        with open(os.path.join(outdir, name + ".txt"), "w", encoding="utf-8") as f:
            f.write(full)
    with open(os.path.join(outdir, name + ".meta.json"), "w", encoding="utf-8") as f:
        json.dump({"file": src, "pages": n_total, "processed": n,
                   "metadata": {k: v for k, v in (doc.metadata or {}).items() if v},
                   "furniture_removed": sorted(black),
                   "per_page": per_page}, f, ensure_ascii=False, indent=2)
    if a.outline:
        with open(os.path.join(outdir, name + ".outline.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(outlines))

    print(LINE)
    print(f"  file    : {os.path.basename(src)}")
    print(f"  pages   : {n_total}  processed {n}")
    print(f"  2-col   : {multi} / {n}")
    print(f"  chars   : {len(full):,}")
    print(f"  furniture removed : {len(black)} pattern(s)")
    print(f"  outdir  : {outdir}")
    print(LINE)
    if black:
        for s in sorted(black)[:12]:
            print("   - " + s[:70])
    if sparse:
        print(f"  warning: pages with almost no text (OCR needed?): {sparse[:20]}")
    if len(full) / max(n, 1) < 300:
        print("  warning: very few chars per page -> scanned/image PDF.")


if __name__ == "__main__":
    main()
