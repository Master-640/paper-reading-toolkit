# pdf-tool —— PDF 自动抓取与清洗

把论文 PDF 一键转成**能直接拿去分析**的干净文本：分栏不错行、页眉页脚不混入、断词能还原。

```
python extract.py "..\..\..\papers\UniCM.pdf" -o out\UniCM --outline --pages
```

---

## 一、能力总表

| 能力 | 实现方式 | 状态 |
|---|---|---|
| **PDF → 文本** | 分栏感知：找中缝 → 按 `(横带, 栏, y)` 重排 | ✅ |
| **页眉页脚清理** | 跨页统计页面上下 8.5% 边缘带的重复行，归一化后拉黑（出现率 ≥34%） | ✅ |
| **行尾断词还原** | `cli-\nmate` → `climate` | ✅ |
| **元数据** | 标题 / 作者 / DOI / 页数 → `<name>.meta.json` | ✅ |
| **章节标题推断** | 按字号 + 粗体 → `<name>.outline.txt`（**候选清单**，非完整目录） | ✅ |
| **逐页文本** | `pages/pNNN.txt` | ✅ |
| **页面渲染成图** | `--render 150` → `png/pNNN.png`，可直接用视觉读**公式与图表** | ✅ |
| **扫描件 OCR** | 需要 tesseract | ❌ 本机未装 |
| **表格精确提取** | pdfplumber（已装，可另写） | ⚙️ 备用 |
| **图里读数值** | 需要视觉判读 | ⚙️ 手动 |

---

## 二、它解决的三个通病

| # | 通病 | 做法 |
|---|---|---|
| 1 | **双栏按“行”交错**，左右栏内容串在一起（`pdftotext` 的老毛病） | 用 PyMuPDF 取每个文本块的 bbox → 在页面中段（28%–72%）找一条**不穿过任何块的中缝** → 按 `(横带, 栏, y)` 重排 |
| 2 | **页眉页脚混进正文**（期刊名、DOI、页码） | 跨页统计边缘带重复行，归一化（数字→`#`、小写）后拉黑，长度 < 4 的短行不参与（避免误删图注字母） |
| 3 | **行尾连字符断词** | 正则还原 |

### 效果对照（UniCM 的 Discussion）

```
修复前（pdftotext -layout）
  driver of ENSO onset than the MJO29. Conversely, in the lead-up to      Discussion
  normal years, the interaction matrix remains largely uniform, indicat-  This work demonstrates that ...
  ↑ 左栏 Results 的尾巴和右栏 Discussion 挤在同一行，读不了

修复后（extract.py）
  Discussion
  This work demonstrates that a substantial portion of the global climate’s predictability is
  an emergent property of the interactions among climate modes, ...   ← 完整连续
```

断词同时修好：`global cli-mate’s` → `global climate’s`，`fund-amental` → `fundamental`。

---

## 三、做不到的（先看清楚，别踩坑）

| 场景 | 原因 |
|---|---|
| **扫描件 / 纯图片 PDF** | 本机**没装 tesseract**，无法 OCR。脚本会检测「每页字符 < 40」并告警 |
| **公式** | 会被打散成乱码（如 `𝒪𝒪((TN)2)`）。要读公式请 `--render` 出图，用视觉读 |
| **复杂表格行列** | 文本能出来但对齐会乱；要精确提取需用 pdfplumber 单独写 |
| **图里的数值** | 无法从曲线读数据，只能看图 |
| **二级小标题** | `--outline` 目前只稳定抓到与正文不同字号的一级标题（如 UniCM 的 `Multi-view architecture...` 这类同字号小标题会漏）→ **是候选清单，不是目录** |

---

## 四、启动方式

```powershell
cd "D:\collections2026\硕博申请\中国科学院计算所\学习材料\刘睿涵\刘睿涵\汇报合集\week2汇报\pdf-tool"

# 最简
python extract.py "..\..\..\papers\UniCM.pdf"

# 常用：指定输出目录 + 章节清单
python extract.py "..\..\..\papers\UniCM.pdf" -o out\UniCM --outline

# 全套：章节清单 + 逐页文本 + 页面图（看公式/图表用）
python extract.py "..\..\..\papers\UniCM.pdf" -o out\UniCM --outline --pages --render 150

# 批量（PowerShell）
$p = "..\..\..\papers"
foreach ($n in 'UniCM','ENSO-complexity','ENSO-CausalNet','FINDER','TritionCast') {
  python extract.py "$p\$n.pdf" -o "out\$n" --outline
}
```

> ⚠️ `TritionCast.pdf` 有 **47.9 MB / 154 页**，单独跑约 1–2 分钟，建议用 `run_in_background` 或加 `--maxpages`。

### 参数

| 参数 | 说明 |
|---|---|
| `-o, --outdir DIR` | 输出目录（默认 `./out/<pdf名>`） |
| `--outline` | 推断章节标题 → `<name>.outline.txt` |
| `--pages` | 逐页文本 → `pages/pNNN.txt` |
| `--render DPI` | 页面渲染成 PNG → `png/pNNN.png` |
| `--no-clean` | 关闭页眉页脚清理与断词还原 |
| `--maxpages N` | 只处理前 N 页 |
| `--no-txt` | 不输出全文 txt |

### 输出结构

```
out/<name>/
├── <name>.txt          清理后的全文（按页分段，带 [page i / n] 标记）
├── <name>.meta.json    元数据 + 每页栏数/中缝位置/字符数 + 被拉黑的页眉页脚清单
├── <name>.outline.txt  候选章节标题（--outline）
├── pages/pNNN.txt      逐页文本（--pages）
└── png/pNNN.png        页面图片（--render）
```

---

## 五、本机实测结果（本轮 5 篇全部跑通）

| 论文 | 页数 | 判定为双栏的页 | 抓出字符 | 清掉的页眉页脚 |
|---|---|---|---|---|
| UniCM | 12 | 9 | 74,725 | 3 |
| ENSO-complexity | 11 | 8 | 76,974 | 6 |
| ENSO-CausalNet | 12 | 0（单栏） | 55,780 | 4 |
| FINDER | 8 | 6 | 46,990 | 3 |
| TritionCast | 154 | 2 | 318,949 | 1 |

UniCM 自动推断出的章节清单（节选）：

```
p  1  26.0  B  Learning the coupled dynamics of global climate modes
p  2  10.8  B  Results
p  7  10.8  B  Discussion
p  8  10.8  B  Methods
p 10  10.8  B  Data availability / Code availability
p 11  10.8  B  References
```

---

## 六、抓取 vs 分析的边界（重要）

| 环节 | 自动化程度 |
|---|---|
| **抓取 + 清洗 → 干净文本 + 元数据 + 章节清单 + 页面图** | **全自动**，一条命令跑完 5 篇 |
| **分析**（概述 / 贡献 / 框架 / 实验 / ablation / 批判） | **由模型逐篇阅读后撰写**。可以再包一层自动生成**骨架笔记**，但结论段必须逐篇核对，**不能全自动** |

也就是说：这个工具负责"把论文变成可分析的材料"，**结论仍然要人（或模型）读过之后才能下**。

---

## 七、环境依赖

```powershell
python -m pip install pymupdf          # 必需
# 可选：pdfplumber（复杂表格）、pypdf（加密/合并）
```

本机已装：**PyMuPDF 1.28.0**、pdfplumber、pypdf、PIL、matplotlib。**未装 tesseract**。

---

## 八、配套工具

| 工具 | 位置 | 作用 |
|---|---|---|
| **论文 DAG 阅读地图** | `..\paper-dag\` | 把 5 篇论文串成 DAG，点节点看概述、打开 PDF / 飞书文档 |
| **本工具** | `.\` | 把 PDF 变成可分析的干净文本 |
