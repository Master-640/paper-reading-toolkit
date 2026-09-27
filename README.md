# 文献阅读工具箱 · Paper Reading Toolkit

把「**读一批论文 → 输出可汇报的材料**」这件事工具化。

这个仓库来自一次真实的文献精读工作：5 篇气候 / 复杂网络方向的论文（ENSO 动力学、气候模态耦合预报、图神经网络强化学习、地球系统长时预报），配套两个自建工具：

| 工具 | 作用 | 依赖 |
|---|---|---|
| **[`paper-dag/`](paper-dag/)** | 把多篇论文串成**有向图（DAG）**，点节点看概述，一键打开 PDF / 飞书文档 / 本地笔记 | 无（纯原生 HTML/CSS/JS） |
| **[`pdf-tool/`](pdf-tool/)** | 把 PDF 转成**能直接分析**的干净文本：分栏不错行、页眉页脚已清理、断词已还原 | PyMuPDF |
| **[`audit/`](audit/)** | **复现审计的可执行证据**：论文正文 / 补充材料 / 官方代码三者的逐项对照，每个数字都有脚本可复跑 | PyMuPDF（+ torch 仅一个脚本） |

---

## ✨ 为什么需要这两个工具

**`pdf-tool`** 解决 `pdftotext` 的三个通病：

| 通病 | 做法 |
|---|---|
| **双栏按「行」交错**，左右栏内容串在一起，读不了 | 取每个文本块的 bbox → 在页面中段找一条**不穿过任何块的中缝** → 按 `(横带, 栏, y)` 重排 |
| **页眉页脚混进正文**（期刊名、DOI、页码） | 跨页统计上下边缘带的重复行，归一化后拉黑 |
| **行尾连字符断词** `cli-\nmate` | 还原成 `climate` |

修复前后（UniCM 的 Discussion）：

```
修复前：driver of ENSO onset than the MJO29. ...        Discussion
        normal years, the interaction matrix ...         This work demonstrates that ...
        ↑ 左栏 Results 的尾巴和右栏 Discussion 挤在同一行

修复后：Discussion
        This work demonstrates that a substantial portion of the global climate's
        predictability is an emergent property of the interactions among climate modes, ...
```

**`paper-dag`** 解决另一个问题：读完 5 篇之后，**它们之间的关系只在自己脑子里**。这里把它画成图，并且每条边都标了**依据强度**（明确 / 同题 / 方法 / 推断），点开就能看到概述与批判性备注。

```
地基                  技术                应用·对标
┌────────────────┐
│ ENSO complexity │──明确──►┌────────────────┐
│   理论地基       │──同题──►│ ENSO-CausalNet │
└────────────────┘        │    同期对标      │
┌────────────────┐        └────────────────┘
│ FINDER          │──方法──►       ▲ 同题（双向）
│   方法来源       │──方法──┐       ▼
└────────────────┘        └─►┌────────────┐
┌────────────────┐           │   UniCM    │
│ TritonCast      │──同题────►│  当前主攻   │
│   技术对标       │           └────────────┘
└────────────────┘
```

---

## 🚀 快速开始

```bash
git clone <this-repo>
cd <repo>
```

### 1. 论文 DAG 阅读地图

```bash
cd paper-dag

# 方式 A：直接打开（无需依赖）
start index.html          # Windows
open  index.html          # macOS

# 方式 B：本地服务器（推荐，所有相对链接都能用）
python serve.py           # 自动打开浏览器，默认 8765 端口
python serve.py 9000      # 换端口
```

### 2. PDF 自动抓取

```bash
cd pdf-tool
python -m pip install pymupdf

# 最简
python extract.py your.pdf

# 常用：指定输出目录 + 生成章节标题清单
python extract.py your.pdf -o out/paper --outline

# 全套：章节清单 + 逐页文本 + 页面图（用来看公式与图表）
python extract.py your.pdf -o out/paper --outline --pages --render 150
```

输出：

```
out/<name>/
├── <name>.txt          清理后的全文（按页分段，带 [page i / n] 标记）
├── <name>.meta.json    元数据 + 每页栏数/中缝位置/字符数 + 拉黑的页眉页脚
├── <name>.outline.txt  候选章节标题（--outline）
├── pages/pNNN.txt      逐页文本（--pages）
└── png/pNNN.png        页面图片（--render）
```

---

## 📄 关于论文全文

**本仓库不包含任何论文原文（PDF）或全文文本。** 原因：

- 论文 PDF 受版权保护（Nature / Nature Machine Intelligence / GRL 等），不能二次分发；
- 提取出的全文文本性质相同。

仓库里保留的是**自己写的阅读笔记与批判性分析**。若你需要全文：

1. 从出版方 / 学校图书馆获取 PDF；
2. 用本仓库的 `pdf-tool` 自己提取：`python extract.py your.pdf --outline`

> `paper-dag` 里每个节点的「本地笔记」链接指向 `*.txt` / `*.md`。
> 其中 `.md` 笔记随仓库提供；`.txt` 全文需按上面步骤自行生成。

---

## 📁 目录结构

```
.
├── paper-dag/                 # 工具一：论文 DAG 阅读地图
│   ├── index.html             #   界面（零依赖）
│   ├── data.js                #   ★ 全部内容：节点 / 连线 / 概述
│   ├── start.cmd              #   双击启动
│   ├── serve.py               #   本地服务器
│   ├── _check.js              #   自检脚本
│   └── README.md
├── pdf-tool/                  # 工具二：PDF 自动抓取
│   ├── extract.py
│   └── README.md
├── audit/                     # 工具三：复现审计的可执行证据
│   ├── _paths.py              #   路径解析（PDF 不在仓库内）
│   ├── 01_…13_*.py            #   13 个探针脚本，见 audit/README.md
│   ├── README.md              #   每个脚本证明了什么
│   └── out/                   #   输出（gitignore）
├── FINDER/                    # 第三方：FINDER 官方实现（MIT，见 LICENSE）
├── GRACO/                     # 图组合优化的深度 RL 框架（MIT）
├── PPT素材/ 代码截图/          # 组会材料
├── UniCM-复现审计.md           # ★ 正文 / SI / 官方代码 三者不一致的审计记录
├── UniCM-*.md                 # 其余阅读笔记：UniCM
├── FINDER-*.md                # 阅读笔记：FINDER
├── enso_oscillator_logic_chain.md / .png   # ENSO 振子逻辑链图
├── feishu_week2_summary.md    # FINDER 组会笔记
├── LICENSE
└── README.md
```

---

## 🔍 `audit/` —— 为什么还要审计

读完一篇论文只是第一步。**论文正文、补充材料、官方代码三者经常互相矛盾**，
而这类矛盾恰恰是复现时最致命的。`audit/` 把这些矛盾逐条落地成可复跑的检查：

| 类型 | 例（UniCM） | 手段 |
|---|---|---|
| 配置不一致 | 正文说 FFN 中间维度 512，发布配置实际 256 | 读 `config.py` + 跑 `torch` 实例化量参数量 |
| 机制不存在 | 正文 Eq 6/7 的每模态投影矩阵 `W_m`，代码里是广播裸加 | 全仓库 grep + 逐行读 `models.py` |
| 数据口径不一致 | 5 个输入通道来自 **3 个不同 CMIP6 模式**，样本内物理量不属于同一气候态 | 逐行读 `LoadData.py` 的硬编码路径 |
| 图与文反向 | SI 正文说「模型越大越好」，图上最小的 S-1 在 6/7 模态最好 | 渲染 SI 图 + 从 PDF 矢量数据读等值线 |
| caption 缺信息 | 主图 Fig.3b 漏写横轴（实为 12 个目标季节），靠 SI 的同款图才补上 | 量条带几何 + 找到 SI 里标全轴的孪生图 |

```bash
python -m pip install pymupdf
python audit/11_extract_fig3b_contours.py     # 例：抽 Fig.3b 的 0.5 等值线位置
```

> **注意**：`audit/` 的输出是**从图形读出的近似值**（像素读数、矢量几何换算），
> 不等同于论文的原始数值。凡引用都带精度限定，见 `audit/README.md` 末节。

---

## 🧩 扩展 `paper-dag`

只改 `paper-dag/data.js`，不用碰 `index.html`：

| 想改什么 | 改哪里 |
|---|---|
| 增删论文 | `NODES` 数组（`col` / `row` 控制位置） |
| 改概述 | 节点的 `overview`（支持 `**粗体**` 与 `` `代码` ``） |
| 改连线 | `EDGES` 数组：`from` / `to` / `relation` / `basis` |
| 加飞书文档 | `feishu: { wiki, docx, label }` |
| 加本地笔记 | `notes: [["显示名", "相对路径"]]` |

改完自检：

```bash
cd paper-dag && node _check.js
```

> 飞书域名默认 `https://feishu.cn`，页面右上角可改（存 localStorage）。

---

## ⚠️ 已知限制

| 限制 | 说明 |
|---|---|
| 扫描件 / 图片型 PDF | `pdf-tool` 无法处理（需 OCR，本仓库未包含 tesseract） |
| 公式提取 | 会打散成乱码，建议 `--render` 出图用视觉读 |
| 复杂表格 | 文本能出来但行列对齐会乱 |
| `--outline` | 只稳定抓一级标题，**是候选清单不是目录** |
| `paper-dag` 连线 | **是阅读地图，不代表论文之间的引用关系** |
| 分析环节 | 抓取可全自动；**结论必须逐篇读过才能下，不能全自动** |

---

## 🙏 致谢与许可

- **代码与笔记**：MIT License，见 [`LICENSE`](LICENSE)。
- **`FINDER/`**：FINDER 官方实现，版权归原作者，MIT License —
  Changjun Fan, Li Zeng, Yizhou Sun, Yang-Yu Liu,
  *Finding key players in complex networks through deep reinforcement learning*,
  Nature Machine Intelligence **2**, 317–326 (2020).
  [doi:10.1038/s42256-020-0177-2](https://doi.org/10.1038/s42256-020-0177-2)
- **`GRACO/`**：MIT License。
- 本仓库**不包含**任何受版权保护的论文原文或全文文本。
