# audit/ —— UniCM 复现审计的可复现证据脚本

本目录里每个脚本都只做一件事：**把一个结论从"我说"变成"你可以自己跑出来"**。
所有结论的汇总见仓库根目录的 `UniCM-复现审计.md`。

## 为什么需要这些脚本

论文正文、补充材料、官方代码三者之间有若干不一致。像素读数、参数量、
等值线位置这类结论如果只写在文档里，读者无法验证。所以每个数字都配一个脚本。

## 前置条件

```bash
pip install pymupdf          # 只用到一个第三方库
```

另外两类资源**不在本仓库内**（版权与体积原因）：

| 资源 | 默认位置 | 覆盖方式 |
|---|---|---|
| 论文主文 `UniCM.pdf` | `<外层>/papers/UniCM.pdf`（仓库之外） | 环境变量 `UNICM_PDF` |
| 补充材料 `UniCM_src_SI_MOESM1.pdf` | 仓库根目录 | 环境变量 `UNICM_SI_PDF` |
| 官方代码仓库（脚本 01 需要） | 仓库根目录 `_UniCM_code/` | 见下 |

```bash
# 官方代码（MIT，见论文 Code availability）
git clone --depth 1 https://github.com/tsinghua-fib-lab/UniCM-Global-Climate-Modes.git _UniCM_code
python -m pip install torch        # 脚本 01 需要
```

输出一律写到 `audit/out/`，已被 `.gitignore` 忽略。

```bash
python audit/11_extract_fig3b_contours.py
```

## 脚本清单

| 脚本 | 它证明了什么 | 依赖 |
|---|---|---|
| `01_ffn_and_decoder_probe.py` | FFN 的真实结构是 `Linear(256→256)-ReLU-Linear(256→256)`（**扩张率 1**，非正文所说的中间维度 512）；残差是 **Post-LN** 且 dropout 只在残差分支输出上；`miniDecoder` 克隆的 3 份 add-norm 中 **`sublayer[2]` 从未被调用**（每层 512 死参数，4 层 × 2 分支 = 4,096），且 `sublayer[1]` 被复用两次导致交叉注意力与 FFN 的 LayerNorm 绑定；全模型 **12,726,037** 参数，attention 占 82.7%、FFN 占 16.5% | `_UniCM_code/`、torch |
| `02_render_si_figures.py` | SI Fig.37–41 的数值**只存在于图形里**（页面内嵌文本仅图注），必须渲染后才能读数 | SI PDF |
| `03_crop_si_figure_panels.py` | 把 Fig.37a/37b、Fig.38 放大到 500 dpi，使柱高可读 | SI PDF |
| `04_crop_si_fig38c.py` | Fig.38c：**最小的 S-1 在 6/7 模态最好**，与 SI 正文"capacity 越大越好、模型未饱和"的表述方向相反 | SI PDF |
| `05_locate_main_fig3.py` | 定位并渲染主文 Fig.3（主文 12 页，Fig.3 在第 5 页） | 主文 PDF |
| `06_crop_main_fig3b.py` | 裁出 Fig.3b 并打印**完整图注原文**：只写 "as a function of lead time"，既未交代横轴，也未定义菱形标记 | 主文 PDF |
| `07_crop_main_fig3b_tight.py` | 打印 y 轴刻度位置，建立"像素 y → 预报提前期"的映射（lead 0 在 y=255.5 pt，每 3 个月 ≈10.75 pt） | 主文 PDF |
| `08_zoom_main_fig3b_columns.py` | 打印条带下方文字：**只有 7 个模态名，没有任何横轴刻度标签**；并放大单条带查看内部结构 | 主文 PDF |
| `09_probe_fig3b_drawings.py` | Fig.3b 内**没有内嵌位图**，说明色场是矢量绘制的（第一版探测） | 主文 PDF |
| `10_probe_fig3b_cell_geometry.py` | 量出 7 条条带精确位置与宽度（x=62.4/96.5/130.6/164.8/198.9/233.0/267.1，宽 **31.28 pt**，间距 34.1 pt）；色场由**多顶点填充多边形**构成 → 是 2 维场而非 1 维曲线。**这是判定"存在未标注横轴"的关键证据** | 主文 PDF |
| `11_extract_fig3b_contours.py` | 从矢量数据抽出 0.5 等值线几何并换算成提前期；**TNA 的 0.5 线被切成 4 段**（4.4/15.1/15.9/19.1 个月），说明其技巧在阈值附近抖动，不应引用单一数字 | 主文 PDF |
| `12_locate_main_fig2.py` | 定位 Fig.2 并列出其季节刻度：**12 个三个月滚动季节 DJF…NDJ，间距 8.1 pt** | 主文 PDF |
| `13_locate_si_fig13_23_25.py` | **定论性证据**：SI Fig.13 是同一批图的 XRO 版，横轴明确标注 **"Target season"** 并列出 12 个季节、色标同为 −1.00…+1.00 → 由此确证主文 Fig.3b 被删掉的横轴就是目标季节。同时得到 SI Fig.23 对显著性点的定义 | SI PDF |

## 两条走了弯路的脚本（已删除，记录在此）

| 脚本 | 为什么放弃 |
|---|---|
| `_test_12cols.py` | 想通过统计色块左沿的量化间距来判断横轴列数，但 31.28 pt 宽度下 **2.6 pt 与 1.3 pt 两种量子都能拟合观测**，无法区分 12 列还是 24 列 —— 方法本身不具判别力 |
| `_compare_fig2b_fig3b.py` | 想把 Fig.2b 旋转 90° 后与 Fig.3b 的 ENSO 条带比对，但**取矩形时裁到了 Fig.2a 的折线图**，比对对象错误。最终由脚本 13 直接给出答案 |

## 一个诚实的限制

`06`、`11`、`04` 里的数值是**从图形读出来的**（像素读数或矢量几何换算），
不是论文给出的表格数据。凡引用必须带"约 ±0.01"之类的限定，
**不能当作论文的原始数值使用**。论文的原始数值应当以正文/SI 给出的数字为准。
