# UniCM 材料清单与阅读索引

> 生成时间：2026-09-25 ｜ 论文：*Learning the coupled dynamics of global climate modes*（UniCM）
> Nature Machine Intelligence, Vol. 8, pp. 930–941 (2026) ｜ DOI: 10.1038/s42256-026-01245-5
> 配套文件：`UniCM-论文阅读.md`（此前的六段框架笔记）、`UniCM-审稿争议要点.md`（审稿文件提炼）

---

## 0. 一句话结论

**正文 PDF 一直就在用户磁盘上**：`D:\collections2026\硕博申请\中国科学院计算所\学习材料\刘睿涵\刘睿涵\papers\UniCM.pdf`（2.84 MB，12 页），已转成 `UniCM_main.txt`（135.8 KB）。
此前我判断"正文拿不到"是**搜索范围错误**：只搜了会话工作区 `week2汇报\`，没往上一层的 `papers\` 找。公开渠道无开放版本这一结论本身没错（OpenAlex `is_oa=false`、Europe PMC 命中 0、Semantic Scholar `isOpenAccess=false`、Unpaywall 无 OA 链接），但本地有文件就不需要它。

现在本地是**全套无缺口**：正文全文 + 摘要/图题/67 refs/声明 + 5 张主图高清图 + 补充材料全文（4 表 41 图）+ 完整审稿文件（审稿人意见与作者逐条回应）。

> 版本说明：这份 PDF 是 **accepted article**（首页 `Published online: xx xx xxxx`，PDF 属性 CreationDate 2026-05-26；Received 12 Oct 2025 / Accepted 20 Apr 2026）。因此卷期页码（Vol. 8, 930–941）来自出版社网页，PDF 本身未印。
> `papers\` 目录里还有 4 篇：`ENSO-CausalNet.pdf`、`ENSO-complexity.pdf`、`TritonCast.pdf`、`FINDER.pdf`。

---

## 1. 本地文件清单

| 文件 | 体积 | 内容 | 来源 / 可信度 |
|---|---|---|---|
| **`UniCM_main.txt`** | **135.8 KB** | **正文全文文本**（12 页，含 Introduction / Results / Methods / 图注 / 参考文献）；双栏排版导致左右栏文字在同一行交错，引用前需回读上下文 | 由 `papers\UniCM.pdf` 用 `pdftotext -layout -enc UTF-8` 转出（官方 accepted 版） |
| `UniCM_正文网页可获取部分.md` | 15.3 KB | 书目信息、**摘要逐字**、章节骨架、5 个图题、补充材料说明、Data/Code availability、致谢、伦理与审稿声明、**67 条参考文献全表** | nature.com 论文页（官方） |
| `UniCM_src_SI_MOESM1.pdf` | 9.3 MB | 补充材料原 PDF | nature.com / Springer（官方） |
| `UniCM_SI1.txt` | 101 KB | SI 全文文本：Supplementary Notes 1–6、Table 1–4、Fig 1–41、SI 参考文献 | 由上一行 PDF 用 `pdftotext -layout -enc UTF-8` 转出 |
| `UniCM_src_SI_MOESM2.pdf` | 7.0 MB | **Peer Review File 原 PDF**（3 位审稿人 + 作者逐条回应） | Springer（官方） |
| `UniCM_SI2.txt` | 109 KB | Peer Review File 全文文本 | 同上 |
| `UniCM_Fig1.png` … `UniCM_Fig5.png` | 0.3–0.9 MB/张 | 5 张主图**高清原图**（Fig.1 架构图已目视确认清晰） | media.springernature.com |
| `UniCM_nature_page.html` | 363 KB | 论文页原始 HTML，供复查或二次提取 | 抓取留存 |
| `UniCM-审稿争议要点.md` | — | 审稿人关切 / 作者回应 / 软肋清单（含行号） | 由 `UniCM_SI2.txt` 提炼 |
| `UniCM-SI图表页码索引.md` | 7.3 KB | **41 张补充图 + 4 张补充表的页码定位表**（页脚印刷页 / 渲染用物理页），附 `pdftoppm` 渲染命令 | 由 `UniCM_SI1.txt` 生成并逐页回验 |
| `_probe_unicm_sources.mjs` / `_fetch_unicm_page.mjs` / `_extract_unicm_page.mjs` / `_probe_unicm_more.mjs` / `_index_si_figures.mjs` | — | 抓取、提取、页码索引脚本（可复现）；`_probe_unicm_sources.json`、`_unicm_repo_tree.json` 为探测记录 | 本地 |

图题（用于做 PPT 图注，原文）：
- **Fig. 1** Overview of the UniCM architecture.
- **Fig. 2** ENSO forecast performance of UniCM.
- **Fig. 3** UniCM's performance on forecasting global climate modes.
- **Fig. 4** Spatial prediction skill of SST across different lead times.
- **Fig. 5** UniCM's attention mechanism reveals event-specific precursors to major ENSO events.

代码仓库（官方）：https://github.com/tsinghua-fib-lab/UniCM-Global-Climate-Modes —— 只有 `src/`（`models.py`、`Trainer.py`、`LoadData.py`、`app_train.py`…）与 `assets/framework.png`，**没有正文 PDF、没有数据**。

---

## 2. 正文文本的获取方式（已完成）

```powershell
pdftotext -layout -enc UTF-8 "D:\collections2026\硕博申请\中国科学院计算所\学习材料\刘睿涵\刘睿涵\papers\UniCM.pdf" UniCM_main.txt
```

- `papers\` 在会话工作区**之外**：读没问题（本次已验证），但**写受限于工作区**，所以产物一律落在 `week2汇报\` 里。
- 双栏 PDF 用 `-layout` 转出后，左右两栏会挤在同一行（例如 L64 同时含左栏"existing approaches…"与右栏"UniCM's ability…"）。**引用某一句前务必回读上下文**，不要直接摘半行。

---

## 3. 关键信息定位表（行号可直接在本目录文本里检索）

### `UniCM_SI1.txt`（补充材料）

| 想找什么 | 位置 |
|---|---|
| 四个 baseline 的架构/训练配置（CNN、ResoNet、XRO、DESN） | **L484–510**（Supplementary Table 1） |
| UniCM 全部超参：D=256、8 层（4 enc + 4 dec）、4 头、dropout 0.2、5°×5°、patch 2×2、12→24、AdamW 5e-4、warmup 3 epoch、cosine、wd 1e-6、batch 32、grad clip 1.0、200 epoch、seed 1–20、单卡 A100 80 GB | **L516–538**（Supplementary Table 2） |
| 数据集与时段：CMIP6 1850–2014 训练；ORAS5 1958–1980 验证；ERA5 / GODAS / ORAS5 / SODA v2.2.4 测试 | **L543–557**（Supplementary Table 3） |
| **7 个模态的精确 SSTA 定义与经纬度框**（ENSO 170°–120°W,5°S–5°N；NPMM；SPMM；IOB；IOD；SIOD；TNA） | **L563–572**（Supplementary Table 4） |
| 为什么 DESN 没走 OOD 协议（CMIP6 预训练后性能严重退化 → 改为 ORAS5 1980 前训练、1980 后测试，16 个月） | **L67–74** |
| ENSO 有效提前期 19 个月（ERA5 验证，baseline 8–14 个月） | **L89** |
| 春季可预报性障碍：目标 MAM/AMJ 仍维持 ACC>0.5 到 **14 个月**；MAM 目标 12 个月 lead ACC≈0.6；baseline 9 个月内崩塌 | **L100–113** |
| 事件检测：El Niño/La Niña 阈值定义（±0.5 °C 连续 ≥5 个月）、Accuracy + CSI 指标 | **L120–140** |
| **强事件幅度收缩（intensity shrinkage）**：|Index|>1.5 °C 时预测幅度系统性偏小 | **L151–156** |
| XRO 的季节性局限：IOB 约 6 个月、SIOD/TNA 5–6 个月内跌破 0.5、NPMM 8 个月后骤降 | **L179–190** |
| 场级对比：12 个月**全球平均 ACC=0.379** vs FuXi-S2S、CAS-Canglong | **L271, L287** |
| 滞后相关与遥相关复现（IOD–SPMM 等） | **L302–350** |
| 消融 / 超参 / 数据量 / 分辨率敏感性 | **L351–458**（5.1–5.4） |
| 预训练与微调分析 | **L459 起**（第 6 节） |
| 数据效率：**随机 40% 子集≈全量性能；时间连续子集反而退步** | **L424–430** |

### `UniCM_SI2.txt`（Peer Review File）

- 总体评价 + 第一位审稿人的两条 major 关切（"写给气候学家而非 Nat MI 读者"、"像是 XRO 的升级版"）→ **L24–54**
- 逐条 minor：CMIP6 选择、5° 分辨率、验证只做 ORAS5 是否只用海洋变量、DESN 协议 → **L156, L264, L376, L429, L507–537**
- `Reviewer #1 (Remarks on code availability)` → **L539**
- 第二位审稿人总体意见（**recommend major revisions**，指出基线公平性、独立性、不确定度量化、物理一致性未充分处理）→ **L553–560**；`Major comments/suggestions` 从 **L567** 起
- 审稿人姓名（论文页 Peer review 声明）：**Annalisa Bracco、Maximilian Gelbrecht、Jing-Jia Luo**
- 逐条关切/回应/软肋的完整整理见 `UniCM-审稿争议要点.md`

### `UniCM_main.txt`（正文，12 页；accepted 版）

| 想找什么 | 位置 |
|---|---|
| **前人工作及其不足的总述（Introduction）** | **L64–75** |
| 摘要 | L20–40 |
| 双视角架构总览（Globalformer / Modeformer、cross-view guidance） | L79–115 |
| 方法细节：5 个物理量、5° 网格、mode-to-patch guidance 公式 | L1035–1148 |
| ÎG vs ÎM、additive vs replacement 消融 | L1132–1144 |
| 训练设置全套（AdamW 5e-4、wd 1e-6、warmup 3、cosine、batch 32、clip 1.0、dropout 0.2、200 epoch、20 seeds、A100 80 GB、~6 GPU-hours） | L1133–1147 |
| 结果：ENSO 19 个月 / IOD 7 个月、多模态统一预报 | L115–117、L234–238、L944 |
| 春季可预报性障碍 | L235–238 |
| 事件多样性（1997–98 极端 El Niño、2020–2023 三重 La Niña） | L335、L356–367 |
| 滞后相关（NPMM–ENSO 约 4 个月、TNA–ENSO、SIOD–IOB 3 个月 r ≈ +0.4） | L541–567 |
| 场级对比 FuXi-S2S / CAS-Canglong（0.379 vs 0.262 / 0.207） | L662–666 |
| 注意力可解释性与事件先兆（65.3% / 76.9% / 53.8% / 69.2%；5.04× / 3.14× / 3.79× / 2.64×） | L648–675 |
| 数据 / 代码可用性、致谢 | L1161 起 |

---

## 4. 既有笔记 `UniCM-论文阅读.md` 的数字核对结果

用 SI 原件逐条验过，**已证实**：

| 笔记里的说法 | 核对结果 |
|---|---|
| ENSO 有效提前期 19 个月 | ✓ SI1 **L89** |
| 场级 12 个月全球平均 ACC 0.379（FuXi-S2S 0.262 / CAS-Canglong 0.207） | ✓ SI1 **L287**（0.379 与对比模型名一致） |
| 165 年、12+24 滑窗 → **1,945 个样本** | ✓ SI1 **L426**；SI2 **L338** 给出公式 N=1980−12−24+1 |
| **40% 训练样本≈全量性能**，时间连续子集反而退化 | ✓ SI1 **L429**；SI2 **L355** |
| CMIP6 筛选后保留 **CESM2 与 EC-Earth3 两家族共 4 个模拟** | ✓ SI2 **L383**：CESM2-FV2、CESM2-WACCM-FV2、EC-Earth3-CC、EC-Earth3-… |
| 春季障碍下维持 ACC>0.5 到 14 个月；baseline 9–12 个月失效 | ✓ SI1 **L105–111** |
| XRO 对 TNA/SIOD/IOB 5–6 个月跌破 0.5 | ✓ SI1 **L184–186** |
| Table 2 的全部超参（D=256、8 层、4 头、5°、batch 32、20 seeds、A100 80 GB、6 GPU-hours 量级） | ✓ SI1 **L516–538**（"6 GPU-hours"本身在 SI 里未出现，属待正文核对） |

另有两条**部分被 SI 支持**：
- "RMSE 相对 baseline 降低 14.1–17.9%"：SI1 **L90** 给出的同量级结果是「12 个月 lead 时 RMSE 相对 baseline 降低约 **16.7%**」；场级 12 个月全球平均 RMSE 为 **0.465**，FuXi-S2S 0.505、CAS-Canglong 0.515（SI1 **L289–290**）。
- "ENSO 19 个月"与"场级 ACC 0.379"的**对比对象**也顺带确认：CAS-Canglong 在 **9 个月**、FuXi-S2S 在 **6 个月**跌破 0.5（SI1 **L271–272**）。

**原先"待正文核对"的项，现已在正文中全部证实**（`UniCM_main.txt`；行号为关键词定位用，双栏交错）：

| 项目 | 正文位置 |
|---|---|
| IOD 有效提前期 **7 个月**（ENSO 19 个月） | L117、L944 |
| RMSE 相对 baseline 降低 **14.1–17.9%** | L229 |
| 注意力占比 **65.3% / 76.9%**（1983、1997 El Niño）、**53.8%**（1995 La Niña，热带大西洋）、**69.2%**（西太平洋） | L648、L658、L662 |
| 交互强度放大 **5.04× / 3.14×**（TNA）、**3.79× / 2.64×**（NPMM） | L673、L675 |
| 训练 **~6 GPU-hours**、单卡 A100 80 GB、PyTorch v2.0 | L1146–1147 |
| Received 12 Oct 2025 / Accepted 20 Apr 2026 | 首页 L10–13 |
| "**Globalformer 导出指数 ÎG 优于 Modeformer 直接输出 ÎM**" | L1132–1144（"the Globalformer-derived indices outperform direct Modeformer outputs (Supplementary Fig. 37b)"） |
| 损失权重 λ1 = λ3 = 1.0、λ2 = 0.01 | 正文 Training objective 节（与 SI1 L396–405 一致）→ 笔记 L67 的 λ2/λ3 仍属**笔误** |

结论：`UniCM-论文阅读.md` 中可核验的数字**全部正确**，唯一需要修的就是 λ 系数的写法。

---

### 4.1 ⚠️ 发现一处需要修正的地方（损失权重写反了）

`UniCM-论文阅读.md` 第 67 行写的是：`λ1=1, λ2=1, λ3=0.01`。
SI 原文（**SI1 L396–405**，式 (7)）为：

```
L_total = λ1 · L_field + λ2 · L_global-mode + λ3 · L_mode-aux
其中 L_field      = Globalformer 的物理场 MSE
     L_global-mode = 由 Globalformer 预测场导出的模态指数 MSE
     L_mode-aux    = Modeformer 的模态指数 MSE
灵敏度分析确认 λ1 = λ3 = 1.0，λ2 = 0.01
```

即 **是二级项（Globalformer 导出的模态指数）取 0.01，Modeformer 辅助项取 1.0**，而笔记把 λ2 与 λ3 的数值对调了。
（SI2 里同一段文字因公式是图片而缺失数值，无法二次验证；以 SI1 的式 (7) 与文字为准。汇报时若被问损失权重，按 SI1 表述。）

---

## 5. SI 里值得补进汇报的新细节（既有笔记未覆盖）

1. **强事件幅度收缩（intensity shrinkage）**：|Index|>1.5 °C 时 UniCM 预测幅度系统性偏小，作者解释为极端样本少 + 方差惩罚损失所致，并强调符号与相对强度仍可靠（SI1 L151–156）。——汇报里主动讲这个，比只讲 ACC 更显扎实，也直接对应审稿人的"物理一致性"关切。
2. **事件化评估指标**：除 ACC/RMSE 外，还有 Accuracy + **CSI（Threat Score）**，并指出 XRO 的 CSI 随 lead time 显著下降、有"偏向 Neutral"的保守偏差（SI1 L120–140）。
3. **lagged correlation / 遥相关的 SI 层面证据**（SI1 L302–350），比主文 Fig.5 更细。
4. **对比的公平性细节**：所有 DL 基线都做了 lr 网格搜索（1e-3 / 5e-4 / 1e-4），200 epoch + early stopping patience 10（SI1 L52–54）。
5. **评估协议的关键区别**：UniCM 与 baseline（除 DESN）走的是 **out-of-distribution** 协议（CMIP6 训练 → 观测再分析测试），DESN 因分布差异只能走 in-distribution（SI1 L55–74）。这是理解"为什么和 DESN 的 16 个月不能直接比"的关键。
6. **5° 分辨率是做过敏感性验证的**（SI1 L440–457，5.4 节）：更细的网格会显著增加空间 token 数与注意力开销，并引入在这些 lead time 上基本随机的次网格变率，**对可预报性收益有限**——即 5° 不只是"对齐模态尺度"的说辞，有实验支撑。
7. **微调收益有限有原文依据**（SI1 L459–470，第 6 节 + Supplementary Fig. 41）：论文自己解释为"预训练已学到基本动力学 + 20 年再分析数据对年代际振荡的采样周期太少"，因此微调只是局部校准而非物理重构。

---

## 6. 建议的阅读顺序（面向组会汇报）

1. `UniCM-论文阅读.md` + 本目录摘要 → 先建立全局印象（30 分钟）
2. `UniCM_Fig1.png` + Supplementary Table 2 → 把方法讲清楚（架构 + 超参）
3. `UniCM_SI1.txt` 第 1–5 节（L44–458）→ 实验设置、鲁棒性、消融、数据效率
4. `UniCM-审稿争议要点.md` → 准备问答环节（审稿人问了什么、作者怎么答、哪里仍是软肋）
5. `UniCM_Fig2–5.png` → 挑 2–3 张做主结果页与可解释性页
6. 需要补充图（如消融、季节性热图、滞后相关）时查 `UniCM-SI图表页码索引.md` 定位后渲染：**渲染用物理页 = 页脚印刷页 + 1**（封面无页码）
