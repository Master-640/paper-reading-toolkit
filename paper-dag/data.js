/* ============================================================
 *  论文 DAG 阅读地图 —— 数据文件
 *  改这里就能更新工具内容，不用动 index.html
 *
 *  ⚠️ 连线是「我梳理的阅读地图」，不代表论文之间的引用关系。
 *     每条边都带 relation 与依据强度 basis，请按需核对。
 * ============================================================ */

/* 飞书域名：如果你公司的飞书不是 feishu.cn，把下面这行换成
 * 例如 'https://your-tenant.feishu.cn'（前端也可以随时改，会存到 localStorage） */
const FEISHU_BASE_DEFAULT = "https://feishu.cn";

/* 论文 PDF 的相对路径根：工具在 week2汇报/paper-dag/，
 * PDF 在 刘睿涵/刘睿涵/papers/  →  所以是 ../../../papers/ */
const PDF_ROOT = "../../../papers/";

/* 工作区本地笔记的相对路径根（工具在 paper-dag/，笔记在上一级） */
const NOTE_ROOT = "../";

/* ---------- 依据强度图例 ---------- */
const BASIS = {
  explicit: { label: "明确", color: "#dc2626", dash: "", desc: "论文中明确对标 / 引用" },
  same:     { label: "同题", color: "#2563eb", dash: "", desc: "解决同一问题，方法不同" },
  method:   { label: "方法", color: "#16a34a", dash: "8 6", desc: "方法/技术亲缘，非同一问题" },
  infer:    { label: "推断", color: "#94a3b8", dash: "2 5", desc: "我的阅读地图推断，需自行核对" },
};

/* ---------- 节点 ---------- */
const NODES = [
  {
    id: "enso-complexity",
    title: "ENSO complexity",
    subtitle: "Timmermann et al. · Nature 559:535–545 · 2018",
    col: 0, row: 0,
    kind: "理论地基",
    accent: "#7c3aed",
    oneline: "给「ENSO 事件为什么每次都不一样」一个统一的动力学解释框架。",
    type: "Nature Review（约 40 位作者，通讯 Axel Timmermann）",
    cid: "DOI 10.1038/s41586-018-0252-6",
    pdf: "ENSO-complexity.pdf",
    feishu: {},
    overview: [
      "**定义了 ENSO complexity**：在「空间型多样性」之外，纳入时间尺度（天气→年循环→年际→年代际）、动力学、可预报性与全球影响。",
      "**旧理论的不足**：线性充放电振子只用 `Te`（东太平洋海温）+ `h`（赤道热含量）两个变量，解释不了空间多样性、正偏态与热带外/大西洋/印度洋的远程影响。",
      "**统一框架（Fig. 5）**：复杂性 = 一对耦合本征模态（**QQ**≈4 年 / 温跃层反馈 → EP 型；**QB**≈2 年 / 纬向平流反馈 → CP 型）× 激发过程（WWE、NPMM/SPMM、南太平洋 booster、TIW、年循环）× 非线性 × 跨时间尺度相互作用。",
      "**关键洞见**：两个模态都处在**临界（增长率≈0）**，所以极易被外部过程激发；**EP/CP 是连续谱，不是两类事件**。",
      "**可预报性分层**：9–15 个月＝先兆（西太暖池 T300），6–9 个月＝触发（随机 WWE）；**热含量是必要非充分条件**；**拉尼娜可预报性系统性低于厄尔尼诺**。",
      "**逆耳结论**：几十年预报技巧并未稳定提升（21 世纪初甚至下降）；现有气候模式**低估 ENSO 多样性**。",
    ],
    numbers: [
      ["事件样本", "1958–2015 共 17 次 El Niño"],
      ["组合音 C-mode 谱峰", "9 个月、15–18 个月"],
      ["夏末起报的冬季技巧", "ACC > 0.6"],
      ["EOF2 相对 EOF1", "仅解释其方差的 25%（原文表述含糊，慎引）"],
    ],
    why: "UniCM 的最强基线 **XRO** 就是本文充放电振子的非线性升级版；UniCM 把 `hT`、`T300` 放进 5 个输入物理场，并用 2020–2023 三重拉尼娜回应「拉尼娜难报」这一条。",
    caveats: [
      "双本征模态只在单一中等复杂度模式（Zebiak–Cane）里得到，真实海洋中能否分离未验证。",
      "框架四支柱是定性罗列：无权重、无判据。",
      "Fig. 4 的 ACC 只对「事后判定为 ENSO 事件」的初值条件计算 → 选择性偏差、技巧被高估。",
    ],
    notes: [
      ["全文纯文本（可检索）", "ENSO-complexity.txt"],
      ["振子/偏态逻辑链（Mermaid + PNG）", "enso_oscillator_logic_chain.md"],
    ],
  },

  {
    id: "finder",
    title: "FINDER",
    subtitle: "Fan et al. · Nature Machine Intelligence 2 · 2020",
    col: 0, row: 1,
    kind: "方法来源",
    accent: "#0891b2",
    oneline: "用深度强化学习在小合成网络上训练，直接迁移到真实网络，找出「关键节点」。",
    type: "Research Article（UCLA 等）",
    cid: "DOI 10.1038/s42256-020-0177-2",
    pdf: "FINDER.pdf",
    feishu: { wiki: "WdbpwCG8giJsxYkjckVce9Y0nce", docx: "W9xNduWQQovJtyxZ10Zcqk8cnbV", label: "Week2 Finder" },
    overview: [
      "**问题**：找一组「关键节点」（激活或移除能最大增强/削弱某种网络功能）是网络科学的基础问题，但其一般形式是 **NP-hard**，此前只有针对特定场景的近似/启发式方法，**缺统一框架**。",
      "**方法**：深度强化学习框架 **FINDER** —— **纯粹在 toy model 生成的小合成网络上训练**，然后应用到广泛场景。",
      "**目标函数**：最常用的是最小化 **GCC（最大连通分量）**，它与线性阈值传播下的「最优扩散问题」互为对偶。",
      "**结果**：在各种问题设定下解质量**显著优于既有方法**；在大网络上**快几个数量级**。",
      "**意义**：开启了用深度学习理解复杂网络组织原理的方向，可用于设计更鲁棒的网络。",
    ],
    numbers: [
      ["训练数据", "仅小型合成网络（BA 模型等）"],
      ["解质量", "显著优于既有方法"],
      ["速度", "大网络上快几个数量级"],
    ],
    why: "本工作区的 **GRACO** 代码框架就是重新实现并扩展了 FINDER 的思想；`feishu_week2_summary.md` 是这篇的组会笔记（含 GraphSAGE encoder / 外积 decoder / ε-greedy 拆点全流程）。",
    caveats: [
      "核心是「图上的序贯决策」而非物理建模，与气候这条线是方法亲缘，不是同一问题。",
      "小合成网络 → 真实网络的迁移依赖分布假设，原文用大量场景验证，但边界仍需留意。",
    ],
    notes: [
      ["组会笔记（三要素 / 架构 / 疑问）", "feishu_week2_summary.md"],
      ["FINDER 代码架构图解", "FINDER-代码架构图解.md"],
      ["强化学习清单与学习路径", "FINDER-强化学习清单与学习路径.md"],
      ["GRACO 代码框架（重新实现 FINDER 思想）", "GRACO/README.md"],
      ["全文纯文本（可检索）", "FINDER.txt"],
    ],
  },

  {
    id: "tritoncast",
    title: "TritonCast",
    subtitle: "Wu, Gao, Gou et al. · arXiv:2505.19432v3 · 2026",
    col: 1, row: 0,
    kind: "技术对标",
    accent: "#ea580c",
    oneline: "用「潜动力学核心 + 外结构融合」解决 AI 地球系统模型在长时自回归中的失稳与谱偏差。",
    type: "Preprint（清华 + 腾讯等，通讯 hxm@tsinghua.edu.cn）",
    cid: "arXiv:2505.19432v3 [cs.LG]",
    pdf: "TritionCast.pdf",
    feishu: {},
    overview: [
      "**问题**：数据驱动的 AI 模型在做**长时间自回归**时不稳定，误差失控放大；根源是**固有的谱偏差**——高频、小尺度过程表征不足。",
      "**灵感**：来自数值模式的**嵌套网格**（用不同分辨率解析不同尺度）。",
      "**架构**：一个专门的**潜动力学核心**保证粗尺度宏观演化的长期稳定；**外层结构**再把这条稳定趋势与细粒度局地细节融合。该设计有效缓解了跨尺度相互作用导致的谱偏差。",
      "**大气结果**：在 SOTA 基准上达到 SOTA 精度；年尺度自回归全球预报表现出**极好的长期稳定性**；多年**无漂移**气候模拟跨越**整个 2500 天测试期**。",
      "**海洋结果**：涡旋预报技巧**延至 120 天**；展现**零样本跨分辨率泛化**。",
      "**消融**：性能来自架构核心组件的**协同作用**（不是单一模块）。",
    ],
    numbers: [
      ["无漂移模拟时长", "整个 2500 天测试期"],
      ["海洋涡旋预报", "120 天"],
      ["额外能力", "零样本跨分辨率泛化"],
    ],
    why: "UniCM 同样要在 **24 个月**上自回归滚动，面临完全相同的「长时稳定性 / 误差累积」问题；TritonCast 提供了另一条（潜动力学核心 + 多尺度融合）的解法，是很好的对照与可借鉴架构。",
    caveats: [
      "它是**地球系统通用模型**（大气 + 海洋场），不是模态/指数预报，评价口径与 UniCM 不同。",
      "PDF 有 47.9 MB，打开可能较慢。",
    ],
    notes: [
      ["全文纯文本（可检索）", "TritionCast.txt"],
    ],
  },

  {
    id: "enso-causalnet",
    title: "ENSO-CausalNet",
    subtitle: "Cui, Mu, Yuan & Qin · GRL · 2025",
    col: 2, row: 0,
    kind: "同期对标",
    accent: "#059669",
    oneline: "把因果推断引入深度学习，让 ENSO 预报建立在真实因果关系而非相关性上。",
    type: "Research Letter（上海大学 / 同济 / 复旦）",
    cid: "DOI 10.1029/2025GL118701",
    pdf: "ENSO-CausalNet.pdf",
    feishu: {},
    overview: [
      "**出发点**：气候模态之间的相互作用**本质上是因果的**；而深度学习虽擅长气候预测，其**可解释性却仅限于验证已知的相关性**。",
      "**新范式**：把**因果推断整合进数据驱动建模**，使预测建立在**真实因果关系**之上。",
      "**结果**：据此构建的 ENSO-CausalNet 实现 Niño3.4 指数**长达 22 个月**的有效 ENSO 预报。",
      "**机制发现**：影响 ENSO 的主导物理过程**随提前期而变**；揭示了 Bjerknes 反馈与**热带外太平洋、大西洋、印度洋**海气相互作用驱动 ENSO 的不同**因果路径**。",
      "**一个反直觉结论**：当**输入维度增大**时，模型反而可能学到**不完整的因果关系**，导致预报技巧下降。",
      "**结论**：预报能力关键取决于对因果关系的完整理解，这也反向验证了模型的物理有效性。",
    ],
    numbers: [
      ["ENSO 有效预报", "Niño3.4 达 22 个月"],
    ],
    why: "与 UniCM 是**同期的直接对标**：两者都追求「可解释的长提前期 ENSO 预报」，但路线不同 —— CausalNet 用**因果推断**（22 个月），UniCM 用**多模态耦合 + 注意力**（19 个月）。另外它的「热带外/大西洋/印度洋因果路径」正好呼应 ENSO complexity 的先兆分层。",
    caveats: [
      "只做 ENSO 单一模态，不做多模态统一预报。",
      "「输入维度越大因果学得越不完整」这一结论对高维输入方案（如 UniCM 的 5 变量场）是个值得追的问题。",
    ],
    notes: [
      ["全文纯文本（可检索）", "ENSO-CausalNet.txt"],
    ],
  },

  {
    id: "unicm",
    title: "UniCM",
    subtitle: "Yuan, Ding, Qiu, Fan & Li · Nature MI 8:930–941 · 2026",
    col: 2, row: 1,
    kind: "当前主攻",
    accent: "#2563eb",
    oneline: "把 7 个全球气候模态当作一个互联系统，统一做 24 个月预报。",
    type: "Research Article（清华 / 北师大 / PIK）",
    cid: "DOI 10.1038/s42256-026-01245-5",
    pdf: "UniCM.pdf",
    feishu: { wiki: "SpSTwsApuipnh5kkt7fc0cKZnkf", docx: "Q5YSdUu7PoIsIkxQPSac3eY7n6g", label: "UniCM——论文阅读" },
    overview: [
      "**架构**：双分支时空 Transformer —— **Globalformer**（5 个物理场：SST、τx、τy、hT、T300，5°×5° 网格、切 patch）+ **Modeformer**（7 个模态指数：ENSO、IOD、TNA、NPMM、SPMM、IOB、SIOD）。",
      "**核心机制**：`mode-to-patch guidance` —— 把模态表征作为**加性偏置**注入 Globalformer 的**编码器与解码器**（解码器侧用的是**预测出的未来模态轨迹**）。",
      "**输出层级**：最终评估用的是**从 Globalformer 预测 SST 场反算出的指数 I_G**；Modeformer 直接输出的 Î_M 只是**辅助输出**。",
      "**结果**：ENSO **19 个月**、IOD **7 个月**、其他少被研究模态平均 **+22%**、RMSE **↓14–18%**。",
      "**数据**：CMIP6 四套模拟（1850–2014）预训练 → ORAS5/ERA5/GODAS/SODA 测试（**OOD 协议**）；其中**只有 GODAS、SODA 算独立验证**。",
      "**已知缺口**：最核心的 `mode-to-patch guidance` **从未被单独消融**；λ 取值**正文与 SI 互相矛盾**。",
    ],
    numbers: [
      ["ENSO 有效提前期", "19 个月（DESN 16、CNN/Transformer 15）"],
      ["12 个月 lead 的 ACC", "0.78–0.8"],
      ["训练样本", "1,945（12 月输入 + 24 月目标）"],
      ["对 S2S 大模型（12 月）", "0.379 vs CAS-Canglong 0.262 / FuXi-S2S 0.207"],
    ],
    why: "这是本工作区的**主线论文**，其余四篇分别是它的理论地基（ENSO complexity）、方法来源（FINDER）、技术对标（TritonCast）与同期对标（ENSO-CausalNet）。",
    caveats: [
      "`mode-to-patch guidance` 只被实现了「区域内空间均匀的加性偏置」，且从未单独消融 → 增益归因不成立。",
      "四个 CMIP6 模拟实质只等于 **2 个独立模式家族**；且筛选标准未公开。",
      "辅助物理场（风应力、温跃层、上层海洋温度）**不作独立验证目标**。",
    ],
    notes: [
      ["论文阅读（六段框架）", "UniCM-论文阅读.md"],
      ["材料清单与阅读索引", "UniCM-材料清单与阅读索引.md"],
      ["审稿争议要点", "UniCM-审稿争议要点.md"],
      ["SI 图表页码索引", "UniCM-SI图表页码索引.md"],
    ],
  },
];

/* ---------- 边 ---------- */
const EDGES = [
  { from: "enso-complexity", to: "unicm",
    relation: "理论前提：充放电振子 / 双本征模态 / 连续谱 / 春季障碍",
    basis: "explicit" },

  { from: "enso-complexity", to: "enso-causalnet",
    relation: "同题：ENSO 可预报性 + 热带外/大西洋/印度洋路径",
    basis: "same" },

  { from: "tritoncast", to: "unicm",
    relation: "共同难题：长时自回归的稳定性与误差累积",
    basis: "same" },

  { from: "tritoncast", to: "enso-causalnet",
    relation: "共同难题：把有效预报推到 20 个月以上",
    basis: "same" },

  { from: "finder", to: "unicm",
    relation: "方法亲缘：图神经网络表征 + 结构上的学习",
    basis: "method" },

  { from: "finder", to: "enso-causalnet",
    relation: "方法亲缘：图/因果结构上的决策与表征",
    basis: "method" },

  { from: "enso-causalnet", to: "unicm",
    relation: "同期对标：因果可解释 vs 耦合可解释",
    basis: "same", bidirectional: true },
];
