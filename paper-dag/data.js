/* 自动生成，请勿手改 —— 源文件在 papers/*.json 与 relations.json
 * 重新生成： node build.js
 * 生成时间： 2026-09-26 14:11:06
 */

const FEISHU_BASE_DEFAULT = "https://feishu.cn";
const PDF_ROOT = "../../../papers/";
const NOTE_ROOT = "../";

const BASIS = {
  explicit: { label: "明确", color: "#dc2626", dash: "", desc: "论文显式引用（附可核对原文）" },
  same:     { label: "同题", color: "#2563eb", dash: "", desc: "解决同一问题，方法不同" },
  method:   { label: "方法", color: "#16a34a", dash: "8 6", desc: "方法/技术亲缘" },
  infer:    { label: "推断", color: "#94a3b8", dash: "2 5", desc: "阅读地图推断，无文献证据" },
};

const NODES = [
  {
    "id": "enso-causalnet",
    "title": "ENSO-CausalNet",
    "fullTitle": "ENSO-CausalNet: Integrating Causal Inference Into Deep Learning for Robust ENSO Prediction",
    "subtitle": "Yuehan Cui et al. · Geophysical Research Letters · 2025",
    "col": 2,
    "row": 0,
    "kind": "同期对标",
    "accent": "#059669",
    "oneline": "把因果推断引入深度学习，让 ENSO 预报建立在真实因果关系而非相关性上。",
    "type": "Research Letter",
    "cid": "10.1029/2025GL118701",
    "pdf": "ENSO-CausalNet.pdf",
    "feishu": {},
    "problem": {
      "statement": "气候模态之间的相互作用本质上是因果的；而深度学习虽擅长气候预测，其可解释性却仅限于验证已知的相关性。作者要让预测建立在真实因果关系之上。",
      "difficulty": "从观测数据中辨识真正的因果关系（而非相关）本身困难；且输入维度增大时因果结构的学习会退化，存在精度与可解释性的取舍。",
      "domain": [
        "enso-dynamics",
        "climate-mode-forecasting"
      ],
      "task": "multi-step-forecasting",
      "object": "ENSO（Niño3.4 指数）"
    },
    "methods": [
      "causal-inference",
      "deep-learning",
      "causal-pathway-analysis",
      "attention-attribution"
    ],
    "data": {
      "pretrain": [],
      "eval": [
        "ENSO 观测/再分析资料（Niño3.4）",
        "多海盆海气变量（热带外太平洋、大西洋、印度洋）"
      ]
    },
    "contributions": [
      "提出「因果推断整合进数据驱动建模」的新范式，使预测基于真实因果关系",
      "ENSO-CausalNet 实现 Niño3.4 指数长达 22 个月的有效 ENSO 预报",
      "揭示影响 ENSO 的主导物理过程随提前期变化；给出 Bjerknes 反馈与热带外太平洋、大西洋、印度洋海气相互作用驱动 ENSO 的不同因果路径",
      "反直觉结论：输入维度增大时模型可能学到不完整的因果关系，导致预报技巧下降",
      "结论：预报能力关键取决于对因果关系的完整理解"
    ],
    "numbers": [
      [
        "ENSO 有效预报",
        "Niño3.4 达 22 个月"
      ]
    ],
    "baselines": [
      "传统深度学习 ENSO 预报模型",
      "该组自己的 ENSO-ASC / ENSO-GTC / ENSO-MC / CAU 系列"
    ],
    "limitations": [
      "只做 ENSO 单一模态，不做多模态统一预报",
      "「输入维度越大因果学得越不完整」这个结论对高维输入方案（如 UniCM 的 5 变量场）是个值得追的问题"
    ],
    "reusable": [
      "「因果路径随 lead time 变化」的分析角度，可以用来审视 UniCM 注意力权重是否也只是相关性",
      "22 个月 vs UniCM 19 个月 —— 两者是同一问题的两条路线的直接对照数据点",
      "该组从 ENSO-ASC(2021) → ENSO-GTC(2022) → ENSO-MC(2022) → CAU(2024) → CausalNet(2025) 的演进线，是一条完整的「ENSO 深度学习」方法谱系"
    ],
    "notes": [
      [
        "全文纯文本（可检索）",
        "ENSO-CausalNet.txt"
      ]
    ],
    "cites": [
      {
        "target": "enso-complexity",
        "relation": "background",
        "where": "正文引言 + 参考文献",
        "evidence": "正文：\"ENSO variability arises from complex air-sea feedbacks (Rasmusson & Carpenter, 1982; Timmermann et al., 2018), such as the Bjerknes feedback (Bjerknes, 1969)\"；参考文献：\"Timmermann, A., An, S.-I., Kug, J.-S., Jin, F.-F., Cai, W., Capotondi, A., et al. (2018). El Niño–Southern Oscillation complexity. Nature, 559(7715), 535–545.\""
      }
    ]
  },
  {
    "id": "enso-complexity",
    "title": "ENSO complexity",
    "fullTitle": "El Niño–Southern Oscillation complexity",
    "subtitle": "Axel Timmermann et al.（约 40 位作者） et al. · Nature 559, 535–545 · 2018",
    "col": 0,
    "row": 0,
    "kind": "理论地基",
    "accent": "#7c3aed",
    "oneline": "给「ENSO 事件为什么每次都不一样」一个统一的动力学解释框架。",
    "type": "Review",
    "cid": "10.1038/s41586-018-0252-6",
    "pdf": "ENSO-complexity.pdf",
    "feishu": {},
    "problem": {
      "statement": "经典理论把 ENSO 当单一模态的规则循环，但观测显示事件在强度、空间型、时间演化、可预报性、全球影响上差异极大。作者要给这种「复杂性」一个正式定义与统一的动力学解释。",
      "difficulty": "1958–2015 仅 17 次 El Niño；高质量观测约 40 年而现象跨年代际；地球系统只有一个实现，无对照实验，噪声与低频确定性动力学难以分离；CGCM 有系统性偏差不能替代观测。",
      "domain": [
        "enso-dynamics",
        "climate-mode-forecasting"
      ],
      "task": "synthesis-review",
      "object": "ENSO 及全球气候模态系统"
    },
    "methods": [
      "recharge-oscillator",
      "coupled-eigenmode-analysis",
      "eof-analysis",
      "kernel-density-probability",
      "observational-diagnosis"
    ],
    "data": {
      "pretrain": [],
      "eval": [
        "ERSST.v5",
        "TAO/TRITON Z20",
        "merged Z20/T300 product",
        "NMME (9 coupled models)",
        "CGCM experiments",
        "paleoclimate reconstructions"
      ]
    },
    "contributions": [
      "给出 ENSO complexity 的正式定义：在空间型多样性之外，纳入时间尺度、动力学、可预报性与全球影响",
      "统一框架（Fig. 5）：复杂性 = 一对耦合本征模态（QQ≈4yr 温跃层反馈→EP；QB≈2yr 纬向平流反馈→CP）× 激发过程 × 非线性 × 跨时间尺度相互作用",
      "关键洞见：两个本征模态都处在临界（增长率≈0），极易被外部过程激发；EP/CP 是连续谱不是两类",
      "可预报性分层：9–15 月先兆 / 6–9 月触发；热含量是必要非充分条件；拉尼娜可预报性系统性低于厄尔尼诺",
      "方法升级：用核密度概率密度随滞后时间的演化替代事件合成，把先兆从个例叙事变成概率陈述",
      "逆耳结论：几十年预报技巧并未稳定提升；现有气候模式低估 ENSO 多样性"
    ],
    "numbers": [
      [
        "事件样本",
        "1958–2015 共 17 次 El Niño"
      ],
      [
        "组合音 C-mode 谱峰",
        "9 个月、15–18 个月"
      ],
      [
        "夏末起报的冬季技巧",
        "ACC > 0.6"
      ],
      [
        "两个本征模态周期",
        "准四年（3–7 年）/ 准两年"
      ],
      [
        "EOF2 相对 EOF1",
        "仅解释其方差的 25%（原文表述含糊，慎引）"
      ]
    ],
    "baselines": [
      "composite analysis",
      "linear recharge oscillator",
      "single-mode linear theory",
      "CGCMs"
    ],
    "limitations": [
      "双本征模态只在单一中等复杂度模式（Zebiak–Cane）里得到，真实海洋中能否分离未验证",
      "框架四支柱是定性罗列：无权重、无判据",
      "Fig. 4 的 ACC 只对「事后判定为 ENSO 事件」的初值条件计算 → 选择性偏差、技巧被高估",
      "观测记录太短，年代际多样性的来源无法定量归因"
    ],
    "reusable": [
      "XRO 的理论祖先：充放电振子 + 双本征模态，正是 UniCM 最强基线的来源",
      "「热含量必要非充分」直接解释了为什么纯统计模型会有可预报性上限",
      "概率化先兆分析（kernel density vs lead time）是一套可迁移的诊断方法"
    ],
    "notes": [
      [
        "全文纯文本（可检索）",
        "ENSO-complexity.txt"
      ],
      [
        "振子/偏态逻辑链（Mermaid + PNG）",
        "enso_oscillator_logic_chain.md"
      ]
    ],
    "cites": []
  },
  {
    "id": "finder",
    "title": "FINDER",
    "fullTitle": "Finding key players in complex networks through deep reinforcement learning",
    "subtitle": "Changjun Fan et al. · Nature Machine Intelligence 2, 317–326 · 2020",
    "col": 0,
    "row": 1,
    "kind": "方法来源",
    "accent": "#0891b2",
    "oneline": "用深度强化学习在小合成网络上训练，直接迁移到真实网络，找出「关键节点」。",
    "type": "Research Article",
    "cid": "10.1038/s42256-020-0177-2",
    "pdf": "FINDER.pdf",
    "feishu": {
      "wiki": "WdbpwCG8giJsxYkjckVce9Y0nce",
      "docx": "W9xNduWQQovJtyxZ10Zcqk8cnbV",
      "label": "Week2 Finder"
    },
    "problem": {
      "statement": "寻找一组「关键节点」（激活或移除能最大增强/削弱某种网络功能）是网络科学的基础问题，但其一般形式是 NP-hard，此前只有针对特定场景的近似/启发式方法，缺统一框架。",
      "difficulty": "问题是 NP-hard，无法用多项式时间的精确算法；不同应用场景的最优策略差异大，难以用单一启发式覆盖。",
      "domain": [
        "network-science",
        "combinatorial-optimization"
      ],
      "task": "combinatorial-optimization",
      "object": "复杂网络（Internet、社交网、电网、食物网、生物分子网络等）"
    },
    "methods": [
      "deep-reinforcement-learning",
      "graph-neural-network",
      "graphsage-encoder",
      "n-step-q-learning",
      "synthetic-to-real-transfer"
    ],
    "data": {
      "pretrain": [
        "小型合成网络（BA 模型等 toy model）"
      ],
      "eval": [
        "真实网络：Internet / 社交 / 基础设施 / 生物网络等多场景"
      ]
    },
    "contributions": [
      "提出深度强化学习框架 FINDER：纯在 toy model 生成的小合成网络上训练，再迁移到广泛真实场景",
      "目标函数常用最小化 GCC（最大连通分量），与线性阈值传播下的最优扩散问题互为对偶",
      "解质量显著优于既有方法；在大网络上快几个数量级",
      "开启用深度学习理解复杂网络组织原理的方向，可用于设计更鲁棒的网络"
    ],
    "numbers": [
      [
        "训练数据",
        "仅小型合成网络"
      ],
      [
        "解质量",
        "显著优于既有方法"
      ],
      [
        "速度",
        "大网络上快几个数量级"
      ]
    ],
    "baselines": [
      "启发式中心性方法（degree / betweenness / k-shell 等）",
      "近似算法"
    ],
    "limitations": [
      "核心是「图上的序贯决策」，与物理建模无关",
      "小合成网络 → 真实网络的迁移依赖分布假设，边界需留意"
    ],
    "reusable": [
      "GraphSAGE 编码器 + 外积解码器 + 虚拟节点维护全局状态的架构，可迁移到任何「图上的序贯决策」任务",
      "「小合成数据训练 → 真实场景零样本迁移」这一范式，与 CMIP6 预训练 → 再分析迁移同构",
      "GCC 作为可微/可评估的目标函数设计"
    ],
    "notes": [
      [
        "组会笔记（三要素 / 架构 / 疑问）",
        "feishu_week2_summary.md"
      ],
      [
        "FINDER 代码架构图解",
        "FINDER-代码架构图解.md"
      ],
      [
        "强化学习清单与学习路径",
        "FINDER-强化学习清单与学习路径.md"
      ],
      [
        "GRACO 代码框架（重新实现 FINDER 思想）",
        "GRACO/README.md"
      ],
      [
        "全文纯文本（可检索）",
        "FINDER.txt"
      ]
    ],
    "cites": []
  },
  {
    "id": "tritoncast",
    "title": "TritonCast",
    "fullTitle": "Advanced Long-term Earth System Forecasting (TritonCast)",
    "subtitle": "Hao Wu et al. · arXiv:2505.19432v3 [cs.LG] · 2026",
    "col": 1,
    "row": 0,
    "kind": "技术对标",
    "accent": "#ea580c",
    "oneline": "用「潜动力学核心 + 外结构融合」解决 AI 地球系统模型在长时自回归中的失稳与谱偏差。",
    "type": "Preprint",
    "cid": "arXiv:2505.19432v3",
    "pdf": "TritionCast.pdf",
    "feishu": {},
    "problem": {
      "statement": "数据驱动的 AI 模型在做长时间自回归时不稳定，误差失控放大；根源是固有的谱偏差——高频、小尺度过程表征不足。",
      "difficulty": "跨尺度相互作用导致的谱偏差难以用单一分辨率网络解决；要在数月至数年尺度上保持无漂移，需要架构层面的机制而非调参。",
      "domain": [
        "earth-system-forecasting",
        "long-horizon-autoregression"
      ],
      "task": "multi-step-forecasting",
      "object": "大气与海洋场（全球地球系统）"
    },
    "methods": [
      "latent-dynamical-core",
      "nested-grid-inspired-architecture",
      "multi-scale-fusion",
      "spectral-bias-mitigation",
      "autoregressive-rollout"
    ],
    "data": {
      "pretrain": [],
      "eval": [
        "大气 SOTA 基准",
        "海洋涡旋预报",
        "2500 天连续测试期"
      ]
    },
    "contributions": [
      "诊断长时自回归失稳的根源：固有谱偏差导致高频小尺度过程表征不足、误差失控放大",
      "借鉴数值模式嵌套网格，设计：潜动力学核心保证粗尺度宏观演化的长期稳定 + 外层结构融合细粒度局地细节",
      "大气：SOTA 精度 + 年尺度自回归稳定性 + 跨越整个 2500 天测试期的多年无漂移气候模拟",
      "海洋：涡旋预报技巧延至 120 天；展现零样本跨分辨率泛化",
      "消融显示性能来自架构核心组件的协同作用，而非单一模块"
    ],
    "numbers": [
      [
        "无漂移模拟时长",
        "整个 2500 天测试期"
      ],
      [
        "海洋涡旋预报",
        "120 天"
      ],
      [
        "额外能力",
        "零样本跨分辨率泛化"
      ],
      [
        "篇幅",
        "154 页"
      ]
    ],
    "baselines": [
      "现有 AI 地球系统模型（如 FuXi / GraphCast 类）",
      "数值模式"
    ],
    "limitations": [
      "是地球系统通用模型（大气 + 海洋场），不是模态/指数预报，评价口径与 UniCM 不同",
      "PDF 47.9 MB / 154 页，本仓库不收录原文"
    ],
    "reusable": [
      "「潜动力学核心 + 外结构融合」是解决长时自回归漂移的现成配方，可直接对照 UniCM 的 24 个月 rollout",
      "谱偏差诊断视角：它解释了为什么纯场级模型在长 lead 上会退化",
      "零样本跨分辨率泛化 → 与「5° 网格够用」的论证可以互相印证"
    ],
    "notes": [
      [
        "全文纯文本（可检索）",
        "TritionCast.txt"
      ]
    ],
    "cites": []
  },
  {
    "id": "unicm",
    "title": "UniCM",
    "fullTitle": "Learning the coupled dynamics of global climate modes (UniCM)",
    "subtitle": "Yuan Yuan et al. · Nature Machine Intelligence 8, 930–941 · 2026",
    "col": 2,
    "row": 1,
    "kind": "当前主攻",
    "accent": "#2563eb",
    "oneline": "把 7 个全球气候模态当作一个互联系统，统一做 24 个月预报。",
    "type": "Research Article",
    "cid": "10.1038/s42256-026-01245-5",
    "pdf": "UniCM.pdf",
    "feishu": {
      "wiki": "SpSTwsApuipnh5kkt7fc0cKZnkf",
      "docx": "Q5YSdUu7PoIsIkxQPSac3eY7n6g",
      "label": "UniCM——论文阅读"
    },
    "problem": {
      "statement": "全球气候模态（ENSO、IOD、TNA、NPMM、SPMM、IOB、SIOD）通过跨洋盆遥相关耦合为一个系统；现有方法要么孤立预测单个模态，要么只做简化配对建模，把整个耦合系统当作整体预测仍是未解难题。",
      "difficulty": "模态跨越大范围时空尺度，通过非线性、状态依赖的耦合相互作用，耦合强度随季节与背景态变化；非 ENSO 模态的物理理解本身有限；训练数据只能来自气候模式（再分析记录太短）。",
      "domain": [
        "climate-mode-forecasting",
        "enso-dynamics"
      ],
      "task": "multi-step-forecasting",
      "object": "7 个全球气候模态 + 5 个物理场"
    },
    "methods": [
      "spatiotemporal-transformer",
      "dual-branch-architecture",
      "cross-view-guidance",
      "mode-to-patch-guidance",
      "attention-interpretability",
      "sim-to-real-transfer"
    ],
    "data": {
      "pretrain": [
        "CMIP6 historical 1850–2014（筛选后 4 套模拟：CESM2-FV2, CESM2-WACCM-FV2, EC-Earth3-CC, EC-Earth3-Veg-LR）"
      ],
      "eval": [
        "ORAS5（主）",
        "ERA5",
        "GODAS",
        "SODA v2.2.4"
      ]
    },
    "contributions": [
      "把 7 个全球气候模态放在一个模型里统一预测，而不是 one-mode-one-model",
      "双分支时空 Transformer：Globalformer（5 个物理场）+ Modeformer（7 个模态指数），显式建模两个层级的耦合",
      "mode-to-patch guidance：把模态表征作为加性偏置注入 Globalformer 的编码器与解码器（解码器侧用预测出的未来模态轨迹）",
      "物理真实性：不只预测抽象指数，还给场级 SST 预报，并能重现观测的跨模态滞后相关",
      "可解释性：用注意力找出极端事件的先兆与跨模态相互作用结构",
      "数据效率 + sim-to-real：CMIP6 训练直接迁移再分析；40% 训练样本即达 >90% 性能"
    ],
    "numbers": [
      [
        "ENSO 有效提前期",
        "ACC > 0.5 维持 19 个月（DESN 16、CNN/Transformer 15）"
      ],
      [
        "IOD 有效提前期",
        "7 个月"
      ],
      [
        "其他少被研究模态",
        "平均提升 >22%"
      ],
      [
        "误差",
        "RMSE 相对 baseline 降低 14.1–17.9%"
      ],
      [
        "12 个月 lead 的 ACC",
        "0.78–0.8（ERA5 / GODAS / SODA）"
      ],
      [
        "春季可预报性障碍",
        "目标落在 MAM/AMJ 时维持 ACC>0.5 到 14 个月"
      ],
      [
        "对 S2S 大模型（12 月全球平均）",
        "ACC 0.379 / RMSE 0.465 vs CAS-Canglong 0.262/0.505、FuXi-S2S 0.207/0.515"
      ],
      [
        "训练样本",
        "1,945（12 月输入 + 24 月目标，1 月步长）"
      ]
    ],
    "baselines": [
      "XRO (ref. 13)",
      "DESN (ref. 25)",
      "CNN (ref. 34)",
      "ResoNet (ref. 35)",
      "FuXi-S2S",
      "CAS-Canglong"
    ],
    "limitations": [
      "最核心的 mode-to-patch guidance 从未被单独消融：GlobalFormer-only 对照同时删掉了引导 + 模态分支 + 辅助损失，增益归因不成立",
      "四个 CMIP6 模拟实质只等于 2 个独立模式家族，且筛选标准未公开",
      "辅助物理场（风应力、温跃层、上层海洋温度）不作独立验证目标",
      "λ 取值正文（λ1=1, λ2=1, λ3=0.01）与 SI（λ1=λ3=1.0, λ2=0.01）互相矛盾",
      "样本账（1,945）与「retained four simulations」的说法对不上",
      "经纬域三处口径不一致（40°S–0°N / 40°S–40°N / Np=12×72）"
    ],
    "reusable": [
      "「宏观状态变量 ↔ 细粒度物理场」的双向耦合建模，可平移到海洋气象大模型（大尺度指数/环流模态 ↔ 高分辨率温盐流场）",
      "CMIP6 预训练 → 再分析零样本迁移，且微调无显著提升：对「缺长期观测、仿真数据充足」的场景极有价值",
      "可解释性作为卖点：注意力图 → 空间先兆区域；跨模态注意力 → 相互作用强度的时间放大",
      "可复用 baseline 清单：XRO、DESN、ResoNet、CNN/Transformer、FuXi-S2S、CAS-Canglong"
    ],
    "notes": [
      [
        "论文阅读（六段框架）",
        "UniCM-论文阅读.md"
      ],
      [
        "材料清单与阅读索引",
        "UniCM-材料清单与阅读索引.md"
      ],
      [
        "审稿争议要点",
        "UniCM-审稿争议要点.md"
      ],
      [
        "SI 图表页码索引",
        "UniCM-SI图表页码索引.md"
      ],
      [
        "正文网页可获取部分",
        "UniCM_正文网页可获取部分.md"
      ]
    ],
    "cites": [
      {
        "target": "enso-complexity",
        "relation": "theoretical-basis",
        "where": "ref. 8",
        "evidence": "ref. 8: Timmermann, A. et al. El Niño–Southern Oscillation complexity. Nature 559, 535–545 (2018)."
      }
    ]
  }
];

const EDGES = [
  {
    "from": "enso-causalnet",
    "to": "enso-complexity",
    "basis": "explicit",
    "relation": "引用（background）",
    "evidence": "正文：\"ENSO variability arises from complex air-sea feedbacks (Rasmusson & Carpenter, 1982; Timmermann et al., 2018), such as the Bjerknes feedback (Bjerknes, 1969)\"；参考文献：\"Timmermann, A., An, S.-I., Kug, J.-S., Jin, F.-F., Cai, W., Capotondi, A., et al. (2018). El Niño–Southern Oscillation complexity. Nature, 559(7715), 535–545.\"",
    "where": "正文引言 + 参考文献",
    "source": "derived:citation"
  },
  {
    "from": "unicm",
    "to": "enso-complexity",
    "basis": "explicit",
    "relation": "引用（theoretical-basis）",
    "evidence": "ref. 8: Timmermann, A. et al. El Niño–Southern Oscillation complexity. Nature 559, 535–545 (2018).",
    "where": "ref. 8",
    "source": "derived:citation"
  },
  {
    "from": "tritoncast",
    "to": "unicm",
    "basis": "same",
    "relation": "共同难题：长时自回归的稳定性与误差累积",
    "reason": "两者都要在数十步到数百步上自回归滚动，且都把「长时漂移」当作核心问题；问题域不同（地球系统通用场 vs 模态指数），但任务类型同为 multi-step-forecasting。",
    "bidirectional": false,
    "source": "declared"
  },
  {
    "from": "tritoncast",
    "to": "enso-causalnet",
    "basis": "same",
    "relation": "共同难题：把有效预报推到 20 个月以上",
    "reason": "TritonCast 追求年尺度无漂移、CausalNet 追求 22 个月 ENSO —— 都属「延长有效预报时长」这一类目标。无引用关系。",
    "bidirectional": false,
    "source": "declared"
  },
  {
    "from": "enso-causalnet",
    "to": "unicm",
    "basis": "same",
    "relation": "同期对标：因果可解释 vs 耦合可解释",
    "reason": "同一问题（长提前期 ENSO）的两条路线，发表时间接近（2025 / 2026），指标可直接比较（22 个月 vs 19 个月）。查证：两文互不引用。",
    "bidirectional": true,
    "source": "declared"
  },
  {
    "from": "finder",
    "to": "unicm",
    "basis": "infer",
    "relation": "阅读地图邻接：结构化学习解决科学问题（弱关联）",
    "reason": "FINDER 是「图上的序贯决策」，UniCM 是「场上的时空建模」，两者的方法族（GNN+RL vs 时空 Transformer）在本仓库的标签体系下并不重叠，且两文互不引用。这条边只表示「本次阅读把两者放在同一批里」，不表示学术关联 —— 前端可一键隐藏 infer 边。",
    "bidirectional": false,
    "source": "declared"
  }
];
