# 论文 DAG 阅读地图

把 week2 的几篇论文串成一张有向图（DAG），点节点就能看概述，并直接打开 **论文 PDF** 与 **飞书文档**。

---

## 🚀 怎么启动

### 方式 A：最简单（双击即可，无需联网、无需服务器）

双击本目录下的 **`start.cmd`**，或者直接双击 **`index.html`**。

> 用 `file://` 打开，PDF 与笔记链接都能点开。
> 唯一限制：部分浏览器对 `file://` 下的本地文件跳转有额外限制，如果遇到打不开的情况请用方式 B。

### 方式 B：本地服务器（推荐，所有链接都正常）

在本目录打开 PowerShell / CMD，运行：

```powershell
python serve.py
```

它会自动打开浏览器，地址形如：

```
http://127.0.0.1:8765/汇报合集/week2汇报/paper-dag/index.html
```

换端口：`python serve.py 9000`　停止：`Ctrl + C`

> 服务器把根目录设在 `刘睿涵\刘睿涵`，所以 `../../../papers/*.pdf` 这类相对链接全都能取到。

---

## 🖱 界面怎么用

| 操作 | 效果 |
|---|---|
| **点节点** | 右侧面板显示：一句话 / 概述 / 关键数字 / 为什么在地图里 / 待验证点 / 上下游关系 / 本地笔记 |
| **悬停节点** | 高亮它相关的连线与邻居，其余淡出 |
| **📄 打开论文 PDF** | 新标签页打开 `papers\*.pdf` |
| **飞书文档** | 新标签页打开对应飞书文档（UniCM、FINDER 有；其余暂缺） |
| **右上角「飞书域名」** | 默认 `https://feishu.cn`。若打不开，改成你租户的域名（形如 `https://xxx.feishu.cn`），会记在浏览器里 |

### 图的读法

```
列 0（地基）            列 1（技术）          列 2（应用 / 对标）
┌──────────────┐
│ ENSO complexity│──── explicit ────────┐
│  理论地基      │     同题 ─────────┐   │
└──────────────┘                   │   ▼
┌──────────────┐   ┌────────────┐  │  ┌────────────────┐
│ FINDER        │   │ TritonCast │──┴─▶│ ENSO-CausalNet │
│  方法来源      │──▶│  技术对标   │────▶│  同期对标       │
└──────────────┘   └────────────┘     └────────────────┘
        │  method                            ▲  同题（双向对标）
        └────────────────────────────────────┴──►┌────────────┐
                                                 │   UniCM     │
                                                 │  当前主攻    │
                                                 └────────────┘
```

**连线颜色的含义**（每条边都标了依据强度）：

| 颜色 | 标签 | 含义 |
|---|---|---|
| 🔴 红 | **明确** | 论文中明确对标 / 引用 |
| 🔵 蓝 | **同题** | 解决同一问题，方法不同 |
| 🟢 绿（虚线） | **方法** | 方法 / 技术亲缘，非同一问题 |
| ⚪ 灰（点线） | **推断** | 我的阅读地图推断，需自行核对 |

> ⚠️ **连线是我梳理的阅读地图，不代表论文之间的引用关系。** 每条边的文字说明都会在悬停时显示。

---

## 📁 文件结构

```
paper-dag/
├── index.html     界面（无任何外部依赖，纯原生 HTML/CSS/JS）
├── schema.json    论文解析框架（Paper Parse Schema）
├── papers/        ★ 每篇论文的解析结果，一篇一个 JSON
│   ├── unicm.json  enso-complexity.json  enso-causalnet.json
│   └── tritoncast.json  finder.json
├── relations.json ★ 显式声明的「非引用」关系（必须给理由）
├── build.js       ★ 构建器：cites → 推导引用边；+ 声明边 → 生成 data.js
├── data.js        【自动生成】前端直接加载，请勿手改
├── start.cmd      双击启动（file://）
├── serve.py       本地服务器启动器
├── _check.js      自检：解析结果 / 关系声明 / 生成物 / 前端
└── README.md      本文件
```

---

## 🧠 解析框架（重点）

### 为什么需要它

一开始 DAG 的边是我「觉得」的，别人无法复核。**框架的目标是让边从解析结果里长出来**：

```
PDF ──pdf-tool──► 干净文本 ──人工/模型解析──► papers/*.json ──build.js──► data.js ──► 网页
                                                   │
                                          schema.json 规定字段
```

### 两条来源，泾渭分明

| 边的来源 | basis | 证据要求 |
|---|---|---|
| **推导**：某篇 `cites` 里写了仓库内另一篇 | `explicit` 明确 | **必须有 `evidence`**（引用原文），构建时缺证据会报错 |
| **声明**：写在 `relations.json` | `same` 同题 / `method` 方法 / `infer` 推断 | **必须有 `reason`**（判定理由） |

`build.js` 会自动：
- 校验 `cites.target` 必须存在于仓库内；
- 校验 `explicit` 边必带引用原文、声明边必带理由，否则**构建失败**；
- 输出**缺口报告**：字段（domain / methods / task）高度重叠却没有边的论文对；
- 输出**弱边提示**：已声明但字段无重叠的边（只能靠 reason 支撑，需人工复核）。

### schema 里哪些字段会「生成边」

| 字段 | 作用 | 生成 |
|---|---|---|
| `problem.domain` / `problem.task` | 问题域与任务类型 | 同位判定 |
| `methods[]` | 方法族标签 | 方法亲缘判定 |
| `cites[]` | 显式引用（含 `target` / `evidence` / `where`） | **明确边** |
| `baselines[]` | 对标对象 | 对照关系 |

其余字段（`contributions` / `numbers` / `limitations` / `reusable` / `data`）负责「讲清楚一篇论文」，不参与构图。

### 实战：框架纠了哪些错

| 边 | 原标注 | 证据核查后 | 结论 |
|---|---|---|---|
| `enso-complexity → unicm` | 明确 | UniCM **ref. 8** = Timmermann et al. *Nature* 559, 535–545 (2018) | ✅ 维持 |
| `enso-complexity → enso-causalnet` | 同题 | CausalNet **正文 + 参考文献都引了**同一篇 | ❌ **升级为「明确」** |
| `enso-causalnet ↔ unicm` | 同题 | 互不引用（2025 / 2026 同代工作） | ✅ 维持同题 |
| `finder → unicm` | 方法 | **互不引用**，且方法族标签不重叠 | ⚠️ **降级为「推断」** |

最后一条正是框架的价值：**它把"我觉得有关系"和"文献上真有关系"分开了**。界面上可以一键「隐藏推断边」，只留下有证据的骨架。

---

## ✏️ 怎么改内容

**改 `papers/*.json`（讲清楚一篇论文）和 `relations.json`（声明关系），然后重新构建。不要手改 `data.js`。**

```powershell
node build.js      # 重新生成 data.js（含校验与缺口报告）
node _check.js     # 全面自检
```

| 想改什么 | 改哪里 |
|---|---|
| 增删论文 | 在 `papers/` 加一个 JSON（照 `schema.json` 填）；`pos.col/row` 控制位置 |
| 加/改引用边 | 在该论文的 `cites[]` 写 `{ target, relation, where, evidence }` |
| 加/改声明边 | `relations.json` 的 `edges[]`，必须写 `reason` |
| 加飞书文档 | 该论文的 `feishu: { wiki, docx, label }` |
| 加本地笔记 | 该论文的 `notes: [["显示名", "相对路径"]]`（相对 `week2汇报\`） |
| 改飞书默认域名 | `build.js` 里的 `FEISHU_BASE_DEFAULT` |


改完跑一下自检：

```powershell
node _check.js
```

它会检查：`data.js` 语法、5 个 PDF 是否存在、所有笔记路径是否存在、边的引用完整性、`index.html` 内联脚本语法、关键 DOM id。

---

## 🔗 配套工具：pdf-tool

同级的 **`..\pdf-tool\`** 负责「把 PDF 变成可分析的干净文本」：

```powershell
cd "..\pdf-tool"
python extract.py "..\..\..\papers\UniCM.pdf" -o out\UniCM --outline --pages
```

它解决三件事：**双栏错行**（找中缝重排）、**页眉页脚混入**（跨页统计拉黑）、**行尾断词**（`cli-mate` → `climate`）。
本轮 5 篇论文的全部文本都由它抓取；详细能力与限制见 `..\pdf-tool\README.md`。

> 本工具里每个节点的「本地笔记」链接，指向的就是那些抓取/整理后的文本。

---

## 📌 注意事项

1. **TritonCast.pdf 有 47.9 MB**，点开可能慢几秒。
2. **飞书文档只有 2 篇有 token**（UniCM、FINDER，从本工作区已有脚本里取出）；
   其余 3 篇在 `papers/*.json` 里是空对象 `{}`，把文档链接给我就能补上。
3. 内容基于**论文原文摘要 / 正文**整理；`ENSO complexity`、`UniCM` 两篇有完整的本地精读笔记，
   另两篇（CausalNet、TritonCast）目前只到摘要层级，**还没做逐段精读**。
4. 界面在 Chrome / Edge 上验证过；`file://` 模式下如果个别链接打不开，改用 `python serve.py`。
5. `cites` 字段目前**只覆盖「本仓库内这 5 篇互相引用」的情形**；论文引用的外部文献（XRO、DESN 等）
   写在 `baselines` 里，尚未建成节点。
