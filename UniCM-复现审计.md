# UniCM 复现审计

> **性质**：内部审计记录，用于组会汇报。不是论文的正式评述，也不对外发布。
>
> **证据来源（本文件只使用以下材料）**
> - 正文：`papers\UniCM.pdf`（12 页），抽取文本 `UniCM_main.txt`
> - SI（Supplementary Information，50 页 / 9.08 MB）：`UniCM_src_SI_MOESM1.pdf`，抽取文本 `UniCM_SI1.txt`
> - 同行评审文件（Peer Review File，6.83 MB）：`UniCM_src_SI_MOESM2.pdf`，抽取文本 `UniCM_SI2.txt`
> - 官方代码：`github.com/tsinghua-fib-lab/UniCM-Global-Climate-Modes`（MIT），本地 `_UniCM_code/`，`git clone --depth 1`
> - 代码探针：`audit/01_ffn_and_decoder_probe.py`（实跑，PyTorch 2.11.0+cpu）
> - PDF 矢量/像素读数：`audit/02_render_si_figures.py`、`audit/03_crop_si_figure_panels.py`、`audit/04_crop_si_fig38c.py`、`audit/11_extract_fig3b_contours.py` 等
>
> **脚本位置**：本文件引用的所有探针脚本都在 `audit/` 下，用法与依赖见 `audit/README.md`。输出写在 `audit/out/`（已 gitignore）。
>
> **标注约定**
> - `【实证】` 直接从论文 / 代码 / PDF 矢量数据读出
> - `【推断】` 基于证据的推理，附推理链
> - `【待核】` 尚未确认
> - 凡像素读数一律标注 `±0.01`
> - 代码版本注意：`_UniCM_code/` 为 `--depth 1` 克隆，**未记录 commit hash** `【待核】`

---

## 一、结论摘要

严重度基准：**是否动摇论文核心主张「学到了耦合动力学」（learning the coupled dynamics）**。
高 = 直接动摇；中 = 影响可复现性或削弱某条支撑论证；低 = 实现细节 / 表述瑕疵。

| # | 发现 | 类别 | 出处 | 严重度 |
|---|---|---|---|---|
| F1 | 同一条训练样本的 5 个输入通道来自 **3 个不同 CMIP6 模式**（SST/风应力 = CESM2-FV2，hT = EC-Earth3-Veg-LR，T300 = EC-Earth3-CC），样本内物理量不属于同一气候态 | 【实证】 | `_UniCM_code/src/LoadData.py L802-826` | **高** |
| F2 | "four simulations from two model families" 指**文件池**而非样本池；样本数 N = 1,945 只等于**单条 165 年序列** | 【推断】 | 正文 L1003-1006；SI1 L416-419；SI2 L382-385；LoadData.py L802-826 | **高** |
| F3 | mode-to-patch guidance 的每模态投影矩阵 `W_m` 在发布代码中**完全不存在**，实际是广播裸加 | 【实证】 | 正文 Eq 6/7（L1100-1101、L1111-1112）、L1116、L1119-1120；`models.py L223`、`L248`；全仓库 grep `W_m`/`projection_head` 零命中 | **高** |
| F4 | `mode-to-patch guidance` **从未被单独消融**；`GlobalFormer-only` 同时移除了模态分支 + guidance + 辅助损失 | 【实证】 | SI1 L372-384（尤其 L375） | **高** |
| F5 | 损失权重 λ 存在**两组不可调和的版本**：正文 + Fig 38b 一组，SI 5.2 + `train.sh` 一组，λ2/λ3 整体对调 | 【实证】 | 正文 L1178；SI1 L399、L404；SI Fig 38b 图例（像素读数）；`src/script/train.sh L3` | 中 |
| F6 | `config.py` 的 λ help 文本与 `Trainer.py` 的实际用法**是反的**，很可能是 F5 的源头 | 【实证】 | `config.py L62-65`；`Trainer.py L515-532` | 中 |
| F7 | FFN 中间维度正文说 **512**，发布配置实际是 **256**（无扩张）；SI Table 2 完全没有这一行 | 【实证】 | 正文 L1115；`config.py L42`；`src/script/train.sh L3`；SI1 L516-539 | 低 |
| F8 | `miniDecoder` 克隆 3 个 add-norm，`sublayer[2]` **从未调用**（每 decoder 层 512 死参数，4 层 × 2 分支 = 4,096），且 `sublayer[1]` 被复用两次导致交叉注意力 LayerNorm 与 FFN LayerNorm **绑定** | 【实证】 | `my_tools.py L208`、`L227-232`；`audit/01_ffn_and_decoder_probe.py` 梯度实测 | 低 |
| F9 | Dropout 实际只加在残差分支输出上一次，两个 Linear 之间没有 | 【实证】 | `my_tools.py L105-112`；`audit/01_ffn_and_decoder_probe.py` 实测 | 低 |
| F10 | `Embed.py` 是死文件（无任何模块 import），且依赖 `torch_geometric`/`timm` | 【实证】 | `Embed.py L4-5`；全仓库 import 图 | 低 |
| F11 | 空间 token 数：正文说 Np = 12 × 72，代码实际是 12×72/(2×2) = **216** | 【推断】 | 正文 L1117-1118；`config.py L32`；`settings.py L114`；README L118 | 中 |
| F12 | 发布版 `train.sh` 用 `--epochs 2`，论文说 200 epochs | 【实证】 | `src/script/train.sh L3`；SI1 L535；正文 L1138-1139 | 中 |
| F13 | SI Section 5 只有两组架构消融；**时空注意力消融、FFN 消融、guidance 单独消融、λ3 单独置 0 全都没有** | 【实证】 | SI1 L351-457；SI1 L356-384 | **高** |
| F14 | Section 5 几乎**没有数值**：只有 Fig 40 给出 0.473 → 0.464，其余全靠图；且**全部没有误差棒**（明明有 20 个种子） | 【实证】 | SI1 L440-457；SI1 L536；渲染图 `audit/out/si_figs/` | 中 |
| F15 | Fig 37b 上 **NPMM：ModeFormer-only 0.55 > UniCM 0.47**（加 Globalformer 反而变差）；TNA：GlobalFormer-only 0.49 略胜 0.48 | 【实证】 | SI Fig 37b，像素读数 ±0.01 | 中 |
| F16 | Fig 37b 内部**混了两种评估口径**（ModeFormer-only 用自身直接输出指数；UniCM / GlobalFormer 用从预测 SST 反算的指数） | 【推断】 | SI1 L377；正文 L1132-1136 | 中 |
| F17 | Fig 38a 的 `lr=5e-4` 柱与 Fig 37 的 UniCM **逐模态对不上**（均值 0.474 vs 0.471，逐模态最大差 0.07） | 【实证】 | SI Fig 38a / Fig 37b，像素读数 ±0.01 | 中 |
| F18 | Fig 38c **S-1（最小）在 6/7 模态最好**，而 SI 正文写 "increasing model capacity generally correlates with improved forecast skill"、"not yet saturated" —— **文字与图反向** | 【实证】 | SI Fig 38c 像素读数 ±0.01；SI1 L406-409 | 中 |
| F19 | Fig 3b 主图 caption **漏掉 x 轴**（只说 "as a function of lead time"），且**菱形标记完全未定义**；同一物理量在三个图里用了**两套色标** | 【实证】 | 正文 L546；SI1 Fig 13 caption（SI1 L852-856）；SI1 Fig 23 caption（SI1 L~） | 低 |
| F20 | XRO/CNN/ResoNet 被改成 OOD（在 CMIP6 上预训练再测观测）；XRO 是物理动力模型，系数本应在观测上拟合 → 该设定**削弱了对手** | 【推断】 | SI1 L55-66 | 中 |
| F21 | DESN 因 OOD 失败被改回 in-distribution（ORAS5 1958–1980 训练），**拿到更容易的协议却仍只到 16 个月**（UniCM 19 个月） | 【实证】 | SI1 L67-74；正文 L237-238 | 低 |
| F22 | 峰值相关报 P < 0.001，但 1 个月滑窗导致高度自相关（相邻样本共享 ≈91.7% 输入），**朴素自由度会系统性高估显著性** | 【推断】 | 正文 L569；SI1 L420-421 | 中 |
| F23 | 发布代码仓库**没有 `dataset/` 目录、没有预处理或下载脚本**；LoadData 依赖 xarray 读 `.nc` | 【实证】 | `_UniCM_code/` 文件清单；`LoadData.py L20` | 中 |
| F24 | 辅助物理场（风应力、温跃层深度、T300）**不作验证目标** | 【实证】 | 正文 L1160-1162 | 中 |
| F25 | 最终评估用的是 Globalformer 预测场反算的指数 `I_G`；Modeformer 自己的预报只是**辅助输出** | 【实证】 | 正文 L1130-1136 | 低 |

---

## 二、数据集与样本数（最严重）

### 2.1 论文的说法

| 主张 | 出处 |
|---|---|
| "four simulations from two model families"（正文层面的表述） | 正文（数据集一节） |
| 筛选后保留 4 个模式：**CESM2-FV2、CESM2-WACCM-FV2、EC-Earth3-CC、EC-Earth3-Veg-LR** | SI2 L382-385 |
| 训练集来自 **165 年**月尺度 CMIP6 historical runs | SI2 L268-270 |
| N = 1,980 − 12 − 24 + 1 = **1,945** 个训练样本 | SI1 L416-419；正文 L1003-1006；SI2 L268-270 |
| 相邻样本共享 ≈ 91.7%（11/12）的输入特征 | SI1 L420-421 |

### 2.2 代码的实际行为

`_UniCM_code/src/LoadData.py` 的 `create_training_dataloaders`（L802-826）逐行：

| 通道 | 文件 | 模式 | 是否硬编码 |
|---|---|---|---|
| `tos`（SST） | `tos_Omon_{training_data}_...185001_201412.nc` | 参数化，默认 `CESM2-FV2*gr` | 否 |
| `tauu`（τx） | `tauu_Amon_CESM2-FV2_...` | **CESM2-FV2** | **是** |
| `tauv`（τy） | `tauv_Amon_CESM2-FV2_...` | **CESM2-FV2** | **是** |
| `t20d`（hT） | `t20d_Emon_EC-Earth3-Veg-LR_...` | **EC-Earth3-Veg-LR** | **是** |
| `thetaot300`（T300） | `thetaot300_Emon_EC-Earth3-CC_...` | **EC-Earth3-CC** | **是** |

`_UniCM_code/src/LoadData.py L802-826`；默认值见 `config.py L36`（`default='CESM2-FV2*gr'`）与 `src/script/train.sh L3`。

### 2.3 推论

**【推断】样本池 vs 文件池**
推理链：
1. 5 个通道的 NetCDF 文件都覆盖 **185001–201412**（165 年），且都按时间对齐拼接（LoadData.py L802-826）；
2. SI 5.3 从 **165 年 × 12 月 = 1,980 个月**推出 N = 1,945（SI1 L416-419），若真有 4 条独立序列，N 应为 4 倍量级；
3. 因此 "four simulations" 指的是**同时用到了 4 个模式的文件**（CESM2-FV2、EC-Earth3-Veg-LR、EC-Earth3-CC，加上仅在 SI2 回复中点名、代码中未出现的 CESM2-WACCM-FV2），而不是 4 条独立样本序列；
4. 这解释了 N = 1,945 为何自洽。

**⚠️ 需注意**：CESM2-WACCM-FV2 在代码中**未出现** `【实证】`（全仓库 grep 无命中），仅在 SI2 L382-385 的文字回复中列出 → 该模式是否实际参与训练 `【待核】`。

**【推断】物理一致性问题**
推理链：同一样本的第 1 通道（SST）来自 CESM2-FV2，第 4/5 通道（hT、T300）来自 EC-Earth3-Veg-LR / EC-Earth3-CC。不同 GCM 的内部变率与 ENSO 相位互不对齐，因此**同一时刻的 SST 与温跃层深度、上层海洋热含量不属于同一个气候态**。

### 2.4 为什么这动摇核心主张

论文的核心主张是 "learning the coupled dynamics of global climate modes"（标题），以及 Discussion 的 "a substantial portion of the global climate's predictability is an emergent property of the interactions among climate modes"（正文 L934-939）。

- 模型的输入张量是 `(Th, Np, 5)`，5 个通道就是论文所称的"fine-grained physical fields"；模型被要求从这 5 个通道里学出**场与模态之间的反馈回路**。
- 若通道之间的耦合关系是**跨模式拼接**产生的，那么模型学到的 SST ↔ T300 关系**不是任何一个 GCM 的真实耦合，也不是观测的耦合**。论文中"学到耦合"的证据（Fig 3e-g 的滞后相关、Fig 5 的注意力）无法排除"学的是拼接残留的统计关联"这一替代解释。
- 进一步：这条链路同时解释了为什么"辅助物理场不作验证目标"（正文 L1160-1162）是必要的 —— 因为那些场本身不构成物理自洽状态。

---

## 三、损失函数权重 λ 的三方矛盾

正文 Eq 9（L1154-1159）与 SI1 Eq (7)（L399）对 λ 的定义一致：

```
L_total = λ1 · L_field + λ2 · L_global-mode + λ3 · L_mode-aux
```

| 来源 | λ1 | λ2 (global-mode) | λ3 (mode-aux) |
|---|---|---|---|
| **正文 L1178** | 1 | **1** | **0.01** |
| **SI1 5.2 L404** | 1 | **0.01** | **1** |
| **SI Fig 38b 图例默认项 `1.0-1.0-0.01`** | 1 | 1 | 0.01 |
| **`_UniCM_code/src/script/train.sh L3`** | 1 | **0.01** | **1** |

- 正文数值出处：正文 L1177-1178（"these were set to λ1 = 1, λ2 = 1 and λ3 = 0.01"）。
- SI1 数值出处：SI1 L403-405（"setting λ1 = λ3 = 1.0, and λ2 = 0.01"）。
- Fig 38b 图例：五个条目为 `1.0-1.0-0.01`、`0.01-1.0-0.01`、`0.1-1.0-0.01`、`1.0-0.01-0.01`、`1.0-0.1-0.01`（像素读数）。**五个条目的第三位全部是 0.01**。
- `train.sh L3`：`--lambda3 1 --lambda2 0.01 --lambda1 1`。README L116 / L147 中记录的同一组参数也是 `--lambda3 1 --lambda2 0.01 --lambda1 1`。

### 矛盾源头（很可能）

| 文件 | 行 | 内容 |
|---|---|---|
| `config.py` | L63 | `--lambda1` default 1.0，help="Coefficient for the main reconstruction loss" |
| `config.py` | L64 | `--lambda3` default 1.0，help="Coefficient for the **climate mode skill** loss" |
| `config.py` | L65 | `--lambda2` default 0.0，help="Coefficient for the **explicit mode prediction** loss" |
| `Trainer.py` | L515-532 | `loss = loss*λ1 + loss2*λ2 + mode_loss*λ3`，其中 `loss` = 物理场 MSE，`loss2` = `climate_mode_sc(...)`（即 global-mode skill），`mode_loss` = `MSE(pred_mode, target_mode)`（即 mode-aux） |

**【实证】** `lambda3` 的 help 说自己是 "climate mode skill loss"，但在 L532 它乘的是 `mode_loss`（辅助损失）；`lambda2` 的 help 说自己是 "explicit mode prediction loss"，但它乘的是 `loss2`（global-mode skill）。**两个 help 文本互相写反了。**

### 结论

- 【实证】**正文 + Fig 38b** 是一组（λ2=1、λ3=0.01）；**SI 5.2 + 发布 train.sh** 是另一组（λ2=0.01、λ3=1）。两组把 λ2 与 λ3 整体对调，**不可调和**。
- 【推断】以发布代码为准更可靠（代码是实际执行的产物），则真值为 λ1=1、λ2=0.01、λ3=1，**正文的 λ 数值写错了**；这也与 SI1 L404 完全一致。
- 【实证】`config.py` 的默认值是 λ2=0.0、λ3=1.0，与 `train.sh` 的 λ2=0.01 又差一个量级 —— 即**默认值、脚本值、论文值三处不同**。

---

## 四、架构实现与论文描述的五处不符

### 4.1 FFN 中间维度

| 项 | 内容 | 出处 |
|---|---|---|
| 论文 | "hidden dimension D = 256… a feedforward network of **intermediate dimension 512**" | 正文 L1110-1115 |
| 代码默认 | `--dim_feedforward` type=int, **default=256**，help 写 `"(default: d_size * 4)"` | `config.py L42` |
| 实际运行值 | `train.sh` **不传** `--dim_feedforward` → 取默认 256 = **无扩张（d_ff = D）** | `src/script/train.sh L3` |
| 超参表 | SI Table 2 罗列了 D=256、8 层、4 头、dropout 0.2、patch 2×2、Th=12、Tp=24，**但没有 FFN 中间维度这一行** | SI1 L516-539（尤其 L520-528） |
| 实测代价 | d_ff=256 时 FFN 参数量 **131,584**；若按论文 512 则为 **263,168**（×2.00） | `audit/01_ffn_and_decoder_probe.py` 实测 |

**【实证】** `config.py L42` 的 help 文本与其自身 default 自相矛盾（`default=256` 却写 `d_size * 4 = 1024`）。

### 4.2 `W_m` 投影矩阵不存在

| 项 | 内容 | 出处 |
|---|---|---|
| 论文 Eq 6 | `z^enc_{t,n} ← z^enc_{t,n} + Σ_{m:n⊂R_m} W^enc_m × h_t^(m)` | 正文 L1100-1101 |
| 论文 Eq 7 | `z^dec_{t,n} ← z^dec_{t,n} + Σ_{m:n⊂R_m} W^dec_m × h_t^(m)` | 正文 L1111-1112 |
| 论文 | "`W^enc_m , W^dec_m ∈ ℝ^{D×D}` are learned projection matrices specific to each mode m" | 正文 L1116 |
| 论文 | "Mode-to-patch guidance is implemented via **learnable linear projection heads of shape 256 for each climate mode**" | 正文 L1119-1120 |
| 代码 | `enc_out[:,start1:end1,start2:end2] = enc_out[...] + enc_out_mode[:,index:index+1].unsqueeze(1)` —— **纯广播加，无任何投影** | `models.py L223` |
| 代码 | 同型：`outvar_pred_emb[...] + outvar_pred_emb_mode[...]` | `models.py L224`、`L248-251` |
| 代码 | 注入点：`if bias_enc is not None: predictor += bias_enc`（在 encoder 之前） | `models.py L276-278` |
| grep | `W_m` / `Wm` / `projection_head` / `mode_head` 在 `_UniCM_code/src/*.py` 中**零命中** | 全仓库 grep |

**【推断】** 论文公式描述了一个代码中不存在的机制。"每个模态一个投影头"的效果实际被吸收进 Modeformer 自身的权重（Modeformer 每个模态是一个独立 token，见 `models.py L37-45` 的 `predictor_emb_mode`，`emb_spatial_size = len(val_relative) + t20d_mode`）。

**补充**：代码里偶极模态（IOD / SIOD）的引导是**带符号**的 —— 一个盒子 `+=`、另一个 `-=`（`models.py L248-251`），与 `Trainer.data2mode` 的 `mean(box1) − mean(box2)`（`Trainer.py L510`）一致。这一点与物理定义相符 `【实证】`。

### 4.3 `miniDecoder` 的死参数与 LayerNorm 绑定

```python
# my_tools.py L208
self.sublayer = clone_layer(layerConnect(d_size, dropout), 3)   # 克隆 3 份

# my_tools.py L227-232
def forward(self, x, en_out, tgt_mask, memory_mask):
    x = self.sublayer[0](x, lambda x: self.divided_TS_attn(x, x, x, tgt_mask))  # [0]
    x = self.sublayer[1](x, lambda x: self.encoder_attn(x, en_out, en_out, memory_mask))  # [1]
    return self.sublayer[1](x, self.FC)   # 又是 [1]；[2] 从未被调用
```

`audit/01_ffn_and_decoder_probe.py` 梯度实测：

```
sublayer[0]  params=512  tensors_with_nonzero_grad=2/2
sublayer[1]  params=512  tensors_with_nonzero_grad=2/2
sublayer[2]  params=512  tensors_with_nonzero_grad=0/2
```

- 每个 decoder 层 **512 个死参数**；4 个 decoder 层 × 2 个分支（Globalformer / Modeformer）= **4,096 个**。
- `sublayer[1]` 被复用两次 ⇒ **交叉注意力的 LayerNorm 与 FFN 的 LayerNorm 是同一个模块**（权重绑定），与 Fig 1b 画的独立 "Layer norm" 框不符（正文 L205-208 图注排布）。
- 对照：encoder 侧 `clone_layer(layerConnect(size=d_size, dropout=dropout), 2)`（`my_tools.py L178`）与 `forward` 里的 `sublayer[0]` / `sublayer[1]`（`L201-203`）**用法正确**。

### 4.4 Dropout 位置

| 项 | 内容 | 出处 |
|---|---|---|
| 论文 | "Dropout with a rate of 0.2 is used in **both the attention and feedforward sublayers**" | 正文 L1137-1138；SI1 L523；SI2 L1225 |
| 代码 | `layerConnect.forward: return self.norm(x + self.dropout(sublayer(x)))` —— dropout 加在**残差分支输出**上 | `my_tools.py L105-112` |
| 代码 | `enc.FC` 内部**没有** Dropout；两个 Linear 之间无 dropout；激活是 **ReLU**（非 GELU/SwiGLU） | `my_tools.py L182-186`；`audit/01_ffn_and_decoder_probe.py` 实测 |

**【实证】** dropout 确实作用于 FFN 的**输出**（因为 `sublayer` 就是 `self.FC`），所以论文表述不算错；但"FFN 子层内部有 dropout"这一常见读法是错的 `【推断】`。

### 4.5 `Embed.py` 是死文件

- `_UniCM_code/src/Embed.py` 中定义了 `TokenEmbedding_S`、`TokenEmbedding_ST`、`get_2d_sincos_pos_embed` 等，但**全仓库没有任何模块 import 它** `【实证】`（grep 所有 `import` 语句，`models.py L4` 只从 `my_tools` 导入）。
- 它依赖 `from torch_geometric.nn.conv import GCNConv` 与 `from timm.models.vision_transformer import PatchEmbed, Attention, Mlp`（`Embed.py L4-5`）—— 两个依赖都不属于本论文的技术栈 `【实证】`。

### 附：全模型参数分布（`audit/01_ffn_and_decoder_probe.py` 实测）

| 项 | 数值 |
|---|---|
| 全模型参数量（绑定/共享模块计一次） | **12,726,037**（≈12.7 M） |
| attention 参数 | 10,526,720（**82.7%**） |
| FFN 参数 | 2,105,344（**16.5%**） |
| FFN 块数 | 16 个唯一模块（4 encoder + 4 decoder）× 2 分支，权重互不共享 |

**【推断】** 这意味着 mode-to-patch guidance **不引入任何独立参数**（只是一个广播加），因此"单独消融 guidance"的代价极低 —— 把 `models.py L276-278` 的 `predictor += bias_enc` 去掉即可。

### 补充：不在上列五处之内、但同样来自代码/正文对照的不符

| # | 项 | 论文 | 代码 | 类别 |
|---|---|---|---|---|
| S1 | 空间 token 数 | "Spatial attention in Globalformer is applied over **Np = 12 × 72** spatial patches, corresponding to a 5° × 5° resolution"（正文 L1117-1118） | `emb_spatial_size = 12 * 72 // (patch_size[0] * patch_size[1])`（`settings.py L114`），`patch_size` 默认 `'2-2'`（`config.py L32`），README 也用 `'2-2'`（README L118） ⇒ Np = **216** | 【推断】12×72 是原始网格数，不是 patch 数 |
| S2 | 训练轮数 | 200 epochs（正文 L1138-1139；SI1 L535） | `train.sh L3` 传 `--epochs 2` | 【实证】 |
| S3 | Warm-up | "linear warm-up over the first **three epochs**"（正文 L1135；SI1 L530） | `--warmup_steps` default **5**，单位是 **step** 不是 epoch（`config.py L54`；`Trainer.py L548-552`） | 【实证】 |
| S4 | 学习率默认值 | 5×10⁻⁴（正文 L1134；SI1 L530） | `config.py L48` default **5e-5**；`train.sh` 传 `5e-4` | 【实证】 |
| S5 | 空间域 | 正文三处互不相容：**L83** 写 40°S–0°N（8 行）、**L1068-1069** 写 40°S–40°N（16 行）、**L1117-1118** 写 Np = 12×72（12 行） | `settings.py L114`：`12*72//(2*2)` = 216 个 patch | 【实证】三处口径不一致 |

---

## 五、SI Section 5 的覆盖度

### 5.1 目录（SI1 L29-33）

```
5 Ablation studies and sensitivity analysis          7
  5.1 Architectural ablation study                   7
  5.2 Hyperparameter sensitivity analysis            7
  5.3 Quantitative assessment of training data requirements  8
  5.4 Grid-resolution sensitivity                    8
```

### 5.2 你以为有的消融 / SI 里到底有没有

| 你以为有的消融 | SI 里有没有 | 出处 |
|---|---|---|
| 融合策略 additive vs replacement | ✅ 有（Fig 37a） | SI1 L360-370 |
| 双分支：GlobalFormer-only / ModeFormer-only / UniCM(hybrid) | ✅ 有（Fig 37b） | SI1 L372-384 |
| 学习率敏感性 | ✅ 有（Fig 38a） | SI1 L390-392 |
| 损失权重敏感性 | ✅ 有（Fig 38b） | SI1 L396-405 |
| 模型规模敏感性 S-1..S-4 | ✅ 有（Fig 38c） | SI1 L406-409 |
| 数据量 / 采样策略 | ✅ 有（Fig 39） | SI1 L411-438 |
| 网格分辨率 5° vs 2.5° | ✅ 有（Fig 40） | SI1 L440-457 |
| **时空注意力消融（T-attn / S-attn）** | ❌ **完全没有** | SI1 L351-457 全文无 |
| **FFN 消融（去掉 / 换激活）** | ❌ **完全没有** | 同上 |
| **mode-to-patch guidance 单独消融** | ❌ **没有**（唯一涉及 guidance 的是 Fig 37a，那是"additive vs replacement"，即 `z+h` 对 `z=h`，不是 `z+0`） | SI1 L360-370 |
| **λ3 单独置 0（关掉辅助损失）** | ❌ **没有**；Fig 38b 五个图例第三位全是 0.01 | SI Fig 38b 图例（像素读数） |
| 去掉模态分支 | ⚠️ 只有"整支删掉"（GlobalFormer-only） | SI1 L375 |

**关于 `GlobalFormer-only` 的口径**（SI1 L375 原文）：

> "**GlobalFormer-only**: A single-branch model processing only the high-resolution global spatio-temporal fields (5° × 5°)."

**【实证】** 这是一个完整的单分支模型，**按构造同时缺失**：模态分支、mode-to-patch guidance、Modeformer 辅助损失。因此 Fig 37b 的"hybrid 优于 GlobalFormer-only"**不能归因到 guidance 机制本身**。

### 5.3 Section 5 几乎没有数值

| 图 | PDF 页（SI 标签） | 页内嵌文本长度 | 有数值吗 |
|---|---|---|---|
| Fig 37 架构消融 | SI p45（PDF 索引 45） | **326 字符 = 只有图注** | ❌ |
| Fig 38 超参敏感 | SI p46（PDF 索引 46） | **213 字符 = 只有图注** | ❌ |
| Fig 39 数据量 | SI p47（PDF 索引 47） | **374 字符 = 只有图注** | ❌ |
| Fig 40 网格分辨率 | SI p48（PDF 索引 48） | 424 字符（含轴标） | ✅ **唯一两个数：0.473 → 0.464** |
| Fig 41 预训练/微调 | SI p49（PDF 索引 49） | 517 字符（含轴标） | ❌ |

**注意页码偏移**：PDF 共 51 页，SI 页标签 1..50，因为标题页不编号 ⇒ **PDF 索引 N = SI 页标签 N−1**。Fig 36 在 PDF 索引 45（标 "44/50"），故 Fig 37 在 PDF 索引 46（标 "45/50"）。

**【实证】** 上述五张图（渲染件见 `audit/out/si_figs/`）**均无误差棒**，尽管 SI1 L536 明确写 "Random Seeds 1–20 (Results are means of 20 independent runs)"。

**【实证】** 正文 L557-558（以及 SI2 L313-314、L977 的修订版）说 "Further ablation studies on model architecture design and hyperparameter sensitivity are provided in Supplementary Section 5"。但如上表，**"architecture design" 只覆盖了分支数与融合方式，注意力结构、FFN、引导机制都不在其中**。

---

## 六、Fig 37 / Fig 38 的像素读数（±0.01）

> ⚠️ 全部为**从渲染图柱高读出的近似值，精度约 ±0.01**。原始渲染件在 `audit/out/si_figs/`（整页 220 dpi + 局部 500 dpi）。**这些数不是论文给出的数值**，论文只给了图。

### 6.1 Fig 37b：UniCM / GlobalFormer-only / ModeFormer-only

| 模态 | UniCM（双分支） | GlobalFormer-only | ModeFormer-only | 谁最好 |
|---|---|---|---|---|
| ENSO | **0.73** | 0.70 | 0.65 | UniCM |
| **NPMM** | 0.47 | 0.36 | **0.55** | **ModeFormer-only（高 0.08）** |
| SPMM | **0.25** | 0.22 | 0.21 | UniCM |
| IOB | **0.53** | 0.48 | 0.47 | UniCM |
| IOD | **0.29** | 0.26 | 0.27 | UniCM |
| SIOD | **0.55** | 0.50 | 0.52 | UniCM |
| **TNA** | 0.48 | **0.49** | 0.47 | GlobalFormer-only（高 0.01） |
| **均值** | **0.471** | 0.430 | 0.433 | UniCM |

**要点**

1. **NPMM 是明确反例**：ModeFormer-only 0.55 ≫ UniCM 0.47 ≫ GlobalFormer-only 0.36。**加入 Globalformer 把 NPMM 拉低 0.08。** 而正文 L558 恰有一段专门讲 NPMM（"UniCM accurately reproduces the ~4-month lead of NPMM on ENSO"）。SI 5.1 的叙述（SI1 L381-384）完全未提及这个反例。
2. **TNA 上 GlobalFormer-only 略胜**（0.49 vs 0.48）—— 差异仅 0.01，在无误差棒的情况下不可解释。
3. **SI 原文的说法**（SI1 L381）："The hybrid UniCM demonstrates a performance gain over both single-branch variants" —— 就**均值**而言成立（0.471 vs 0.430 / 0.433），但**逐模态不成立**。
4. **【推断】这不是同类比较**。推理链：
   - 正文 L1132-1136：Modeformer 直接产生模态指数预测 `Î_M`，但被指定为**辅助输出**；最终评估基于从 Globalformer 预测物理场反算的指数 `Î_G`。
   - SI1 L377：ModeFormer-only 是"a model focusing exclusively on the interactions between predefined climate indices"，**没有物理场输出**。
   - ⇒ ModeFormer-only 的 ACC 只能来自**它自己直接输出的指数**，而 UniCM / GlobalFormer-only 的 ACC 来自**从预测 SST 反算的指数**。两种评估口径混在同一张柱状图里。

### 6.2 Fig 37a：additive vs replacement

| 模态 | additive（默认） | replacement | 差 |
|---|---|---|---|
| ENSO | **0.73** | 0.70 | +0.03 |
| NPMM | **0.47** | 0.39 | +0.08 |
| SPMM | **0.25** | 0.23 | +0.02 |
| IOB | **0.53** | 0.50 | +0.03 |
| IOD | **0.29** | 0.28 | +0.01 |
| SIOD | **0.55** | 0.52 | +0.03 |
| TNA | 0.48 | **0.51** | −0.03 |
| **均值** | **0.471** | 0.447 | +0.024 |

**【实证】** additive 在 6/7 模态上更好，TNA 例外。此结论与正文 L1141-1145 一致。
**【推断】** 但这只证明"加法优于替换"，**不证明"引导有用"** —— 缺 `z + 0` 这一档。

### 6.3 Fig 38a：学习率（`lr=5e-4` 红 / `lr=1e-4` 蓝 / `lr=1e-3` 黄）

| 模态 | 5e-4 | 1e-4 | 1e-3 | 与 Fig 37b 的 UniCM 之差 |
|---|---|---|---|---|
| ENSO | 0.73 | 0.67 | 0.70 | 0.73 → 0.73 ✅ |
| NPMM | **0.41** | 0.32 | 0.36 | **0.47 → 0.41（−0.06）** |
| SPMM | **0.21** | 0.24 | 0.23 | **0.25 → 0.21（−0.04）** |
| IOB | **0.58** | 0.54 | 0.54 | **0.53 → 0.58（+0.05）** |
| IOD | 0.27 | 0.24 | 0.26 | 0.29 → 0.27（−0.02） |
| SIOD | 0.57 | 0.55 | 0.56 | 0.55 → 0.57（+0.02） |
| TNA | **0.55** | 0.57 | 0.53 | **0.48 → 0.55（+0.07）** |
| **均值** | **0.474** | 0.447 | 0.454 | 0.471 → 0.474 |

**【实证】** 默认学习率就是 5×10⁻⁴（正文 L1134；SI1 L530；`train.sh` 传 `5e-4`），所以 Fig 38a 的红柱**理应与 Fig 37b 的 UniCM 完全相同**。实测**只有 ENSO 一致；均值接近（0.474 vs 0.471）但逐模态最大差 0.07（TNA）**。
**【推断】** 两张图来自不同的实验协议（lead-time 聚合窗口 / 种子 / 数据切分之一不同），论文未说明。

### 6.4 Fig 38b：损失权重

五个图例字符串（像素读数）：

```
1.0-1.0-0.01
0.01-1.0-0.01
0.1-1.0-0.01
1.0-0.01-0.01
1.0-0.1-0.01
```

**【实证】** 五个条目**第三位全部是 0.01** ⇒ 这张图**从未单独变动过辅助损失权重**（见 F13）。
**【推断】** 若按自然的 (λ1-λ2-λ3) 顺序读，`1.0-1.0-0.01` 与**正文 L1178** 的 λ 一致；而 SI1 L404 声称的最优（λ2=0.01、λ3=1.0，即 `1.0-0.01-1.0`）**根本不在图例里**。这与第三节的矛盾同源。图例顺序是否就是 (λ1-λ2-λ3) `【待核】`。

### 6.5 Fig 38c：模型规模 S-1..S-4

| 模态 | S-1 | S-2 | S-3 | S-4 |
|---|---|---|---|---|
| ENSO | **0.73** | 0.70 | 0.69 | 0.64 |
| NPMM | **0.48** | 0.35 | 0.35 | 0.37 |
| SPMM | **0.25** | 0.24 | 0.24 | 0.22 |
| IOB | **0.53** | 0.50 | 0.49 | 0.39 |
| IOD | **0.29** | 0.28 | 0.28 | 0.27 |
| SIOD | **0.56** | 0.51 | 0.50 | 0.48 |
| TNA | 0.48 | **0.54** | 0.50 | 0.43 |

**【实证】**

1. **S-1（最小）在 6/7 模态上最好**；S-4 显著最差（ENSO 0.73→0.64，IOB 0.53→0.39）。
2. **文字与图反向**：SI1 L406-409 写 "increasing model capacity generally correlates with improved forecast skill at long lead times. This suggests that the UniCM architecture is **not yet saturated** and possesses the expressive power to benefit from larger simulation ensembles or higher-resolution data"。但图只用**一根柱/模态**、标注是 "Mean forecast skill"（SI1 L1601-1602 图注），**没有 lead-time 分层** ⇒ "at long lead times" 这个限定在图里看不到依据。
3. **S-1..S-4 从未定义** `【实证】`：SI 里没有表说明各档的层数或 d_model ⇒ "模型规模"这根轴**不可辨识**。
4. **S-1 的七个数与 Fig 37b 的 UniCM 吻合**（0.73 / 0.48≈0.47 / 0.25 / 0.53 / 0.29 / 0.56≈0.55 / 0.48）⇒ **S-1 就是默认配置**，即 Fig 38c 的 baseline 没有换配置。

---

## 七、Fig 3b 的图示缺陷

### 7.1 已确证的编码

| 编码 | 内容 | 出处 |
|---|---|---|
| y 轴 | Forecast lead (months)，刻度 0, 3, …, 21；框顶约在 lead 23 | PDF 页索引 4 矢量数据：lead 0 在 y=255.5 pt，lead 21 在 y=180.2 pt ⇒ 3.586 pt/月；框顶 y=173.5 pt ⇒ ≈lead 23 |
| x 轴（每条带内部） | **12 个三个月目标季节**：`DJF, JFM, FMA, MAM, AMJ, MJJ, JJA, JAS, ASO, SON, OND, NDJ` | SI Fig 13（SI1 L852-856 图注 + 页面矢量文字） |
| 颜色 | ACC / Correlation Skill，**发散色标 −1.00 … +1.00**，刻度 1.00 / 0.60 / 0.20 / −0.20 / −0.60 / −1.00 | 主图与 SI Fig 13 色标（像素读数） |
| 黑线 | **0.5 技巧阈值**（另有 −0.5） | 正文 L546；SI1 L856 |
| 菱形标记 | 显著性 | 见 7.3 |

**确证依据**：SI Fig 13（SI1 L852-856）是同一批图的 **XRO 版本**，6 个非 ENSO 模态 2×3 排列，其 caption 明确写 "The horizontal axis indicates the **target season**, and the vertical axis represents the forecast lead time in months… with the black contour line indicating the 0.5 skill threshold"，页面上 12 个季节标签齐全。

**主图几何**（从 PDF 矢量数据量得）：7 个条带起始 x = 62.4 / 96.5 / 130.6 / 164.8 / 198.9 / 233.0 / 267.1 pt，**每条带宽 31.28 pt**，条带中心间距 **≈34.1 pt**。
**【推断】** 31.28 pt 里放不下 12 个季节标签，所以主图把 x 轴标签整个删了。

### 7.2 缺陷

| # | 缺陷 | 出处 |
|---|---|---|
| D1 | **caption 漏掉 x 轴**：只写 "b, Heatmap of monthly forecast skill (ACC) for each climate mode **as a function of lead time**. Contour lines indicate correlation levels at 0.5 intervals." | 正文 L546 |
| D2 | **菱形标记 caption 完全未定义** | 正文 L546（全文无提及） |
| D3 | **同一物理量两套色标**：Fig 3b（−1…+1，标 "Anomaly correlation coefficient (ACC)"）；SI Fig 13（−1…+1，标 "Correlation Skill"）；SI Fig 23（**截成 0.0…1.0**，标 ACC） | 正文 L546；SI1 L856；SI1 Fig 23 页面（PDF 索引 31） |
| D4 | 正文靠这个 x 轴得出结论，但 caption 未交代 ⇒ **只看主图无法复现结论** | 正文 L559-564；正文 L546 |

正文的用法（L559-564）：

> "we analysed the **seasonal** forecast skill for each mode (Fig. 3b). The results reveal distinct predictability windows: the IOD peaks in **boreal autumn (September, October and November)**, the IOB in **early spring (February and March)** and the TNA in **late summer (July, August and September)**."

**【推断】** 这段读的是"沿竖直方向看哪一列最红"，即 x 轴。这是 x 轴必然存在的直接证据。

### 7.3 菱形标记

**【实证】** 主图 Fig 3b 内至少 ENSO、NPMM 条带内存在**斜向排列的菱形标记链**（黑色描边 + 深红实心两种），caption 未提及。
**【推断】** 它们与 SI Fig 23 的显著性点同类。推理链：SI Fig 23 是"同一布局的转置版"（y = Climate mode 7 行，x = Forecast lead 1..24），其 caption 明确写 "Dots mark statistically significant skill after field-significance testing with multiple-comparison control"（SI1 Fig 23 caption，PDF 索引 31）。
**【待核】** 主图菱形链**呈斜向**（而非散布），而 SI Fig 23a 的点是散布的。斜线在 (目标季节, lead) 平面里对应**恒定初始化月份**（因为 初始化月 = 目标月 − lead），这与春季可预报性障碍有关，但论文未说明 ⇒ 该解释未经证实。

### 7.4 从 PDF 矢量数据抽出的 0.5 等值线位置

> 方法：定位条带内文字为 `0.5` 的标签，用上面的 y→lead 映射换算。

| 模态 | 条带内 "0.5" 标签数 | 标签处 lead | 矢量 stroke 覆盖的 lead 区间 |
|---|---|---|---|
| ENSO | 1 | ≈15.9 | −0.1 … 22.9（4 条 path） |
| NPMM | 1 | ≈9.1 | 4.4 … 8.4 |
| SPMM | 1 | ≈3.1 | 0.7 … 4.3 |
| IOB | 2 | ≈8.9、≈18.9 | **未检出** |
| IOD | 1 | ≈4.3 | 0.0 … 5.9 |
| SIOD | 2 | ≈10.4、≈20.0 | 5.9 … 22.9 |
| TNA | **4** | ≈4.4、15.1、15.9、19.1 | 5.0 … 22.9（1 条 path） |

**【实证】** 标签数量与 lead 位置。
**【待核】** stroke 覆盖区间：IOB **未检出**，TNA **标签 4 个但只抽出 1 条 path** —— 抽取时用矩形包含做过滤，`matplotlib` 的内联标签会把一条等值线切成多段 path，`bbox` 可能被误排除。**stroke 段数不可靠，需人工看图确认。**

**【实证】** 论文对这两个模态给出的 headline（非本表读出）：ENSO 19 个月（正文 L237）、IOD 7 个月（正文 L117、L944）。与上表交叉核对：IOD 的 0.5 线在 lead 0–5.9 之间、标签在 4.3，与"7 个月"大致相容。
**【推断】** **TNA 的 0.5 线被切成 4 段（4 个标签）说明它的技巧正好在阈值附近上下抖动，处于噪声量级 ⇒ 不应引用单一数字。** 这也解释了正文为什么只对 ENSO 给硬数字（19 个月），对其他模态一律模糊（"+22%"、"ACC ≈ 0.5"）。

### 7.5 附：读这类图的方法（供汇报使用）

| 步骤 | 做什么 | 本图示例 |
|---|---|---|
| ① 确认三个编码 + 色标中心 | 发散色标 ⇒ 蓝色 = **负相关**（比无技巧更差），不是"低技巧" | SI Fig 13 的 IOB 长提前期有大片深蓝 |
| ② 找阈值线，读"在哪里断" | 读 0.5 等值线的位置，不读颜色深浅 | ENSO 线顶到框顶；IOD 线止于 lead≈6 |
| ③ 数等值线被切成几段 | **段数越多越不可信** | TNA 4 段 ⇒ 技巧在阈值附近抖 |
| ④ 看边界走向 | 竖线 = 目标季节锁相；水平线 = 纯记忆衰减；**斜线（斜率 +1）= 初始化月份** | Fig 3b 有斜边界与斜向菱形链 |
| ⑤ 长提前期锯齿 = 样本量告急 | 25 个月 × 12 季节的格子，20+ 个月处只有个位数独立事件 | lead > 15 个月的孤立色块按噪声看 |

---

## 八、基线协议问题

### 8.1 XRO / CNN / ResoNet 被改成 OOD

SI1 L55-66 原文要点：

> "In an in-distribution evaluation, a model is trained and tested on data drawn from the same observational reanalysis product. For instance, **XRO is fitted to ORAS5 reanalysis over 1979–2022** and verified via hindcasts over the same period… In contrast, **UniCM and its baselines (except DESN) in this study are evaluated under an out-of-distribution protocol: models are pre-trained exclusively on CMIP6 climate simulations and then tested on observational reanalysis data**."

**【推断】** XRO 是**物理动力模型**（SI1 Table 1 列其为 "Extended nonlinear recharge oscillator"，求解器 "Multivariate regression"，"Prescribed nonlinear coefficients"，SI1 L497-505）。这类模型的系数**本应在观测上拟合**。把它挪到 CMIP6 上训练再去测观测，是对手被人为削弱；用它论证"线性假设不足"（正文 L365-368：XRO 在 TNA/SIOD/IOB 上 5–6 个月内跌破 0.5）不够公平。

### 8.2 DESN 拿到更容易的协议

SI1 L67-74：

> "for DESN, we initially attempted to follow the same out-of-distribution protocol. However, pre-training on CMIP6 simulations led to substantially degraded performance… We therefore trained DESN directly on ORAS5 reanalysis data from 1958 to 1980 and evaluated it on ORAS5 data after 1980, which constitutes an **out-of-sample but in-distribution** setting."

**【实证】** DESN 得到 16 个月的技巧领先期（SI1 L71-72），与原文报告一致。正文 L237-238："surpassing DESN at 16 months"。
**【推断】** 这一条对 UniCM **有利** —— DESN 拿到了更容易的协议（in-distribution）却仍落后于 UniCM 的 19 个月。汇报时应同时说出这一条，否则批评 8.1 会显得片面。

### 8.3 作者自认的模态依赖性

正文 L1012-1014：

> "While UniCM marks an advance, its improvement is **mode-dependent**, with **smaller gains for modes such as the IOD** whose predictability is intrinsically constrained by tight seasonal phase-locking and short decorrelation timescales."

**【实证】** 与正文 L117 / L944 的"ENSO 19 个月、IOD 7 个月"一致（差 2.7 倍）。

### 8.4 显著性检验的自由度问题

| 项 | 内容 | 出处 |
|---|---|---|
| 论文声明 | "All reported peak correlations were statistically significant (**P < 0.001**)" | 正文 L569 |
| 场显著性 | BH-FDR 校正后：UniCM 在 ENSO 上几乎覆盖整个 24 个月，NPMM/IOB/TNA 达 12–18 个月；XRO 大多 9–13 个月后显著区萎缩 | SI1 L246-255 |
| 自相关 | 1 个月滑窗 ⇒ 相邻样本共享 **≈91.7%（11/12）** 输入特征 | SI1 L420-421 |

**【推断】** 论文未报告有效自由度（effective degrees of freedom）的估计。在高度自相关的序列上使用朴素自由度计算 P 值会**系统性高估显著性** ⇒ P < 0.001 这个数不可直接引用。SI1 L254 声称 "robust against both temporal autocorrelation and the risks of multiple testing"，但只说明了多重比较（BH-FDR），**没有说明时间自相关如何处理**。

### 8.5 数据集独立性

**【实证】** 正文 L231-232 作者自己声明："**ORAS5 and ERA5 share the same prescribed SST boundary conditions and thus do not constitute fully independent validations for SST-derived indices.**" 四个再分析测试集（GODAS / ERA5 / ORAS5 / SODA v2.2.4，正文 L228-233）中，只有 **GODAS 与 SODA 是独立的**。

### 8.6 与 S2S 基础模型的对比（供参考）

| 项 | UniCM | FuXi-S2S | CAS-Canglong | 出处 |
|---|---|---|---|---|
| ENSO ACC > 0.5 的月数 | **19** | 6 | 9 | SI1 L270-272 |
| 12 个月全球平均 ACC | **0.379** | 0.262 | 0.207 | SI1 L287-288 |
| 12 个月全球平均 RMSE | **0.465** | 0.505 | 0.515 | SI1 L288-290 |

**【实证】** 改造方式：修改输入层以容纳 5 个物理变量，使用与 UniCM **相同的数据集与自回归 rollout 策略**，模态指数由预测 SST 场**事后反算**（SI1 L262-267）。

---

## 九、未决问题

1. **4 个 CMIP6 模式是否真的只用于拼通道？** 具体地：CESM2-WACCM-FV2（仅在 SI2 L382-385 被点名）在代码中完全未出现，它是否参与过任何训练？
2. **是否做过"通道同源"的对照实验**（即 5 个通道全部取自同一模式）？若有，在哪一节？若无，是否有未公布的结果？`【待核】`
3. **λ 到底以哪组为准？** 需要向作者确认，或从发布代码的 commit history / 训练日志反查。目前只能确定"正文+Fig 38b"与"SI 5.2+train.sh"两组不可调和。
4. **Fig 38b 图例的三元组顺序**是否为 (λ1-λ2-λ3)？若顺序不同，第三节的矛盾描述需要修改。`【待核】`
5. **Fig 37b 中 ModeFormer-only 的评估口径**：它是用自身直接输出的指数评分，还是也做了某种"反算"？论文与 SI 均未说明。`【待核】`
6. **Fig 3b 菱形标记的确切定义**在论文（含 SI）里是否另有说明？目前只找到 SI Fig 23 对"点 = 显著性"的定义。
7. **Fig 3b 菱形链为何呈斜向**？"斜线 = 恒定初始化月份"这一解释未经证实。`【推断】` + `【待核】`
8. **`train.sh` 的 `--epochs 2`** 是发布时的调试残留，还是训练确实只跑 2 个 epoch（例如 `epoch` 的含义在 `settings.py` 里被重定义）？需要通读 `Trainer.py` 的 epoch 循环才能确认。`【待核】`
9. ~~模型空间域的准确范围~~ → **已核对完毕，转为已确认的不一致**。正文共出现三种互不相容的说法，行号如下：
   - 正文 **L83**："a 5° × 5° grid covering **0° E–360° E and 40° S–0° N**" → 5° 网格下为 **8 行**
   - 正文 **L1068-1069**："interpolated onto a regular 5° × 5° grid covering the **global oceans from 40° S to 40° N**" → 5° 网格下为 **16 行**
   - 正文 **L1117-1118**："Spatial attention in Globalformer is applied over **Np = 12 × 72** spatial patches, corresponding to a 5° × 5° resolution" → 12 行
   三者两两不等（8 / 16 / 12 行）；且第三种把「原始网格数」当成了「patch 数」（代码里 `emb_spatial_size = 12*72//(2*2) = 216`，见 S1）。
   **剩余待核**：真实训练域到底是哪一个？（`LoadData.py` 的经纬切片逻辑未逐行读）`【待核】`
10. **`_UniCM_code/` 的 commit hash 未记录**（`--depth 1` 克隆）。所有代码结论在重新核对时需锁定同一个 commit。
11. **Fig 37b / 38a / 38c 的精确数值**：本文件全部为像素读数（±0.01）。若能拿到作者的数据或图源文件，应替换为精确值。
12. **PDF 矢量抽取中 IOB 的 0.5 等值线未检出、TNA 的 4 个标签只对应 1 条 path** —— 需人工看图确认段数，或用更宽松的 bbox 过滤重跑 `audit/11_extract_fig3b_contours.py`。
13. **辅助物理场（τx、τy、hT、T300）是否有未公布的验证**？正文 L1160-1162 说它们"are not used as separate validation targets in our current setup"，"current setup" 这一措辞留了口子。`【待核】`
14. **Fig 37a 的 replacement 策略具体实现**：SI1 L363 只说 "Substituting specific latent spatial representations with mode-focused embeddings"，没有给出替换的是哪些位置、用什么算子。无法仅凭文字判断该对照是否公平。`【待核】`
15. **DESN 的 16 个月**是在 in-distribution 下取得，而正文 L237 的 19 个月是 OOD。两者的可比性需在汇报中显式说明（本文件已在 8.2 标注方向，但严格的可比性仍存疑）。`【待核】`
