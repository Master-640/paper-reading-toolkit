# FINDER 论文的强化学习内容清单与学习路径

> 本文档回答一个问题：**FINDER 这篇论文到底用了哪些强化学习内容，以及该按什么顺序、去哪里学。**
>
> 所有行号均经过实际核对：
> - `FINDER_text.txt` / `FINDER_supp_text.txt` = 论文正文 / 补充材料的 pdftotext 提取（同目录）
> - `FINDER/code/FINDER_ND/` = 官方源码（TensorFlow 1.x + Cython）
>
> 整理范围：RL 部分。编码器/解码器的架构问题另有专文（见 `FINDER-代码架构图解.md`）。

---

## 一、总览：范围很窄，但很扎实

FINDER 的 RL 构成可以一句话概括：

> **一个标准 MDP + n-step Q-learning + 经验回放 + 目标网络 + ε-greedy 退火。**

关键事实是它**只采用了 n-step 这一个 DQN 改进**，其余全部实验后排除。这意味着学习清单可以显著收窄——Double/Dueling/Distributional/Prioritized/NoisyNet 都不必深读，只需知道"被测过且无增益"。

---

## 二、A 类：用到并起作用的 RL 内容

### A-1 问题建模层

| # | 概念 | 论文原话 / 位置 | 代码位置 |
|---|---|---|---|
| A1 | **MDP（马尔可夫决策过程）** | p.319 Methods：*"the finding of key players as a Markov decision process"* | `mvc_env.cpp:40-292` |
| A2 | **状态 s** = 残余网络 | *"the state is defined as the residual network"* | `covered_set` + `graph` |
| A3 | **动作 a** = 移除一个节点 | *"the action is to remove (or activate) the identified key player (node)"* | `step(a)`，`mvc_env.cpp:53` |
| A4 | **奖励 r** | *"the reward is the decrease of ANC after taking the action"* | `getReward()`，`mvc_env.cpp:289-292` ⚠️ **与论文不符，见第五节** |
| A5 | **回合 episode** | *"complete a whole key-player finding process (denoted as an episode)"* | `isTerminal()`，`mvc_env.cpp:282-286` |
| A6 | **终止状态** | Suppl. `:1166`：*"a terminal state can be an user-defined state, e.g., maximum budget nodes or minimum connectivity threshold"* | 代码口径是"所有边被覆盖" |

> **A6 值得单独注意**：补充材料明确说终止状态是**用户自定义**的，并给了两种典型设定（最大预算节点数 / 最小连通阈值）。而代码 `isTerminal()` 用的是"所有边都被覆盖"（MVC 式）。论文正文 `:286` 又说 *"the residual graph becomes completely disconnected"*。三处口径不同，但都属于"用户可定义终止"这个框架内——汇报时统一到代码口径即可。

### A-2 学习算法层

| # | 概念 | 论文位置 | 代码位置 |
|---|---|---|---|
| B1 | **Q 函数 / 动作价值函数** | p.319 Decoding 段 | `q_pred` / `q_on_all`，`FINDER.pyx:278,330` |
| B2 | **Bellman 最优方程 / 值迭代** | Suppl. **Eq. S19**（`:855`）：$Q^*_k(s_t,a_t)=Q^*_{k-1}+\eta[\cdot]$ | 由 `:528` 的 TD 目标隐式实现 |
| B3 | **Q-learning**（off-policy，用 max） | *"n-step Q-learning loss"*，p.319 | `IsMultiStepDQN = True`，`:75` |
| B4 | **TD 学习 / 自举 bootstrapping** | Suppl. **Eq. S20**（`:881`）：$(r_t+\gamma\max_a\hat Q(s_{t+1},a)-Q(s_t,a_t))$ | `:520-530` |
| B5 | **n-step 回报** | **Eq. S22**（`:905`）；Table S4 *"Q-learning steps 3,4,5"* | `N_STEP = 5`，`:40` |
| B6 | **损失函数** = Q-learning loss + 重构正则 | Suppl. **Eq. S27**（`:1136-1146`） | `:295,297` |
| B7 | **折扣因子 γ** | Table S4（`:2136`）写 **0.99** | `GAMMA = 1`，`:26` ⚠️ **冲突，见第五节** |
| B8 | **目标网络 Q̂**（硬更新） | **Algorithm S3** 第 3、17 行；Table S4 *"update time 10³"* | `:121,123,502,643` |

**Eq. S27 的完整形式**（学习时对照这段最清楚）：

$$\text{Loss}=\underbrace{\mathbb{E}_{(s_t,a_t,r_{t,t+n},s_{t+n})\sim U(\mathcal{D})}\Big[\big(r_{t,t+n}+\gamma\max_{a'}\hat Q(s_{t+n},a';\hat\Theta_Q)-Q(s_t,a_t;\Theta_Q)\big)^2\Big]}_{\text{Q-learning loss}}+\alpha\underbrace{\sum_{i,j}^{N}s_{i,j}\|y_i-y_j\|_2^2}_{\text{Graph reconstruction loss}}$$

补充材料 `:1148-1162` 对每一项都有文字解释，包括 γ 的作用：

> *"γ is the discount factor that determines the importance of future rewards. A factor of 0 will make the agent short-sighted by only considering current rewards, while a factor approaching 1 will make it strive for a long-term high reward. If it exceeds 1, the action values may diverge."*

### A-3 数据与探索层

| # | 概念 | 论文位置 | 代码位置 |
|---|---|---|---|
| C1 | **经验回放池** | *"experience replay buffer—a queue that maintains the most recent M 4-tuples"* | `nstep_replay_mem.cpp:20-33` |
| C2 | **容量 M** | Table S4 *"5×10⁵"* | `MEMORY_SIZE = 500000`，`:31` |
| C3 | **均匀随机 mini-batch** | **Algorithm S3** 第 14 行；Eq. S27 的 $U(\mathcal{D})$ | `Sampling()`，`:82-105` |
| C4 | **ε-greedy 探索** | *"select the highest-Q node with probability (1−ε) and take a random action otherwise"* | `:413-422` |
| C5 | **ε 线性退火 1.0 → 0.05** | Table S4 *"exploration steps 10⁴"* | `:603-605,620` |
| C6 | **暖池（no-op）** | Table S4 *"no op max 10³ — number of episodes to be played by the agent without updating parameters"* | `:600-601` |
| C7 | **训练/应用策略分离** | 训练 ε-greedy → 应用 **batch nodes selection**（前 1%） | `Predict` 后排序截断 |

---

## 三、D 类：**没有**用到的 RL 内容（论文明确说明）

代码里有一整套 DQN 变体开关（`FINDER.pyx:70-78`），**只有一个打开**：

```python
71:  self.IsHuberloss = False
72:  self.IsDoubleDQN = False           # ← 关
73:  self.IsPrioritizedSampling = False # ← 关
74:  self.IsDuelingDQN = False          # ← 关
75:  self.IsMultiStepDQN = True         # ← 唯一打开的
76:  self.IsDistributionalDQN = False   # ← 关
77:  self.IsNoisyNetDQN = False         # ← 关
78:  self.Rainbow = False               # ← 关
```

补充材料把结论写得很明确（`FINDER_supp_text.txt:892-898`）：

> *"DQN has been a great improvement compared to traditional Q-learning ones, and afterward, several extensions have been proposed... Such as Double DQN[42], prioritized sampling[43], n-step DQN[44], dueling DQN[45], distributional DQN[46], etc. **We have implemented all these DQN extensions in our problem and found that only n-step Q-learning improves the results, while other extensions bring along merely marginal effects or even hinder the performance.** Therefore, we only employ the n-step DQN model as the learning part of our framework."*

| 未采用 | 论文结论 |
|---|---|
| Double DQN | 边际效应或有害 |
| Prioritized sampling | 同上（代码实现了但关闭） |
| Dueling DQN | 同上 |
| Distributional DQN | 同上 |
| NoisyNet | 同上 |
| Rainbow | 未采用 |
| Huber loss | 未采用（用 MSE） |

**n-step 为什么有效**（`:899-902`）：

> *"The n-step DQN waits for n steps, rather than the traditional one step, before updating the parameters, to **collect a more accurate estimate of the future rewards**."*

即：n 步累计奖励 $\sum_{i=0}^{n-1}r(s_{t+i},a_{t+i})$ 比单步奖励更接近真实回报，降低了 TD 目标的偏差。

---

## 四、超参核对：Table S4 vs 代码

逐项对齐结果如下。**大部分精确一致**，这是论文超参表基本可信的信号。

| 超参 | Table S4（`:2122-2150`） | 代码 | 一致？ |
|---|---|---|---|
| replay memory size | 5×10⁵ | `MEMORY_SIZE = 500000`（`:31`） | ✅ |
| learning rate | 1×10⁻⁴ | `LEARNING_RATE = 0.0001`（`:30`） | ✅ |
| embedding dimension | 32,64,128 | `EMBEDDING_SIZE = 64`（`:28`） | ✅（取 64） |
| update time | 10³ | `UPDATE_TIME = 1000`（`:27`） | ✅ |
| layer iterations | 3,4 | `max_bp_iter = 3`（`:51`） | ✅（取 3） |
| maximum episodes | 10⁶ | `MAX_ITERATION = 1000000`（`:29`） | ✅ |
| Q-learning steps | 3,4,5 | `N_STEP = 5`（`:40`） | ✅（取 5） |
| mini-batch size | 16,32,64 | `BATCH_SIZE = 64`（`:44`） | ✅（取 64） |
| initial exploration | 1 | `eps_start = 1.0`（`:603`） | ✅ |
| final exploration | 5×10⁻² | `eps_end = 0.05`（`:604`） | ✅ |
| exploration steps | 10⁴ | `eps_step = 10000.0`（`:605`） | ✅ |
| no op max | 10³ | `PlayGame(100, 1)` × 10（`:600-601`） | ✅（1000 局） |
| **discount factor** | **0.99** | **`GAMMA = 1`**（`:26`） | ❌ **冲突** |

> **术语说明**：Table S4 用 "maximum episodes"（10⁶），但代码是 `MAX_ITERATION` 即**训练步数**（`:615` 的 `for iter in range(...)`）。论文正文 `:286` 又说 ε 在 "10,000 episodes" 内退火，而代码 `eps_step` 是**迭代数**。这是论文用词不严谨（episode ≠ iteration），不是实质矛盾。

---

## 五、三处论文-代码出入（带着问题去读）

这三处是精读时的**重点核对项**，也是汇报时可能被追问的点。

### 出入 ①：折扣因子 γ（最严重）

| 来源 | 说法 |
|---|---|
| Table S4（`:2136`） | `discount factor 0.99` |
| Suppl. II.D.1（`:560`） | `γ ∈ (0, 1)` |
| 代码 `FINDER.pyx:26` | `cdef double GAMMA = 1` |

代码 `:528` 直接使用：`q_rhs = GAMMA * self.Max(list_pred[i])`，即 $\gamma=1$（不打折），且 n-step 时等效 $\gamma^n=1$。

$\gamma=1$ 与 $0.99$ 在 $n=5$ 时差 $0.99^5\approx0.951$，即后续价值被折扣 5% vs 完全不打折。

**判断**：倾向以代码为准（`γ=1`）。理由——其余 12 项超参精确吻合，而论文正文中另两处描述（奖励、终止条件）经核实也与代码不符，说明论文的文字段落有几处未与最终代码同步。

**如何验证**：把 `GAMMA` 改成 0.99 跑一轮对比。若结果显著变化，说明该常量确实生效（必然是生效的，`tf` 图里直接引用）；若要判断作者真实用的是哪个，需要看其报告结果对应的 checkpoint 与配置。

### 出入 ②：奖励定义

论文正文（`:281`）：

> *"the reward is **the decrease of ANC** after taking the action"*

代码（`mvc_env.cpp:289-292`）：

```cpp
double MvcEnv::getReward() {
    return -(double)getMaxConnectedNodesNum() / (graph->num_nodes * graph->num_nodes);
}
```

返回**当前状态的 LCC 占比取负**，不是差值。紧跟其后有一段被注释掉的**差值版**：

```cpp
294: //double MvcEnv::getReward(double oldCcNum)
296: //    return (CcNum - oldCcNum) / CcNum*graph->num_nodes ;
```

**奖励设计的演进史**（补充材料里有完整记录）：

| 代 | 形式 | 出处 |
|---|---|---|
| 第 1 代 | $r_t=-1$ | Suppl. `:1747`：*"At first, we simply changed the reward to be −1"* |
| 第 2 代 | ANC/LCC **差值** | 被注释掉的 `getReward(double oldCcNum)`；论文正文写的是这代 |
| 第 3 代（现行） | $-\mathrm{LCC}(s_t)/N^2$ **绝对值** | `getReward()` |

**为什么绝对值也能工作**：$\gamma=1$ 时，n 步累计恰为

$$\sum_{k=0}^{n-1}r_{t+k}=-\sum_{k}\frac{\mathrm{LCC}(s_{t+k})}{N^{2}}$$

即"持续压低 LCC"。与"最大化 ANC 下降总量"在形式上是同一个优化目标，但**信号性质不同**——每步都有稠密信号，而非只在有下降时才有信号。

### 出入 ③：终止条件

| 来源 | 说法 |
|---|---|
| 论文正文（`:286`） | *"the residual graph becomes completely disconnected"* |
| 代码（`mvc_env.cpp:285`） | `return graph->num_edges == numCoveredEdges;`（所有边被覆盖） |
| Suppl. `:1166` | *"a terminal state can be an user-defined state"* |
| Suppl. `:1773` | *"the terminal state is set to be the absence of 2-core"* |

**关键澄清**：`:1773` 那处 2-core 的设定属于**补充材料 IV.F 描述的变体**（网络拆解 minimal percolation threshold 问题），**不是主实验 FINDER 的设定**。原文 `:1771-1776`：

> *"building upon the vanilla framework, we define the graph state in this problem as the 2-core of the residual graph, rather than the previous whole residual graph; the terminal state is set to be the absence of 2-core. For the reward function, we set as −1 at each step... **Those are the only changes made to FINDER.**"*

所以主实验口径应取"所有边被覆盖"（代码）。这个出入影响很小（都是"图被打破到某程度就停"），但汇报时口径要统一。

---

## 六、材料定位：读什么，读哪里

### 6.1 论文（精读范围，按优先级）

| 优先级 | 位置 | 内容 |
|---|---|---|
| ⭐⭐⭐ | **正文 p.319 Methods** | RL 建模、奖励、n-step、ε、回放池、应用阶段策略 |
| ⭐⭐⭐ | **Fig. 2** | MDP 状态-动作-奖励循环图；标注 "Repeat N episodes"、"Experience replay buffer" |
| ⭐⭐⭐ | **Fig. 3** | 两阶段框架（离线训练 → 真实应用） |
| ⭐⭐⭐ | **Suppl. Algorithm S3**（`:1250-1294`） | 完整训练伪代码，17 行，**最值得逐行读** |
| ⭐⭐ | **Suppl. II.C**（`:908-917`） | S2V-DQN 简介（FINDER 的前身） |
| ⭐⭐ | **Suppl. II.D.1**（`:1036-1108`） | Decoding + Greedy selection（含 batch nodes selection 动机） |
| ⭐⭐ | **Suppl. II.D.2**（`:1112-1184`） | 训练算法、$\Theta_E/\Theta_D$ 两个参数集、Eq. S19/S20/S22/S27 |
| ⭐⭐ | **Suppl. Eq. S25**（`:960-975`） | S2V-DQN 的 Q-learning 损失（对比着读） |
| ⭐⭐ | **Suppl. Table S4**（`:2115-2150`） | 全部超参（⚠️ γ 那行注意） |
| ⭐⭐ | **Suppl. 消融段**（`:892-898`） | 为何只采用 n-step |
| ⭐ | **Suppl. II.D.3**（`:1205+`） | 复杂度分析 |
| ⭐ | **Suppl. IV.F**（`:1740-1784`） | 奖励函数与终止条件的演进史（出入 ②③ 的原始记录） |

### 6.2 代码

| 文件 | 行数 | 对应 RL 内容 |
|---|---|---|
| `FINDER_ND/mvc_env.pyx` + `src/lib/mvc_env.cpp` | 460 | **环境**：MDP 的状态/动作/奖励/终止 |
| `FINDER_ND/nstep_replay_mem.cpp` | 106 | **回放池** + n-step 构造（`:53-80` 是关键） |
| `FINDER_ND/FINDER.pyx:395-426` | — | **`Run_simulator`**：rollout + ε-greedy |
| `FINDER_ND/FINDER.pyx:504-535` | — | **`Fit`**：Bellman 目标 + 损失 |
| `FINDER_ND/FINDER.pyx:595-646` | — | **`Train`**：主循环 + ε 退火 + 目标网络同步 |
| `FINDER_ND/FINDER.pyx:70-78` | — | **DQN 变体开关**（看清哪些关着） |
| `FINDER_ND/nstep_replay_mem_prioritized.cpp` | — | 优先级采样（**未启用，可跳过**） |

---

## 七、学习路径（按顺序）

### 第 1 层：必需地基

不理解这些就看不懂论文的 RL 部分。

- MDP 五元组、回报（return）、折扣因子 γ
- 状态价值 $V(s)$ / 动作价值 $Q(s,a)$
- Bellman 期望方程与最优方程、值迭代
- 探索-利用困境（exploration vs exploitation）

### 第 2 层：FINDER 直接使用的算法

- **Q-learning**（off-policy，用 $\max$）← 核心
- **TD 学习与自举**（bootstrapping）← 核心
- **n-step 回报**（n-step TD）← FINDER 唯一采用的改进
- ε-greedy 及其线性退火
- **经验回放**（Experience Replay）
- **目标网络**（Target Network）

### 第 3 层：知道"被排除了"即可，不必深读

- Double DQN / Dueling DQN / Distributional DQN
- Prioritized Experience Replay
- NoisyNet / Rainbow
- on-policy vs off-policy 的取舍（读 SARSA 做对比即可）

### 第 4 层：本论文特有（非通用 RL）

- **为什么这是 RL 而不是组合优化**：动作改变状态 → 序贯决策 → 长期最优 ≠ 单步局部最优之和
- **训练/推理策略分离**：训练 ε-greedy 逐个删点 → 应用前 1% 批量删点

---

## 八、自测问题（能答上就说明读懂了）

1. FINDER 的奖励为什么不直接用 Q 值，而要设计成 $-\mathrm{LCC}/N^2$？绝对值与差值的区别是什么？
2. 为什么必须用 RL 而不能用贪心启发式？叙述"长期最优 ≠ 单步最优之和"在这个问题上的具体表现。
3. n-step 回报相对 1-step 的优劣各是什么？为什么补充材料说它"collect a more accurate estimate of the future rewards"？
4. 目标网络 Q̂ 解决的是什么问题？为什么不能直接用 Q 算 target？
5. 经验回放池为什么能提升稳定性？（提示：样本相关性、数据分布漂移）
6. 均匀随机抽样为什么是对期望 TD 损失的**无偏**估计？
7. ε 从 1.0 退火到 0.05 而不是直接设为 0，原因是什么？
8. 训练时逐个删点、应用时一次删 1%，为什么性能基本不受影响？
9. γ=1 在本问题中是否会导致发散？补充材料为什么说 "if it exceeds 1, the action values may diverge"？
10. 论文说奖励是"ANC 的下降"，代码返回的是"LCC 占比取负"，这两者在 γ=1、n=5 的设定下是什么关系？

> 第 1、5、6、10 题的详细推导见对话记录；第 4、7 题见 `FINDER-代码架构图解.md` 与本文 §二。

---

## 九、待办核对项

- [ ] 验证 `GAMMA` 实际取值（改 0.99 跑对比，或检查论文报告结果的对应配置）
- [ ] 确认 `MAX_ITERATION` 是迭代数还是回合数（代码是迭代数；Table S4 写 "maximum episodes"）
- [ ] 核对 `MEMORY_SIZE`：代码 500000 = 5×10⁵，与 Table S4 一致；但 `FINDER-代码架构图解.md:139` 记的是"代码为 500,000，论文写 50,000"——**该处笔记需修正，论文正文 `:291` 写 M=50,000、Table S4 写 5×10⁵，两处不一致**
- [ ] 奖励演进史（Suppl. IV.F）与主实验奖励的关系，确认主实验确实用绝对值版

---

*整理依据：`FINDER.pdf`（Nature Machine Intelligence, VOL 2, June 2020, 317–324）与 `FINDER_supplementary.pdf`，以及官方源码 `FINDER/code/FINDER_ND/`。所有行号已核对。*
