# ENSO 充放电振子 → 正偏态 → 两种机制（Mermaid 源码）

> 用法：把下面的代码块贴到任何支持 Mermaid 的地方即可出图
> （飞书文档 / Typora / Obsidian / GitHub / mermaid.live）。
> 也可以直接看同目录下的 `enso_oscillator_logic_chain.png`（已渲染好的静态图）。

## 主图：完整逻辑链

```mermaid
flowchart TD
    A["① 充放电振子<br/>两个状态变量：Te 温度 + h 电量<br/>F 与 α 构成延迟负反馈 → 自我振荡"]
    A -.->|"提供长提前期先兆"| L["西太暖池热含量 T300<br/>9–15 个月可预报性的主要来源"]
    L -.->|"但只是必要条件"| M["2012 / 2014 / 2017<br/>热含量有利，却因缺 WWE 而未发展"]

    A --> B["② 观测事实<br/>SSTA 分布正偏：El Niño 比 La Niña 更强"]

    B --> C{"线性振子 + 对称噪声<br/>能否给出正偏？"}
    C -->|"不能：数学上必然给出高斯，偏度 = 0"| D["⇒ ENSO 必然含非线性<br/>或状态依赖噪声"]
    C -->|"能"| E["与观测矛盾"]

    D --> F["路线 A：非线性 Bjerknes 反馈<br/>深对流阈值/饱和 · 海洋热平流非线性"]
    D --> G["路线 B：乘性随机噪声<br/>WWE 随背景 SST 变暖而增多"]
    F --> F1["暖态增长快、冷态加深难"]
    G --> G1["暖态噪声大、极值尾部被拉长"]

    F1 --> H["共同结果：分布右偏"]
    G1 --> H
    H --> I["El Niño 约 1 年（快起快止）"]
    H --> J["La Niña 可维持 2–3 年，且更难预报"]
    H --> K["事件间差异大 = ENSO complexity"]

    classDef osc fill:#dbeafe,stroke:#2563eb
    classDef obs fill:#fee2e2,stroke:#dc2626
    classDef mech fill:#dcfce7,stroke:#16a34a
    classDef out fill:#fef3c7,stroke:#d97706
    class A,L,M osc
    class B,C,D,E obs
    class F,G,F1,G1 mech
    class H,I,J,K out
```

## 细节图 1：振子的内部结构（Eq. 1）

```mermaid
flowchart LR
    H["h　赤道平均温跃层深度异常<br/>= 赤道热含量库存"]
    T["Te　赤道东太平洋 SST 异常"]
    H -- "F 温跃层反馈<br/>h 变大 ⇒ 次表层暖水上涌 ⇒ Te 变大" --> T
    T -- "α 充放电速率<br/>Te 变暖 ⇒ 信风减弱 ⇒ 热量外排 ⇒ h 变小" --> H
    T -. "I_BJ 正反馈（自我增强）" .-> T
    H -. "ε 阻尼（自行衰减回零）" .-> H
```

## 细节图 2：一次完整事件（一个充放电周期）

```mermaid
flowchart TD
    A["充电态 h > 0<br/>西太暖池热含量充足（满电）"] --> B["温跃层反馈 F·h<br/>暖水上涌到海表"]
    B --> C["Te > 0<br/>中东部太平洋增暖"]
    C --> D["信风减弱 → 进一步增暖<br/>Bjerknes 正反馈 I_BJ·Te"]
    D --> C
    C --> E["风应力旋度反向<br/>Sverdrup 输运把赤道热量外排"]
    E --> F["放电 h 下降"]
    F --> G["温跃层反馈失去燃料<br/>Te 转负 = La Niña"]
    G --> H["反向风应力旋度<br/>热量重新汇入赤道"]
    H --> A
    C --> I["冬季达峰：El Niño"]
    G --> J["La Niña 可维持 2–3 年<br/>可预报性低于 El Niño"]
```

## 一句话总结

> 海洋记忆给了 ENSO 预报的**可能性**，状态依赖的非线性与噪声给了它的**上限**。
