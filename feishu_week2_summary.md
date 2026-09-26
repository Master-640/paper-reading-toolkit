# Week2 Finder
(wiki node=WdbpwCG8giJsxYkjckVce9Y0nce -> docx=W9xNduWQQovJtyxZ10Zcqk8cnbV)

Week2 Finder
三要素
insight：寻找关键
背景知识
专有名词
GCC:最大的联通分量（多次映射之后得到最终的网络？）
什么是小型合成网络
小型合成网络是什么？
image.png

Finder在小型合成网络上进行学习，然后直接迁移到大型网络
图神经网络
见下方
权重 VS 超参数
image.png

非平凡和遗传性质
image.png

如何生成向量
image.png

重点问题
Finder是怎么学习的
image.png


image.png

Finder的设计实现
什么是graphsage
image.png

encoder
为什么要用encoder？为什么要用decoder？
encoder：为了更好地去描述一个节点的对应信息，所以使用encoder。具体而言，就是在这个编码器中，我会计算这一个节点的初始特征H0，它的邻居特征是H1，然后它的第二条邻居是H2，把这些东西都整合到同一个向量中(逐层进行整合与处理），这样就可以描述这个节点在这个网络中的所处位置以及其他信息。所以我们需要encoder。
H0 = [1，1]，实际代码中跳了三层
聚合邻居的H0，可能是通过求平均值之类的方法得到H0'
然后拼接起来，得到一个新的向量
接着（乘以权重矩阵 W + 激活函数）
image.png

然后我们得到的一个N*d的矩阵
Decoder
flowchart LR
    subgraph Input [输入特征]
        S[全局状态向量 S\n来自虚拟节点]
        Hv[节点动作特征 h_v\n来自真实节点]
    end

    subgraph Interaction [状态-动作交互]
        S --> Outer[外积操作\nS ⊗ h_v^T]
        Hv --> Outer
        Outer --> Matrix[生成 D×D 交互矩阵\n捕捉细粒度依赖]
    end

    subgraph Scoring [Q值打分]
        Matrix --> Flatten[展平/双线性池化]
        Flatten --> MLP[多层感知机 MLP\n+ ReLU 激活]
        MLP --> QValue[输出标量 Q 值\n评估移除该节点的长期收益]
    end

    style S fill:#f9f,stroke:#333,stroke-width:2px
    style Hv fill:#bbf,stroke:#333,stroke-width:2px
    style QValue fill:#f96,stroke:#333,stroke-width:2px
一个非常巧妙的设计！虚拟节点来描述当前图的整体情况！从而维护全局向量。（这也是算法竞赛中网络流的一个常见技巧！）
整体脉络：
image.png

image.png

整体架构
flowchart TD
    subgraph Env ["环境：小型合成网络 BA模型"]
        State["当前状态 S_t<br>残余网络 + 虚拟节点"]
    end

    subgraph Forward ["前向传播"]
        State --> Encoder["Encoder<br>GraphSAGE"]
        Encoder --> Decoder["Decoder<br>外积 + MLP"]
        Decoder --> PredQ["预测所有节点的 Q 值"]
    end

    subgraph Action ["动作选择与执行"]
        PredQ --> Epsilon["ε-greedy 策略<br>训练时探索，应用时贪心"]
        Epsilon --> Remove["移除 Q 值最高的节点<br>或批量移除前 1%"]
        Remove --> Reward["获取即时奖励 R_t<br>网络连通性下降量"]
        Reward --> NextState["得到新状态 S_t+1"]
    end

    subgraph Replay ["经验回放池 Experience Replay"]
        Transition[("(S_t, A_t, R_t, S_t+1)")]
        NextState --> Transition
    end

    subgraph Train ["模型更新"]
        Transition --> Sample["随机采样 Mini-batch"]
        Sample --> TargetQ["计算目标 Q 值<br>Target = R_t + γ * max Q_next"]
        TargetQ --> Loss["计算损失 Loss<br>n-step Q-learning Loss<br>+ Graph Reconstruction Loss"]
        Loss --> Backprop["反向传播"]
        Backprop -. "更新权重 W" .-> Encoder
        Backprop -. "更新权重 W" .-> Decoder
    end

    NextState -. "循环下一回合" .-> State

    style State fill:#bbf,stroke:#333,stroke-width:2px
    style PredQ fill:#f96,stroke:#333,stroke-width:2px
    style Loss fill:#f66,stroke:#333,stroke-width:2px
    style Backprop fill:#f66,stroke:#333,stroke-width:2px

一些疑问
虚拟节点在代码中的具体实现方式？
全连接神经网络？
当前层和下层的每一个节点都是相连的
什么是L2归一化？向量中除以所有元素平方和的sqrt
image.png

什么是外积？
两个向量乘得到一个矩阵
image.png

在这里什么是MLP？
一开始输入一个向量，然后过几层全连接层（4096 -> 128 -> 64 -> 1）得到最后的Q值
