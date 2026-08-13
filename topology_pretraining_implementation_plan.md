# 纯拓扑图预训练模型实现流程

> 目标：训练一个与具体数据集、节点语义、任务标签无关的 **Topology Pretrained Graph Encoder**。  
> 模型输入 rooted local graph，输出固定维度 topology embedding；随后可选地通过 Projector 对齐到冻结 LLM，生成 GraphTokens。

---

# 1. 总体框架

整个训练流程分为两个阶段：

```text
Stage 1：Topology Pretraining
Synthetic Rooted Graph
        │
        ├── Isomorphic Pair
        │
        └── Perturbed Pair
                │
                ↓
      Continuous Topology Distance
                │
                ↓
        2-layer SUM-GIN
                │
                ↓
        Topology Embedding
                │
      ┌─────────┼──────────┐
      ↓         ↓          ↓
Similarity   Ranking   Structure Stat
  Loss        Loss        Loss
      └─────────┼──────────┘
                ↓
      Pretrained Topology Encoder
                │
              Freeze
                │
        ┌───────┴────────┐
        ↓                ↓
 Standalone Graph     Stage 2
    Encoder         LLM Projector
```

核心模型：

\[
f_\theta:(G,c)\rightarrow z_G\in\mathbb R^D
\]

其中：

- \(G=(V,E)\)：局部图；
- \(c\)：中心节点；
- \(z_G\)：rooted topology embedding；
- \(D\)：embedding 维度，第一版建议 \(D=128\)。

目标：

\[
G_1\cong G_2
\Rightarrow
\cos(z_1,z_2)\approx 1
\]

同时：

\[
d_{\text{topo}}(G_1,G_2)\uparrow
\Rightarrow
\cos(z_1,z_2)\downarrow
\]

最终学习一个连续的 topology metric space。

---

# 2. Stage 1：Synthetic Graph 生成

## 2.1 图类型

随机生成 rooted 2-hop graph：

\[
G=(V,E,c)
\]

要求中心节点 \(c\) 到所有保留节点的最短路径距离不超过 2。

建议训练图覆盖：

- Erdős–Rényi random graph
- Tree
- Star
- Cycle
- Barbell / bridge-like graph
- Scale-free / Barabási–Albert graph
- Community graph
- Dense local graph
- Sparse local graph
- Irregular mixed graph

建议随机变化：

- 节点数量；
- 边数量；
- 平均度；
- density；
- degree distribution；
- 1-hop / 2-hop 节点比例。

目的是扩大 synthetic topology distribution，而不是模拟某个特定真实数据集。

---

## 2.2 Rooted 2-hop 约束

生成原始图以后：

1. 随机选中心节点 \(c\)；
2. 计算以 \(c\) 为中心的 2-hop ego graph；
3. 只保留：

\[
\{v\mid dist(v,c)\leq2\}
\]

4. 重新编号用于 batch，但保留 root index。

---

# 3. 节点输入特征

第一版采用纯拓扑输入。

基础输入：

\[
x_v=\mathbf 1_D
\]

推荐：

\[
D=128
\]

即：

\[
X=\mathbf1_{|V|\times128}
\]

这样模型无法利用：

- 文本；
- 用户信息；
- 商品信息；
- 标签；
- 时间；
- 数据集特定属性。

所有差异只能来自 adjacency matrix。

---

## 3.1 Root Indicator

因为目标是 rooted topology：

\[
(G,c)\rightarrow z_{G,c}
\]

建议额外加入 root indicator：

\[
r_v=
\begin{cases}
1,&v=c\\
0,&v\neq c
\end{cases}
\]

输入可写为：

\[
x_v=[\mathbf1_D\Vert r_v]
\]

也可以将 root indicator 单独经过一个小 embedding 后与全 1 特征相加。

建议做消融：

- All-one
- All-one + Root Indicator

---

# 4. Topology Encoder

推荐第一版采用：

\[
\boxed{\text{2-layer SUM-GIN + MLP}}
\]

第 \(l\) 层：

\[
h_v^{(l+1)}
=
MLP_l
\left(
(1+\epsilon_l)h_v^{(l)}
+
\sum_{u\in N(v)}h_u^{(l)}
\right)
\]

其中：

- \(MLP_1\neq MLP_2\)；
- 不同层允许不同参数；
- **不同图必须共享同一套 encoder 参数**。

即：

\[
\theta=
\{MLP_1,MLP_2,\epsilon_1,\epsilon_2\}
\]

对所有 synthetic / YelpZip / Amazon / 其他图均共享。

---

## 4.1 为什么使用 SUM

全 1 输入下：

\[
\sum_{u\in N(v)}\mathbf1
=
\deg(v)\mathbf1
\]

因此第一层可以自然感知 degree。

第二层进一步聚合：

\[
\{\!\{\deg(u):u\in N(v)\}\!\}
\]

从而学习：

- root degree；
- 1-hop 节点数；
- 邻居 degree distribution；
- 2-hop branching；
- local connectivity。

不建议第一版使用 MEAN，因为：

\[
\frac1{|N(v)|}\sum_{u\in N(v)}\mathbf1
=
\mathbf1
\]

会弱化重要的数量信息。

---

## 4.2 MLP 推荐结构

每层 GIN 内部可使用：

```text
Linear(D, 2D)
→ GELU
→ Linear(2D, D)
```

例如：

```text
128 → 256 → 128
```

第一版不需要太深。

---

# 5. Topology Embedding

2-hop 图传播两层：

\[
H^{(0)}
\rightarrow
H^{(1)}
\rightarrow
H^{(2)}
\]

第一版输出中心节点：

\[
\boxed{
z_G=h_c^{(2)}
}
\]

暂时不加入额外 projection head。

目的：让 topology supervision 直接作用到 encoder。

后续可消融：

\[
z_G=
[h_c^{(2)}\Vert \operatorname{SUMPOOL}(H^{(2)})]
\]

但不作为第一版默认设置。

---

# 6. Contrastive Pair 构造

对于每个 anchor graph：

\[
G_1
\]

生成两类训练样本。

---

## 6.1 Isomorphic Positive Pair

对节点做随机 permutation：

\[
G_{\text{iso}}=\pi(G_1)
\]

同时保持 root correspondence。

要求：

\[
d(G_1,G_{\text{iso}})=0
\]

目标：

\[
\cos(z_1,z_{\text{iso}})\rightarrow1
\]

作用：强化 permutation / isomorphism consistency。

---

## 6.2 Perturbed Pair

从：

\[
G_1=(V,E_1)
\]

出发生成：

\[
G_2=(V,E_2)
\]

第一版固定：

\[
V_1=V_2
\]

只进行：

- edge addition；
- edge deletion。

暂时不加入：

- node addition；
- node deletion。

这样节点 correspondence 已知，无需求一般意义上的昂贵 GED。

---

# 7. 扰动记录

不能直接把“操作次数”作为最终距离。

例如：

```text
add(u,v)
delete(u,v)
```

执行了两次操作，但最终 topology 未发生变化。

因此定义：

\[
\Delta E=E_1\triangle E_2
\]

其中 \(\triangle\) 表示 symmetric difference。

训练数据生成过程中建议维护 changed-edge hash set。

若实际执行 \(k\) 次操作，则维护复杂度平均约：

\[
O(k)
\]

这里使用的是：

> **fixed-correspondence edge edit distance**

而不是一般意义的 exact Graph Edit Distance。

---

# 8. 连续拓扑距离

第一版采用 normalized edge Jaccard distance：

\[
\boxed{
d(G_1,G_2)
=
\frac{|E_1\triangle E_2|}
{|E_1\cup E_2|}
}
\]

满足：

\[
0\leq d\leq1
\]

解释：

- \(d=0\)：边集合完全一致；
- \(d\rightarrow1\)：边结构差异越来越大。

等价于：

\[
d
=
1-
\frac{|E_1\cap E_2|}
{|E_1\cup E_2|}
\]

优势：

- 自动归一化图规模；
- 不依赖固定最大扰动次数；
- 不需要人为划分轻 / 中 / 重扰动。

---

# 9. Isomorphism Check

可能出现：

\[
E_1\neq E_2
\]

但：

\[
G_1\cong G_2
\]

即节点编号上的边集合发生变化，但 topology 实际同构。

这会产生错误监督。

因此对 perturbation pair 建议进行 rooted graph isomorphism check。

若：

\[
(G_1,c_1)\cong(G_2,c_2)
\]

则：

\[
d=0
\]

或者直接把该 pair 放入 isomorphic-positive pool。

对于小型 2-hop graph，这一步通常可接受。

---

# 10. 连续目标相似度

定义：

\[
\boxed{
s=1-d
}
\]

因此：

\[
d=0\Rightarrow s=1
\]

\[
d=0.25\Rightarrow s=0.75
\]

\[
d=0.5\Rightarrow s=0.5
\]

\[
d=1\Rightarrow s=0
\]

模型预测：

\[
\hat s=
\cos(z_1,z_2)
\]

目标：

\[
\boxed{
\cos(z_1,z_2)\approx1-d(G_1,G_2)
}
\]

第一版不使用：

\[
s=1-2d
\]

因为 topology 不相似不意味着两个向量必须方向完全相反。

---

# 11. 扰动距离采样

不能让训练数据大量集中在：

\[
d\approx0
\]

否则模型只会学习区分非常相似的 topology。

建议使整体 \(d\) 分布尽可能覆盖：

\[
[0,1]
\]

理想近似：

\[
d\sim U(0,1)
\]

实现方式：

1. 采样：

\[
d_{\text{target}}\sim U(0,1)
\]

2. 从 \(G_1\) 开始不断 add/delete edge；
3. 每次更新 \(d_{\text{actual}}\)；
4. 当：

\[
|d_{\text{actual}}-d_{\text{target}}|<\delta
\]

时停止。

例如：

\[
\delta=0.02
\]

对于小图无法精确达到 target 时，选择最近可达值即可。

---

# 12. Loss 1：Similarity Alignment

计算：

\[
z_1=f_\theta(G_1)
\]

\[
z_2=f_\theta(G_2)
\]

预测：

\[
\hat s=\cos(z_1,z_2)
\]

teacher：

\[
s=1-d
\]

推荐：

\[
\boxed{
L_{\text{sim}}
=
Huber(\hat s-s)
}
\]

即：

\[
L_{\text{sim}}
=
Huber
\left[
\cos(z_1,z_2)
-
(1-d)
\right]
\]

相比 MSE，Huber 对 topology-distance teacher 中不可避免的噪声更稳健。

---

# 13. Loss 2：Ranking Loss

同一个 anchor：

\[
G
\]

生成两个不同扰动：

\[
G_a,\quad G_b
\]

若：

\[
d(G,G_a)<d(G,G_b)
\]

则要求：

\[
\cos(z_G,z_a)>
\cos(z_G,z_b)
\]

定义：

\[
\boxed{
L_{\text{rank}}
=
\max
\left[
0,\,
m-(\cos_a-\cos_b)
\right]
}
\]

其中 \(m\) 为 margin。

推荐初始：

\[
m=0.05
\]

或：

\[
m=0.1
\]

Similarity Loss 负责连续绝对尺度；

Ranking Loss 负责局部顺序。

---

# 14. Loss 3：结构统计自监督

为了保证 topology embedding 保留基础 graph statistics，引入训练期辅助预测头：

\[
g_{\text{stat}}(z_G)
\]

预测：

\[
q_G=
[
|V|,
|E|,
n_1,
n_2
]
\]

其中：

\[
n_1=
|\{v:dist(v,c)=1\}|
\]

\[
n_2=
|\{v:dist(v,c)=2\}|
\]

训练：

\[
\boxed{
L_{\text{stat}}
=
Huber
(
g_{\text{stat}}(z_G),
q_G
)
}
\]

建议先对这些统计量做归一化，例如：

\[
\tilde{|V|}
=
\frac{|V|}{V_{\max}}
\]

避免不同量纲导致 loss 不平衡。

注意：

> 这些统计量只作为训练 target，不作为 GNN 输入。

训练完成后：

\[
g_{\text{stat}}
\]

直接删除。

---

# 15. Stage 1 总损失

最终：

\[
\boxed{
L_{\text{Stage1}}
=
L_{\text{sim}}
+
\lambda_rL_{\text{rank}}
+
\lambda_sL_{\text{stat}}
}
\]

第一轮建议：

\[
\lambda_r=1
\]

\[
\lambda_s=1
\]

后续再做：

- no ranking；
- no structure stat；
- similarity only；

等消融。

---

# 16. Stage 1 训练伪代码

```python
for step in range(num_steps):

    # --------------------------------------------------
    # 1. Generate synthetic rooted graph
    # --------------------------------------------------
    G1, root = sample_synthetic_rooted_graph()

    # --------------------------------------------------
    # 2. Generate isomorphic positive
    # --------------------------------------------------
    G_iso, root_iso = permute_graph(G1, root)

    # --------------------------------------------------
    # 3. Sample continuous target perturbation strength
    # --------------------------------------------------
    d_target_a = uniform(0.0, 1.0)
    d_target_b = uniform(0.0, 1.0)

    G_a = perturb_until_distance(G1, d_target_a)
    G_b = perturb_until_distance(G1, d_target_b)

    # --------------------------------------------------
    # 4. Compute actual normalized edit distance
    # --------------------------------------------------
    d_a = edge_jaccard_distance(G1, G_a)
    d_b = edge_jaccard_distance(G1, G_b)

    if rooted_isomorphic(G1, G_a):
        d_a = 0.0

    if rooted_isomorphic(G1, G_b):
        d_b = 0.0

    # --------------------------------------------------
    # 5. Encoder
    # --------------------------------------------------
    z1    = encoder(G1, root)
    z_iso = encoder(G_iso, root_iso)
    z_a   = encoder(G_a, root)
    z_b   = encoder(G_b, root)

    # --------------------------------------------------
    # 6. Similarity targets
    # --------------------------------------------------
    s_iso = 1.0
    s_a   = 1.0 - d_a
    s_b   = 1.0 - d_b

    cos_iso = cosine(z1, z_iso)
    cos_a   = cosine(z1, z_a)
    cos_b   = cosine(z1, z_b)

    # --------------------------------------------------
    # 7. Similarity loss
    # --------------------------------------------------
    L_sim = (
        huber(cos_iso, s_iso)
        + huber(cos_a, s_a)
        + huber(cos_b, s_b)
    )

    # --------------------------------------------------
    # 8. Ranking loss
    # --------------------------------------------------
    if d_a < d_b:
        L_rank = relu(margin - (cos_a - cos_b))
    else:
        L_rank = relu(margin - (cos_b - cos_a))

    # --------------------------------------------------
    # 9. Structure statistics prediction
    # --------------------------------------------------
    q1 = normalized_graph_stats(G1, root)
    q_hat = stat_head(z1)

    L_stat = huber(q_hat, q1)

    # --------------------------------------------------
    # 10. Total loss
    # --------------------------------------------------
    loss = L_sim + lambda_rank * L_rank + lambda_stat * L_stat

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

---

# 17. Stage 1 推荐默认配置

| 参数 | 推荐初值 |
|---|---:|
| Radius | 2-hop |
| Hidden dim \(D\) | 128 |
| GNN | 2-layer GIN |
| Aggregation | SUM |
| GIN MLP | 128 → 256 → 128 |
| Activation | GELU |
| Node feature | All-one + Root Indicator |
| Perturbation | Edge add / delete |
| Distance | Edge Jaccard Distance |
| Target similarity | \(s=1-d\) |
| Similarity loss | Huber |
| Ranking margin | 0.05–0.10 |
| \(\lambda_r\) | 1.0 |
| \(\lambda_s\) | 1.0 |
| Optimizer | AdamW |
| Initial LR | \(10^{-3}\) |
| Weight decay | \(10^{-5}\sim10^{-4}\) |
| Embedding normalize | L2 before cosine |
| d sampling | approximately Uniform(0,1) |

以上仅作为第一轮默认配置，最终需通过 validation 确定。

---

# 18. Stage 1 验证

训练完成后不要只看 loss。

至少测试以下指标。

---

## 18.1 Isomorphism Consistency

对大量：

\[
G,\pi(G)
\]

计算：

\[
\cos(f(G),f(\pi(G)))
\]

希望：

\[
\rightarrow1
\]

记录：

- mean；
- std；
- minimum；
- P5 / P50 / P95。

---

## 18.2 Similarity Alignment

对 validation graph pairs 计算：

\[
d(G_i,G_j)
\]

以及：

\[
1-\cos(z_i,z_j)
\]

报告：

- Pearson correlation；
- Spearman correlation；
- MAE；
- Huber loss。

Spearman 特别重要，因为它反映 topology ranking 是否保持。

---

## 18.3 Ranking Accuracy

构造：

\[
d(G,G_a)<d(G,G_b)
\]

统计：

\[
\cos(G,G_a)>\cos(G,G_b)
\]

的比例。

---

## 18.4 Structure Recoverability

冻结 encoder。

使用：

\[
z_G
\]

预测：

\[
|V|,\ |E|,\ n_1,\ n_2
\]

观察：

- MAE；
- \(R^2\)；
- correlation。

如果这些统计量无法从 embedding 恢复，说明 encoder 丢失了重要结构信息。

---

## 18.5 Embedding Collision

选明显不同的 topology pair：

- star vs cycle；
- tree vs dense graph；
- sparse vs community graph；
- bridge vs clique-like graph。

检查是否大量出现：

\[
\cos(z_i,z_j)\approx1
\]

若 collision 严重，需要增强 encoder 或修改 topology supervision。

---

## 18.6 Dimension Ablation

测试：

\[
D\in\{32,64,128\}
\]

如果：

\[
D=64\approx128
\]

则优先选择 64。

---

# 19. OOD Topology 测试

为了验证模型是否真正是 graph pretraining model，而不是记住 graph generator：

训练阶段故意留出某一类 topology，例如：

```text
Train:
ER + Tree + BA + Community

Test:
Cycle / Barbell
```

或者反过来。

检查：

- similarity alignment；
- ranking；
- topology retrieval；
- graph statistics recovery。

---

# 20. Real Graph Transfer

训练完成后：

\[
\boxed{
f_\theta\text{ Frozen}
}
\]

对真实图中的每个目标节点 \(c\)：

1. 提取 2-hop ego graph；
2. 所有节点使用 all-one + root indicator；
3. 输入 encoder；
4. 得到：

\[
z_c^{topo}
\]

然后用简单 downstream head：

\[
z_c^{topo}
\rightarrow
LR/MLP
\]

测试真实任务。

推荐优先使用：

- Linear Probe；
- Logistic Regression；
- 小 MLP。

不要第一步就使用复杂 downstream 模型，否则无法判断预训练表示本身质量。

---

# 21. Stage 1 最终产物

训练完成以后删除：

- perturbation generator；
- distance calculator；
- isomorphism pair generator；
- structure-stat prediction head。

只保存：

\[
\boxed{
f_\theta
}
\]

最终接口：

```python
z = topology_encoder(graph, root)
```

输出：

\[
z\in\mathbb R^D
\]

这就是最终的：

> **Pretrained Topology Encoder**

---

# 22. Stage 2：可选 LLM Projector

Stage 1 验证通过以后，再进入 LLM 对齐。

此阶段：

\[
f_\theta
\]

冻结。

指定一个目标 LLM，并冻结 LLM。

只训练：

\[
P_\phi
\]

---

## 22.1 Projector

输入：

\[
z_G\in\mathbb R^D
\]

映射到：

\[
T_G
=
P_\phi(z_G)
\in
\mathbb R^{K\times d_{\text{LLM}}}
\]

其中：

- \(K\)：GraphToken 数量；
- 第一版建议 \(K=4\) 或 \(8\)。

推荐：

```text
Topology embedding
        │
        ↓
      Linear
        ↓
       GELU
        ↓
      Linear
        ↓
reshape → K × d_LLM
```

---

# 23. Projector 自监督任务

全部可以由 synthetic graph 自动生成，无需人工标签。

---

## 23.1 Topology Description

根据 adjacency 自动计算：

- \(|V|\)
- \(|E|\)
- root degree
- \(n_1\)
- \(n_2\)
- density
- cycle existence
- triangle count 等

自动生成 topology description。

输入：

```text
<GraphTokens>

Describe the topology of the rooted graph.
```

目标输出对应描述。

只更新 Projector。

---

## 23.2 Topology QA

自动生成问题，例如：

```text
How many one-hop neighbors does the root have?
```

```text
How many nodes are located at distance two from the root?
```

```text
Does the graph contain a cycle?
```

答案全部由程序从图中计算。

---

## 23.3 Topology Comparison

利用 Stage 1 的连续距离：

\[
d(G,G_a)<d(G,G_b)
\]

构造：

```text
Which graph is structurally more similar to Graph A,
Graph B or Graph C?
```

目标答案自动得到。

该任务用于让 LLM 理解 GraphTokens 之间的 topology similarity。

---

# 24. Stage 2 Loss

可使用 frozen LLM 的 token-level cross entropy：

\[
L_{\text{desc}}
\]

\[
L_{\text{QA}}
\]

\[
L_{\text{compare}}
\]

最终：

\[
\boxed{
L_{\text{Stage2}}
=
L_{\text{desc}}
+
L_{\text{QA}}
+
L_{\text{compare}}
}
\]

训练：

```text
Topology Encoder: Frozen
Projector:         Trainable
LLM:               Frozen
```

---

# 25. 最终使用模式

## 模式 A：独立图预训练模型

```text
Graph
  ↓
Pretrained Topology Encoder
  ↓
Topology Embedding
  ↓
Downstream Head
```

适用于：

- node classification；
- graph retrieval；
- anomaly detection；
- fraud detection；
- graph clustering；
- similarity search。

---

## 模式 B：Graph + LLM

```text
Graph
  ↓
Frozen Topology Encoder
  ↓
Topology Embedding
  ↓
Projector
  ↓
GraphTokens
  ↓
Frozen LLM
  ↓
ICL / Few-shot Inference
```

---

# 26. 推荐开发顺序

建议严格按照下面顺序实现。

### Phase 1：Graph Generator

完成：

- 多种 synthetic graph family；
- rooted 2-hop extraction；
- graph serialization；
- reproducible random seed。

---

### Phase 2：Perturbation Engine

完成：

- edge add；
- edge delete；
- changed-edge tracking；
- normalized edit distance；
- continuous \(d\) sampling；
- rooted isomorphism check。

---

### Phase 3：Topology Encoder

完成：

- all-one input；
- root indicator；
- 2-layer SUM-GIN；
- center embedding。

---

### Phase 4：Stage 1 Loss

完成：

- Similarity Huber Loss；
- Ranking Loss；
- Structure Statistics Loss。

---

### Phase 5：Synthetic Validation

完成：

- isomorphism consistency；
- Pearson / Spearman；
- ranking accuracy；
- embedding collision；
- structure recoverability；
- dimension ablation。

---

### Phase 6：OOD Topology Test

使用 unseen graph family 验证迁移能力。

---

### Phase 7：Real Graph Transfer

冻结 topology encoder，在真实图上进行 linear probe。

---

### Phase 8：LLM Projector

只有 Stage 1 验证通过以后再开发：

- multi-token projector；
- topology description；
- topology QA；
- topology comparison；
- frozen LLM alignment。

---

# 27. 必做消融实验

建议至少包含：

| Ablation | 目的 |
|---|---|
| All-one vs Root Indicator | 判断 rooted information 是否必要 |
| Linear GNN vs GIN+MLP | 判断非线性结构编码能力 |
| \(D=32/64/128\) | 判断 embedding 容量 |
| Similarity only | 基线 |
| + Ranking | 判断排序监督价值 |
| + Structure Stat | 判断结构信息保留能力 |
| Uniform d vs Natural perturbation | 判断连续距离覆盖的重要性 |
| Center only vs Center+Pooling | 判断全局统计信息是否必要 |

---

# 28. 实现检查清单

## 数据生成

- [ ] 支持多种 synthetic graph family
- [ ] 每张图具有 root
- [ ] 半径限制为 2-hop
- [ ] 节点数和边数具有足够变化范围
- [ ] 图生成可复现

## Perturbation

- [ ] 支持 edge add
- [ ] 支持 edge delete
- [ ] 不允许 self-loop（除非实验明确需要）
- [ ] 不允许重复 edge
- [ ] 维护最终 changed-edge set
- [ ] 计算 edge Jaccard distance
- [ ] 支持 \(d_{\text{target}}\) 采样
- [ ] 支持 rooted isomorphism check

## Encoder

- [ ] All-one input
- [ ] Root indicator
- [ ] SUM aggregation
- [ ] 2-layer GIN
- [ ] 每层独立 MLP
- [ ] 输出 center embedding
- [ ] L2 normalize before cosine

## Loss

- [ ] Similarity Huber
- [ ] Ranking loss
- [ ] Structure-stat prediction
- [ ] loss 权重可配置

## Validation

- [ ] Isomorphic cosine
- [ ] Pearson
- [ ] Spearman
- [ ] Ranking accuracy
- [ ] Statistic recoverability
- [ ] Collision analysis
- [ ] Dimension ablation
- [ ] OOD topology test
- [ ] Real graph linear probe

## LLM Alignment

- [ ] Freeze topology encoder
- [ ] Freeze LLM
- [ ] Train projector only
- [ ] Topology Description
- [ ] Topology QA
- [ ] Topology Comparison
- [ ] GraphToken 数量消融

---

# 29. 最终核心公式

Topology encoder：

\[
z_G=f_\theta(G,c)
\]

连续 edit distance：

\[
\boxed{
d(G_1,G_2)
=
\frac{|E_1\triangle E_2|}
{|E_1\cup E_2|}
}
\]

target similarity：

\[
\boxed{
s=1-d
}
\]

embedding similarity：

\[
\boxed{
\hat s=\cos(z_1,z_2)
}
\]

Stage 1：

\[
\boxed{
L_{\text{Stage1}}
=
Huber(\hat s-s)
+
\lambda_rL_{\text{rank}}
+
\lambda_sL_{\text{stat}}
}
\]

最终产物：

\[
\boxed{
G,c
\xrightarrow{f_\theta}
z_{\text{topo}}
}
\]

可选 LLM：

\[
\boxed{
G
\rightarrow
f_\theta
\rightarrow
z_{\text{topo}}
\rightarrow
P_\phi
\rightarrow
GraphTokens
\rightarrow
Frozen\ LLM
}
\]

---

# 30. 一句话总结

> **利用 synthetic rooted graphs 和连续结构扰动构造自监督 topology metric，训练 SUM-GIN 将局部图压缩为保持结构相似性的固定维度 embedding；该 encoder 本身作为 dataset-independent topology pretrained graph model 使用，并可进一步通过自监督 Projector 转换为 Frozen LLM 可理解的 GraphTokens。**
