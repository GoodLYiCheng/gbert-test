# Stage 2：Topology-to-LLM GraphToken Alignment 实现框架

> 目标：冻结 Stage 1 Topology Encoder 与目标 LLM，仅训练 Projector，将 topology embedding 转换为 LLM 可以功能性利用的连续 GraphTokens。Stage 2 训练数据继续全部来自 synthetic graph，与具体数据集无关。

## 1. 总体目标

Stage 1 输出：

\[
z_G=f_\theta(G,c)\in\mathbb R^D
\]

当前默认：

\[
D=128
\]

Stage 2 训练：

\[
P_\phi:\mathbb R^D\rightarrow\mathbb R^{K\times d_{LLM}}
\]

得到：

\[
T_G=[t_1,\ldots,t_K]
\]

完整路径：

\[
\boxed{
G
\xrightarrow{Frozen\ Encoder}
z_G
\xrightarrow{Trainable\ Projector}
T_G
\xrightarrow{Frozen\ LLM}
Output
}
\]

训练策略：

- Topology Encoder：Frozen
- Projector：Trainable
- LLM：Frozen
- 数据：100% synthetic graph
- 监督：由图结构自动生成

---

## 2. Stage 2 的核心任务

Stage 2 不是简单做：

\[
128\rightarrow d_{LLM}
\]

的维度转换，而是学习：

\[
\boxed{
\text{Topology Embedding Space}
\rightarrow
\text{LLM-usable Continuous Prompt Space}
}
\]

成功标准不是 Projector 输出维度正确，而是：

> Frozen LLM 能够仅依赖 GraphTokens 正确回答结构相关问题。

---

## 3. 为什么主方案只训练 Projector

推荐：

```text
Topology Encoder   Frozen
Projector          Trainable
LLM                Frozen
```

原因：

1. 保留 Stage 1 已经学习到的 topology metric space；
2. 防止 Encoder 为语言任务重新组织 embedding；
3. 防止 LLM 自己适应陌生 GraphToken；
4. 可以明确验证 Projector 是否完成 topology-to-LLM alignment。

全模型微调只作为后续 ablation。

---

## 4. 数据生成

继续使用 Stage 1 synthetic graph generator，例如：

- ER random graph
- Tree
- Star
- Cycle
- BA / scale-free
- Community
- Sparse / dense graph
- Irregular graph
- rooted 2-hop graph

不使用：

- YelpZip / Amazon 等真实数据集
- fraud label
- review text
- 用户/商家属性
- 时间信息

---

## 5. 加载并冻结 Stage 1 Encoder

```python
encoder.eval()

for p in encoder.parameters():
    p.requires_grad = False
```

对每张 synthetic graph：

\[
z_G=f_\theta(G,c)
\]

可选择：

- 在线计算 topology embedding；
- 或预计算并缓存 topology embedding。

如果 Stage 2 synthetic 图集合固定，推荐缓存以提高效率。

---

## 6. Projector

推荐第一版：

```text
128
→ Linear
→ 512
→ GELU
→ Linear
→ K × d_LLM
→ reshape
```

数学形式：

\[
h=GELU(W_1z_G+b_1)
\]

\[
u=W_2h+b_2
\]

最终：

\[
T_G\in\mathbb R^{K\times d_{LLM}}
\]

推荐：

\[
K=4
\]

后续消融：

\[
K\in\{1,4,8\}
\]

示例：

```python
class GraphProjector(nn.Module):
    def __init__(self, graph_dim, hidden_dim, llm_dim, num_tokens):
        super().__init__()
        self.num_tokens = num_tokens
        self.llm_dim = llm_dim

        self.net = nn.Sequential(
            nn.Linear(graph_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_tokens * llm_dim),
        )

    def forward(self, z):
        x = self.net(z)
        return x.view(z.size(0), self.num_tokens, self.llm_dim)
```

---

## 7. GraphToken 注入 LLM

GraphTokens 不经过 tokenizer，而是直接插入 LLM 的 `inputs_embeds`。

逻辑 prompt：

```text
The following vectors represent the topology of a rooted graph:

<GRAPH>

Question:
How many direct neighbors does the root have?
```

实际 embedding 序列：

\[
[
E(prefix),
T_G,
E(question)
]
\]

即：

\[
[e_1,\ldots,e_m,t_1,\ldots,t_K,e_{m+1},\ldots,e_n]
\]

关键要求：

> GraphTokens 必须成为问题所需 topology information 的唯一来源。

---

## 8. 节点编号

Stage 2 可以为 synthetic graph 临时生成节点名称：

```text
Root
Node A
Node B
Node C
...
```

但编号只用于自然语言问题中的指代，不得输入 Stage 1 Encoder。

建议每次随机重命名，例如同一 topology 可使用：

```text
Root, A, B, C
```

或：

```text
Root, F, H, D
```

避免固定名字形成 shortcut。

---

## 9. Task A：Topology Attribute QA

这是最基础的训练任务。

自动计算：

\[
|V|,|E|,deg(c),n_1,n_2
\]

以及：

- 是否存在 cycle
- 是否存在 triangle
- density
- 是否为 tree
- 其他 root-centered structural facts

示例问题：

```text
How many nodes are in this graph?
```

```text
How many edges are in this graph?
```

```text
How many direct neighbors does the root have?
```

```text
How many nodes are exactly two hops away from the root?
```

```text
Does the graph contain a cycle?
```

答案全部由程序自动计算。

---

## 10. 每次一个问题：推荐动态采样

每个 training sample 对应一个问题是可行的。

关键要求是：

> 同一个 graph 在整个训练过程中要随机遇到不同问题。

定义：

\[
Q(G)=\{Q_1,Q_2,\ldots,Q_m\}
\]

每次随机采：

\[
Q_i\sim Q(G)
\]

例如：

```python
QUESTION_TYPES = [
    "num_nodes",
    "num_edges",
    "root_degree",
    "num_hop1",
    "num_hop2",
    "has_cycle",
    "has_triangle",
    "density",
]
```

训练时：

```python
qtype = random.choice(QUESTION_TYPES)
question, answer = build_question(G, root, qtype)
```

建议 question type 尽可能均衡采样。

---

## 11. Prompt 模板随机化

同一个任务准备多个等价模板。

例如 root degree：

```text
How many direct neighbors does the root have?
```

```text
What is the degree of the root node?
```

```text
How many nodes are directly connected to Root?
```

这样减少语言模板 shortcut。

---

## 12. Task B：Topology Description

输入 GraphTokens，要求 Frozen LLM 输出整体结构描述。

例如：

```text
The graph contains 18 nodes and 24 edges.
The root has 4 direct neighbors.
There are 4 nodes at hop one and 13 nodes at hop two.
The graph contains at least one cycle.
```

作用：

> 迫使同一个 GraphToken representation 同时支持多个 topology facts，而不是只映射某一个统计量。

Description 模板也应随机化。

---

## 13. Task C：Topology Comparison

复用 Stage 1 的 topology distance。

给定：

\[
G_A,G_B,G_C
\]

若：

\[
d(G_A,G_B)<d(G_A,G_C)
\]

构造：

```text
Graph A:
<GRAPH_A>

Graph B:
<GRAPH_B>

Graph C:
<GRAPH_C>

Which graph is structurally more similar to Graph A,
Graph B or Graph C?
```

目标：

```text
Graph B
```

用于训练：

\[
\boxed{
GraphToken\rightarrow Topology\ Similarity\ Reasoning
}
\]

建议要求：

\[
|d_{AB}-d_{AC}|>\delta
\]

第一版：

\[
\delta=0.1
\]

避免构造过于模糊的 comparison sample。

---

## 14. 节点级 QA

可以作为辅助任务，例如：

```text
Is node A one hop or two hops away from Root?
```

```text
Do Root, A and B form a triangle?
```

但不建议作为主任务。

因为 Stage 1 当前输出的是压缩后的 rooted graph embedding：

\[
z_G=h_c
\]

它未必完整保留任意节点对的精确 adjacency。

因此 Stage 2 主目标仍应是：

\[
\boxed{
\text{graph-level / root-centered topology understanding}
}
\]

---

## 15. Stage 2 Loss

Frozen LLM 使用标准 causal language modeling loss。

三类任务：

\[
L_{QA},\quad L_{Desc},\quad L_{Compare}
\]

总损失：

\[
\boxed{
L_{Stage2}
=
\lambda_Q L_{QA}
+
\lambda_D L_{Desc}
+
\lambda_C L_{Compare}
}
\]

第一版：

\[
\lambda_Q=\lambda_D=\lambda_C=1
\]

---

## 16. 推荐任务比例

第一版：

| Task | 比例 |
|---|---:|
| Topology QA | 40% |
| Topology Description | 30% |
| Topology Comparison | 30% |

后续根据 validation 调整。

---

## 17. 梯度路径

Forward：

\[
G
\rightarrow
f_\theta
\rightarrow
z_G
\rightarrow
P_\phi
\rightarrow
T_G
\rightarrow
LLM
\rightarrow
Loss
\]

Backward：

\[
Loss
\rightarrow
LLM
\rightarrow
T_G
\rightarrow
P_\phi
\]

但：

\[
\nabla_\theta=0
\]

\[
\nabla_{LLM}=0
\]

仅：

\[
\boxed{
\nabla_\phi\neq0
}
\]

---

## 18. 冻结策略

```python
encoder.eval()
llm.eval()

for p in encoder.parameters():
    p.requires_grad = False

for p in llm.parameters():
    p.requires_grad = False

for p in projector.parameters():
    p.requires_grad = True
```

Optimizer 只注册 Projector：

```python
optimizer = AdamW(
    projector.parameters(),
    lr=projector_lr,
)
```

---

## 19. Stage 2 训练伪代码

```python
for step in range(num_steps):

    task_type = sample_task(
        qa_prob=0.4,
        desc_prob=0.3,
        compare_prob=0.3,
    )

    if task_type == "qa":

        G, root = sample_graph()
        question, answer = sample_dynamic_qa(G, root)

        with torch.no_grad():
            z = encoder(G, root)

        graph_tokens = projector(z)

        inputs_embeds, labels = build_llm_input(
            graph_tokens,
            question,
            answer,
        )

    elif task_type == "description":

        G, root = sample_graph()
        description = build_topology_description(G, root)

        with torch.no_grad():
            z = encoder(G, root)

        graph_tokens = projector(z)

        inputs_embeds, labels = build_description_input(
            graph_tokens,
            description,
        )

    elif task_type == "comparison":

        G_a, G_b, G_c = sample_comparison_triplet()

        with torch.no_grad():
            z_a = encoder(G_a)
            z_b = encoder(G_b)
            z_c = encoder(G_c)

        t_a = projector(z_a)
        t_b = projector(z_b)
        t_c = projector(z_c)

        question, answer = build_comparison_question(
            G_a, G_b, G_c
        )

        inputs_embeds, labels = build_comparison_input(
            t_a, t_b, t_c,
            question, answer
        )

    outputs = llm(
        inputs_embeds=inputs_embeds,
        labels=labels,
    )

    loss = outputs.loss

    optimizer.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        projector.parameters(),
        1.0,
    )

    optimizer.step()
```

---

## 20. 必须避免的 shortcut

Prompt 中不能同时出现：

- adjacency list
- edge list
- node count
- edge count
- root degree
- hop count
- topology statistics
- 答案本身

例如不能输入：

```text
Root-A
Root-B
Root-C
```

然后问：

```text
How many neighbors does Root have?
```

否则 LLM 可以直接从文本读取答案。

Stage 2 必须保证：

\[
\boxed{
GraphTokens\ 是唯一 topology\ information\ source
}
\]

---

## 21. Stage 2 核心验证

训练 loss 不是最终证明。

必须验证 Frozen LLM 是否真正使用 GraphTokens。

### 21.1 Correct GraphToken

正常：

\[
T_G=P(z_G)
\]

记录正常性能：

\[
Acc_{normal}
\]

### 21.2 Zero GraphToken

替换：

\[
T_G\rightarrow0
\]

得到：

\[
Acc_{zero}
\]

要求：

\[
Acc_{normal}\gg Acc_{zero}
\]

### 21.3 Random GraphToken

替换：

\[
T_G\rightarrow T_{random}
\]

要求：

\[
Acc_{normal}\gg Acc_{random}
\]

### 21.4 Shuffle GraphToken

将 Graph A 的 token 换成 Graph B：

\[
T_A\rightarrow T_B
\]

要求：

\[
\boxed{
Acc_{normal}\gg Acc_{shuffle}
}
\]

这是验证：

\[
GraphToken\leftrightarrow Topology
\]

真实绑定关系的关键实验。

---

## 22. GraphToken 数量消融

测试：

\[
K\in\{1,4,8\}
\]

比较：

- QA accuracy
- Description attribute accuracy
- Comparison accuracy
- 参数量
- 训练稳定性

第一版推荐：

\[
\boxed{K=4}
\]

---

## 23. Projector Capacity Ablation

比较：

### Linear Projector

\[
128\rightarrow Kd_{LLM}
\]

### 2-layer MLP

\[
128\rightarrow512\rightarrow Kd_{LLM}
\]

如果 Linear 已足够，优先使用 Linear。

若 MLP 明显更强，则保留小型 MLP。

---

## 24. Stage 2 OOD 验证

训练时保留 unseen graph family，例如：

```text
Train:
ER + Tree + BA

Test:
Cycle + Community
```

也可以测试：

- unseen node-count range
- unseen edge-count range
- unseen density range

检查 Frozen LLM 是否仍能正确读取 GraphToken。

---

## 25. 推荐指标

### QA

- Exact Match Accuracy
- Numeric Accuracy
- Yes/No Accuracy

### Description

程序化抽取：

- \(|V|\)
- \(|E|\)
- root degree
- \(n_1\)
- \(n_2\)
- cycle

然后计算 attribute accuracy。

不建议只使用 BLEU / ROUGE。

### Comparison

- Pairwise / triplet comparison accuracy

### GraphToken Dependency

必须报告：

\[
Acc_{normal}
\]

\[
Acc_{zero}
\]

\[
Acc_{random}
\]

\[
Acc_{shuffle}
\]

---

## 26. 推荐默认配置

| 参数 | 第一版建议 |
|---|---|
| Topology Encoder | Stage 1 pretrained |
| Encoder status | Frozen |
| Topology dim | 128 |
| LLM | 单一目标 LLM |
| LLM status | Frozen |
| Projector | 2-layer MLP |
| Hidden dim | 512 |
| Activation | GELU |
| GraphToken 数 | 4 |
| GraphToken dim | \(d_{LLM}\) |
| QA ratio | 40% |
| Description ratio | 30% |
| Comparison ratio | 30% |
| Question sampling | Dynamic |
| Node names | 临时随机命名 |
| Loss | Token-level CE |
| Optimizer | AdamW |
| Projector LR | \(10^{-4}\sim10^{-3}\) 起始测试 |
| Gradient clipping | 1.0 |
| Encoder grad | Off |
| LLM grad | Off |

---

## 27. 推荐开发顺序

### Phase 1：加载 Stage 1 Encoder

- [ ] 加载 checkpoint
- [ ] Freeze Encoder
- [ ] 验证输出与 Stage 1 一致

### Phase 2：加载 Frozen LLM

- [ ] 加载目标 LLM
- [ ] Freeze 全部参数
- [ ] 获取 \(d_{LLM}\)
- [ ] 验证 `inputs_embeds` forward

### Phase 3：实现 Projector

- [ ] Linear baseline
- [ ] 2-layer MLP
- [ ] 输出 \(K\times d_{LLM}\)
- [ ] GraphToken embedding 注入

### Phase 4：先实现基础 QA

- [ ] num_nodes
- [ ] num_edges
- [ ] root_degree
- [ ] num_hop1
- [ ] num_hop2

先确认：

\[
Loss
\rightarrow
Frozen\ LLM
\rightarrow
Projector
\]

梯度链工作正常。

### Phase 5：扩展 QA

- [ ] cycle
- [ ] triangle
- [ ] density
- [ ] 更多 root-centered structure facts

### Phase 6：Topology Description

- [ ] 自动 description generator
- [ ] 多模板随机化
- [ ] attribute-level evaluation

### Phase 7：Topology Comparison

- [ ] triplet generator
- [ ] 复用 Stage 1 distance
- [ ] distance gap control
- [ ] comparison accuracy

### Phase 8：GraphToken Ablation

- [ ] Zero
- [ ] Random
- [ ] Shuffle
- [ ] K=1/4/8
- [ ] Linear vs MLP
- [ ] single-task vs multi-task
- [ ] fixed question vs dynamic question

### Phase 9：OOD

- [ ] unseen graph family
- [ ] unseen graph size
- [ ] unseen density

---

## 28. 可选 Stage 2B：联合微调

只有满足以下条件才考虑：

1. Stage 1 embedding 已证明包含足够 topology information；
2. Projector-only 长期无法对齐；
3. 已排除 GraphToken 注入、prompt、训练实现问题。

可尝试：

```text
Encoder    Trainable with very small LR
Projector  Trainable
LLM        Frozen
```

同时保留 Stage 1 topology loss：

\[
L=
L_{LLM}
+
\lambda_{topo}L_{Stage1}
\]

防止 topology embedding space 漂移。

---

## 29. 不建议第一版训练整个 LLM

若：

```text
Encoder    Trainable
Projector  Trainable
LLM        Trainable
```

那么 LLM 可能直接适应陌生 GraphToken。

此时难以证明：

> GraphToken 已经被映射到了 LLM 原本可利用的空间。

因此：

\[
\boxed{
Projector-only
}
\]

应作为主实验。

---

## 30. Stage 2 最终公式

Stage 1：

\[
z_G=f_\theta(G,c)
\]

其中：

\[
\theta
\]

冻结。

Projector：

\[
\boxed{
T_G=P_\phi(z_G)
}
\]

LLM：

\[
p(y\mid T_G,q)
\]

训练：

\[
\boxed{
\phi^*
=
\arg\min_\phi L_{Stage2}
}
\]

其中：

\[
\boxed{
L_{Stage2}
=
\lambda_Q L_{QA}
+
\lambda_D L_{Desc}
+
\lambda_C L_{Compare}
}
\]

并保持：

\[
\boxed{
\nabla_\theta=0
}
\]

\[
\boxed{
\nabla_{LLM}=0
}
\]

---

## 31. Stage 2 成功标准

第二阶段至少需要满足：

- [ ] 正常 GraphToken 下 topology QA 明显高于随机基线
- [ ] Zero GraphToken 后性能明显下降
- [ ] Random GraphToken 后性能明显下降
- [ ] Shuffle GraphToken 后性能明显下降
- [ ] Comparison task 能保持 Stage 1 topology ranking
- [ ] OOD synthetic topology 上仍有较好的迁移能力
- [ ] Projector-only 可以工作，无需修改 Encoder 和 LLM

---

## 32. 一句话总结

> **Stage 2 通过 synthetic topology 自动生成 QA、Description 和 Comparison 监督，在冻结 Stage 1 Topology Encoder 与冻结 LLM 的前提下，仅训练一个 Projector，将 topology embedding 映射为 Frozen LLM 可以按问题动态读取和利用的连续 GraphTokens。**
