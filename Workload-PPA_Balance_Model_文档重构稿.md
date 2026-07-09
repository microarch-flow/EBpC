# 面向大模型推理演进的 Workload-PPA Balance Model
## 一套用于指导可重构 AI 芯片设计的资源配比方法论

### 1. 背景：AI 芯片竞争正在从峰值算力转向有效 PPA

过去 AI 芯片通常用 TOPS、FLOPS、HBM 带宽、显存容量等峰值指标进行比较。但在大模型推理阶段，尤其是长上下文、MoE、KV Cache、稀疏注意力、投机解码和 agentic workflow 出现后，单一峰值指标已经无法判断芯片是否真正高效。

同一颗芯片，在 dense GEMM 上可能算力利用率很高，在长上下文 decode 上可能受限于带宽，在 MoE 或稀疏 attention 上又可能受限于片上调度、数据搬运和非规则算子映射。

因此，未来推理芯片的核心问题不是“峰值算力有多高”，而是：

> 在目标 workload 下，芯片的算力、带宽、片上存储、调度和数据流能力是否匹配；在达到匹配时，所需面积、功耗和成本是否最优。

我们将这一问题抽象为 **Workload-PPA Balance Model**。

---

### 2. 核心观点：Workload 决定资源需求，架构决定有效供给

我们把大模型推理芯片设计拆成三层：

1. **Workload Demand**
   描述模型在某个上下文长度、batch、精度和注意力机制下，对计算、带宽、状态容量和控制流的需求。

2. **Hardware Supply**
   描述芯片能提供多少峰值算力、峰值带宽、片上存储、片间通信和调度能力。

3. **Mapping Efficiency**
   描述该 workload 在该硬件上的实际利用率。峰值资源只有被有效映射，才转化为有效 PPA。

因此，真正应比较的不是：

```text
Peak TOPS / Peak Bandwidth
```

而是：

```text
Effective Compute / Effective Bandwidth / Effective State Capacity / Effective Mapping Efficiency
```

---

### 3. 核心模型：Demand-Supply-Gap

对任意 workload `w` 和硬件架构 `h`，定义：

```text
Demand(w) = workload 对 compute / bandwidth / state / control 的需求
Supply(h,w) = 硬件在该 workload 下的有效供给
Gap(w,h) = Demand(w) / Supply(h,w)
```

其中：

```text
Supply_compute = P_peak * u_c(w,h)
Supply_bandwidth = BW_peak * u_b(w,h)
Supply_state = S_onchip * u_s(w,h)
Supply_control = C_sched * u_m(w,h)
```

`u_c, u_b, u_s, u_m` 分别表示算力、带宽、片上存储和映射/调度的有效利用率。

判断方式：

```text
Gap > 1：该资源成为约束
Gap ≈ 1：资源配比较平衡
Gap < 1：该资源存在冗余
```

这个模型的目标不是预测每个 kernel 的精确运行时间，而是回答芯片设计早期最关键的问题：

> 面向未来 workload，PPA budget 应该分配给算力、带宽、片上存储、NoC，还是可重构数据流能力？

---

### 4. EBpC：带宽-算力配比的核心切片

在完整模型中，最重要的一个切片是带宽与算力的配比，即：

```text
EBpC = Effective Bandwidth per Compute
```

定义为：

```text
EBpC(w,h) = Bytes_per_token(w) / FLOPs_per_token(w) * u_c(w,h) / u_b(w,h)
```

含义是：

> 对某个 workload，芯片每单位有效算力需要配多少有效带宽，才能避免算力被带宽饿死。

若芯片实际带宽/算力配比低于该需求，则 workload 受带宽约束；若高于该需求，则带宽存在冗余，瓶颈可能转向算力、片上存储或映射效率。

这使我们可以直观看到不同芯片在不同 workload 下的约束来源。

---

### 5. LLM 推理 workload 的演进趋势

大模型推理的核心变量之一是上下文长度 `C`。不同注意力机制对上下文的访问复杂度不同，可以抽象为：

```text
KV_Read(C) ∝ C^λ
```

其中：

```text
λ = 1      传统 attention / GQA / MLA decode，仍随上下文线性增长
0 < λ < 1  稀疏、压缩、检索式 attention
λ = 0      固定窗口、linear attention、SSM，长期趋近常数
```

这给出一个重要趋势：

> 随着 λ 从 1 下降到 0，芯片瓶颈会从 HBM 带宽和 KV 容量，逐步迁移到片上存储组织、控制流、非 GEMM 算子和动态数据流映射。

这也是可重构计算的机会所在。

---

### 6. 为什么可重构计算是合理路线

固定架构的风险在于：它通常针对某一类 dominant workload 做 PPA 最优设计。一旦 workload 从 dense GEMM 转向 MoE、稀疏 attention、动态 routing、SSM 或 agentic 多阶段任务，原有计算阵列、存储层级和数据流会出现利用率下降。

可重构计算的核心价值不是“也能做 AI 加速”，而是：

```text
当 workload 变化时，通过数据流、片上存储、算子融合和执行拓扑重构，
保持更高的 u_c、u_b、u_s 和 u_m。
```

也就是说，可重构架构不只提升峰值性能，而是提升 **workload 演进过程中的有效 PPA 稳定性**。

这可以成为我们的核心技术叙事：

> 未来大模型算法仍在快速变化，固定架构押注单一 workload，而可重构架构提供面向不确定 workload 的架构适配能力。

---

### 7. 架构类型对比

| 架构类型 | 优势 | 主要风险 | 在本模型中的典型表现 |
|---|---|---|---|
| GPU | 软件生态强，通用性好 | 面积和功耗开销高，非规则 workload 有效利用率下降 | `u_c/u_b` 随 workload 变化波动大 |
| 固定 ASIC | 单一 workload PPA 极高 | 算法变化后适配能力弱 | 某些点 Gap 很低，换 workload 后 Gap 快速升高 |
| Dataflow 架构 | 数据搬运效率高，适合流式计算 | 编译和映射复杂 | 对部分 workload 可保持较高 `u_b` |
| 可重构计算 | 可随 workload 调整数据流和资源组织 | 需要强编译器和架构经验 | 在 workload 演进中维持更稳定的有效 PPA |

我们的差异化应表达为：

> 不是简单重复海外 dataflow/reconfigurable 路线，而是用 Workload-PPA Balance Model 指导可重构资源如何配置、何时配置、为哪些未来 workload 配置。

---

### 8. 设计闭环

该模型可以形成芯片设计闭环：

```text
目标模型与业务场景
→ workload 需求建模
→ compute / bandwidth / state / control gap 分析
→ PPA budget 分配
→ 可重构阵列、SRAM、NoC、调度单元设计
→ 编译器/仿真器验证有效利用率
→ 反馈下一轮架构设计
```

最终输出不是单一指标，而是一组设计曲线：

```text
不同 context 长度下，每 PFLOPS 需要多少 TB/s
不同 attention λ 下，瓶颈如何迁移
不同 batch / MoE 路由下，片上存储和带宽如何平衡
不同架构下，effective PPA 如何变化
```

---

## 技术附录雏形

### A1. Decode 阶段简化模型

```text
O_tok(C) = O_param + O_attn(C)
Q_tok(C,B) = Q_weight(B) + Q_KV(C) + Q_act
```

其中：

```text
O_param ≈ 2 * N_act
Q_weight ≈ N_weight_read(B) * b_W / B
Q_KV(C) ∝ C^λ * s_KV
```

对 dense 模型，`N_weight_read(B)` 近似为参与参数量。
对 MoE 模型，更严格应使用 batch 内激活专家并集，而不是简单 `N_act / B`。

---

### A2. 有效带宽-算力配比

```text
EBpC(w,h) = Q_tok(w) / O_tok(w) * u_c(w,h) / u_b(w,h)
```

如果硬件实际配比为：

```text
HW_BpC = BW_peak / P_peak
```

则：

```text
EBpC > HW_BpC：带宽不足
EBpC < HW_BpC：带宽相对冗余
EBpC ≈ HW_BpC：带宽-算力配比较平衡
```

---

### A3. PPA 成本模型

```text
Cost = α_P * P_peak + α_B * BW_peak + α_S * SRAM + α_M * Mem + α_N * NoC
```

这里的重点不是得到绝对面积，而是比较不同资源投入的边际收益：

```text
增加带宽是否比增加 SRAM 更划算？
增加算力是否会被带宽浪费？
增加可重构 fabric 是否能提高多个 workload 的综合利用率？
```

---

### A4. 模型边界

本模型用于架构早期设计和 BP 级趋势判断，不直接替代 cycle-level 仿真。
更精确的参数，如 `u_c, u_b, u_s, u_m`，需要通过内部仿真器、编译器映射结果或真实芯片测试获得。

参考锚点：Roofline 模型的基本思想来自对 compute peak、memory bandwidth 和 arithmetic intensity 的可视化约束分析；SambaNova SN40L 论文也强调 dataflow 和多级存储体系用于缓解 AI memory wall。

Sources:
- https://en.wikipedia.org/wiki/Roofline_model
- https://arxiv.org/abs/2405.07518
