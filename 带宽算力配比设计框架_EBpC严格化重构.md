# 大模型推理的带宽—算力配比设计框架
## ——"有效带宽算力比（EBpC）"的严格化重构

> 目标读者：芯片架构与算法团队内部讨论稿
> 性质：对《大模型推理中有效带宽算力比（EBpC）的技术调研报告》的分析、修订与升级方案

---

## 摘要

原报告的目标是正确且有价值的：**在 Roofline 的基础上引入修正参数，使一个指标能同时刻画 workload 需求并指导芯片 PPA（面积、带宽、精度、片上存储）设计。**

本文的核心结论是：这个目标可以严格实现，但实现形式不是"一个标量"，而是"**三层模型 + 一个均衡公式**"。原因在于该目标同时要求三种性质——①给定 workload 谁算都一样（workload 内蕴性）；②包含"有效"即利用率（硬件×映射属性）；③关联 PPA（硬件成本属性）——这三者分属三个不同的定义域，任何单一标量无法同时携带三者而不产生自由参数（原报告中 θ、β、α、γ 四个悬空系数即由此产生）。

本文给出的框架中，原报告的四个修正因子**全部保留其物理直觉，但各归其位**：

| 原报告因子 | 原定位 | 严格化后的归属 |
|---|---|---|
| α（面积等效因子） | 计算量分母的折算系数 | **第 2 层：硬件成本模型的面积单价**（语义完全保留） |
| β_IO（I/O 削减因子） | 拍定的比例系数 | **第 1 层：I/O 下界曲线 Q_min(S) 的可计算读数**，β_IO = Q_min(S)/Q_min(0) |
| θ_mem（内存对齐因子） | 拍定的放大系数 | **第 3 层：带宽利用率 u_b 的组成部分**，由仿真/实测产出 |
| γ（MoE 动态置换因子） | 拍定的比例系数 | **第 3 层：调度策略参数**，由仿真产出 |

最终，"有效带宽算力比"获得严格定义（第 6 节）：

$$\Big(\frac{BW}{P_c}\Big)^{*} = \frac{Q_{min}(S, B^* \mid C)}{O_{tok}(C)} \cdot \frac{u_c}{u_b}$$

它是均衡（成本最优）设计点上的硬件带宽算力配比，等于 workload 需求比乘以利用率修正。它**不是常数，而是（上下文分布、SLA、片上容量、内存容量）的函数**——建议的最终交付物是这个函数的可视化：一族"每 PFLOPS 应配多少 TB/s"的设计曲线，每个点可复算、可审计。

---

## 1. 问题的结构：为什么标准指标"谁算都一样"，而"有效"版本定义不出来

**标准"带宽算力比"的可计算性来自一组行业共享的隐式约定**：权重与 KV 驻留片外（HBM/LPDDR）、每 token 各流式读取一次、token 内片上完美复用（FlashAttention 类算法，tile 不 spill）、batch 按声明取值。在这组约定下，每 token 计算量与搬运量唯一确定，谁算都一样。

但注意：**被这组约定冻结的参数——片上驻留容量 S、batch B、每次搬运服务的 token 数——恰好就是芯片设计要决策的自由度。**三个例证：

- 权重常驻片上的架构（Groq/Cerebras 路线），每 token 权重片外搬运量为 0，不是标准值 N·b_W；
- batch 从 1 到 128，每 token 权重搬运量 N·b_W/B 变化两个数量级；
- MTP/投机解码一次参数读取服务多个 token（原报告 6.2 节自己指出了这一点），直接击穿"每 token 搬一次"。

一个通过冻结所有设计变量才获得唯一性的数，原理上无法对这些变量给出指导——它对所有设计方案输出同一个值。**"有效"二字试图解冻这些参数，又不愿放弃唯一性，只能靠悬空系数缝合——这就是原指标定义困难的结构性根源。**

出路：把这些冻结参数升格为**显式自变量**。workload 属性从"一个数"升级为"一个函数"，标准指标是该函数在约定点上的一次求值。以下三层模型即此思路的展开。

---

## 2. 第 1 层：Workload 需求模型（给定参数，谁算都一样）

### 2.1 符号表

| 符号 | 含义 | 来源 |
|---|---|---|
| L, d, H_Q, H_KV, d_h | 层数、隐藏维（d=H_Q·d_h）、Query/KV 头数、单头维度 | 模型配置 |
| C | 当前上下文长度（tokens） | 运行时变量 |
| N_act | 单 token 前向实际参与矩阵乘的参数量（MoE 取激活参数） | 模型配置 |
| b_W, b_KV | 权重 / KV 每元素字节数（FP16=2, FP8=1, INT4=0.5） | 部署选择 |
| B | batch size：本 decode step 同时前进的请求数 | **调度选择（显式参数）** |
| S | 片上可用于跨 token 驻留的存储容量（Bytes） | **设计变量（显式参数）** |
| d_c, d_R | MLA 的压缩 latent 维度、RoPE 分离维度 | 模型配置 |

### 2.2 每 token 计算量 O_tok(C)

按"是否含参数"切分（而非按注意力/FFN 的模块边界切分），因为这条切线与搬运的两条流精确对齐：

$$O_{tok}(C) = \underbrace{2 N_{act}}_{\text{含参：QKV/O 投影、FFN、embedding 全在内}} + \underbrace{O_{attn}(C)}_{\text{无参：} qK^\top \text{与} AV\text{，随 } C \text{ 线性增长}}$$

- 含参部分：decode 是矩阵×向量，每个参与的参数恰做 1 次 MAC = 2 FLOPs，故为 2N_act，自动覆盖全部含参矩阵乘（无遗漏、无重复）。
- 无参部分（MHA/GQA）：O_attn = 4·L·d·C。**注意 H_KV 不出现——GQA 只减 KV 存储与搬运，不减注意力计算量**（计算量跟 Query 头数走）。
- 无参部分（MLA 吸收式）：O_attn = 2·L·H_Q·C·(2d_c + d_R)。MLA 的计算量高于同规模 GQA——这是"以计算换带宽"的设计本质。

### 2.3 每 token 搬运量：两条性质迥异的流

$$B_{tok}(C, B) = \underbrace{\frac{N_{act} \cdot b_W}{B}}_{\text{权重流：随 batch 摊薄}} + \underbrace{s \cdot \Phi_{read}(C)}_{\text{KV 流：每请求私有，batch 不摊薄}} + \varepsilon_{act}$$

两条流的每字节复用度（arithmetic intensity）相差可达两个数量级：权重流 = 2B/b_W（随 batch 线性增长）；KV 流 = 2H_Q/(H_KV·b_KV)（由架构固定：MHA≈2、GQA-8≈16、MLA≈数百）。**任何试图对二者之和的"比值"寻找普适规律的努力都会失败——规律在分量里，不在商里。**

### 2.4 注意力优化技术的统一参数化（四元组 + 组合规则）

所有注意力技术可写成统一规范形式，每种技术是四元组空间中的一个点：

$$B_{tok}^{KV} = s \cdot \Phi_{read}(C) + B_{idx}, \qquad O_{attn} = \omega \cdot \Phi_{read}(C) + O_{idx}, \qquad \text{KV 存占用} = s_{store} \cdot \Phi_{store}(C)$$

| 技术 | Φ_read(C) 每 token 访问条目数 | Φ_store(C) 驻留条目数 | s 每条目字节 | ω 每条目 FLOPs |
|---|---|---|---|---|
| MHA | L·C | L·C | 2·H_Q·d_h·b | 4·H_Q·d_h |
| MQA/GQA | L·C | L·C | 2·H_KV·d_h·b | 4·H_Q·d_h（不变） |
| MLA | L·C | L·C | (d_c+d_R)·b | 2·H_Q·(2d_c+d_R) |
| SWA（窗口 W） | L·min(C,W) | L·min(C,W) | 继承宿主 | 继承宿主 |
| DSA/top-k | L·k | **L·C（全存）**+ 索引 | 继承宿主 | 继承宿主 + 索引开销 |
| 块压缩（率 m） | L·C/m | L·C/m | 压缩条目字节 | 对应压缩注意力 |
| Linear/SSM | O(1) 常数 | O(1) 常数 | 状态字节 | 状态更新 FLOPs |

要点：
1. **单一"相对 MHA 的影响率"不足以刻画技术**：MLA 对字节 ×0.08 而对 FLOPs 反而上升（反向组合）；SSM 的影响率是 C 的函数而非常数；SWA 与 DSA 读流量可以相同但存占用差 C/W 倍。四元组捕获全部差异，"影响率"可作为导出量随时投影出来供汇报使用。
2. **组合规则**：量化只乘 s；分组/latent 改 (s, ω)；窗口/稀疏/压缩是对 Φ 的变换且可复合（CSA = 块压缩 ∘ top-k）；层间异构模型按层求和。
3. **KV 优化的收益是双通道的**：直接通道降每 token 流量（Φ_read·s）；间接通道降存占用（Φ_store·s_store）→ 同容量下 batch 上限提高 → 权重流摊得更薄。只看流量比会漏掉后一半收益。

### 2.5 解冻驻留约定：Q_min(S, B | C) 曲线

引入设计变量 S 后，每 token 最小片外流量是一个**驻留分配优化**的解：

$$Q_{min}(S,B \mid C) = \min_{S_W + S_{KV} \le S} \left[ \frac{\max(N_{act}b_W - S_W,\ 0)}{B} + s\,\Phi_{read}(C)\Big(1 - \frac{S_{KV}}{B\, s\,\Phi_{store}(C)}\Big)_{+} \right]$$

该曲线关于 S 分段线性、凸、单调不减小，三个区段各有明确设计含义：

```
Q_min(S)  [Bytes/token]
  │
  │●  区段 I：S≈0 —— 取值 = 标准约定点（现行"带宽算力比"即此点读数）
  │  ↘
  │    ↘   区段 II：权重部分驻留，斜率 = -1/B
  │      ↘        （SRAM 换带宽的汇率，见下）
  │        ↘
  │          ●———————————————  区段 III：S ≥ N_act·b_W
  │        （拐点）              权重流=0，平台高度 = s·Φ_read(C)
  │                             ——只能靠换注意力架构压低
  └───────────┬───────────────────► S
          N_act·b_W
```

- **区段 II 的汇率判据（PPA 接口一）**：驻留 1 字节权重 = 每 step 省 1 字节片外读取。设 TPOT = 20ms，则 1 MB SRAM ≡ 省 50 MB/s 片外带宽。两边代入面积单价（α_S 对比 α_B×50MB/s）即可判断该不该加 SRAM。对数十 GB 级云端权重，此汇率通常不划算（这是云端大模型不做权重驻留的定量原因）；对百 MB 级端侧模型则可行（端侧大 SLC 路线的依据）。
- **拐点位置（PPA 接口二）**：S = N_act·b_W 是"Groq 模式"的门槛，直接读出两种商业模式（流式 vs 驻留）的分界线。
- **原报告 β_IO 的严格化**：β_IO = Q_min(S)/Q_min(0)，从拍定系数变为曲线上的可计算读数。

### 2.6 Prefill 阶段的需求模型

Prefill 与 decode 共用同一套记账原则（含参/无参切分、流式约定），但三个结构性差异使公式形态不同：①一次处理 S_p 个 token（矩阵×矩阵而非矩阵×向量）；②不读历史 KV，而是**写出**整个 KV cache；③注意力中间量随 S_p² 增长，token 内完美复用（约定 A）不再自动成立。

**新增符号**：

| 符号 | 含义 |
|---|---|
| S_p | prompt 长度（tokens） |
| B_p | 并发 prefill 的请求数 |
| N_res | 片外驻留参数量（dense = N；MoE = N_total 全量） |
| b_act | 激活值每元素字节数 |
| n_pass(S) | 注意力计算中 KV 被重复读取的遍数（见下） |

#### (1) 计算量（整个 prefill 阶段的总量）

含参部分：每个参与的参数对 S_p 个 token 各做 1 次 MAC：

$$O_{pre}^{lin} = 2 \cdot N_{act} \cdot S_p$$

无参部分（因果注意力，第 i 个 query 只看前 i 个位置，求和得 S_p²/2）：

$$O_{pre}^{attn} = \sum_{i=1}^{S_p} 4 L d \cdot i \approx 2 \cdot L \cdot d \cdot S_p^2 \qquad \text{（MLA 吸收式：} \approx L \cdot H_Q \cdot S_p^2 \cdot (2d_c + d_R)\text{）}$$

$$\boxed{O_{pre}(S_p) = 2 N_{act} S_p + 2 L d S_p^2}$$

#### (2) 搬运量（整个 prefill 阶段的总量）

四项，性质各不相同：

$$Q_{pre}(S, B_p \mid S_p) = \underbrace{N_{res} \cdot b_W}_{\text{①权重：整个阶段读一遍}} + \underbrace{s \cdot \Phi_{store}(S_p) \cdot B_p}_{\text{②KV 写出（decode 没有的项）}} + \underbrace{L \cdot s \cdot \tfrac{S_p}{2} \cdot n_{pass}(S) \cdot B_p}_{\text{③注意力 KV 重读}} + \underbrace{\kappa \cdot L \cdot S_p \cdot d \cdot b_{act} \cdot B_p}_{\text{④激活溢出（}\kappa \in [0,2]\text{）}}$$

逐项说明：

- **① 权重项与 MoE 陷阱**：dense 模型权重读一遍即 N·b_W。但 **MoE 在 prefill 时几乎必然全量激活**——S_p 个 token 各选各的专家，并集趋于全部专家，故读取量为 N_total·b_W 而非 N_act·b_W·S_p。摊到每 token 为 N_res·b_W/(B_p·S_p)：**S_p 在 prefill 中扮演了 batch 在 decode 中的角色**（权重摊薄因子），这是两阶段的深层对称性。
- **② KV 写出**：decode 每 token 只写 1 条（可忽略），prefill 要写出全部 s·Φ_store(S_p) 字节——这是 decode 公式中不存在、prefill 必须新增的项。
- **③ 注意力重读与 n_pass(S)**：S_p 大时，一层的全部 K/V（s·S_p 字节）放不进片上，FlashAttention 类算法按 query 分块、每块对 K/V 过一遍，重读遍数 n_pass ≈ ⌈(每 query 块驻留需求 × 块数) / 有效 S⌉，量级为 Θ(S_p·s / S)。**即 prefill 的片外流量天然含 S_p²/S 项**——约定 A（token 内完美复用）在 prefill 中只是 S 足够大时的近似，Q_min 对 S 的依赖在 prefill 中比 decode 更本质（这正是 FlashAttention 论文 I/O 复杂度 Θ(S_p²d²/M) 结论的框架内表述）。
- **④ 激活溢出**：层间激活 S_p·d·b_act 在 S_p 大时超出片上容量，每层边界需写回+读入（κ=2）；片上装得下则 κ=0。

#### (3) Prefill 何时 compute-bound：一个可计算的判据

权重项主导搬运、含参项主导计算时（典型区间），prefill 的计算强度为：

$$\text{AI}_{pre} \approx \frac{2 N_{act} S_p}{N_{res}\, b_W}$$

与硬件 ridge point R（FLOPs/Byte）比较，得 **compute-bound 的最小 prompt 长度**：

$$\boxed{S_p^{*} = \frac{R \cdot b_W}{2} \cdot \frac{N_{res}}{N_{act}}}$$

代表值（R = 400 FLOPs/Byte，FP8 级硬件）：
- **Dense**（N_res = N_act）：S_p* ≈ 200 token——很短的 prompt 就 compute-bound，"prefill 必然算力受限"的常识来源；
- **MoE**（以 N_total/N_act ≈ 18 的配置为例）：S_p* ≈ 3,600 token——**短 prompt 的 MoE prefill 是带宽受限的**，"prefill 必然 compute-bound"对 MoE 不成立。这直接影响 PD 分离中 prefill 机的配比：若业务 prompt 分布集中在数千 token 以下，prefill 机同样需要高带宽配比。

#### (4) TTFT 约束的完整形式

$$t_{pre} = \max\Big(\frac{O_{pre}(S_p)}{P_c \cdot u_c^{pre}},\ \frac{Q_{pre}(S, B_p \mid S_p)}{BW \cdot u_b^{pre}}\Big) \le \text{TTFT}$$

compute-bound 区（S_p > S_p*）时退化为 P_c 下限：P_c ≥ O_pre/(u_c^pre·TTFT)。注意 u_c^pre 与 decode 的 u_c 是不同的数——prefill 是大矩阵乘，利用率显著高于 decode（这也是两阶段必须分别标定利用率的原因）。

工程补充：chunked prefill（分块预填充与 decode 混排）不改变以上总量公式，只改变 ③④ 项的 n_pass 与 κ（分块降低单次驻留需求）以及调度层的 u 值——在本框架中属第 3 层参数，不影响第 1 层结构。

---

## 3. 第 2 层：硬件成本模型（纯硬件，无 workload）

$$\text{Area}(P_c, BW, S) = \sum_P P_c^{(P)} \cdot \alpha_P \;+\; S \cdot \alpha_S \;+\; BW \cdot \alpha_B$$

| 符号 | 含义 | 单位 |
|---|---|---|
| P_c^(P) | 精度 P 下的峰值算力（设计变量） | TOPS |
| BW / Mem / S | 片外带宽 / 片外容量 / 片上驻留容量（设计变量） | — |
| α_P | 精度 P 的算力面积单价（FP16 : FP8 : FP4 ≈ 1 : 0.5 : 0.15） | mm²/TOPS |
| α_S | SRAM 面积单价 | mm²/MB |
| α_B | 带宽的 PHY+控制器面积单价 | mm²/(GB/s) |

**原报告 α 因子的物理直觉在此完整保留**——"量化越激进，同面积算力越高"正是 α_P 表达的内容。修订仅在于位置：α 是**成本系数**，进入面积目标函数；它不进入任何 workload 公式的分母（否则会产生第 8.1 节所述的恒等式问题）。功耗与 BOM 成本可同构地增加目标项。

---

## 4. 第 3 层：映射层——"有效"二字的严格定义

| 符号 | 定义 | 来源 |
|---|---|---|
| u_c（MFU） | 实际 FLOPs/s ÷ P_c | **仿真（archax S2）或实测** |
| u_b（MBU） | 实际 Bytes/s ÷ BW | 同上 |
| TPOT / TTFT | 每输出 token 时延 / 首 token 时延 SLA | 产品需求 |

decode 单步时间（计算与访存充分重叠假设，同 Roofline）：

$$t_{step} = \max\Big(\frac{B \cdot O_{tok}}{P_c\, u_c},\ \frac{B \cdot Q_{min}(S,B|C)}{BW\, u_b}\Big) \le \text{TPOT}$$

容量约束（MoE 时 N_res 为全量参数而非激活参数——原报告 7.1 节"MoE 失真"的严格表述）：

$$N_{res} \cdot b_W + B \cdot s_{store}\Phi_{store}(C) \le Mem$$

**u_c、u_b 不是拍定常数，是 (workload, 硬件, 调度) 三元组的函数**——原报告中的 θ_mem、非 GEMM 算子开销、TLB 效应、反量化惩罚全部折叠于此。它们的产出方式只有两种：真实硬件实测，或架构仿真。**对尚不存在的候选架构，仿真是唯一途径——这正是 archax S1/S2 在本框架中的位置：它是该指标体系中利用率修正项的生成器。**

---

## 5. 合成：设计问题与均衡公式

### 5.1 优化问题

$$\min_{P_c,\, BW,\, S,\, Mem,\, B} \ \text{Area}(P_c, BW, S) \quad \text{s.t.} \quad t_{step} \le \text{TPOT},\quad \text{TTFT 约束},\quad \text{容量约束}$$

batch B 进入决策变量：硬件配比的合理性依赖于"将来用多大 batch 跑"，二者必须联合决定。

### 5.2 均衡引理：最优点上算力与带宽同时踩线

反证：若最优解上带宽是瓶颈（搬 20ms > 算 10ms），则将 P_c 砍半，t_step 不变、SLA 照满足、面积净省——与最优矛盾。故只要面积单价为正，最优解必有两支路相等（真实硬件为离散档位，结论弱化为"尽可能接近相等"）：

$$\frac{B \cdot O_{tok}}{P_c\, u_c} = \frac{B \cdot Q_{min}}{BW\, u_b}$$

**直觉：闲置的资源就是白花的面积，最优设计不养闲人。**"均衡设计"不是审美偏好，是成本最优性的必要条件。

### 5.3 主公式

整理上式（B 在两侧消去，但通过 Q_min 的权重项继续起作用）：

$$\boxed{\ \Big(\frac{BW}{P_c}\Big)^{*} \;=\; \underbrace{\frac{Q_{min}(S, B^{*} \mid C)}{O_{tok}(C)}}_{\text{workload 需求比（第 1 层，可复算）}} \;\cdot\; \underbrace{\frac{u_c}{u_b}}_{\text{利用率修正（第 3 层，仿真/实测）}}\ }$$

- 左边：硬件 spec 上的带宽算力配比（每 TOPS 配多少 GB/s）。
- 右一：workload 在设计点 (S, B*) 的每 FLOP 字节需求——即标准指标从原点取值升级为在设计点取值。
- 右二：利用率修正——**这两个字就是"有效"的全部严格内容**。若算力只能发挥 40% 而带宽能发挥 80%，配带宽时按实际可用算力配，比值乘 0.5。

**该等式把 spec 表（左）、workload 数学（右一）、仿真产出（右二）三个世界接在一个等号上——这就是"Roofline + 修正参数 + PPA 关联"的严格实现。**

### 5.4 B* 的闭环与 prefill 锚点

- B* 不自由：上限由容量约束给出（KV 池装不下更多请求），经济性驱动优化器把 B 推到约束顶住为止。依赖链成环：Mem, C ⇒ B* ⇒ Q_min ⇒ (BW/P_c)* ⇒ 面积 ⇒ 反馈 Mem/S 预算。实操即在 (S, B) 网格上扫描求面积最小格。
- decode 均衡只定**比值**，绝对规模由 prefill 定：由 2.6 节，S_p > S_p* 时 prefill compute-bound，TTFT 约束给出 P_c 下限 P_c ≥ O_pre/(u_c^pre·TTFT)。**完整解法：TTFT 定 P_c 绝对值 → 均衡公式配 BW → 容量约束配 Mem → Q_min 汇率判据配 S。**注意 MoE 短 prompt 场景（S_p < S_p*）prefill 亦带宽受限，此时 TTFT 同时约束 BW，两阶段张力减弱、PD 分离收益下降——分离与否本身成为可用本框架计算的决策。
- 同一芯片被 prefill 往高 P_c 拉、被 decode 往高 BW/P_c 拉——两锚打架即 PD 分离架构的存在性证明，也是本框架应产出的第二张设计图。

---

## 6. 数值示例（全链条演算，Llama-8B 级）

配置：L=32, d=4096, H_KV=8, d_h=128, N_act=8B，权重与 KV 均 FP8（b=1），C=32K，B=64：

- O_tok = 2×8e9 + 4×32×4096×32768 ≈ 16.0 + 17.2 = **33.2 GFLOPs/token**
- B_tok = 8e9/64 + 2×32×8×128×32768 ≈ 0.125 + 2.15 = **2.27 GB/token**（KV 流主导）
- 需求比 = 2.27/33.2 ≈ **0.068 Byte/FLOP**；设 u_c/u_b = 0.5（示例值，实际由仿真产出），则 (BW/P_c)* ≈ 0.034 Byte/FLOP，即 **1 PFLOPS 应配约 34 TB/s**。

对照现实：H200 的物理配比为 4.8 TB/s / 2 PFLOPS ≈ 0.0024 Byte/FLOP——比该 workload 需求低一个数量级以上。这定量解释了长上下文 decode 为何深度带宽受限、以及 KV 压缩类技术（MLA/稀疏/压缩）的产业价值：它们把需求比往硬件可达的 0.002~0.005 Byte/FLOP 区间压。

**Prefill 侧（同模型，S_p = 8K，B_p = 1）**：

- O_pre = 2×8e9×8192 + 2×32×4096×8192² ≈ 131 + 17.6 = **149 TFLOPs**（含参项主导，注意力占 12%）
- Q_pre ≈ 权重 8 GB + KV 写出 0.54 GB + 注意力重读 ~1 GB（n_pass≈1~2）+ 激活溢出 ~4 GB ≈ **13 GB 量级**
- AI_pre ≈ 149e12 / 13e9 ≈ **1.1 万 FLOPs/Byte** ≫ ridge 400——深度 compute-bound，验证 S_p = 8K ≫ S_p* = 200（dense）。
- TTFT = 1s、u_c^pre = 0.5 时：P_c ≥ 149/0.5 ≈ **300 TFLOPS**——这就是"TTFT 定 P_c 绝对值"的具体形态。
- 反例感受 MoE 差异：若同为 8K prompt 但 N_res/N_act = 18，则 S_p* ≈ 3,600，8K 仍 compute-bound 但裕量仅约 2 倍；prompt 缩到 2K 即翻入带宽受限区。

---

## 7. 建议交付物

1. **设计平面图（核心交付物）**：横轴上下文 C（或业务 C 分布），纵轴 (BW/P_c)*（TB/s per PFLOPS），按模型/注意力架构/SLA 画曲线族。架构师读"我们该怎么配比"，投资人读"竞品配比在什么 workload 下失衡"。现有 requirement_engine 工具已具备 80% 骨架（三需求模型 + per-model SLA 参数），需增加：成本层（α 表）、batch 扫描、Q_min(S) 曲线。
2. **PD 分离判据图**：prefill 锚点与 decode 锚点在 (P_c, BW) 平面上的张力可视化。
3. **利用率修正项 (u_c, u_b) 的产出管线**：对候选架构由 archax S2 仿真生成，对现有硬件由实测标定——本指标体系与仿真器形成互为输入输出的闭环。

---

## 8. 附录：原报告需修订的技术点

以下各点均可独立验证，列出以便讨论时逐条核对。

### 8.1 "收敛于 1.0" 是精度表的恒等式

权重流主导时 EBpC ≈ θ·b_W/(2α_W)。代入原报告精度表：FP16 = 2.0/(2×1.0) = 1.0；FP8 = 1.0/(2×0.5) = 1.0；INT4 = 0.5/(2×0.15) = 1.67。乘 θ=1.04 后正是报告中 V4-Flash 1.01、V4-Pro 1.02、RWKV 1.73 等全部"理论值"。由于精度表恰取 α_P = b_P/2（FP16/FP8），**"KV 压掉后 EBpC 锁定 1.0"等价于"MAC 面积与操作数字节宽度成正比"这一表格假设本身**，与模型架构无关。将 α 移至成本层（第 3 节）后此循环自动消除。

### 8.2 第 8 章配比建议存在约三个数量级的单位换算问题

1 TB/s per PFLOPS = 0.001 Byte/OP。若 workload 需求为 1 Byte/nOP（报告计算值），对应配比应为数百 TB/s per PFLOPS 而非 1.0~1.2 TB/s。按报告自身数字，H200（≈0.005 Byte/nOP）相对该需求带宽不足两个数量级，正确推论方向与"带宽已冗余、应提升算力密度"相反。实际均衡须通过大 batch 将需求比压至硬件可达区间（见第 6 节），而报告设定 Batch=1。

### 8.3 H100 实测表内部数据不自洽

以 Llama 行为例：MBU 94.2% × 3.35 TB/s ≈ 3.16 TB/s；MFU 0.8% × 3958 TFLOPS ≈ 31.7 TFLOPS；按定义反推实测 EBpC 量级应为 ~100 Byte/nOP，表中记为 1.45，相差两个数量级，三列互不兼容。另 DeepSeek-V3 FP8 权重约 671 GB，加 36.8 GB KV，单张 80 GB H100 无法容纳，"单卡 1M 实测"物理上不可行；多卡场景的带宽核算方式不同。建议实测部分重做或删除。

### 8.4 注意力计算量公式

报告以 4·L·(d/G)·C 计注意力 FLOPs，使用了 KV 维度。QK/AV 计算量跟 Query 头数走，与 GQA 分组无关，应为 4·L·d·C（Llama 行少计 4 倍）。对 MLA 影响更大：吸收式计算下每 latent 字节被全部 H_Q 个头复用，计算量为 2·L·H_Q·C·(2d_c+d_R)，远高于按 KV 字节比例折算的值——报告公式隐含"计算量正比于 KV 字节"，恰好抹掉了 MLA"以计算换带宽"的本质区别。

### 8.5 PagedAttention 条目混淆了容量与带宽

显存碎片浪费的是**容量**（预留未用），不产生额外读**带宽**；"回收接近 30% 无用带宽"的表述应改为容量利用率提升 → 同容量下可容纳更大 batch → 间接摊薄权重带宽（即 2.4 节的双通道机制）。

### 8.6 Batch=1 设定使权重流人为主导

真实 serving 通过 continuous batching 将权重读取摊薄 1~2 个数量级，KV 流重新成为主导项。任何不含 batch 参数的 decode 带宽结论对设计不具指导性；本框架将 B 升格为显式变量并由容量/SLA 反解 B*（第 5.4 节）。

---

## 9. 本框架的可挑战假设（诚实声明）

1. max 重叠模型：真实执行是部分重叠，t_step 介于 max 与 sum 之间；
2. u_c、u_b 作为标量：真实随 (B, C) 漂移，严格做法是函数（正是仿真要扫的面）；
3. 算法类固定为 Flash 类：换算法类则 Q_min 曲线整体移动；
4. Q_min 的驻留假设为静态全有全无分配：跨层预取（换带宽不换容量）是另一条路线，严格版应对"驻留 vs 预取"再取 min；
5. KV 驻留命中率假设均匀：对动态稀疏检索（top-k 类）不成立；
6. 多芯片场景需重新定义 S 与 BW 的边界（片间链路归属），3D 堆叠（CUBE 类）的本质是压低 α_B，会翻转区段 II 的汇率判据。
